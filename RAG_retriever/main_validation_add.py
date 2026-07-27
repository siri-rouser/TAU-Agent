import ast
import json
import os
import random

from captioning import Captioning
from tools import Toolkit
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from prompt_library import prompt_lib

# Keep one observation roughly every FRAME_STRIDE frames (by frame id, not by
# list index) so spacing stays consistent even when a track has detection gaps.
FRAME_STRIDE = 5

def _stride_by_frame(observations, frame_stride=FRAME_STRIDE):
    """Subsample observations so consecutive kept frames are >= frame_stride apart.

    Striding on the list index ([::N]) is unreliable because tracks can drop
    frames; this keeps a stable temporal spacing based on each observation's
    "frame" value instead.
    """
    stride = max(1, int(frame_stride))
    kept = []
    last_frame = None
    for obs in observations:
        frame = obs.get("frame")
        if last_frame is None or frame - last_frame >= stride:
            kept.append(obs)
            last_frame = frame
    return kept or observations

def preprocess(video_path_list, captioning_agent, openrouter_api_key):
    base_dir = "/data/captions"
    assert captioning_agent in ["Gemini31pro", "Gemini35Flash"], "Pls select captioning agents from ['Gemini31pro', 'Gemini35Flash']."
    captioning = Captioning(
        video_path_list=video_path_list,
        base_dir=base_dir,
        captioning_agent=captioning_agent,
        openrouter_api_key=openrouter_api_key,
    )
    captioning.run()


def RAG_Retriever(video_path, question, gt_answer, gt_reasoning, task_type, openrouter_api_key, captioning_agent, main_agent=None):
    assert main_agent in ["gpt-5.4"], "main_agent must be 'gpt-5.4'."
    toolkit = Toolkit(
        video_path=video_path,
        task_type=task_type,
        captioning_agent=captioning_agent,
    )
    prompts = prompt_lib()

    total_segments = len(toolkit.captions)
    @tool(description=(
            "Ground Truth Retrieval Tool.\n"
            "Purpose: Retrieve the ground truth answer and reasoning for a specific question.\n"
            "Input: The complete question text.\n\n"
            "Output: ground truth answer and reasoning.\n"))
    def ground_truth_retrieval(input_question):
        """Retrieve ground truth for validation/comparison with RAG results."""
        # Normalize questions for flexible matching (case-insensitive, whitespace-tolerant)
        normalized_input = " ".join(input_question.strip().lower().split())
        normalized_target = " ".join(question.strip().lower().split())
        
        if normalized_input != normalized_target:
            return (
                f"\n⚠️  Question Mismatch!\n"
                f"Expected: {question[:80]}...\n"
                f"Received: {input_question[:80]}...\n"
            )
        
        gt_data = toolkit.ground_truth_retrieval(input_question)
        if not gt_data:
            return "\n❌ Ground truth not found in dataset.\n"
        
        # Format output for clarity
        output = (
            f"\n{'='*70}\n"
            f"GROUND TRUTH REFERENCE\n"
            f"{'='*70}\n"
            f"Answer: {gt_data.get('answer', 'N/A')}\n\n"
            f"Reasoning:\n{gt_data.get('reasoning', 'N/A')}\n"
            f"{'='*70}\n"
        )
        return output

    @tool(description=(
        f"Retrieve caption segments and the video-level summary.\n\n"
        f"Input: a string tuple \"(start_segment_id, end_segment_id)\".\n"
        f"Returns the whole-video summary and captions for all segments in the requested range.\n\n"
        f"Each segment includes temporal information and frame ranges.\n\n"
        f"Total segments in this video: {total_segments} (valid IDs: 0–{total_segments - 1}).\n\n"
        f"Use for event understanding, temporal context, and identifying relevant frame ranges."))
    def caption_retrieval(input_tuple):
        if isinstance(input_tuple, (list, tuple)):
            parsed = input_tuple
        else:
            try:
                parsed = ast.literal_eval(input_tuple)
            except Exception:
                return "\nInvalid input tuple!\n"
        if len(parsed) != 2:
            return "\nInvalid input tuple!\n"
        answer = toolkit.caption_retrieval(int(parsed[0]), int(parsed[1]))
        return '\n'+answer+'\n'

    @tool(description=(
        "Detect and track objects matching one or more text queries.\n\n"
        "Supports both predefined object classes and open-vocabulary queries.\n\n"
        "Input: a non-empty string."
        "To track several objects at once, separate them with ';' (e.g. \"black sedan; white suv; pedestrian\"); "
        "All queries are resolved in a single efficient video pass.\n"
        "Returns trajectories of matched objects (frame stride = 5)."))
    def free_text_tracking(cls: str):
        if not isinstance(cls, str) or not cls.strip():
            return "\nInvalid input: input must be a non-empty string.\n"
        queries = [q.strip().lower() for q in cls.split(";") if q.strip()]
        if not queries:
            return "\nInvalid input: input must be a non-empty string.\n"
        answer = toolkit.free_text_tracking(queries if len(queries) > 1 else queries[0])
        return '\n'+answer+'\n'

    question = question.strip()

    model = ChatOpenAI(
        model="openai/gpt-5.4",
        openai_api_key=openrouter_api_key,
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0.0,
    )
    tools = [caption_retrieval, free_text_tracking, ground_truth_retrieval]

    agent = create_react_agent(model, tools, prompt=prompts.retrieval_system_prompt)

    result = agent.invoke({"messages": [("human", question)]})
    return result["messages"][-1].content


def _model_subdir(captioning_agent):
    return {"Gemini31pro": "gemini31", "Gemini35Flash": "gemini35"}.get(captioning_agent, "default")


def _rag_info_path(video_path):
    """Mirror the source path under /data/RAG_Info_new/, replacing .mp4 with .json.

    e.g. /data/so-tad/test/38.mp4                          →  /data/RAG_Info_new/so-tad/test/38.json
         /data/UCF_Crimes/Videos/RoadAccidents/foo.mp4     →  /data/RAG_Info_new/UCF_Crimes/Videos/RoadAccidents/foo.json
    """
    rel = video_path.split("/data/", 1)[-1].replace(".mp4", ".json")
    return os.path.join("/data/RAG_Info_new", rel)


def _enrich_result(entry, captioning_agent):
    """Resolve segment captions and track bboxes referenced in retrieval_result."""
    video_path = entry["video_path"]
    try:
        parsed = json.loads(entry["retrieval_result"])
    except (json.JSONDecodeError, TypeError):
        return entry  # plain-text answer — nothing structured to enrich

    # --- captions ---
    rel = video_path.split("/data/")[-1].replace(".mp4", ".json")
    caption_path = os.path.join("/data/captions", _model_subdir(captioning_agent), rel)
    segment_captions = {}
    video_summary = ""
    if os.path.exists(caption_path):
        with open(caption_path) as f:
            captions_dict = json.load(f)
        video_summary = captions_dict.get("summary", "")
        ordered = [(k, v) for k, v in captions_dict.items() if k != "summary" and "_" in k]
        for seg in parsed.get("relevant_segments", []):
            sid = seg["segment_id"]
            if 0 <= sid < len(ordered):
                k, text = ordered[sid]
                segment_captions[sid] = {"key": k, "caption": text, "importance": seg.get("importance")}

    # --- tracks ---
    track_rel = video_path.split("/data")[-1].replace(".mp4", "").lstrip("/")
    tracks_dir = os.path.join("/data/tracks", track_rel)
    relevant_track_data = []
    for rt in parsed.get("relevant_tracks", []):
        safe_prompt = rt["category"].replace(" ", "_")[:80]
        track_file = os.path.join(tracks_dir, f"{safe_prompt}.json")
        if os.path.exists(track_file):
            with open(track_file) as f:
                tdata = json.load(f)
            matched = [t for t in tdata["tracks"] if t["track_id"] == rt["track_id"]]
            observations = matched[0]["observations"] if matched else []
            observations = _stride_by_frame(observations)
            relevant_track_data.append({
                "track_id": rt["track_id"],
                "category": rt["category"],
                "importance": rt.get("importance"),
                "observations": observations,
            })

    return {
        **entry,
        "relevant_frame_ranges": parsed.get("relevant_frame_ranges", []),
        "video_summary": video_summary,
        "segment_captions": segment_captions,
        "relevant_tracks_data": relevant_track_data,
    }


def main(video_path_list, video_question_list, gt_answer_list, gt_reasoning_list, task_type_list, openrouter_api_key,
         save_results=False, save_dir="/workspace/TAU-R1/RAG_retriever/results"):
    captioning_agent = "Gemini35Flash"
    main_agent = "gpt-5.4"

    preprocess(video_path_list=video_path_list, captioning_agent=captioning_agent, openrouter_api_key=openrouter_api_key)

    save_path = None
    results = []
    done_keys = set()

    if save_results:
        out_dir = os.path.join(save_dir, main_agent)
        os.makedirs(out_dir, exist_ok=True)
        save_path = os.path.join(out_dir, "retrieval_results.json")
        if os.path.exists(save_path):
            with open(save_path) as f:
                results = json.load(f)
            done_keys = {(r["video_path"], r["question"]) for r in results}

    for vp, q, gt_a, gt_r, qtype in zip(video_path_list, video_question_list, gt_answer_list, gt_reasoning_list, task_type_list):
        if (vp, q) in done_keys:
            print(f"[SKIP] {os.path.basename(vp)}")
            continue
        output = RAG_Retriever(vp, q, gt_a, gt_r, qtype, openrouter_api_key, captioning_agent, main_agent)
        entry = {"video_path": vp, "question": q, "retrieval_result": output, "gt_answer": gt_a, "gt_reasoning": gt_r, "task_type": qtype}
        results.append(entry)
        done_keys.add((vp, q))
        if save_results:
            with open(save_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"[SAVED] {os.path.basename(vp)} → {save_path}")

            video_entries = [r for r in results if r["video_path"] == vp]
            enriched_entries = [_enrich_result(e, captioning_agent) for e in video_entries]
            out_path = _rag_info_path(vp)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(enriched_entries, f, indent=2)
            print(f"[RAG_Info] {os.path.basename(vp)} → {out_path}")

if __name__ == '__main__':
    openrouter_api_key = os.environ.get('OPENROUTER_API_KEY')
    if not openrouter_api_key:
        print("Error: OPENROUTER_API_KEY environment variable not set.")
        exit(1)

    selected_video_path_list = [
        "/data/Accident-Bench/land_space/medium/videos/MxiC2UdOzE8.mp4",
        "/data/TAD/Normal_239.mp4",
        "/data/TAD-benchmark/train/accident_小于1M/video6.mp4",
        "/data/UCF_Crimes/Videos/RoadAccidents/RoadAccidents006_x264.mp4",
        "/data/so-tad/test/45.mp4",
        "/data/TAD/01_Accident_082.mp4",
        "/data/UCF_Crimes/Videos/RoadAccidents/RoadAccidents132_x264.mp4",
        "/data/TAD-benchmark/train/normal/20220620_acci-bg33.mp4",
        "/data/TAD-benchmark/train/accident_小于1M/video153.mp4",
        "/data/TAD/Normal_199.mp4",
        "/data/TAD-benchmark/train/accident_大于1M小于5M/videox12_1.mp4",
        "/data/so-tad/train/136.mp4",
        "/data/Accident-Bench/land_space/long/videos/000004.mp4",
        "/data/TAD-benchmark/train/accident_大于1M小于5M/video27.mp4",
        "/data/Accident-Bench/land_space/long/videos/000001.mp4",
        "/data/Accident-Bench/land_space/long/videos/000002.mp4",
        "/data/Accident-Bench/land_space/long/videos/000039.mp4",
        "/data/so-tad/train/227.mp4",
        "/data/so-tad/test/65.mp4",
        "/data/so-tad/train/151.mp4",
        "/data/so-tad/train/126.mp4",
        "/data/UCF_Crimes/Videos/RoadAccidents/RoadAccidents089_x264.mp4",
    ]

    video_path_list = []
    video_question_list = []
    gt_answer_list = []
    gt_reasoning_list = []
    task_type_list = []

    random_seed = 42
    question_types = [
        "bcq",
        "bcq_openended",
        "causal_linkage",
        "mcq",
        "mcq_openended",
        "open_qa",
        "scene_description",
        "temporal_description",
        "temporal_localization",
        "video_summarization",
    ]

    rng = random.Random(random_seed)
    dataset_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "dataset", "train_backup"))

    question_bank = {}
    for qtype in question_types:
        qpath = os.path.join(dataset_dir, f"{qtype}.json")
        if not os.path.exists(qpath):
            print(f"[WARN] Missing question file: {qpath}")
            question_bank[qtype] = []
            continue
        with open(qpath) as f:
            data = json.load(f)
        question_bank[qtype] = data.get("items", [])

    for video_path in selected_video_path_list:
        video_id = video_path.split("/data/", 1)[-1]
        sampled_types = rng.sample(question_types, k=5)

        for qtype in sampled_types:
            matches = [
                item
                for item in question_bank.get(qtype, [])
                if item.get("video_id") == video_id and item.get("question")
            ]
            if not matches:
                print(f"[WARN] No {qtype} question for {video_id}")
                continue

            picked = rng.choice(matches)
            video_path_list.append(video_path)
            video_question_list.append(picked["question"])
            gt_answer_list.append(picked.get("answer"))
            gt_reasoning_list.append(picked.get("reasoning"))
            task_type_list.append(qtype)

    print(f"Prepared {len(video_question_list)} questions from {len(selected_video_path_list)} videos.")


    main(
        video_path_list=video_path_list,
        video_question_list=video_question_list,
        gt_answer_list=gt_answer_list,
        gt_reasoning_list=gt_reasoning_list,
        task_type_list=task_type_list,
        openrouter_api_key=openrouter_api_key,
        save_results=True,
    )
