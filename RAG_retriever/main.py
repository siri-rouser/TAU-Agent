import argparse
import ast
import json
import os

from captioning import Captioning
from tools import Toolkit
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from prompt_library import prompt_lib

# Resolve paths relative to the repo root (parent of this RAG_retriever/ dir)
# so everything works regardless of where the repo is cloned/mounted.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
CAPTIONS_BASE_DIR = os.path.join(_REPO_ROOT, "data", "captions")
TRACKS_BASE_DIR = os.path.join(_REPO_ROOT, "data", "tracks")

# Keep one observation roughly every FRAME_STRIDE frames (by frame id, not by
# list index) so spacing stays consistent even when a track has detection gaps.
FRAME_STRIDE = 5


def _dataset_rel_path(path_str):
    """Strip an absolute video/track path down to its dataset-relative portion.

    Videos live under .../data/videos/<dataset>/..., while captions/tracks are
    organized as .../data/<captions|tracks>/.../<dataset>/... (no "videos"
    segment). Strip through the first "data/videos/" if present (current
    layout), else the first "data/" (legacy layout without a videos/ subdir).
    """
    for anchor in ("data/videos/", "data/"):
        idx = path_str.find(anchor)
        if idx != -1:
            return path_str[idx + len(anchor):]
    return path_str.lstrip("/")

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

def preprocess(video_path_list, captioning_agent, api_key):
    base_dir = CAPTIONS_BASE_DIR
    assert captioning_agent in ["Gemini31pro", "Gemini35Flash"], "Pls select captioning agents from ['Gemini31pro', 'Gemini35Flash']."
    captioning = Captioning(
        video_path_list=video_path_list,
        base_dir=base_dir,
        captioning_agent=captioning_agent,
        api_key=api_key,
    )
    captioning.run()

def RAG_Retriever(video_path, question, api_key, captioning_agent, main_agent=None):
    assert main_agent in ["gpt-5.4"], "main_agent must be 'gpt-5.4'."
    toolkit = Toolkit(
        video_path=video_path,
        captioning_agent=captioning_agent,
    )
    prompts = prompt_lib()

    total_segments = len(toolkit.captions)

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
        model="gpt-5.4",
        api_key=api_key,
        base_url="https://api.modelsell.com/v1",
        temperature=0.0,
    )
    tools = [caption_retrieval, free_text_tracking]

    agent = create_react_agent(model, tools, prompt=prompts.retrieval_system_prompt)

    result = agent.invoke({"messages": [("human", question)]})
    return result["messages"][-1].content


def _model_subdir(captioning_agent):
    return {"Gemini31pro": "gemini31", "Gemini35Flash": "gemini35"}.get(captioning_agent, "default")


def _enrich_result(entry, captioning_agent):
    """Resolve segment captions and track bboxes referenced in retrieval_result."""
    video_path = entry["video_path"]
    try:
        parsed = json.loads(entry["retrieval_result"])
    except (json.JSONDecodeError, TypeError):
        return entry  # plain-text answer — nothing structured to enrich

    # --- captions ---
    rel = _dataset_rel_path(video_path).replace(".mp4", ".json")
    caption_path = os.path.join(CAPTIONS_BASE_DIR, _model_subdir(captioning_agent), rel)
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
    track_rel = _dataset_rel_path(video_path).replace(".mp4", "")
    tracks_dir = os.path.join(TRACKS_BASE_DIR, track_rel)
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


def main(video_path, questions, api_key, output_dir):
    captioning_agent = "Gemini35Flash"
    main_agent = "gpt-5.4"

    preprocess(video_path_list=[video_path], captioning_agent=captioning_agent, api_key=api_key)

    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, "retrieval_results.json")
    rag_info_path = os.path.join(output_dir, "rag_info.json")
    results = []

    for question in questions:
        output = RAG_Retriever(video_path, question, api_key, captioning_agent, main_agent)
        entry = {"video_path": video_path, "question": question, "retrieval_result": output}
        results.append(entry)
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[SAVED] {os.path.basename(video_path)} → {results_path}")

    enriched_entries = [_enrich_result(e, captioning_agent) for e in results]
    with open(rag_info_path, "w") as f:
        json.dump(enriched_entries, f, indent=2)
    print(f"[RAG_Info] {os.path.basename(video_path)} → {rag_info_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Example script demonstrating the TAU-Agent RAG retriever on a single video.")
    parser.add_argument("--video-path", required=True,
                         help="Path to the input video, e.g. data/videos/tar_test/example.mp4")
    parser.add_argument("--question", required=True, nargs="+",
                         help="One or more questions to ask about the video (space-separated, quote each one).")
    parser.add_argument("--output-dir", default="./RAG_retriever/results",
                         help="Directory to save retrieval_results.json and rag_info.json.")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    api_key = os.environ.get("MODELSELL_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: MODELSELL_API_KEY or OPENROUTER_API_KEY environment variable not set.")
        exit(1)

    main(video_path=args.video_path,
         questions=args.question,
         api_key=api_key,
         output_dir=args.output_dir)