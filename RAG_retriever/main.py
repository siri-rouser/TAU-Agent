import ast
import json
import os

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

def RAG_Retriever(video_path, question, openrouter_api_key, captioning_agent, main_agent=None):
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
        model="openai/gpt-5.4",
        openai_api_key=openrouter_api_key,
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0.0,
    )
    tools = [caption_retrieval, free_text_tracking]

    agent = create_react_agent(model, tools, prompt=prompts.retrieval_system_prompt)

    result = agent.invoke({"messages": [("human", question)]})
    return result["messages"][-1].content


def _model_subdir(captioning_agent):
    return {"Gemini31pro": "gemini31", "Gemini35Flash": "gemini35"}.get(captioning_agent, "default")


def _rag_info_path(video_path):
    """Mirror the source path under /data/RAG_Info/, replacing .mp4 with .json.

    e.g. /data/so-tad/test/38.mp4                          →  /data/RAG_Info/so-tad/test/38.json
         /data/UCF_Crimes/Videos/RoadAccidents/foo.mp4     →  /data/RAG_Info/UCF_Crimes/Videos/RoadAccidents/foo.json
    """
    rel = video_path.split("/data/", 1)[-1].replace(".mp4", ".json")
    return os.path.join("/data/RAG_Info", rel)


def _enrich_result(entry, captioning_agent):
    """Resolve segment captions and track bboxes referenced in retrieval_result."""
    video_path = entry["video_path"]
    try:
        parsed = json.loads(entry["retrieval_result"])
    except (json.JSONDecodeError, TypeError):
        return entry  # plain-text answer — nothing structured to enrich

    # --- captions ---
    rel = video_path.split("/data")[-1].replace(".mp4", ".json").lstrip("/")
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


def main(video_path_list, video_question_list, openrouter_api_key,
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
        # if os.path.exists(save_path):
        #     with open(save_path) as f:
        #         results = json.load(f)
        #     done_keys = {(r["video_path"], r["question"]) for r in results}

    for vp, q in zip(video_path_list, video_question_list):
        if (vp, q) in done_keys:
            print(f"[SKIP] {os.path.basename(vp)}")
            continue
        output = RAG_Retriever(vp, q, openrouter_api_key, captioning_agent, main_agent)
        entry = {"video_path": vp, "question": q, "retrieval_result": output}
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
        "/data/HTV/cctv052x2004080620x00105.mp4",
        "/data/TAD/Normal_239.mp4",
        "/data/TAD-benchmark/train/accident_小于1M/video6.mp4",
        "/data/UCF_Crimes/Videos/RoadAccidents/RoadAccidents006_x264.mp4",
        "/data/Vad-R1/Vad-Reasoning-RL/SH/Abnormal/07_0009.mp4",
        "/data/barbados_traffic_challenge/normanniles1/normanniles1_2025-10-22-11-26-45.mp4",
        "/data/so-tad/test/45.mp4",
        "/data/tar_test/v=owY0Fg-1Yhk_2-25_2-35.mp4",
        "/data/TAD/01_Accident_082.mp4",
        "/data/so-tad/train/276.mp4",
        "/data/so-tad/train/634.mp4",
        "/data/barbados_traffic_challenge/normanniles1/normanniles1_2025-10-24-10-50-45.mp4",
        "/data/UCF_Crimes/Videos/RoadAccidents/RoadAccidents132_x264.mp4",
        "/data/TAD-benchmark/train/normal/20220620_acci-bg33.mp4",
        "/data/TAD-benchmark/train/normal/20230818_acci-bg_33.mp4",
        "/data/TAD-benchmark/train/accident_小于1M/video153.mp4",
        "/data/so-tad/test/1129.mp4",
        "/data/TAD/Normal_199.mp4",
        "/data/Vad-R1/Vad-Reasoning-SFT/SH/05_0017.mp4",
        "/data/TAD-benchmark/train/accident_大于1M小于5M/videox12_1.mp4",
        "/data/so-tad/train/136.mp4",
        "/data/so-tad/test/1589.mp4",
        "/data/barbados_traffic_challenge/normanniles1/normanniles1_2025-10-20-13-23-45.mp4",
        "/data/tar_test/v=-3nwOfm1Pdk_0-43_0-50.mp4",
    ]

    selected_question_pair_list = [
        [
            "Which of the following best describes the physical motion of the white sedan immediately following the T-bone collision?\nA. It flipped over once and landed on its roof on the right side.\nB. It skidded straight forward and crashed into a traffic pole.\nC. It performed a full 360-degree spin before coming to a stop.\nD. It was pushed backward toward the left side of the frame.\nAnswer with a single letter.",
            "Does the gray sedan drive straight through the intersection without turning?\nAnswer with Yes or No followed by a brief explanation.",
        ],
        [
            "What happened in the video between 00:00.00 and 00:02.00?",
            "Explain the relationship between the event at 00:00.00 and the situation at 00:02.00.",
        ],
        [
            "Does the traffic flow maintain a generally steady, single-file movement throughout the video?\nAnswer with Yes or No followed by a brief explanation.",
            "What are the key events and traffic conditions observed in this video?",
        ],
        [
            "Do two different vehicles collide with the crash barrier in the video?\nAnswer with Yes or No.",
            "Explain the relationship between the event at 00:00.00 and the situation at 00:02.00.",
        ],
        [
            "Is the parked black sedan involved in the collision?\nAnswer with Yes or No followed by a brief explanation.",
            "Does a white car experience a collision in the video?\nAnswer with Yes or No.",
        ],
        [
            "What is the direct consequence of the pedestrian performing a flip in the middle of the road?",
            "What events occur in this video?",
        ],
        [
            "What happened in the video between 00:00:35 and 00:00:40?",
            "Does a yellow bus pass through the intersection on the main road?\nAnswer with Yes or No followed by a brief explanation.",
        ],
        [
            "Does a black car collide with the median?\nAnswer with Yes or No.",
            "What are the key events and any anomalies observed in this traffic video?",
        ],
        [
            "Explain the relationship between the event at 00:04.67 and the situation at 00:07.57.",
            "What happened in the video between 00:04.67 and 00:07.06?",
        ],
        [
            "Does the black car collide with the median barrier?\nAnswer with Yes or No followed by a brief explanation.",
            "When does the sequence of collisions involving the black car, the black bus, and the median barrier occur?\nAnswer with the start and end timestamps in the format {\"start\": \"MM:SS\", \"end\": \"MM:SS\"}.",
        ],
        [
            "What are the key events and any anomalies observed in this traffic video?",
            "Which of the following best describes the physical aftermath of the collision involving the gray sedan?\nA. The vehicle remained immobilized and stalled against the divider.\nB. The vehicle overturned and slid into the adjacent fourth lane.\nC. The vehicle bounced off the divider and returned to its original lane.\nD. The vehicle continued moving forward after striking the traffic sign.\nAnswer with a single letter.",
        ],
        [
            "When does the black car on the right side of the road overtake other vehicles at a high speed?\nAnswer with the start and end timestamps in the format {\"start\": \"MM:SS\", \"end\": \"MM:SS\"}.",
            "How does the behavior of the black car appearing on the right side of the road between 00:05.00 and 00:10.00 differ from the other vehicles in the scene?",
        ],
        [
            "Which of the following best describes the traffic flow dynamics and signal state at this intersection?\nA. Stop-and-go traffic resulting from a malfunctioning traffic signal.\nB. Heavy directional flow from the right, regulated by stop signs.\nC. High congestion with traffic managed by a timed signal cycle.\nD. Moderate, free-flowing traffic at an uncontrolled, yield-based intersection.\nAnswer with a single letter.",
            "What are the key events and traffic patterns observed in this video?",
        ],
        [
            "When does the head-on collision between the white van and the motorcycle occur at the intersection?\nAnswer with the start and end timestamps in the format {\"start\": \"MM:SS\", \"end\": \"MM:SS\"}.",
            "Does the collision occur because the motorcycle ran a red light?\nAnswer with Yes or No.",
        ],
        [
            "Does any vehicle perform a lane change during the recorded period?\nAnswer with Yes or No.",
            "When is the black van visible driving on the right side of the highway?\nAnswer with the start and end timestamps in the format {\"start\": \"MM:SS\", \"end\": \"MM:SS\"}.",
        ],
        [
            "Is the traffic flow on the highway free-flowing throughout the video?\nAnswer with Yes or No.",
            "Explain the relationship between the event at 00:00:04 and the situation at 00:00:08.",
        ],
        [
            "Is there a person on the highway near a stalled vehicle?\nAnswer with Yes or No followed by a brief explanation.",
            "Is there a collision between vehicles in the video?\nAnswer with Yes or No followed by a brief explanation.",
        ],
        [
            "How does the road layout and environmental lighting contribute to the visibility and safety of the intersection at night?",
            "Is there any pedestrian activity at the marked crosswalks during the video?\nAnswer with Yes or No.",
        ],
        [
            "Describe the road layout and environmental surroundings visible in this traffic video.",
            "Do the moving vehicles navigate through a narrow path created by parked vehicles?\nAnswer with Yes or No.",
        ],
        [
            "Based on the sequence of events, how did the pedestrian in the black shirt react to the falling object?\nA. He pointed upward to warn other pedestrians before the object landed on the path.\nB. He walked past the man in the red shirt and ignored the object falling from above.\nC. He ran into the frame while looking upward and caught the object with his arms.\nD. He stopped abruptly and stepped aside to avoid being hit by the object.\nAnswer with a single letter.",
            "Based on the sequence of events, how did the pedestrian in the black shirt react to the falling object?\nA. He pointed upward to warn other pedestrians before the object landed on the path.\nB. He walked past the man in the red shirt and ignored the object falling from above.\nC. He ran into the frame while looking upward and caught the object with his arms.\nD. He stopped abruptly and stepped aside to avoid being hit by the object.\nAnswer with the correct option letter followed by a brief explanation.",
        ],
        [
            "Explain the relationship between the event at 00:02.00 and the situation at 00:05.00.",
            "Which vehicle is at fault for the collision, and what specific traffic violation occurred?",
        ],
        [
            "Which of the following best describes the sequence of events involving the black SUV?\nA. It flips upside down in its own lane, slides into the right road barrier, and finally comes to rest at the central divider.\nB. It loses control, crashes into the right road barrier, deflects across lanes into the central divider, and then flips upside down.\nC. It crashes into the right road barrier and immediately flips upside down without moving toward the central divider.\nD. It crashes into the central divider, deflects toward the right road barrier, and then flips upside down.\nAnswer with a single letter.",
            "Does the black SUV flip over after the collisions?\nAnswer with Yes or No followed by a brief explanation.",
        ],
        [
            "What are the key events and traffic conditions observed in this video?",
            "When does the red car drive straight through the highway lanes?\nAnswer with the start and end timestamps in the format {\"start\": \"MM:SS\", \"end\": \"MM:SS\"}.",
        ],
        [
            "Do vehicles entering the roundabout from the top approach predominantly turn right?\nAnswer with Yes or No followed by a brief explanation.",
            "How is the traffic flow managed at this intersection, and what is the resulting pattern for vehicles entering from the top approach?",
        ],
        [
            "Is there a T-bone collision in the video?\n\nAnswer with only Yes or No.",
            "Does the black SUV run a red light in the video?\n\nAnswer with only Yes or No.",
        ],
    ]

    # selected_video_path_list = ["/data/Accident-Bench/land_space/long/videos/000001.mp4"]
    # selected_question_pair_list = [["Does a white van collide with a white SUV in the video?\nAnswer with Yes or No.", "Does a white van collide with a white SUV in the video?\nAnswer with Yes or No."]]
    
    if len(selected_question_pair_list) != len(selected_video_path_list):
        raise ValueError("selected_question_pair_list must have one 2-question entry per selected video")

    video_path_list = []
    video_question_list = []
    for video_path, question_pair in zip(selected_video_path_list, selected_question_pair_list):
        if len(question_pair) != 2:
            raise ValueError(f"Each question entry must contain exactly 2 questions: {video_path}")
        video_path_list.extend([video_path, video_path])
        video_question_list.extend(question_pair)
    
    main(video_path_list=video_path_list, 
         video_question_list=video_question_list,
         openrouter_api_key=openrouter_api_key,
         save_results=True)