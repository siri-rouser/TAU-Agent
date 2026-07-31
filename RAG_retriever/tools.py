import os
import json
import cv2
from tracking import Tracking

# Resolve paths relative to the repo root (parent of this RAG_retriever/ dir)
# so everything works regardless of where the repo is cloned/mounted.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)

GDINO_CONFIG = os.path.join(_REPO_ROOT, "GroundingDINO", "groundingdino", "config", "GroundingDINO_SwinT_OGC.py")
GDINO_CHECKPOINT = os.path.join(_REPO_ROOT, "GroundingDINO", "weights", "groundingdino_swint_ogc.pth")
CAPTIONS_BASE_DIR = os.path.join(_REPO_ROOT, "data", "captions")
TRACKS_BASE_DIR = os.path.join(_REPO_ROOT, "data", "tracks")
SECONDS_PER_CAPTION = 2


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


class Toolkit:
    def __init__(self, video_path, captioning_agent=None, tracker=None, task_type=None):
        self.video_path = video_path
        self.captioning_agent = captioning_agent
        self.task_type = task_type

        if captioning_agent in ("Gemini31Pro", "Gemini31pro"):
            model_subdir = "gemini31"
        elif captioning_agent == "Gemini35Flash":
            model_subdir = "gemini35"
        else:
            model_subdir = "default"
        captions_root = os.path.join(CAPTIONS_BASE_DIR, model_subdir)

        # Derive caption JSON path using the same logic as Captioning._get_output_path
        rel = _dataset_rel_path(video_path).replace(".mp4", ".json")
        caption_path = os.path.join(captions_root, rel)

        # Load captions and build an ordered list for index-based access
        with open(caption_path, "r") as f:
            captions_dict = json.load(f)

        # Determine fps from the video file to resolve segment IDs
        cap = cv2.VideoCapture(video_path)
        self.fps = round(cap.get(cv2.CAP_PROP_FPS))
        cap.release()

        self.captions = [
            (k, v) for k, v in captions_dict.items()
            if k not in ("summary", "scene_description") and "_" in k
        ]
        self.summary = captions_dict.get("summary", "")

        # Reuse a shared tracker when provided (e.g. one preloaded model per
        # worker process) so GroundingDINO is not reloaded for every video.
        if tracker is not None:
            self.tracker = tracker
        else:
            self.tracker = Tracking(
                model_config=GDINO_CONFIG,
                model_checkpoint=GDINO_CHECKPOINT,
                device="cuda",
            )

    def caption_retrieval(self, start_segment_id: int, end_segment_id: int) -> str:
        """Return the video-level summary followed by captions for segment IDs in
        [start_segment_id, end_segment_id].
        """
        end_segment_id = min(end_segment_id, len(self.captions) - 1)

        lines = []

        if self.summary:
            lines.append(f"[Video Summary] {self.summary}")
            lines.append("")

        selected = self.captions[start_segment_id:end_segment_id + 1]
        if not selected:
            lines.append("No captions found for the requested segment range.")
            return "\n".join(lines)

        for idx, (segment_key, caption) in enumerate(selected, start=start_segment_id):
            start_frame, end_frame = segment_key.split("_")
            start_sec = int(start_frame) // self.fps
            end_sec = int(end_frame) // self.fps
            lines.append(f"[Segment {idx} | frames {start_frame}-{end_frame} | {start_sec}s-{end_sec}s] {caption}")

        return "\n".join(lines)
    
    def free_text_tracking(self, text_prompt) -> str:
        """Detect and track one or more objects in the video.

        ``text_prompt`` may be a single query string or a list of query strings.
        Multiple queries are resolved in a single video pass (YOLO runs once per
        frame on the union of classes). Each query is cached to its own JSON.

        Returns a human-readable trajectory summary. For multiple queries the
        per-query summaries are concatenated under ``=== <query> ===`` headers.
        """
        from tracking import Track

        if isinstance(text_prompt, str):
            prompts = [text_prompt]
        else:
            prompts = list(dict.fromkeys(p for p in text_prompt if p and p.strip()))
        if not prompts:
            return "No valid query provided."

        rel = _dataset_rel_path(self.video_path).replace(".mp4", "")
        save_dir = os.path.join(TRACKS_BASE_DIR, rel)

        def _save_path(prompt):
            safe_prompt = prompt.replace(" ", "_")[:80]
            return os.path.join(save_dir, f"{safe_prompt}.json")

        def _tracks_from_cache(cached):
            tracks = []
            for td in cached["tracks"]:
                if not td["observations"]:
                    continue
                obs = td["observations"]
                t = Track(
                    track_id=td["track_id"],
                    frame_idx=obs[0]["frame"],
                    box_xyxy=obs[0]["box_xyxy"],
                    confidence=obs[0]["confidence"],
                    phrase=obs[0]["phrase"],
                )
                for o in obs[1:]:
                    t.update(o["frame"], o["box_xyxy"], o["confidence"], o["phrase"])
                tracks.append(t)
            return cached["fps"], tracks

        summaries: dict[str, str] = {}
        to_run = []
        for p in prompts:
            sp = _save_path(p)
            if os.path.exists(sp):
                with open(sp, "r") as f:
                    cached = json.load(f)
                fps, tracks = _tracks_from_cache(cached)
                summaries[p] = Tracking.format_tracks(tracks, fps, frame_stride=5)
            else:
                to_run.append(p)

        if to_run:
            # Find the maximum track_id across all cached JSON files in save_dir
            # so that new tracks are assigned non-overlapping IDs.
            max_cached_id = 0
            if os.path.isdir(save_dir):
                for fname in os.listdir(save_dir):
                    if not fname.endswith(".json"):
                        continue
                    try:
                        with open(os.path.join(save_dir, fname), "r") as _f:
                            _cached = json.load(_f)
                        for _td in _cached.get("tracks", []):
                            if _td.get("observations"):
                                max_cached_id = max(max_cached_id, _td["track_id"])
                    except Exception:
                        pass

            results = self.tracker.detect_and_track_multi(
                video_path=self.video_path,
                text_prompts=to_run,
                yolo_conf_threshold=0.25,
                gd_box_threshold=0.4,
                gd_text_threshold=0.3,
                start_track_id=max_cached_id,
            )
            fps = self.tracker._fps
            os.makedirs(save_dir, exist_ok=True)
            for p in to_run:
                tracks = results.get(p, [])
                track_data = [
                    {"track_id": t.track_id, "observations": t.observations}
                    for t in tracks
                ]
                with open(_save_path(p), "w") as f:
                    json.dump({"fps": fps, "text_prompt": p, "tracks": track_data}, f, indent=2)
                summaries[p] = Tracking.format_tracks(tracks, fps, frame_stride=5)

        if len(prompts) == 1:
            return summaries[prompts[0]]
        return "\n\n".join(f"=== {p} ===\n{summaries[p]}" for p in prompts)
    
    def ground_truth_retrieval(self, input_question):
        """Retrieve the ground truth (answer and reasoning) for a given question.
        
        Args:
            input_question: The question text to search for
            
        Returns:
            A dict with 'answer' and 'reasoning' keys if found, else None
        """
        if not self.task_type:
            print("task_type not set for ground truth retrieval")
            return None
        
        # Ground-truth annotation JSONs live under data/dataset/train/track3.
        dataset_dir = os.path.join(_REPO_ROOT, "data", "dataset", "train", "track3")
        question_json = os.path.join(dataset_dir, f"{self.task_type}.json")
        
        if not os.path.exists(question_json):
            print(f"Ground truth JSON file not found: {question_json}")
            return None
        
        with open(question_json, "r") as f:
            data = json.load(f)
        
        for item in data.get("items", []):
            if item.get("question") == input_question:
                return {
                    "answer": item.get("answer"),
                    "reasoning": item.get("reasoning"),
                }
        
        print(f"Question not found in {self.task_type}.json: {input_question[:100]}")
        return None
    


class ToolkitStage2_trainset:
    def __init__(self, video_id,video_path):
        self.annotation_path = os.path.join(_REPO_ROOT, "data", "dataset", "train", "track3")
        self.video_id = video_id
        self.video_path = video_path
        self.time_task_group = ["causal_linkage", "temporal_description"]
        self.info_task_group = ["bcq", "mcq", "bcq_openended", "mcq_openended", "open_qa", "temporal_localization"]
        cap = cv2.VideoCapture(video_path)
        self.fps = round(cap.get(cv2.CAP_PROP_FPS))
        cap.release()

    def get_question_info(self):
        info = ""

        for task in self.info_task_group:
            question_json = os.path.join(self.annotation_path, f"{task}.json")
            if not os.path.exists(question_json):
                print(f"Ground truth JSON file not found: {question_json}")
                continue
            
            with open(question_json, "r") as f:
                data = json.load(f)
            
            for item in data.get("items", []):
                if item.get("video_id") == self.video_id:
                    info += f"{task} task with question: {item.get('question')}\n"
        return info
    
    def get_time_task_info(self):
        info = f"fps: {self.fps}\n"

        for task in self.time_task_group:
            question_json = os.path.join(self.annotation_path, f"{task}.json")
            if not os.path.exists(question_json):
                print(f"Ground truth JSON file not found: {question_json}")
                continue
            
            with open(question_json, "r") as f:
                data = json.load(f)
            
            for item in data.get("items", []):
                if item.get("video_id") == self.video_id:
                    info += f"{task} task with question: {item.get('question')}\n"
        return info


class ToolkitStage2_testset:
    def __init__(self, video_id, video_path):
        self.annotation_path = os.path.join(_REPO_ROOT, "data", "dataset", "test", "tar_test")
        self.video_id = video_id
        self.video_path = video_path
        self.time_task_group = ["causal_linkage", "temporal_description"]
        self.info_task_group = ["bcq", "mcq", "bcq_openended", "mcq_openended", "open_qa", "temporal_localization"]
        cap = cv2.VideoCapture(video_path)
        self.fps = round(cap.get(cv2.CAP_PROP_FPS))
        cap.release()

        test_json = os.path.join(self.annotation_path, "test.json")
        with open(test_json, "r") as f:
            data = json.load(f)
        self._items = [
            item for item in data.get("items", [])
            if item.get("video_id") == self.video_id
        ]

    def get_question_info(self):
        info = ""
        for item in self._items:
            task = item.get("task_type", "")
            if task in self.info_task_group:
                info += f"{task} task with question: {item.get('question')}\n"
        return info

    def get_time_task_info(self):
        info = f"fps: {self.fps}\n"
        for item in self._items:
            task = item.get("task_type", "")
            if task in self.time_task_group:
                info += f"{task} task with question: {item.get('question')}\n"
        return info