import os
import cv2
import json
import math
import base64
import time
from prompt_library import prompt_lib
import requests

# Resolve paths relative to the repo root (parent of this RAG_retriever/ dir)
# so everything works regardless of where the repo is cloned/mounted.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
DEFAULT_CAPTIONS_BASE_DIR = os.path.join(_REPO_ROOT, "data", "captions")


def _dataset_rel_path(path_str):
    """Strip an absolute video path down to its dataset-relative portion.

    Videos live under .../data/videos/<dataset>/...; captions are organized
    as .../data/captions/<agent>/<dataset>/... (no "videos" segment). Strip
    through the first "data/videos/" if present (current layout), else the
    first "data/" (legacy layout without a videos/ subdir).
    """
    for anchor in ("data/videos/", "data/"):
        idx = path_str.find(anchor)
        if idx != -1:
            return path_str[idx + len(anchor):]
    return path_str.lstrip("/")


class Captioning:
    def __init__(self, video_path_list, base_dir=None, captioning_agent=None, api_key=None,
                 dataset="default"):
        base_dir = base_dir or DEFAULT_CAPTIONS_BASE_DIR
        self.prompts = prompt_lib()
        self.captioning_agent = captioning_agent
        self.video_path_list = video_path_list
        self.api_key = api_key
        # Selects which prompt set to use: "default" (TAR/PSI-VQA style anomaly captions) or
        # "fetv" (fisheye traffic-violation captions).
        self.dataset = dataset
        # Save captions under a model-specific subdir to avoid mixing outputs.
        self.base_dir = os.path.join(base_dir, self._model_output_subdir(captioning_agent))
        self.seconds_per_caption = 2   # each caption covers a 2-second window
        self.frames_per_caption = 4   # 4 frames sampled from each 2-second window

        if "Gemini" not in self.captioning_agent:
            raise ValueError(f"Unsupported captioning agent: {captioning_agent}")

    @staticmethod
    def _model_output_subdir(captioning_agent):
        if captioning_agent == "Gemini31pro":
            return "gemini31"
        if captioning_agent == "Gemini35Flash":
            return "gemini35"
        return "default"

    def _get_output_path(self, video_path):
        """Mirror the source path structure under base_dir, replacing .mp4 with .json."""
        rel = _dataset_rel_path(video_path).replace(".mp4", ".json")
        return os.path.join(self.base_dir, rel)

    def _extract_frames(self, cap, fps, caption_id):
        """Seek and read 4 evenly spaced frames from the caption_id-th 2-second window."""
        start_frame = caption_id * fps * self.seconds_per_caption
        frame_interval = (fps * self.seconds_per_caption) // self.frames_per_caption
        frames = []
        for i in range(self.frames_per_caption):
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame + i * frame_interval)
            success, frame = cap.read()
            if success:
                frames.append(frame)
        return frames

    def _frame_to_base64(self, frame):
        """Encode a BGR frame as a base64 JPEG string."""
        _, buffer = cv2.imencode(".jpg", frame)
        return base64.b64encode(buffer).decode("utf-8")

    def _gemini_model_id(self):
        """Return the ModelSell Gemini-native model id for the current agent."""
        if self.captioning_agent == "Gemini31pro":
            return "gemini-3.1-pro-preview"
        if self.captioning_agent == "Gemini35Flash":
            return "gemini-3.5-flash"
        return "qwen-3.5"

    def _image_part(self, frame):
        """Build a Gemini-native inline image part from a BGR frame."""
        return {
            "inline_data": {
                "mime_type": "image/jpeg",
                "data": self._frame_to_base64(frame),
            }
        }

    def _call_gemini(self, parts):
        """POST Gemini-native `parts` to ModelSell generateContent and return the text."""
        model_id = self._gemini_model_id()
        response = requests.post(
            url=f"https://www.modelsell.com/v1beta/models/{model_id}:generateContent",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            data=json.dumps({
                "contents": [{"role": "user", "parts": parts}],
            }),
        )
        response.raise_for_status()
        return response.json()["candidates"][0]["content"]["parts"][0]["text"].strip()

    def _generate_caption(self, frames):
        """Send frames to the captioning model and return a short caption for the segment."""
        caption_prompt = (
            self.prompts.fetv_video_caption_prompt if self.dataset == "fetv"
            else self.prompts.video_caption_prompt
        )
        parts = [{"text": caption_prompt}]
        parts.extend(self._image_part(frame) for frame in frames)
        return self._call_gemini(parts)

    def _generate_scene_description(self, cap, total_frames):
        """Extract 4 evenly spaced frames from the whole video and generate a scene description."""
        frame_indices = [int((total_frames / 4) * (i + 0.5)) for i in range(4)]
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            success, frame = cap.read()
            if success:
                frames.append(frame)

        if not frames:
            return ""

        parts = [{"text": self.prompts.scene_description_prompt}]
        parts.extend(self._image_part(frame) for frame in frames)
        return self._call_gemini(parts)

    def _generate_summary_caption(self, captions):
        """Generate one whole-video summary from chronological segment captions."""
        if not captions:
            return ""

        def _segment_start(segment):
            try:
                return int(segment.split("_")[0])
            except (ValueError, IndexError):
                return 10**12

        ordered_segments = sorted(captions.keys(), key=_segment_start)
        ordered_captions = [captions[s] for s in ordered_segments if captions[s].strip()]
        if not ordered_captions:
            return ""

        content = "\n".join(
            f"{idx + 1}. {cap_text}" for idx, cap_text in enumerate(ordered_captions)
        )
        summary_prompt = (
            self.prompts.fetv_video_summary_prompt if self.dataset == "fetv"
            else self.prompts.video_summary_prompt
        )
        parts = [{"text": f"{summary_prompt}\n{content}"}]
        return self._call_gemini(parts)

    def generate_captions_for_all_videos(self):
        for video_path in self.video_path_list:
            output_path = self._get_output_path(video_path)
            if os.path.exists(output_path):
                with open(output_path, "r") as f:
                    existing_data = json.load(f)

                needs_summary = "summary" not in existing_data
                needs_scene = "scene_description" not in existing_data

                if not (needs_summary or needs_scene):
                    print(f"Skipping {os.path.basename(video_path)}: caption JSON already complete at {output_path}")
                    continue

                updated = False

                if needs_summary:
                    existing_captions = {
                        k: v for k, v in existing_data.items()
                        if isinstance(v, str) and k not in ("summary", "scene_description")
                    }
                    existing_data["summary"] = self._generate_summary_caption(existing_captions)
                    updated = True
                    print(f"Backfilled whole-video summary → {output_path}")

                if needs_scene:
                    cap = cv2.VideoCapture(video_path)
                    if not cap.isOpened():
                        print(f"Error: Unable to open video file for scene description: {video_path}")
                    else:
                        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        existing_data["scene_description"] = self._generate_scene_description(cap, total_frames)
                        cap.release()
                        updated = True
                        print(f"Backfilled scene description → {output_path}")

                if updated:
                    with open(output_path, "w") as f:
                        json.dump(existing_data, f, indent=2)
                continue

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Error: Unable to open video file: {video_path}")
                continue

            fps = round(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            total_captions = math.ceil(total_frames / (fps * self.seconds_per_caption))

            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            captions = {}
            start_time = time.time()

            for caption_id in range(total_captions):
                frames = self._extract_frames(cap, fps, caption_id)
                if not frames:
                    print(f"  Warning: no frames extracted for window {caption_id}, skipping.")
                    continue

                start_f = caption_id * fps * self.seconds_per_caption
                end_f = min((caption_id + 1) * fps * self.seconds_per_caption, total_frames)
                segment = f"{start_f}_{end_f}"

                text = self._generate_caption(frames)
                captions[segment] = text
                print(f"  id: {caption_id}, frames: {segment}, caption: {text}")

            scene_des = self._generate_scene_description(cap, total_frames)
            cap.release()
            elapsed = round(time.time() - start_time, 3)
            base_name = os.path.basename(video_path)
            print(f"Captioning done for {base_name} in {elapsed}s → {output_path}")

            with open(output_path, "w") as f:
                json.dump(captions, f, indent=2)

            summary_text = self._generate_summary_caption(captions)
            captions_with_summary = dict(captions)
            captions_with_summary["summary"] = summary_text
            captions_with_summary["scene_description"] = scene_des
            with open(output_path, "w") as f:
                json.dump(captions_with_summary, f, indent=2)
            print(f"Saved whole-video summary into JSON → {output_path}")

    def run(self):
        self.generate_captions_for_all_videos()
