"""
Parallel video captioning for all training videos referenced in an annotation
JSON file (e.g. bcq.json).

Usage (from inside the container, or locally with the same env vars set):
    python RAG_retriever/caption_generation.py --data-json data/dataset/train/track3/bcq.json [--workers N] [--media-root data/videos]

For datasets that have no question/annotation JSON file to source a video list
from (e.g. FETV), pass --video-dir pointing at the directory of videos instead
of --data-json; the script will glob all .mp4 files under it.

Each video is captioned independently in a thread pool, so N workers run N
ModelSell API request streams concurrently.  Completed videos are skipped
automatically (caption JSON already exists).
"""

import argparse
import glob
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Allow running from the workspace root or from inside RAG_retriever/
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from captioning import Captioning

# Resolve paths relative to the repo root (parent of this RAG_retriever/ dir)
# so everything works regardless of where the repo is cloned/mounted.
_REPO_ROOT = os.path.dirname(_HERE)
DEFAULT_MEDIA_ROOT = os.path.join(_REPO_ROOT, "data", "videos")
DEFAULT_CAPTIONS_BASE_DIR = os.path.join(_REPO_ROOT, "data", "captions")

# ── helpers ──────────────────────────────────────────────────────────────────

_print_lock = threading.Lock()


def safe_print(*args, **kwargs):
    with _print_lock:
        print(*args, **kwargs, flush=True)


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


def caption_output_path(video_path: str, captioning_agent: str, base_dir: str) -> str:
    """Return the expected output JSON path for a given video."""
    subdir_map = {
        "Gemini31pro": "gemini31",
        "Gemini35Flash": "gemini35",
    }
    model_subdir = subdir_map.get(captioning_agent, "default")
    rel = _dataset_rel_path(video_path).replace(".mp4", ".json")
    return os.path.join(base_dir, model_subdir, rel)


def caption_one_video(video_path: str, captioning_agent: str, api_key: str,
                      base_dir: str, dataset: str = "default") -> str:
    """Caption a single video.  Returns a status string."""
    output_path = caption_output_path(video_path, captioning_agent, base_dir)

    # Pre-check so we don't even construct the object for already-done videos.
    if os.path.exists(output_path):
        try:
            with open(output_path) as f:
                data = json.load(f)
            if "summary" in data and "scene_description" in data:
                return f"[SKIP] {os.path.basename(video_path)}"
        except (json.JSONDecodeError, OSError):
            pass  # corrupt file — re-process it

    cap = Captioning(
        video_path_list=[video_path],
        base_dir=base_dir,
        captioning_agent=captioning_agent,
        api_key=api_key,
        dataset=dataset,
    )
    cap.run()
    return f"[DONE] {os.path.basename(video_path)}"


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Parallel training-set captioning")
    parser.add_argument("--data-json", default=None,
                        help="Path to a bcq.json-style annotation file to source video_ids from. "
                             "Mutually exclusive with --video-dir. (default: %(default)s)")
    parser.add_argument("--video-dir", default=None,
                        help="Directory to glob *.mp4 files from directly, for datasets with no "
                             "--data-json. (default: %(default)s)")
    parser.add_argument("--media-root", default=DEFAULT_MEDIA_ROOT,
                        help="Root directory where video datasets live (default: %(default)s)")
    parser.add_argument("--base-dir", default=DEFAULT_CAPTIONS_BASE_DIR,
                        help="Output base directory for caption JSONs (default: %(default)s)")
    parser.add_argument("--agent", default="Gemini35Flash",
                        choices=["Gemini35Flash", "Gemini31pro"],
                        help="Captioning model to use (default: %(default)s)")
    parser.add_argument("--workers", type=int, default=12,
                        help="Number of parallel threads (default: %(default)s)")
    args = parser.parse_args()

    api_key = os.environ.get("MODELSELL_API_KEY")
    if not api_key:
        print("Error: MODELSELL_API_KEY environment variable not set.", file=sys.stderr)
        sys.exit(1)

    if bool(args.data_json) == bool(args.video_dir):
        print("Error: specify exactly one of --data-json or --video-dir.", file=sys.stderr)
        sys.exit(1)

    media_root = args.media_root.rstrip("/")

    if args.video_dir:
        # No annotation file available (e.g. FETV) — glob videos directly.
        video_paths = sorted(glob.glob(os.path.join(args.video_dir, "**", "*.mp4"), recursive=True))
        dataset = "fetv"
    else:
        # ── load video list from annotation file ──
        with open(args.data_json) as f:
            data = json.load(f)

        unique_video_ids = sorted(set(item["video_id"] for item in data["items"]))
        video_paths = [os.path.join(media_root, vid) for vid in unique_video_ids]
        dataset = "default"

    # Filter to only videos that actually exist on disk.
    missing, todo = [], []
    for vp in video_paths:
        if os.path.exists(vp):
            todo.append(vp)
        else:
            missing.append(vp)

    if missing:
        print(f"[WARN] {len(missing)} videos not found on disk — will be skipped.")
        for p in missing[:10]:
            print(f"       {p}")
        if len(missing) > 10:
            print(f"       … and {len(missing) - 10} more.")

    print(f"Videos to process : {len(todo)}")
    print(f"Captioning agent  : {args.agent}")
    print(f"Parallel workers  : {args.workers}")
    print(f"Output base dir   : {args.base_dir}")
    print()

    done_count = 0
    skip_count = 0
    fail_count = 0
    total = len(todo)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                caption_one_video,
                vp,
                args.agent,
                api_key,
                args.base_dir,
                dataset,
            ): vp
            for vp in todo
        }

        for future in as_completed(futures):
            vp = futures[future]
            try:
                result = future.result()
                if result.startswith("[SKIP]"):
                    skip_count += 1
                else:
                    done_count += 1
                finished = done_count + skip_count + fail_count
                safe_print(f"[{finished}/{total}] {result}")
            except Exception as exc:
                fail_count += 1
                finished = done_count + skip_count + fail_count
                safe_print(f"[{finished}/{total}] [ERROR] {os.path.basename(vp)}: {exc}")

    print()
    print(f"Finished — done: {done_count}, skipped: {skip_count}, errors: {fail_count}")


if __name__ == "__main__":
    main()