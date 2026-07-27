"""
RAG_train_fetv.py — run the RAG retrieval framework over every FETV fisheye clip
in parallel, with resumable per-video outputs.

FETV differs from the in-domain TAR / PSI-VQA tasks in three ways, all handled here:

1. There is no per-clip question/annotation JSON. Every clip is asked the SAME
   unified FETV question (``prompt_lib.fetv_unified_question``), which drives the
   downstream VLM to emit the 12 structured submission fields plus a caption.
2. There is no ground truth, so the ``ground_truth_retrieval`` tool is NOT wired
   into the retrieval agent, and a FETV-specific retrieval system prompt is used.
3. Object detection uses a custom Fisheye8K YOLO model (different class ids than
   COCO). The tracker is configured with COCO<->native class maps so the existing
   prompt routing (car / motorcycle / bus / truck / pedestrian, plus vehicle
   colour via the classifier) works unchanged.

One JSON is written per video under ``<output-dir>/fetv/<video_rel>.json``.
Re-running skips clips already saved, so an interrupted run resumes safely.

Captions are assumed to already exist under ``/data/captions/...`` (generate them
with ``caption_train.py --video-dir /data/FETV`` otherwise); clips without
captions are skipped.

Usage
-----
    export MODELSELL_API_KEY=...
    python RAG_train_fetv.py                        # all FETV clips
    python RAG_train_fetv.py --gpus 0 1 --workers-per-gpu 6
    python RAG_train_fetv.py --limit 5              # smoke test: first 5 clips
"""

import argparse
import glob
import json
import multiprocessing as mp
import os
import sys
import tempfile
import time

CAPTIONING_AGENT = "Gemini35Flash"
CAPTION_SUBDIR = "gemini35"        # must match the caption model used for FETV
CAPTIONS_BASE_DIR = "/data/captions"
TRACKS_BASE_DIR = "/data/tracks"
FRAME_STRIDE = 5  # keep one observation roughly every 5 frames (matches main.py)

DEFAULT_VIDEO_DIR = "/data/FETV"
DEFAULT_OUTPUT_DIR = "/data/FETV_rag"

# Single logical task bucket for FETV (used for output pathing).
FETV_TASK = "fetv"

# Custom Fisheye8K detector and its class map. The checkpoint's classes are
# (0: Bike, 1: Bus, 2: Car, 3: Pedestrian, 4: Truck). We express detection
# requests in COCO ids (as the prompt-routing code does) and translate:
#   COCO  -> native : person/bicycle/car/motorcycle/bus/truck -> fisheye class
#   native-> COCO   : representative COCO id for each fisheye class
FETV_YOLO_MODEL = (
    "/workspace/TAU-R1/eval_FETV/runs/detect/runs/fisheye8k/"
    "yolo26x_overfit-2/weights/best.pt"
)
FETV_COCO_TO_NATIVE = {
    0: 3,   # person      -> Pedestrian
    1: 0,   # bicycle     -> Bike
    2: 2,   # car         -> Car
    3: 0,   # motorcycle  -> Bike
    5: 1,   # bus         -> Bus
    7: 4,   # truck       -> Truck
}
FETV_NATIVE_TO_COCO = {
    0: 3,   # Bike        -> motorcycle
    1: 5,   # Bus         -> bus
    2: 2,   # Car         -> car
    3: 0,   # Pedestrian  -> person
    4: 7,   # Truck       -> truck
}


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
def _caption_path(video_path):
    rel = video_path.split("/data")[-1].replace(".mp4", ".json").lstrip("/")
    return os.path.join(CAPTIONS_BASE_DIR, CAPTION_SUBDIR, rel)


def _output_path(out_dir, video_id):
    rel = video_id[:-4] if video_id.endswith(".mp4") else video_id
    return os.path.join(out_dir, FETV_TASK, rel + ".json")


def _atomic_write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f, indent=2)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


# ---------------------------------------------------------------------------
# Jobs — one per FETV clip, each carrying the single unified question
# ---------------------------------------------------------------------------
def load_video_jobs(video_dir, media_root, out_dir, limit=None):
    from prompt_library import prompt_lib
    unified_question = prompt_lib().fetv_unified_question.strip()

    video_files = sorted(glob.glob(os.path.join(video_dir, "**", "*.mp4"), recursive=True))
    if limit is not None:
        video_files = video_files[:limit]

    media_root = media_root.rstrip("/")
    jobs = []
    for vp in video_files:
        # video_id is the path relative to media_root (e.g. "FETV/001_000.mp4").
        vid = os.path.relpath(vp, media_root) if vp.startswith(media_root) else os.path.basename(vp)
        jobs.append({
            "video_id": vid,
            "video_path": vp,
            "out_dir": out_dir,
            "questions": [{
                "task": FETV_TASK,
                "question": unified_question,
            }],
        })
    return jobs


def _pending(job):
    """Questions for this clip not yet saved in its per-clip output JSON."""
    out_path = _output_path(job["out_dir"], job["video_id"])
    done = set()
    if os.path.exists(out_path):
        try:
            with open(out_path) as f:
                done.update(e["question"] for e in json.load(f).get("results", []))
        except (json.JSONDecodeError, OSError):
            pass
    return [q for q in job["questions"] if q["question"] not in done]


# ---------------------------------------------------------------------------
# Enrichment (mirrors RAG_train._enrich)
# ---------------------------------------------------------------------------
def _stride_by_frame(observations, frame_stride=FRAME_STRIDE):
    stride = max(1, int(frame_stride))
    kept, last = [], None
    for obs in observations:
        frame = obs.get("frame")
        if last is None or frame - last >= stride:
            kept.append(obs)
            last = frame
    return kept or observations


def _enrich(video_path, entry):
    try:
        parsed = json.loads(entry["retrieval_result"])
    except (json.JSONDecodeError, TypeError):
        return entry  # plain-text answer; nothing structured to enrich

    rel = video_path.split("/data")[-1].replace(".mp4", ".json").lstrip("/")
    caption_path = os.path.join(CAPTIONS_BASE_DIR, CAPTION_SUBDIR, rel)
    segment_captions, video_summary, scene_description = {}, "", ""
    if os.path.exists(caption_path):
        with open(caption_path) as f:
            captions_dict = json.load(f)
        video_summary = captions_dict.get("summary", "")
        scene_description = captions_dict.get("scene_description", "")
        ordered = [(k, v) for k, v in captions_dict.items() if k not in ("summary", "scene_description") and "_" in k]
        for seg in parsed.get("relevant_segments", []):
            sid = seg["segment_id"]
            if 0 <= sid < len(ordered):
                k, text = ordered[sid]
                segment_captions[sid] = {"key": k, "caption": text, "importance": seg.get("importance")}

    track_rel = video_path.split("/data")[-1].replace(".mp4", "").lstrip("/")
    tracks_dir = os.path.join(TRACKS_BASE_DIR, track_rel)
    relevant_track_data = []
    for rt in parsed.get("relevant_tracks", []):
        track_file = os.path.join(tracks_dir, rt["category"].replace(" ", "_")[:80] + ".json")
        if os.path.exists(track_file):
            with open(track_file) as f:
                tdata = json.load(f)
            matched = [t for t in tdata["tracks"] if t["track_id"] == rt["track_id"]]
            observations = _stride_by_frame(matched[0]["observations"] if matched else [])
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
        "scene_description": scene_description,
        "segment_captions": segment_captions,
        "relevant_tracks_data": relevant_track_data,
    }


# ---------------------------------------------------------------------------
# Worker — one GPU + one preloaded (custom-YOLO) tracker per process
# ---------------------------------------------------------------------------
_WORKER = {"tracker": None, "api_key": None}


def _init_worker(gpu_queue, api_key):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_queue.get())
    from tools import GDINO_CONFIG, GDINO_CHECKPOINT
    from tracking import Tracking
    _WORKER["api_key"] = api_key
    _WORKER["tracker"] = Tracking(
        model_config=GDINO_CONFIG,
        model_checkpoint=GDINO_CHECKPOINT,
        device="cuda",
        # Custom Fisheye8K detector + COCO<->native class maps.
        yolo_model_path=FETV_YOLO_MODEL,
        yolo_coco_to_native=FETV_COCO_TO_NATIVE,
        yolo_native_to_coco=FETV_NATIVE_TO_COCO,
    )
    print(f"[worker pid={os.getpid()}] ready on GPU {os.environ['CUDA_VISIBLE_DEVICES']}", flush=True)


def _run_retrieval(toolkit, question, api_key):
    """Build the ReAct agent + tools and retrieve evidence for one FETV clip.

    No ground_truth_retrieval tool is wired in (FETV has no ground truth), and
    the FETV-specific retrieval system prompt is used.
    """
    import ast
    from langchain_core.tools import tool
    from langchain_openai import ChatOpenAI
    from langgraph.prebuilt import create_react_agent
    from prompt_library import prompt_lib

    prompts = prompt_lib()
    total_segments = len(toolkit.captions)
    question = question.strip()

    @tool(description=(
        f"Retrieve caption segments and the video-level summary.\n\n"
        f"Input: a string tuple \"(start_segment_id, end_segment_id)\".\n"
        f"Returns the whole-video summary and captions for all segments in the requested range.\n\n"
        f"Each segment includes temporal information and frame ranges.\n\n"
        f"Total segments in this video: {total_segments} (valid IDs: 0-{total_segments - 1}).\n\n"
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
        return "\n" + toolkit.caption_retrieval(int(parsed[0]), int(parsed[1])) + "\n"

    @tool(description=(
        "Detect and track objects matching one or more text queries.\n\n"
        "Reliable object classes for this fisheye detector: car, motorcycle, bus, truck, pedestrian "
        "(two-wheelers are detected as motorcycle). A vehicle colour may be prepended (e.g. \"red car\").\n\n"
        "Input: a non-empty string. "
        "To track several objects at once, separate them with ';' (e.g. \"red car; motorcycle; pedestrian\"); "
        "all queries are resolved in a single efficient video pass.\n"
        "Returns trajectories of matched objects (frame stride = 5)."))
    def free_text_tracking(cls: str):
        if not isinstance(cls, str) or not cls.strip():
            return "\nInvalid input: input must be a non-empty string.\n"
        queries = [q.strip().lower() for q in cls.split(";") if q.strip()]
        if not queries:
            return "\nInvalid input: input must be a non-empty string.\n"
        answer = toolkit.free_text_tracking(queries if len(queries) > 1 else queries[0])
        return "\n" + answer + "\n"

    model = ChatOpenAI(
        model="gpt-5.5",
        api_key=api_key,
        base_url="https://api.modelsell.com/v1",
        temperature=0.0,
    )
    agent = create_react_agent(model, [caption_retrieval, free_text_tracking],
                               prompt=prompts.retrieval_system_prompt_fetv)
    result = agent.invoke({"messages": [("human", question)]})
    return result["messages"][-1].content


def process_video(job):
    """Retrieve evidence for one FETV clip and write one JSON.

    Returns (status, job, n_done, detail).
    """
    todo = _pending(job)
    if not todo:
        return "skip", job, 0, None
    if not os.path.exists(_caption_path(job["video_path"])):
        return "no_caption", job, 0, None

    out_dir = job["out_dir"]
    out_path = _output_path(out_dir, job["video_id"])

    # Load existing results so incremental saves stay correct.
    results = []
    if os.path.exists(out_path):
        try:
            with open(out_path) as f:
                results = json.load(f).get("results", [])
        except (json.JSONDecodeError, OSError):
            results = []

    try:
        from tools import Toolkit
        toolkit = Toolkit(video_path=job["video_path"], captioning_agent=CAPTIONING_AGENT,
                          tracker=_WORKER["tracker"])
        for q in todo:
            retrieval = _run_retrieval(toolkit, q["question"], _WORKER["api_key"])
            entry = {
                "task": q["task"],
                "question": q["question"],
                "retrieval_result": retrieval,
            }
            results.append(_enrich(job["video_path"], entry))
            # Save after each question so progress survives an interruption.
            _atomic_write_json(out_path, {
                "video_id": job["video_id"],
                "video_path": job["video_path"],
                "results": results,
            })
        return "ok", job, len(todo), None
    except Exception as e:  # noqa: BLE001 - one bad clip must not kill the pool
        return "error", job, 0, repr(e)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Parallel RAG evidence retrieval over the FETV clips.")
    p.add_argument("--video-dir", default=DEFAULT_VIDEO_DIR,
                   help="Directory of FETV .mp4 clips (default: %(default)s).")
    p.add_argument("--media-root", default="/data",
                   help="Root used to derive each clip's video_id (default: %(default)s).")
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--gpus", nargs="+", type=int, default=[0],
                   help="GPU ids inside the container (default: 0).")
    p.add_argument("--workers-per-gpu", type=int, default=6)
    p.add_argument("--limit", type=int, default=None, help="First N clips only (debug).")
    p.add_argument("--start", type=int, default=0,
                   help="First clip index to process (0-based, inclusive). Default: 0.")
    p.add_argument("--end", type=int, default=None,
                   help="Last clip index to process (exclusive). Default: all.")
    return p.parse_args()


def main():
    args = parse_args()
    api_key = os.environ.get("MODELSELL_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        sys.exit("Error: MODELSELL_API_KEY or OPENROUTER_API_KEY environment variable not set.")

    jobs = load_video_jobs(args.video_dir, args.media_root, args.output_dir, limit=args.limit)
    if args.start or args.end is not None:
        total = len(jobs)
        jobs = jobs[args.start:args.end]
        print(f"[plan] clip range [{args.start}:{args.end}] -> {len(jobs)} of {total} clip(s).")
    todo = [j for j in jobs if _pending(j)]
    n_q = sum(len(_pending(j)) for j in todo)
    print(f"[plan] {len(jobs)} clip(s); {len(todo)} with pending questions "
          f"({n_q} question(s) to run).")

    missing_cap = [j for j in todo if not os.path.exists(_caption_path(j["video_path"]))]
    if missing_cap:
        print(f"[plan] {len(missing_cap)} clip(s) have no captions and will be skipped.")
    if not todo:
        print("[plan] nothing to do.")
        return

    num_workers = max(1, len(args.gpus) * args.workers_per_gpu)
    ctx = mp.get_context("spawn")
    gpu_queue = ctx.Manager().Queue()
    for i in range(num_workers):
        gpu_queue.put(args.gpus[i % len(args.gpus)])

    print(f"[run] {num_workers} worker(s) across GPUs {args.gpus} for {len(todo)} clip(s) ...")
    counts = {}
    start = time.time()
    with ctx.Pool(num_workers, initializer=_init_worker, initargs=(gpu_queue, api_key)) as pool:
        for n, (status, job, k, detail) in enumerate(pool.imap_unordered(process_video, todo), 1):
            counts[status] = counts.get(status, 0) + 1
            extra = f" :: {detail}" if detail else (f" (+{k}q)" if k else "")
            print(f"[{n}/{len(todo)}] {status.upper():9s} "
                  f"{os.path.basename(job['video_path'])}{extra}", flush=True)

    elapsed = round(time.time() - start, 1)
    summary = " ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    print(f"[done] {elapsed}s  {summary}")


if __name__ == "__main__":
    main()

# python RAG_retriever/RAG_train_fetv.py --gpus 0 --workers-per-gpu 6
# python RAG_retriever/RAG_train_fetv.py --video-dir /data/FETV --output-dir /data/FETV_rag --gpus 0 --workers-per-gpu 8
