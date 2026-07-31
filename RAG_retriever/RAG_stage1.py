"""
RAG_train.py — run the RAG retrieval framework over every question in
``dataset/train`` in parallel, with resumable per-video outputs.

One JSON is written per video, named after the video, under
``results/train/<video_rel>.json`` (inside the repo, so it is reviewable over
SSH). That JSON holds the retrieval result for every question asked about the
video across all task files. Re-running skips questions already saved, so an
interrupted run resumes safely.

Each worker is pinned to one GPU and loads the GroundingDINO tracker once,
reusing it for every video it handles. Heavy imports are deferred into the
workers so ``CUDA_VISIBLE_DEVICES`` is set before torch touches the GPU.

Captions are assumed to already exist under ``data/captions/...`` (run
main.py's preprocess otherwise); videos without captions are skipped.

Usage
-----
    export OPENROUTER_API_KEY=...
    python RAG_train.py                       # all tasks, all items
    python RAG_train.py --split test          # use the non-validation system prompt
    python RAG_train.py --captioning-agent Gemini31pro  # use data/captions/gemini31 as evidence
    python RAG_train.py --tasks mcq open_qa   # subset of tasks
    python RAG_train.py --gpus 0 1 --workers-per-gpu 4
    python RAG_train.py --limit 5             # smoke test: 5 items per task
    python RAG_train.py --end 1000            # first 1000 videos
    python RAG_train.py --start 1000 --end 2000  # videos 1000–1999
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
import tempfile
import time

# Resolve paths relative to the repo root (parent of this RAG_retriever/ dir)
# so everything works regardless of where the repo is cloned/mounted.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)

CAPTION_SUBDIR_BY_AGENT = {"Gemini31pro": "gemini31", "Gemini35Flash": "gemini35"}
CAPTIONS_BASE_DIR = os.path.join(_REPO_ROOT, "data", "captions")
TRACKS_BASE_DIR = os.path.join(_REPO_ROOT, "data", "tracks")
DEFAULT_MEDIA_ROOT = os.path.join(_REPO_ROOT, "data", "videos")
FRAME_STRIDE = 5  # keep one observation roughly every 5 frames (matches main.py)

DEFAULT_DATASET_DIR = os.path.join(_REPO_ROOT, "data", "dataset", "train", "track3")
DEFAULT_OUTPUT_DIR = os.path.join(_REPO_ROOT, "data", "RAG_Info", "train")

ALL_TASKS = [
    "bcq", "bcq_openended", "causal_linkage", "mcq", "mcq_openended",
    "open_qa", "temporal_description",
    "temporal_localization", "video_summarization",
] 


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
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


def _caption_path(video_path, caption_subdir):
    rel = _dataset_rel_path(video_path).replace(".mp4", ".json")
    return os.path.join(CAPTIONS_BASE_DIR, caption_subdir, rel)

def _output_path(out_dir, task, video_id):
    rel = video_id[:-4] if video_id.endswith(".mp4") else video_id
    # Some datasets (e.g. tar_test) bake their own folder name into video_id
    # ("tar_test/foo.mp4"), which duplicates the same segment when --output-dir
    # already points at a dataset-specific folder (".../test/tar_test"). Strip
    # it in that case so paths don't nest "tar_test" twice. This is a no-op for
    # datasets like track3 whose video_id (e.g. "Accident-Bench/...") never
    # matches out_dir's trailing folder name.
    out_dir_name = os.path.basename(os.path.normpath(out_dir))
    head, _, tail = rel.partition("/")
    if tail and head == out_dir_name:
        rel = tail
    return os.path.join(out_dir, task, rel + ".json")


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
# Jobs — one per video, carrying all its questions across tasks
# ---------------------------------------------------------------------------
def load_video_jobs(tasks, dataset_dir, media_root, out_dir, limit=None):
    videos = {}
    for task in tasks:
        path = os.path.join(dataset_dir, f"{task}.json")
        if not os.path.exists(path):
            print(f"[warn] missing task file: {path}")
            continue
        with open(path) as f:
            data = json.load(f)
        root = data.get("media_root") or media_root
        items = data.get("items", [])
        if limit is not None:
            items = items[:limit]
        for it in items:
            vid = it["video_id"]
            job = videos.setdefault(vid, {
                "video_id": vid,
                "video_path": os.path.join(root, vid),
                "out_dir": out_dir,
                "questions": [],
            })
            job["questions"].append({
                "task": task,
                "question": it["question"].strip(),
                "gt_answer": it.get("answer"),
                "gt_reasoning": it.get("reasoning"),
            })
    return list(videos.values())


def _pending(job):
    """Questions for this video not yet saved in their per-task output JSONs."""
    out_dir = job["out_dir"]
    done = set()
    for task in {q["task"] for q in job["questions"]}:
        out_path = _output_path(out_dir, task, job["video_id"])
        if os.path.exists(out_path):
            try:
                with open(out_path) as f:
                    done.update(e["question"] for e in json.load(f).get("results", []))
            except (json.JSONDecodeError, OSError):
                pass
    return [q for q in job["questions"] if q["question"] not in done]


# ---------------------------------------------------------------------------
# Enrichment (mirrors main._enrich_result)
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


def _enrich(video_path, entry, caption_subdir):
    try:
        parsed = json.loads(entry["retrieval_result"])
    except (json.JSONDecodeError, TypeError):
        return entry  # plain-text answer; nothing structured to enrich

    rel = _dataset_rel_path(video_path).replace(".mp4", ".json")
    caption_path = os.path.join(CAPTIONS_BASE_DIR, caption_subdir, rel)
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

    track_rel = _dataset_rel_path(video_path).replace(".mp4", "")
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
# Worker — one GPU + one preloaded tracker per process
# ---------------------------------------------------------------------------
_WORKER = {"tracker": None, "api_key": None, "split": None, "captioning_agent": None, "caption_subdir": None}


def _init_worker(gpu_queue, api_key, split, captioning_agent, caption_subdir):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_queue.get())
    from tools import GDINO_CONFIG, GDINO_CHECKPOINT
    from tracking import Tracking
    _WORKER["api_key"] = api_key
    _WORKER["split"] = split
    _WORKER["captioning_agent"] = captioning_agent
    _WORKER["caption_subdir"] = caption_subdir
    _WORKER["tracker"] = Tracking(
        model_config=GDINO_CONFIG,
        model_checkpoint=GDINO_CHECKPOINT,
        device="cuda",
    )
    print(f"[worker pid={os.getpid()}] ready on GPU {os.environ['CUDA_VISIBLE_DEVICES']}", flush=True)

def _run_retrieval(toolkit, question, api_key, split, gt_answer=None, gt_reasoning=None):
    """Build the ReAct agent + tools and answer one question (cf. main.RAG_Retriever)."""
    import ast
    from langchain_core.tools import tool
    from langchain_openai import ChatOpenAI
    from langgraph.prebuilt import create_react_agent
    from prompt_library import prompt_lib

    prompts = prompt_lib()
    total_segments = len(toolkit.captions)
    question = question.strip()

    @tool(description=(
            "Ground Truth Retrieval Tool.\n"
            "Purpose: Retrieve the ground truth answer and reasoning for a specific question.\n"
            "Input: The complete question text.\n\n"
            "Output: ground truth answer and reasoning.\n"))
    def ground_truth_retrieval(input_question):
        """Retrieve ground truth for validation/comparison with RAG results."""
        normalized_input = " ".join(input_question.strip().lower().split())
        normalized_target = " ".join(question.strip().lower().split())
        if normalized_input != normalized_target:
            return (
                f"\n⚠️  Question Mismatch!\n"
                f"Expected: {question[:80]}...\n"
                f"Received: {input_question[:80]}...\n"
            )
        if gt_answer is None and gt_reasoning is None:
            return "\n❌ Ground truth not available for this question.\n"
        output = (
            f"\n{'='*70}\n"
            f"GROUND TRUTH REFERENCE\n"
            f"{'='*70}\n"
            f"Answer: {gt_answer or 'N/A'}\n\n"
            f"Reasoning:\n{gt_reasoning or 'N/A'}\n"
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
        return "\n" + toolkit.caption_retrieval(int(parsed[0]), int(parsed[1])) + "\n"

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
        return "\n" + answer + "\n"

    model = ChatOpenAI(
        model="gpt-5.4",     
        api_key=api_key,
        base_url="https://api.modelsell.com/v1",
        temperature=0.0,
    )
    system_prompt = (
        prompts.retrieval_system_prompt_no_validation if split == "train"
        else prompts.retrieval_system_prompt
    )
    tools = [caption_retrieval, free_text_tracking]
    if split == "train":
        tools.append(ground_truth_retrieval)
    agent = create_react_agent(model, tools, prompt=system_prompt)
    result = agent.invoke({"messages": [("human", question)]})
    return result["messages"][-1].content


def process_video(job):
    """Answer every pending question for one video and write one JSON per task.

    Returns (status, job, n_done, detail).
    """
    todo = _pending(job)
    if not todo:
        return "skip", job, 0, None
    if not os.path.exists(_caption_path(job["video_path"], _WORKER["caption_subdir"])):
        return "no_caption", job, 0, None

    out_dir = job["out_dir"]

    # Load existing per-task results so incremental saves stay correct.
    results_by_task = {}
    for task in {q["task"] for q in job["questions"]}:
        out_path = _output_path(out_dir, task, job["video_id"])
        if os.path.exists(out_path):
            try:
                with open(out_path) as f:
                    results_by_task[task] = json.load(f).get("results", [])
            except (json.JSONDecodeError, OSError):
                results_by_task[task] = []

    try:
        from tools import Toolkit
        toolkit = Toolkit(video_path=job["video_path"], captioning_agent=_WORKER["captioning_agent"],
                          tracker=_WORKER["tracker"])
        for q in todo:
            retrieval = _run_retrieval(toolkit, q["question"], _WORKER["api_key"], _WORKER["split"],
                                        gt_answer=q["gt_answer"], gt_reasoning=q["gt_reasoning"])
            entry = {
                "task": q["task"],
                "question": q["question"],
                "gt_answer": q["gt_answer"],
                "gt_reasoning": q["gt_reasoning"],
                "retrieval_result": retrieval,
            }
            task = q["task"]
            results_by_task.setdefault(task, []).append(_enrich(job["video_path"], entry, _WORKER["caption_subdir"]))
            # Save after each question so progress survives an interruption.
            _atomic_write_json(_output_path(out_dir, task, job["video_id"]), {
                "video_id": job["video_id"],
                "video_path": job["video_path"],
                "results": results_by_task[task],
            })
        return "ok", job, len(todo), None
    except Exception as e:  # noqa: BLE001 - one bad video must not kill the pool
        return "error", job, 0, repr(e)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Parallel RAG retrieval over dataset/train.")
    p.add_argument("--split", choices=["train", "test"], default="train",
                   help="'train' uses retrieval_system_prompt_no_validation (with ground-truth "
                        "validation tool); 'test' uses retrieval_system_prompt. Default: train.")
    p.add_argument("--captioning-agent", choices=list(CAPTION_SUBDIR_BY_AGENT), default="Gemini35Flash",
                   help="Which caption model's outputs to use as RAG evidence, i.e. the subdir under "
                        "data/captions/ (gemini35 or gemini31). Default: Gemini35Flash.")
    p.add_argument("--tasks", nargs="+", default=ALL_TASKS, help="Task stems (default: all 10).")
    p.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR)
    p.add_argument("--media-root", default=DEFAULT_MEDIA_ROOT)
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--gpus", nargs="+", type=int, default=[0],
                   help="GPU ids inside the container (default: 0 1).")
    p.add_argument("--workers-per-gpu", type=int, default=8)
    p.add_argument("--limit", type=int, default=None, help="First N items per task (debug).")
    p.add_argument("--start", type=int, default=0,
                   help="First video index to process (0-based, inclusive). Default: 0.")
    p.add_argument("--end", type=int, default=None,
                   help="Last video index to process (exclusive). Default: all. "
                        "E.g. --end 1000 processes the first 1000 videos; "
                        "--start 1000 --end 2000 processes videos 1000-1999.")
    return p.parse_args()


def main():
    args = parse_args()
    api_key = os.environ.get("MODELSELL_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        sys.exit("Error: MODELSELL_API_KEY or OPENROUTER_API_KEY environment variable not set.")

    jobs = load_video_jobs(args.tasks, args.dataset_dir, args.media_root,
                           args.output_dir, limit=args.limit)
    if args.start or args.end is not None:
        total = len(jobs)
        jobs = jobs[args.start:args.end]
        print(f"[plan] video range [{args.start}:{args.end}] → {len(jobs)} of {total} video(s).")
    todo = [j for j in jobs if _pending(j)]
    n_q = sum(len(_pending(j)) for j in todo)
    print(f"[plan] {len(jobs)} video(s); {len(todo)} with pending questions "
          f"({n_q} question(s) to run).")

    missing_cap = [j for j in todo if not os.path.exists(
        _caption_path(j["video_path"], CAPTION_SUBDIR_BY_AGENT[args.captioning_agent]))]
    if missing_cap:
        print(f"[plan] {len(missing_cap)} video(s) have no captions and will be skipped.")
    if not todo:
        print("[plan] nothing to do.")
        return

    num_workers = max(1, len(args.gpus) * args.workers_per_gpu)
    ctx = mp.get_context("spawn")
    gpu_queue = ctx.Manager().Queue()
    for i in range(num_workers):
        gpu_queue.put(args.gpus[i % len(args.gpus)])

    print(f"[run] {num_workers} worker(s) across GPUs {args.gpus} for {len(todo)} video(s) ...")
    counts = {}
    start = time.time()
    caption_subdir = CAPTION_SUBDIR_BY_AGENT[args.captioning_agent]
    with ctx.Pool(num_workers, initializer=_init_worker,
                  initargs=(gpu_queue, api_key, args.split, args.captioning_agent, caption_subdir)) as pool:
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