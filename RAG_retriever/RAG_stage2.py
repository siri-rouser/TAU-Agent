"""
RAG_stage2.py — Stage 2 RAG: extract video-level context from questions and
enrich per-task training JSONs with that context.

For each video in --media-root, a ReAct agent reads the questions associated
with the video and produces:
  - factual_information: key facts deduced from question context
  - potential_information: additional useful details
  - frame_ranges: relevant frame windows extracted from timestamps in questions

Outputs
-------
  <output_dir>/<stem>.json    raw retrieval cache (resumability marker for the LLM call)
      fields: factual_information, potential_information, frame_ranges
  <stage1_rag_dir>/... .json  the matching stage1 (RAG_stage1.py) JSON, updated *in place*
      adds stage2_factual, stage2_potential; merges relevant_frame_ranges

Resumable: re-running skips videos whose stage1 JSON already carries stage2_factual.

Usage
-----
    export MODELSELL_API_KEY=...
    python RAG_stage2.py                              # train split, all videos
    python RAG_stage2.py --split test                 # test split (tar_test by default)
    python RAG_stage2.py --workers 16                 # 16 parallel workers
    python RAG_stage2.py --limit 5                    # smoke test
    python RAG_stage2.py --start 1000 --end 2000      # videos 1000-1999
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

DEFAULT_MEDIA_ROOT = os.path.join(_REPO_ROOT, "data", "videos")

# Per-split defaults for --json-dir / --stage1-rag-dir / --output_dir (each
# individually overridable). "train" enumerates videos from the train bcq.json
# task file; "test" defaults to the tar_test test set.
SPLIT_DEFAULTS = {
    "train": {
        "json_dir": os.path.join(_REPO_ROOT, "data", "dataset", "train", "track3", "bcq.json"),
        "stage1_rag_dir": os.path.join(_REPO_ROOT, "data", "RAG_Info", "train", "bcq"),
        "output_dir": os.path.join(_REPO_ROOT, "data", "RAG_Stage2", "train"),
    },
    "test": {
        "json_dir": os.path.join(_REPO_ROOT, "data", "dataset", "test", "tar_test", "test.json"),
        "stage1_rag_dir": os.path.join(_REPO_ROOT, "data", "RAG_Info", "test", "tar_test", "test"),
        "output_dir": os.path.join(_REPO_ROOT, "data", "RAG_Stage2", "test"),
    },
}

STAGE2_RAG_DIR = SPLIT_DEFAULTS["train"]["output_dir"]
AICITY_TRAIN_DIR = SPLIT_DEFAULTS["train"]["stage1_rag_dir"]
AICITY_TEST_DIR = SPLIT_DEFAULTS["test"]["stage1_rag_dir"]
AICITY_TRAIN_2STAGE_DIR = os.path.join(_REPO_ROOT, "data", "aicity_train_2stage")
SPLIT = "train"

ALL_TASKS = [
    "bcq", "bcq_openended", "causal_linkage", "mcq", "mcq_openended",
    "open_qa", "temporal_description",
    "temporal_localization", "video_summarization","test"
]


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------
def _output_path(out_dir, video_id):
    rel = video_id[:-4] if video_id.endswith(".mp4") else video_id
    return os.path.join(out_dir, rel + ".json")


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
# Jobs — one per unique video from bcq.json
# ---------------------------------------------------------------------------
def load_video_jobs(json_dir, media_root, limit=None):
    with open(json_dir) as f:
        data = json.load(f)
    seen = set()
    jobs = []
    for item in data["items"]:
        vid = item["video_id"]
        if vid in seen:
            continue
        seen.add(vid)
        jobs.append({
            "video_id": vid,
            "video_path": os.path.join(media_root, vid),
        })
    if limit is not None:
        jobs = jobs[:limit]
    return jobs


def _stage1_src_path(job):
    """Path to the matching stage1 (RAG_stage1.py) JSON for this video.

    Stage1 output layout differs by split: train nests results under
    data/RAG_Info/train/<task>/<video_id-relative>.json, while test is flat
    under data/RAG_Info/test/tar_test/test/<basename>.json.
    """
    video_rel = os.path.splitext(job["video_id"])[0]
    if SPLIT == "train":
        return os.path.join(AICITY_TRAIN_DIR, video_rel + ".json")
    return os.path.join(AICITY_TEST_DIR, os.path.basename(video_rel) + ".json")


def _pending(job):
    """Return True if the stage1 JSON hasn't been merged with stage2 info yet."""
    src_path = _stage1_src_path(job)
    if not os.path.exists(src_path):
        return True  # nothing to merge into yet; process_video will warn
    try:
        with open(src_path) as f:
            results = json.load(f).get("results", [])
    except (json.JSONDecodeError, OSError):
        return True
    return not results or "stage2_factual" not in results[0]


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------
_WORKER = {"api_key": None, "output_dir": STAGE2_RAG_DIR, "stage1_rag_dir": AICITY_TRAIN_DIR}


def _init_worker(api_key, output_dir, stage1_rag_dir, split):
    global STAGE2_RAG_DIR, AICITY_TRAIN_DIR, AICITY_TEST_DIR, SPLIT
    _WORKER["api_key"] = api_key
    _WORKER["output_dir"] = output_dir
    _WORKER["stage1_rag_dir"] = stage1_rag_dir
    STAGE2_RAG_DIR = output_dir
    if split == "train":
        AICITY_TRAIN_DIR = stage1_rag_dir
    else:
        AICITY_TEST_DIR = stage1_rag_dir
    SPLIT = split
    print(f"[worker pid={os.getpid()}] ready", flush=True)


def _run_retrieval(toolkit, api_key):
    """Build the ReAct agent and extract video-level context from questions."""
    from langchain_core.tools import tool
    from langchain_openai import ChatOpenAI
    from langgraph.prebuilt import create_react_agent
    from prompt_library import prompt_lib

    prompts = prompt_lib()

    @tool(description="Retrieve context-rich questions for this video. Returns questions only, not answers.")
    def question_info_retrieval() -> str:
        return toolkit.get_question_info()

    model = ChatOpenAI(
        model="gpt-5.4",
        api_key=api_key,
        base_url="https://api.modelsell.com/v1",
        temperature=0.0,
    )

    agent = create_react_agent(
        model,
        [question_info_retrieval],
        prompt=prompts.option_cross_question_context_prompt,
    )

    result = agent.invoke({"messages": [("human", "Extract video-level context for this video.")]})
    return result["messages"][-1].content


def _normalise_frame_range(r):
    """Coerce a frame range to {"start_frame": int, "end_frame": int}.

    Accepts dicts with start_frame/end_frame keys, or 2-element lists/tuples.
    Returns None if the input cannot be interpreted.
    """
    if isinstance(r, dict):
        if "start_frame" in r and "end_frame" in r:
            return r
        # Handle alternative key names the LLM might use
        start = r.get("start") or r.get("start_frame")
        end = r.get("end") or r.get("end_frame")
        if start is not None and end is not None:
            return {"start_frame": int(start), "end_frame": int(end)}
    elif isinstance(r, (list, tuple)) and len(r) >= 2:
        return {"start_frame": int(r[0]), "end_frame": int(r[1])}
    return None


def _merge_frame_ranges(existing, new_ranges):
    """Merge two frame-range lists, deduplicating by (start_frame, end_frame).

    Both lists may contain dicts or lists/tuples; all are normalised first.
    """
    merged = [_normalise_frame_range(r) for r in existing]
    merged = [r for r in merged if r is not None]
    seen = {(r["start_frame"], r["end_frame"]) for r in merged}
    for r in new_ranges:
        nr = _normalise_frame_range(r)
        if nr is None:
            continue
        key = (nr["start_frame"], nr["end_frame"])
        if key not in seen:
            merged.append(nr)
            seen.add(key)
    return merged


def process_video(job):
    """Run stage2 retrieval for one video and merge the result into stage1's JSON.

    Saves:
      <output_dir>/<video_rel>.json    raw retrieval fields (resumability marker for the LLM call)
      <stage1 JSON for this video>     updated in place: adds stage2_factual, stage2_potential;
          merges relevant_frame_ranges

    Returns (status, job, n_updated, detail).
    """
    if not _pending(job):
        return "skip", job, 0, None

    try:
        video_rel = os.path.splitext(job["video_id"])[0]
        # Raw stage2 cache lives at STAGE2_RAG_DIR/<video_rel>.json, nested by
        # dataset so basenames that repeat across datasets don't collide.
        raw_path = os.path.join(STAGE2_RAG_DIR, video_rel + ".json")

        if os.path.exists(raw_path):
            # Step 1 already done — read cached result
            with open(raw_path) as f:
                cached = json.load(f)
            factual = cached.get("factual_information", [])
            potential = cached.get("potential_information", [])
            new_frame_ranges = cached.get("relevant_frame_ranges", [])
        else:
            # Stage2 only runs over track3 data (train's track3 folder and its
            # test-split counterpart, tar_test). 
            from tools import ToolkitStage2_trainset, ToolkitStage2_testset
            ToolkitCls = ToolkitStage2_trainset if SPLIT == "train" else ToolkitStage2_testset
            toolkit = ToolkitCls(video_id=job["video_id"], video_path=job["video_path"])
            retrieval = _run_retrieval(toolkit, _WORKER["api_key"])

            # 1. Parse retrieval
            try:
                parsed = json.loads(retrieval)
                if isinstance(parsed, list):
                    parsed = parsed[0] if parsed else {}
            except (json.JSONDecodeError, TypeError):
                parsed = {}
            factual = parsed.get("factual_information", [])
            if not isinstance(factual, list):
                factual = []
            potential = parsed.get("potential_information", [])
            if not isinstance(potential, list):
                potential = []
            new_frame_ranges = parsed.get("relevant_frame_ranges", [])
            if not isinstance(new_frame_ranges, list):
                new_frame_ranges = []

            # Save raw stage2 result
            _atomic_write_json(raw_path, {
                "video_id": job["video_id"],
                "factual_information": factual,
                "potential_information": potential,
                "relevant_frame_ranges": new_frame_ranges,
            })

        # 2. Enrich the stage1 RAG_Info JSON *in place* (no separate copy).
        n_updated = 0
        src_path = _stage1_src_path(job)
        if os.path.exists(src_path):
            with open(src_path) as f:
                doc = json.load(f)
            for result in doc.get("results", []):
                result["stage2_factual"] = factual
                result["stage2_potential"] = potential
                result["relevant_frame_ranges"] = _merge_frame_ranges(
                    result.get("relevant_frame_ranges", []), new_frame_ranges
                )
            _atomic_write_json(src_path, doc)
            n_updated = 1
        else:
            print(f"[warn] stage1 src not found: {src_path}", flush=True)

        return "ok", job, n_updated, None
    except Exception as e:  # noqa: BLE001
        return "error", job, 0, repr(e)


def parse_args():
    p = argparse.ArgumentParser(
        description="Stage 2 RAG: enrich aicity_train JSONs with video-level context."
    )
    p.add_argument("--split", choices=["train", "test"], default="train",
                   help="Selects the default --json-dir/--stage1-rag-dir/--output_dir for the "
                        "train or test data (each individually overridable below). Default: train.")
    p.add_argument("--media-root", default=None)
    p.add_argument("--json-dir", default=None)
    p.add_argument("--stage1-rag-dir", default=None)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--workers", type=int, default=32,
                   help="Number of parallel worker processes (default: 32).")
    p.add_argument("--limit", type=int, default=None, help="Process only first N videos (debug).")
    p.add_argument("--start", type=int, default=0,
                   help="First video index (0-based, inclusive).")
    p.add_argument("--end", type=int, default=None,
                   help="Last video index (exclusive). E.g. --start 1000 --end 2000.")
    args = p.parse_args()

    defaults = SPLIT_DEFAULTS[args.split]
    if args.media_root is None:
        args.media_root = DEFAULT_MEDIA_ROOT
    if args.json_dir is None:
        args.json_dir = defaults["json_dir"]
    if args.stage1_rag_dir is None:
        args.stage1_rag_dir = defaults["stage1_rag_dir"]
    if args.output_dir is None:
        args.output_dir = defaults["output_dir"]
    return args


def main():
    args = parse_args()
    api_key = os.environ.get("MODELSELL_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        sys.exit("Error: MODELSELL_API_KEY or OPENROUTER_API_KEY environment variable not set.")

    jobs = load_video_jobs(args.json_dir, args.media_root, limit=args.limit)

    # Apply --output_dir override in the main process so _pending() uses it too.
    global STAGE2_RAG_DIR, AICITY_TRAIN_DIR, AICITY_TEST_DIR, SPLIT
    STAGE2_RAG_DIR = args.output_dir
    if args.split == "train":
        AICITY_TRAIN_DIR = args.stage1_rag_dir
    else:
        AICITY_TEST_DIR = args.stage1_rag_dir
    SPLIT = args.split

    if args.start or args.end is not None:
        total = len(jobs)
        jobs = jobs[args.start:args.end]
        print(f"[plan] video range [{args.start}:{args.end}] -> {len(jobs)} of {total} video(s).")

    todo = [j for j in jobs if _pending(j)]
    print(f"[plan] {len(jobs)} video(s); {len(todo)} pending.")

    if not todo:
        print("[plan] nothing to do.")
        return

    num_workers = max(1, args.workers)
    ctx = mp.get_context("spawn")

    print(f"[run] {num_workers} worker(s) for {len(todo)} video(s) ...")
    counts = {}
    start = time.time()
    with ctx.Pool(num_workers, initializer=_init_worker,
                  initargs=(api_key, args.output_dir, args.stage1_rag_dir, args.split)) as pool:
        for n, (status, job, n_updated, detail) in enumerate(pool.imap_unordered(process_video, todo), 1):
            counts[status] = counts.get(status, 0) + 1
            extra = f" :: {detail}" if detail else (f" ({n_updated} tasks)" if n_updated else "")
            print(f"[{n}/{len(todo)}] {status.upper():9s} "
                  f"{os.path.basename(job['video_path'])}{extra}", flush=True)

    elapsed = round(time.time() - start, 1)
    summary = " ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    print(f"[done] {elapsed}s  {summary}")


if __name__ == "__main__":
    main()
