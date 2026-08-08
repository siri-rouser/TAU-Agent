#!/usr/bin/env python3
"""Deterministic (no-API) builder for the PSI-VQA cross-question context.

The cross-question context (`stage2_factual` / `stage2_potential`) is normally
produced by an LLM agent (see RAG_stage2.py) that reads the *other* questions
asked about the same video and rewrites them into presupposed facts and candidate
hypotheses. For PSI-VQA this context is decisive for the open-ended task, because
on the ambiguous clips it enumerates the candidate crossing-intent cues that the
reference answer is drawn from -- without it the model collapses open-QA to a bare
"None".

That agent only ever sees the QUESTION TEXT (no answers, no frames), so the same
context can be rebuilt deterministically, with no LLM / no API, purely from the
released question JSONs. This script does exactly that and injects the result into
copies of the per-video RAG-evidence files.

Usage:
  python RAG_retriever/build_stage2_ctx_noapi.py \
      --questions-dir data/dataset/test/PSI_VQA \
      --rag-in  data/RAG_Info/test/PSI_VQA \
      --rag-out data/RAG_Info/test/PSI_VQA_ctx \
      [--only-video PSI_VQA/test/videos/ambiguous/video_0205_track_0.mp4]  # smoke test
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict

TASKS = ["bcq", "mcq", "open_qa", "temporal_localization"]

_TIME_RE = re.compile(r"t=([0-9.]+)s\s*to\s*t=([0-9.]+)s")
_TIME_RE2 = re.compile(r"from\s+t=([0-9.]+)s")  # bcq: "from t=0s to t=10.733s"
_OPT_SPLIT_RE = re.compile(r"(?m)^\s*([A-D])\)\s*")


def _clean_bullets(block: str) -> str:
    """Join an option's '- ' bullet lines into one clause."""
    lines = [ln.strip().lstrip("-").strip() for ln in block.splitlines() if ln.strip()]
    lines = [ln for ln in lines if ln and not ln.lower().startswith("answer with")]
    return " ".join(lines).strip()


def _stance(question: str):
    q = question.lower()
    if "not intend to cross" in q or "not to cross" in q:
        return "The pedestrian might not intend to cross because {c}"
    if "uncertain" in q:
        return "The pedestrian's crossing intent may be uncertain because {c}"
    if "intend to cross" in q:
        return "The pedestrian might intend to cross because {c}"
    return None


def _parse_mcq_options(question: str):
    """Return list of option clause strings from an mcq question body."""
    parts = _OPT_SPLIT_RE.split(question)
    # parts = [preamble, 'A', bodyA, 'B', bodyB, ...]
    opts = []
    for i in range(1, len(parts) - 1, 2):
        body = parts[i + 1]
        # cut off the trailing "Answer with the letter..." if it bled into D)
        body = re.split(r"(?m)^\s*Answer with the letter", body)[0]
        clause = _clean_bullets(body)
        if clause:
            opts.append(clause)
    return opts


def build_context(questions):
    """questions: list of (task, question_text). Returns (factual, potential)."""
    factual, potential = [], []
    seen_f, seen_p = set(), set()

    def add_f(s):
        if s and s not in seen_f:
            seen_f.add(s); factual.append({"content": s})

    def add_p(s):
        if s and s not in seen_p:
            seen_p.add(s); potential.append({"content": s})

    # ---- factual: shared preamble facts ----
    joined = " ".join(q for _, q in questions)
    if "egocentric dashcam" in joined.lower():
        add_f("The video is an egocentric dashcam video recorded from the front of a moving car.")
    elif "dashcam" in joined.lower():
        add_f("This is a dashcam video.")
    elif "our car" in joined.lower() or "driver" in joined.lower():
        add_f("The video is from the perspective of our car.")

    m = _TIME_RE.search(joined) or _TIME_RE2.search(joined)
    if m:
        if m.re is _TIME_RE:
            span = f"from t={m.group(1)}s to t={m.group(2)}s"
        else:
            m2 = re.search(r"from\s+t=([0-9.]+)s\s+to\s+t=([0-9.]+)s", joined)
            span = f"from t={m2.group(1)}s to t={m2.group(2)}s" if m2 else f"from t={m.group(1)}s"
        add_f(f"A pedestrian marked with a red bounding box is visible at the start and shown {span}.")
    elif "red bounding box" in joined.lower():
        add_f("A pedestrian marked with a red bounding box is visible at the start of the video.")

    if "disagreed" in joined.lower():
        add_f("Annotators disagreed about the marked pedestrian's crossing intent.")

    # ---- potential: candidate cues from mcq options + bcq stance ----
    has_bcq = any(t == "bcq" for t, _ in questions)
    has_temporal = any(t == "temporal_localization" for t, _ in questions)
    for task, q in questions:
        if task == "mcq":
            tmpl = _stance(q)
            if not tmpl:
                continue
            for clause in _parse_mcq_options(q):
                if clause:
                    clause = clause[0].lower() + clause[1:]
                add_p(tmpl.format(c=clause))
    if has_temporal:
        add_p("A road user such as a pedestrian, cyclist, or another vehicle, or a road "
              "factor such as a traffic signal, sign, or road condition may influence the "
              "driver's decision-making.")
    if has_bcq and not potential:
        add_p("The pedestrian may intend to cross in front of our car.")

    return factual, potential


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions-dir", required=True,
                    help="Dir of per-task PSI question JSONs (bcq.json, mcq.json, ...).")
    ap.add_argument("--rag-in", required=True,
                    help="Per-video RAG-evidence dir to read (with empty stage2 context).")
    ap.add_argument("--rag-out", required=True,
                    help="Output dir with the cross-question context filled in.")
    ap.add_argument("--only-video", default=None,
                    help="If set, only process files whose video_id matches (smoke test); prints the built context.")
    args = ap.parse_args()

    # 1) aggregate questions per video_id across all tasks
    by_vid = defaultdict(list)
    for task in TASKS:
        p = os.path.join(args.questions_dir, f"{task}.json")
        if not os.path.exists(p):
            continue
        doc = json.load(open(p))
        items = doc["items"] if isinstance(doc, dict) else doc
        for it in items:
            by_vid[it["video_id"]].append((task, it["question"]))
    print(f"[ctx] aggregated questions for {len(by_vid)} videos")

    ctx_by_vid = {vid: build_context(qs) for vid, qs in by_vid.items()}

    if args.only_video:
        f, p = ctx_by_vid.get(args.only_video, ([], []))
        print(f"\n=== built context for {args.only_video} ===")
        print("stage2_factual:")
        for x in f:
            print("  F:", x["content"])
        print("stage2_potential:")
        for x in p:
            print("  P:", x["content"])
        return

    # 2) walk rag-in, inject, write to rag-out
    n_files, n_results, n_filled, n_novid = 0, 0, 0, 0
    for src in glob.glob(os.path.join(args.rag_in, "**", "*.json"), recursive=True):
        if os.sep + "_raw_cache" + os.sep in src:
            continue
        try:
            doc = json.load(open(src))
        except Exception:
            continue
        if not (isinstance(doc, dict) and "results" in doc):
            continue
        vid = (doc.get("video_id") or "").strip("/")
        ctx = ctx_by_vid.get(vid)
        n_files += 1
        for r in doc.get("results", []):
            n_results += 1
            if ctx:
                r["stage2_factual"], r["stage2_potential"] = ctx[0], ctx[1]
                if ctx[0] or ctx[1]:
                    n_filled += 1
            else:
                n_novid += 1
        rel = os.path.relpath(src, args.rag_in)
        dst = os.path.join(args.rag_out, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        json.dump(doc, open(dst, "w"), indent=2)

    print(f"[ctx] wrote {n_files} files, {n_results} results, {n_filled} filled, {n_novid} no-vid-match")


if __name__ == "__main__":
    main()
