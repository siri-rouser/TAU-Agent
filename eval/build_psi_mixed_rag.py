#!/usr/bin/env python3
"""Build the PSI-VQA "open_qa-only context" RAG dir (task-selective context).

The cross-question context is a double-edged component on PSI-VQA:
  - It HELPS the open-ended task a lot (Open-QA Cue-F1 rises sharply), because on
    the ambiguous clips the block enumerates the candidate crossing-intent cues the
    reference answer is drawn from.
  - The SAME context HURTS BCQ: on the clear clips it degenerates into a one-sided
    "the pedestrian may intend to cross" restatement that biases the model toward a
    single label (macro-F1 drops). It also nudges MCQ toward the option whose cues
    appear first.

The best setup keeps the context ONLY for open_qa and strips it for every other
task. This script takes the fully-built context dir (produced by
build_stage2_ctx_noapi.py) and, for every non-open_qa result, blanks out
stage2_factual / stage2_potential, leaving the visual evidence (scene description,
summary, captions, tracks) untouched. open_qa files are copied through unchanged.

Usage:
  python eval/build_psi_mixed_rag.py \
      --ctx-dir data/RAG_Info/test/PSI_VQA_ctx \
      --out     data/RAG_Info/test/PSI_VQA_mixed
  # then run eval/eval_aicity_rag_test.py on --rag-dir data/RAG_Info/test/PSI_VQA_mixed
"""
import argparse
import glob
import json
import os
import shutil

CONTEXT_TASKS = {"open_qa"}          # tasks that KEEP the full cross-question context


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx-dir", required=True,
                    help="Fully-built context dir (from build_stage2_ctx_noapi.py).")
    ap.add_argument("--out", required=True, help="Output task-selective dir.")
    ap.add_argument("--strip-fields", default="stage2_factual,stage2_potential",
                    help="Which stage2 fields to blank for non-open_qa tasks. Use "
                         "'stage2_potential' to drop only the candidate hints and "
                         "KEEP the neutral stage2_factual.")
    args = ap.parse_args()
    strip_fields = tuple(f.strip() for f in args.strip_fields.split(",") if f.strip())

    if os.path.abspath(args.ctx_dir) == os.path.abspath(args.out):
        raise SystemExit("--out must differ from --ctx-dir")
    if os.path.exists(args.out):
        shutil.rmtree(args.out)

    n_files = n_stripped = n_open = 0
    for src in glob.glob(os.path.join(args.ctx_dir, "**", "*.json"), recursive=True):
        rel = os.path.relpath(src, args.ctx_dir)
        # task = first path component (open_qa / bcq / mcq / temporal_localization)
        task = rel.split(os.sep, 1)[0]
        dst = os.path.join(args.out, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        try:
            doc = json.load(open(src))
        except (json.JSONDecodeError, OSError):
            shutil.copy2(src, dst)
            continue
        n_files += 1
        if task in CONTEXT_TASKS:
            n_open += 1
            shutil.copy2(src, dst)          # keep context for open_qa
            continue
        # strip cross-question context for every other task
        for r in doc.get("results", []):
            for f in strip_fields:
                if r.get(f):
                    r[f] = []
                    n_stripped += 1
        json.dump(doc, open(dst, "w"), indent=2)

    print(f"wrote {args.out}")
    print(f"  files: {n_files} (open_qa kept: {n_open}), stripped stage2 blocks: {n_stripped}")


if __name__ == "__main__":
    main()
