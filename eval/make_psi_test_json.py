#!/usr/bin/env python3
"""Merge the four per-task PSI-VQA question files (bcq/mcq/open_qa/
temporal_localization.json) into a single combined test JSON with a flat
``items`` list, as expected by eval_aicity_rag_test.py.

Usage:
  python eval/make_psi_test_json.py \
      --questions-dir data/dataset/test/PSI_VQA \
      --out data/dataset/test/PSI_VQA/test.json
"""
import argparse
import json
import os

TASKS = ["bcq", "mcq", "open_qa", "temporal_localization"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    items = []
    for task in TASKS:
        path = os.path.join(args.questions_dir, f"{task}.json")
        if not os.path.exists(path):
            continue
        doc = json.load(open(path))
        items += doc["items"] if isinstance(doc, dict) else doc
    json.dump({"items": items}, open(args.out, "w"), indent=2)

    from collections import Counter
    print(f"wrote {args.out}: {len(items)} items {dict(Counter(it['task_type'] for it in items))}")


if __name__ == "__main__":
    main()
