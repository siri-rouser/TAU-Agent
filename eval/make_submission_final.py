#!/usr/bin/env python3
"""Combine predictions_mcq_fixed.jsonl and predictions_freetext_consensus.jsonl
into a single submission CSV (item_index,prediction), ordered by test.json.

- bcq, mcq, bcq_openended, mcq_openended come from predictions_mcq_fixed.jsonl
- everything else (causal_linkage, open_qa, scene_description,
  temporal_description, temporal_localization, video_summarization) comes
  from predictions_freetext_consensus.jsonl
"""
import argparse
import csv
import json

MCQ_FIXED_TYPES = {"bcq", "mcq", "bcq_openended", "mcq_openended"}


def load_jsonl(path):
    records = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            records[d["item_index"]] = d
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mcq-fixed", default="eval/output/mcq_fixed/predictions_oe_harmonized.jsonl")
    ap.add_argument("--freetext", default="eval/output/freetext_consensus/predictions_freetext_consensus.jsonl")
    ap.add_argument("--test-json", default="data/dataset/test/tar_test/test.json")
    ap.add_argument("--out","-o", default="eval/output/submission_tar_final.csv")
    args = ap.parse_args()

    mcq_fixed = load_jsonl(args.mcq_fixed)
    freetext = load_jsonl(args.freetext)

    test_items = json.load(open(args.test_json))["items"]

    rows = []
    missing = []
    wrong_source_type = []
    for it in test_items:
        idx = it["item_index"]
        task_type = it["task_type"]
        if task_type in MCQ_FIXED_TYPES:
            src, other = mcq_fixed, freetext
        else:
            src, other = freetext, mcq_fixed

        rec = src.get(idx)
        if rec is None:
            rec = other.get(idx)
            if rec is None:
                missing.append(idx)
                continue
        elif rec.get("task_type") != task_type:
            wrong_source_type.append(idx)

        rows.append((idx, rec["prediction"]))

    if missing:
        print(f"WARNING: {len(missing)} item_index missing from both sources: {missing[:10]}")
    if wrong_source_type:
        print(f"WARNING: {len(wrong_source_type)} item_index had mismatched task_type: {wrong_source_type[:10]}")

    print(f"Total test.json items: {len(test_items)}, rows written: {len(rows)}")

    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["item_index", "prediction"])
        for idx, pred in rows:
            writer.writerow([idx, pred])

    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
