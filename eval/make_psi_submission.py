#!/usr/bin/env python3
"""Turn the PSI-VQA prediction JSONL from eval_aicity_rag_test.py into the
official ``item_index,prediction`` submission CSV.

Per the PSI-VQA format, the prediction is:
  bcq                   -> "Yes" / "No"
  mcq                   -> a single option letter A/B/C/D
  open_qa               -> a bulleted cue list, or "None"
  temporal_localization -> {"start": "MM:SS", "end": "MM:SS"}

The QA-VLM answers MCQ as "A) ...", so we extract the leading option letter for
that task and pass every other task's prediction through verbatim.

Usage:
  python eval/make_psi_submission.py \
      --predictions eval/output_psi/predictions/rag_test_predictions.jsonl \
      --out eval/output_psi/submission_psi.csv
"""
import argparse
import csv
import json
import re


def mcq_letter(s: str) -> str:
    m = re.match(r"\s*([A-D])\b", (s or "").strip())
    return m.group(1) if m else (s or "").strip()[:1].upper()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True, help="rag_test_predictions.jsonl")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    preds = [json.loads(l) for l in open(args.predictions)]
    rows = []
    for p in preds:
        pred = (p.get("prediction") or "").strip()
        if p["task_type"] == "mcq":
            pred = mcq_letter(pred)
        rows.append((p["item_index"], pred))

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["item_index", "prediction"])
        w.writerows(rows)
    print(f"wrote {args.out}: {len(rows)} rows")


if __name__ == "__main__":
    main()
