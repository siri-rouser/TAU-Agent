#!/usr/bin/env python3
"""
AI City Challenge 2026 Track 3 — TAR test set evaluator.

Two modes:

1. **Submission validation** (no answers needed).
   Validates that a submission CSV covers every test item, has the right
   columns, and that per-task predictions are in a parseable shape (e.g.
   `bcq` predictions start with Yes/No, `temporal_localization` predictions
   contain a `{"start": "...", "end": "..."}` JSON object, etc.).
   Runs automatically whenever the loaded GT file has its answers redacted
   (``metadata.answers_redacted == true``), or whenever ``--validate`` is
   passed.

2. **Scoring** (requires GT with real answers).
   Computes the standard TAR benchmark metrics (bcq/mcq accuracy,
   temporal_localization mean-IoU, BERTScore F1 for open-ended tasks).
   Runs when the loaded GT contains real answers. The released
   ``test.json`` has its answers redacted, so participants will only see
   the validation report; organizers run this script against the private
   GT to compute the leaderboard scores.

Usage::

    # Validate submission format against the released (redacted) test.json
    python evaluate.py --gt test/test.json --submission my_submission.csv

    # Force validation-only even with a non-redacted GT
    python evaluate.py --gt my_gt.json --submission my_submission.csv --validate

    # Full scoring against a GT with real answers (e.g. your own held-out set)
    python evaluate.py --gt my_gt.json --submission my_submission.csv

Submission CSV format
---------------------
Exactly two columns: ``item_index,prediction``.

  - ``item_index`` — the 16-hex sample id from ``test.json``. Join key.
  - ``prediction`` — the model's raw output text. Multi-line predictions
    are fine; pandas CSV quoting handles them. See
    ``submission.example.csv`` for a concrete reference.

Every ``item_index`` in ``test.json`` must appear in the submission
(960 rows total) unless ``--allow-missing`` is passed.
"""

import argparse
import json
import logging
import re
import sys

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

OPEN_ENDED_TASKS = frozenset({
    "bcq_openended", "mcq_openended", "open_qa",
    "causal_linkage", "scene_description", "temporal_description",
    "video_summarization",
})


# ---- extraction helpers (mirror the scoring code paths so the validation
#      check agrees with how predictions are actually scored) ----

def _extract_yesno(text):
    if pd.isna(text) or not str(text).strip():
        return None
    s = str(text).strip().lower()
    m = re.match(r"^(yes|no)\b", s)
    if m:
        return m.group(1)
    m = re.search(r"\b(yes|no)\b", s)
    return m.group(1) if m else None


def _extract_letter(text):
    if pd.isna(text) or not str(text).strip():
        return None
    s = str(text).strip()
    m = re.match(r"^\(?([A-Za-z])\)?[).\s,:]", s)
    if m:
        return m.group(1).upper()
    if re.fullmatch(r"[A-Da-d]", s):
        return s.upper()
    m = re.search(r"\b([A-D])\b", s)
    return m.group(1) if m else None


def _gt_yesno(answer):
    assert answer and str(answer).strip(), f"GT empty: {answer!r}"
    first = str(answer).strip().lower().split(".")[0].split()[0]
    assert first in ("yes", "no"), f"GT does not start with Yes/No: {answer!r}"
    return first


def _gt_letter(answer):
    assert answer and str(answer).strip(), f"GT empty: {answer!r}"
    m = re.match(r"^([A-Za-z])\)", str(answer).strip())
    assert m, f"GT does not match letter) format: {answer!r}"
    return m.group(1).upper()


def _parse_timestamp(ts):
    parts = str(ts).strip().split(":")
    if len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    return float(ts)


def _extract_json(text):
    if pd.isna(text) or not str(text).strip():
        return None
    s = str(text).strip()
    m = re.search(r"```json\s*(.*?)\s*```", s, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(1))
            if isinstance(obj, list) and obj and isinstance(obj[0], dict) \
                    and "start" in obj[0] and "end" in obj[0]:
                return obj[0]
            return obj
        except json.JSONDecodeError:
            pass
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return None


def _has_text(text):
    return not (pd.isna(text) or not str(text).strip())


# ---- BERTScore (only needed for scoring) ----

_BERTSCORER = None


def _bertscore_f1(predictions, references):
    global _BERTSCORER
    if _BERTSCORER is None:
        import bert_score
        _BERTSCORER = bert_score.BERTScorer(lang="en", rescale_with_baseline=True)
    _, _, f1 = _BERTSCORER.score(predictions, references)
    return float(f1.mean())


# ---- per-task scorers ----

def _score_bcq(df):
    correct = sum(
        _extract_yesno(row["prediction"]) == _gt_yesno(row["answer"])
        for _, row in df.iterrows()
    )
    return {"bcq_accuracy": correct / len(df)}


def _score_mcq(df):
    correct = sum(
        _extract_letter(row["prediction"]) == _gt_letter(row["answer"])
        for _, row in df.iterrows()
    )
    return {"mcq_accuracy": correct / len(df)}


def _score_temporal_localization(df):
    ious = []
    skipped = 0
    for _, row in df.iterrows():
        gt = _extract_json(row["answer"])
        if gt is None:
            logger.warning(f"Failed to parse GT for temporal_localization: {row['answer']!r}")
            continue
        pred = _extract_json(row["prediction"])
        if pred is None or "start" not in pred or "end" not in pred:
            skipped += 1
            continue
        try:
            gs = _parse_timestamp(gt["start"])
            ge = _parse_timestamp(gt["end"])
            ps = _parse_timestamp(pred["start"])
            pe = _parse_timestamp(pred["end"])
        except (KeyError, ValueError, TypeError):
            skipped += 1
            continue
        inter = max(0.0, min(ge, pe) - max(gs, ps))
        union = max(0.0, (ge - gs) + (pe - ps) - inter)
        ious.append(inter / union if union > 0 else 0.0)
    if skipped:
        logger.warning(f"temporal_localization: skipped {skipped}/{len(df)} unparseable predictions")
    return {"temporal_localization_miou": float(np.mean(ious)) if ious else 0.0}


def score(df: pd.DataFrame) -> dict:
    """Score all rows in df, grouped by task_type. Returns flat {metric: value, ..., 'mean': value}."""
    metrics = {}
    for task_type in df["task_type"].unique():
        subset = df[df["task_type"] == task_type]
        if task_type == "bcq":
            metrics.update(_score_bcq(subset))
        elif task_type == "mcq":
            metrics.update(_score_mcq(subset))
        elif task_type == "temporal_localization":
            metrics.update(_score_temporal_localization(subset))
        elif task_type in OPEN_ENDED_TASKS:
            metrics[f"{task_type}_bertscore_f1"] = _bertscore_f1(
                predictions=subset["prediction"].tolist(),
                references=subset["answer"].tolist(),
            )
        else:
            logger.warning(f"Unknown task_type {task_type!r}, skipping")
    if metrics:
        metrics["mean"] = float(np.mean(list(metrics.values())))
    return metrics


# ---- validation (format-only, no scoring) ----

def _check_parseable(df, task_type):
    """Return (n_parseable, list_of_bad_item_indices) for the given task_type."""
    bad = []
    if task_type == "bcq":
        for _, r in df.iterrows():
            if _extract_yesno(r["prediction"]) is None:
                bad.append((r["item_index"], "no Yes/No in prediction"))
    elif task_type == "mcq":
        for _, r in df.iterrows():
            if _extract_letter(r["prediction"]) is None:
                bad.append((r["item_index"], "no parseable letter in prediction"))
    elif task_type == "temporal_localization":
        for _, r in df.iterrows():
            obj = _extract_json(r["prediction"])
            if obj is None or "start" not in obj or "end" not in obj:
                bad.append((r["item_index"], "no {start, end} JSON in prediction"))
    else:
        # Open-ended tasks: just need non-empty text.
        for _, r in df.iterrows():
            if not _has_text(r["prediction"]):
                bad.append((r["item_index"], "empty prediction"))
    return len(df) - len(bad), bad


def validate(gt_df: pd.DataFrame, sub_df: pd.DataFrame, allow_missing: bool = False) -> dict:
    """Validate submission format. Prints a per-task report and returns a
    summary dict. Raises ValueError on hard errors (missing columns,
    duplicate item_index, missing predictions without --allow-missing).
    """
    if "item_index" not in sub_df.columns or "prediction" not in sub_df.columns:
        raise ValueError("submission CSV must have columns: item_index, prediction")

    dup_mask = sub_df["item_index"].duplicated()
    if dup_mask.any():
        dups = sub_df.loc[dup_mask, "item_index"].tolist()[:5]
        raise ValueError(f"submission has duplicate item_index values: {dups} ...")

    gt_keys = set(gt_df["item_index"])
    sub_keys = set(sub_df["item_index"])
    missing = gt_keys - sub_keys
    extra = sub_keys - gt_keys

    print(f"Submission: {len(sub_df)} rows")
    print(f"GT:         {len(gt_df)} items, {len(gt_df['task_type'].unique())} task types")
    print()
    print("Coverage:")
    if missing:
        msg = f"{len(missing)} GT item(s) have no submitted prediction"
        if allow_missing:
            print(f"  ! {msg} (--allow-missing set; will score remainder)")
        else:
            for k in sorted(missing)[:5]:
                print(f"      missing: {k}")
            raise ValueError(f"{msg}; pass --allow-missing to score the remainder.")
    else:
        print(f"  ok   every GT item has a submitted prediction")
    if extra:
        print(f"  warn {len(extra)} submitted prediction(s) have no matching GT (ignored)")

    merged = gt_df.merge(sub_df, on="item_index", how="inner")
    print()
    print("Per-task format parsing:")
    summary = {}
    total_bad = 0
    for tt in sorted(merged["task_type"].unique()):
        subset = merged[merged["task_type"] == tt]
        n_ok, bad = _check_parseable(subset, tt)
        summary[tt] = {"count": len(subset), "parseable": n_ok, "bad": [b[0] for b in bad]}
        marker = "ok  " if not bad else "warn"
        print(f"  {marker} {tt:24} {n_ok}/{len(subset)} parseable")
        for item_idx, reason in bad[:3]:
            print(f"           - {item_idx}: {reason}")
        if len(bad) > 3:
            print(f"           ... and {len(bad) - 3} more")
        total_bad += len(bad)

    print()
    if total_bad:
        print(f"Submission validates with {total_bad} prediction(s) that may not parse cleanly;")
        print("they will receive 0/IoU=0 on the affected tasks but won't block scoring.")
    else:
        print("Submission validates cleanly — all predictions parse for their task type.")

    return {"total_items": len(gt_df), "rows": len(sub_df),
            "missing": len(missing), "extra": len(extra),
            "tasks": summary}


# ---- GT loading + join ----

def _load_gt(gt_path: str) -> tuple[pd.DataFrame, dict]:
    with open(gt_path, "r", encoding="utf-8") as f:
        doc = json.load(f)
    if doc.get("format") != "tao-vl-reason-v1.0":
        raise ValueError(f"Expected format='tao-vl-reason-v1.0', got {doc.get('format')!r}")
    if doc.get("metadata", {}).get("type") != "annotation":
        raise ValueError(f"Expected metadata.type='annotation', got {doc.get('metadata', {}).get('type')!r}")
    df = pd.DataFrame(doc["items"])
    for col in ("item_index", "task_type", "answer"):
        if col not in df.columns:
            raise ValueError(f"GT items missing required column: {col}")
    return df, doc.get("metadata", {})


def _is_redacted(gt_df: pd.DataFrame, metadata: dict) -> bool:
    if metadata.get("answers_redacted"):
        return True
    answers = gt_df["answer"].astype(str).str.strip()
    return (answers == "").all()


def evaluate(gt_path: str, submission_path: str,
             allow_missing: bool = False, validate_only: bool = False) -> dict:
    gt_df, metadata = _load_gt(gt_path)
    sub_df = pd.read_csv(submission_path)

    redacted = _is_redacted(gt_df, metadata)
    do_score = (not validate_only) and (not redacted)

    summary = validate(gt_df, sub_df, allow_missing=allow_missing)

    if do_score:
        merged = gt_df.merge(sub_df, on="item_index", how="inner")
        metrics = score(merged)
        return {"mode": "score", "metrics": metrics, "validation": summary}

    reason = "validate-only flag" if validate_only else "GT answers are redacted"
    return {"mode": "validate", "reason": reason, "validation": summary}


def main(argv=None):
    p = argparse.ArgumentParser(
        description="TAR test set evaluator + submission validator.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--submission", required=True,
                   help="Path to submission CSV (item_index,prediction).")
    p.add_argument("--gt", required=True,
                   help="Path to a tao-vl-reason-v1.0 GT JSON."
                        "Pass your own GT with real answers to compute metrics for validation.")
    p.add_argument("--validate", action="store_true",
                   help="Run only the format-validation report, even if the "
                        "GT contains real answers.")
    p.add_argument("--allow-missing", action="store_true",
                   help="Score remainder when some GT items have no submitted prediction.")
    p.add_argument("--out", default=None,
                   help="Optional CSV path to write the one-row metrics table (scoring mode only).")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    result = evaluate(args.gt, args.submission,
                      allow_missing=args.allow_missing,
                      validate_only=args.validate)

    print()
    if result["mode"] == "score":
        print("=== Metrics ===")
        print(json.dumps(result["metrics"], indent=2, sort_keys=True))
        if args.out:
            pd.DataFrame([result["metrics"]]).to_csv(args.out, index=False)
            print(f"wrote {args.out}", file=sys.stderr)
    else:
        print(f"=== Validation-only mode ({result['reason']}) ===")
        print("No scores computed. Submit your CSV to the evaluation server "
              "to obtain leaderboard metrics.")


if __name__ == "__main__":
    main()

# python dataset/test/tar_test/evaluate.py --gt dataset/test/tar_test/test.json --submission submission_tar_final.csv --allow-missing