#!/usr/bin/env python3
"""
AI City Challenge 2026 Track 8 — PSI VQA out-of-domain evaluator.

PSI VQA is an egocentric dashcam benchmark (40 clips, 328 items total)
built on PSI 2.0.  The four sub-tasks map directly to in-domain task types:

    PSI-T1  bcq                 — crossing-intent binary classification
    PSI-T2  open_qa             — ambiguous-intent cue articulation
    PSI-T3  mcq                 — cue identification with mixed distractors
    PSI-T4  temporal_localization — driver-decision critical interval

The GT lives in four separate JSON files (one per task type) in the
PSI_VQA directory.  Pass the directory path via ``--gt-dir``, or point
``--gt`` to a pre-merged single JSON produced by ``merge_gt``.

Two modes
---------
1. **Submission validation** (no real answers needed).
   Checks coverage, duplicate item_index, and per-task parseability.
   Runs automatically when GT answers are redacted, or with ``--validate``.

2. **Scoring** (requires GT with real answers).
   bcq/mcq accuracy, temporal_localization mean-IoU, BERTScore F1 for
   open_qa.  Organizers run this against the private GT.

Usage::

    # Validate against the released (redacted) GT files in PSI_VQA/
    python evaluate.py --gt-dir dataset/test/PSI_VQA --submission my_sub.csv

    # Force validation-only even with a non-redacted GT
    python dataset/test/PSI_VQA/evaluate.py --gt-dir dataset/test/PSI_VQA --submission eval_PSI_VQA/last_update/submission_psi_vqa_20260710_073850.csv --validate

    # Score against a GT directory that contains real answers
    python evaluate.py --gt-dir my_psi_gt/ --submission my_sub.csv

    # Merge all four GT files into a single JSON, then use --gt
    python evaluate.py --gt-dir dataset/test/PSI_VQA --merge-out merged.json
    python evaluate.py --gt merged.json --submission my_sub.csv

Submission CSV format
---------------------
Exactly two columns: ``item_index,prediction``.

  - ``item_index`` — the 16-hex sample id. Join key.
  - ``prediction`` — raw model output text.

Expected prediction per sub-task:
  PSI-T1 (bcq)                   Text starting with "Yes" or "No".
  PSI-T2 (open_qa)               Bulleted cue list, or "None".
  PSI-T3 (mcq)                   Single letter A, B, C, or D.
  PSI-T4 (temporal_localization) JSON: {"start": "MM:SS", "end": "MM:SS"}.

Example rows::

    item_index,prediction
    af612fe6c7a21ab1,Yes
    490bc8f79dc97e68,A
    81e15aa0170b2aa9,"- The pedestrian faced toward the road.
    - Eye contact was made with the driver."
    bf38384086599009,None
    46242a7a89cabe21,"{""start"": ""00:01"", ""end"": ""00:05""}"
"""

import argparse
import json
import logging
import os
import re
import sys

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# File names that make up the PSI_VQA GT directory
PSI_GT_FILES = [
    "bcq.json",
    "mcq.json",
    "open_qa.json",
    "temporal_localization.json",
]

OPEN_ENDED_TASKS = frozenset({"open_qa"})

# Expected total item counts (for a quick sanity check)
EXPECTED_COUNTS = {
    "bcq": 55,
    "mcq": 91,
    "open_qa": 126,
    "temporal_localization": 56,
}


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

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
    m = re.match(r"^\(?([A-Da-d])\)?[).\s,:]", s)
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


def _is_valid_open_qa(text):
    """open_qa predictions should be a bulleted list or the literal 'None'."""
    if pd.isna(text) or not str(text).strip():
        return False
    s = str(text).strip()
    if s.lower() == "none":
        return True
    # Accept if at least one bullet-style line is present
    return bool(re.search(r"^\s*[-*•]\s+\S", s, re.MULTILINE))


# ---------------------------------------------------------------------------
# BERTScore (lazy import, only needed for scoring)
# ---------------------------------------------------------------------------

_BERTSCORER = None


def _bertscore_f1(predictions, references):
    global _BERTSCORER
    if _BERTSCORER is None:
        import bert_score
        _BERTSCORER = bert_score.BERTScorer(lang="en", rescale_with_baseline=True)
    _, _, f1 = _BERTSCORER.score(predictions, references)
    return float(f1.mean())


# ---------------------------------------------------------------------------
# Per-task scorers
# ---------------------------------------------------------------------------

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
            logger.warning(
                f"Failed to parse GT for temporal_localization: {row['answer']!r}"
            )
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
        logger.warning(
            f"temporal_localization: skipped {skipped}/{len(df)} unparseable predictions"
        )
    return {"temporal_localization_miou": float(np.mean(ious)) if ious else 0.0}


def score(df: pd.DataFrame) -> dict:
    """Score all rows in df, grouped by task_type. Returns flat metrics dict."""
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


# ---------------------------------------------------------------------------
# Validation (format-only, no scoring)
# ---------------------------------------------------------------------------

def _check_parseable(df, task_type):
    """Return (n_parseable, list_of_(item_index, reason)) for a task type."""
    bad = []
    if task_type == "bcq":
        for _, r in df.iterrows():
            if _extract_yesno(r["prediction"]) is None:
                bad.append((r["item_index"], "no Yes/No in prediction"))
    elif task_type == "mcq":
        for _, r in df.iterrows():
            if _extract_letter(r["prediction"]) is None:
                bad.append((r["item_index"], "no parseable letter (A-D) in prediction"))
    elif task_type == "temporal_localization":
        for _, r in df.iterrows():
            obj = _extract_json(r["prediction"])
            if obj is None or "start" not in obj or "end" not in obj:
                bad.append((r["item_index"], 'no {"start","end"} JSON in prediction'))
    elif task_type == "open_qa":
        for _, r in df.iterrows():
            if not _is_valid_open_qa(r["prediction"]):
                bad.append(
                    (r["item_index"],
                     'prediction should be a bulleted list or "None"')
                )
    else:
        for _, r in df.iterrows():
            if not _has_text(r["prediction"]):
                bad.append((r["item_index"], "empty prediction"))
    return len(df) - len(bad), bad


def validate(gt_df: pd.DataFrame, sub_df: pd.DataFrame,
             allow_missing: bool = False) -> dict:
    """Validate submission format. Returns a summary dict."""
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

    total_expected = sum(EXPECTED_COUNTS.values())
    print(f"Submission: {len(sub_df)} rows  (expected {total_expected})")
    print(f"GT:         {len(gt_df)} items, {len(gt_df['task_type'].unique())} task types")
    print()

    # Per-task item counts sanity check
    print("GT item counts per task type:")
    for tt, cnt in sorted(EXPECTED_COUNTS.items()):
        actual = int((gt_df["task_type"] == tt).sum())
        ok = "ok  " if actual == cnt else "warn"
        print(f"  {ok} {tt:26} {actual}  (expected {cnt})")
    print()

    print("Coverage:")
    if missing:
        msg = f"{len(missing)} GT item(s) have no submitted prediction"
        if allow_missing:
            print(f"  ! {msg} (--allow-missing set; will score remainder)")
        else:
            for k in sorted(missing)[:5]:
                print(f"      missing: {k}")
            raise ValueError(
                f"{msg}; pass --allow-missing to score the remainder."
            )
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
        summary[tt] = {
            "count": len(subset),
            "parseable": n_ok,
            "bad": [b[0] for b in bad],
        }
        marker = "ok  " if not bad else "warn"
        print(f"  {marker} {tt:26} {n_ok}/{len(subset)} parseable")
        for item_idx, reason in bad[:3]:
            print(f"           - {item_idx}: {reason}")
        if len(bad) > 3:
            print(f"           ... and {len(bad) - 3} more")
        total_bad += len(bad)

    print()
    if total_bad:
        print(
            f"Submission validates with {total_bad} prediction(s) that may not parse "
            "cleanly; they will receive 0 / IoU=0 on the affected tasks but won't "
            "block scoring."
        )
    else:
        print("Submission validates cleanly — all predictions parse for their task type.")

    return {
        "total_items": len(gt_df),
        "rows": len(sub_df),
        "missing": len(missing),
        "extra": len(extra),
        "tasks": summary,
    }


# ---------------------------------------------------------------------------
# GT loading
# ---------------------------------------------------------------------------

def _load_single_json(path: str) -> tuple[pd.DataFrame, dict]:
    """Load one tao-vl-reason-v1.0 JSON and return (items_df, metadata)."""
    with open(path, "r", encoding="utf-8") as fh:
        doc = json.load(fh)
    if doc.get("format") != "tao-vl-reason-v1.0":
        raise ValueError(
            f"{path}: expected format='tao-vl-reason-v1.0', got {doc.get('format')!r}"
        )
    if doc.get("metadata", {}).get("type") != "annotation":
        raise ValueError(
            f"{path}: expected metadata.type='annotation', "
            f"got {doc.get('metadata', {}).get('type')!r}"
        )
    df = pd.DataFrame(doc["items"])
    for col in ("item_index", "task_type", "answer"):
        if col not in df.columns:
            raise ValueError(f"{path}: GT items missing required column: {col!r}")
    return df, doc.get("metadata", {})


def load_gt_from_dir(gt_dir: str) -> tuple[pd.DataFrame, dict]:
    """Load and merge all four PSI_VQA GT files from *gt_dir*."""
    frames = []
    merged_meta: dict = {}
    for fname in PSI_GT_FILES:
        fpath = os.path.join(gt_dir, fname)
        if not os.path.isfile(fpath):
            raise FileNotFoundError(
                f"Expected GT file not found: {fpath}\n"
                f"Make sure --gt-dir points to the PSI_VQA directory containing "
                f"{', '.join(PSI_GT_FILES)}."
            )
        df, meta = _load_single_json(fpath)
        frames.append(df)
        merged_meta.update(meta)  # last-write wins for duplicate keys (all same here)
        logger.info(f"Loaded {len(df)} items from {fname}")
    combined = pd.concat(frames, ignore_index=True)
    # Confirm no duplicate item_index across files
    dups = combined[combined["item_index"].duplicated()]["item_index"].tolist()
    if dups:
        raise ValueError(f"Duplicate item_index across GT files: {dups[:5]}")
    return combined, merged_meta


def load_gt_from_file(gt_path: str) -> tuple[pd.DataFrame, dict]:
    """Load GT from a single (merged) JSON file."""
    return _load_single_json(gt_path)


def _is_redacted(gt_df: pd.DataFrame, metadata: dict) -> bool:
    if metadata.get("answers_redacted"):
        return True
    answers = gt_df["answer"].astype(str).str.strip()
    return (answers == "").all()


# ---------------------------------------------------------------------------
# Merge utility
# ---------------------------------------------------------------------------

def merge_gt_files(gt_dir: str, out_path: str) -> None:
    """Merge all four GT JSON files into a single tao-vl-reason-v1.0 JSON."""
    frames = []
    first_meta: dict = {}
    for fname in PSI_GT_FILES:
        fpath = os.path.join(gt_dir, fname)
        df, meta = _load_single_json(fpath)
        frames.append(df)
        if not first_meta:
            first_meta = meta
    combined = pd.concat(frames, ignore_index=True)
    doc = {
        "format": "tao-vl-reason-v1.0",
        "metadata": {**first_meta, "task": "all", "source": "PSI_VQA merged"},
        "items": combined.to_dict(orient="records"),
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2, ensure_ascii=False)
    print(f"Merged {len(combined)} items into {out_path}")


# ---------------------------------------------------------------------------
# Top-level evaluate
# ---------------------------------------------------------------------------

def evaluate(gt_df: pd.DataFrame, metadata: dict, submission_path: str,
             allow_missing: bool = False, validate_only: bool = False) -> dict:
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(
        description="PSI VQA evaluator + submission validator (Track 8).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    gt_group = p.add_mutually_exclusive_group()
    gt_group.add_argument(
        "--gt-dir",
        metavar="DIR",
        help="Path to the PSI_VQA directory containing the four GT JSON files "
             "(bcq_questions.json, mcq_questions.json, open_qa_questions.json, "
             "temporal_localization_questions.json).",
    )
    gt_group.add_argument(
        "--gt",
        metavar="FILE",
        help="Path to a single pre-merged tao-vl-reason-v1.0 GT JSON "
             "(produced by --merge-out).",
    )

    p.add_argument(
        "--submission",
        metavar="CSV",
        help="Path to submission CSV (item_index,prediction). "
             "Required unless --merge-out is used alone.",
    )
    p.add_argument(
        "--validate", action="store_true",
        help="Run only the format-validation report, even if GT contains real answers.",
    )
    p.add_argument(
        "--allow-missing", action="store_true",
        help="Score remainder when some GT items have no submitted prediction.",
    )
    p.add_argument(
        "--out", default=None, metavar="CSV",
        help="Optional CSV path to write the one-row metrics table (scoring mode only).",
    )
    p.add_argument(
        "--merge-out", default=None, metavar="JSON",
        help="Merge the four GT files from --gt-dir into a single JSON and write "
             "to this path. No scoring is performed.",
    )

    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    # --- merge-only mode ---
    if args.merge_out:
        if not args.gt_dir:
            p.error("--merge-out requires --gt-dir")
        merge_gt_files(args.gt_dir, args.merge_out)
        return

    # --- normal evaluate mode ---
    if not args.submission:
        p.error("--submission is required for evaluation")
    if not args.gt_dir and not args.gt:
        p.error("one of --gt-dir or --gt is required")

    if args.gt_dir:
        gt_df, metadata = load_gt_from_dir(args.gt_dir)
    else:
        gt_df, metadata = load_gt_from_file(args.gt)

    result = evaluate(
        gt_df, metadata, args.submission,
        allow_missing=args.allow_missing,
        validate_only=args.validate,
    )

    print()
    if result["mode"] == "score":
        print("=== Metrics ===")
        print(json.dumps(result["metrics"], indent=2, sort_keys=True))
        if args.out:
            pd.DataFrame([result["metrics"]]).to_csv(args.out, index=False)
            print(f"wrote {args.out}", file=sys.stderr)
    else:
        print(f"=== Validation-only mode ({result['reason']}) ===")
        print(
            "No scores computed. Submit your CSV to the evaluation server "
            "to obtain leaderboard metrics."
        )


if __name__ == "__main__":
    main()
