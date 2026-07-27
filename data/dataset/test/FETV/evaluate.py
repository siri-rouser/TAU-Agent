#!/usr/bin/env python3
"""
AI City Challenge 2026 Track 7 — FETV (FishEye Traffic Violation) evaluator.

FETV is an optional out-of-domain leaderboard scored as Track 7. The test
set consists of 200 short fisheye video clips covering 7 violation types.

Two modes
---------
1. **Submission validation** (no real GT answers needed).
   Checks that all 200 clips are covered, that all required fields are
   present, and that every categorical value is drawn from its allowed set.
   Runs automatically when GT answers are redacted, or with ``--validate``.

2. **Scoring** (requires GT with real answers).
   Computes per-field macro-F1 (categorical + date exact-match +
   time 7-second-tolerance match), BERTScore F1 and normalised CIDEr for
   the free-form description, then combines into the official FETV score:

       S_FETV = 0.25 · CIDErnorm + 0.25 · BERTScore + 0.5 · MacroF1

   MacroF1 = mean of the 12 per-field scores (10 categorical macro-F1 +
   date accuracy + time accuracy).

Usage::

    # Validate submission format (works even with redacted GT)
    python evaluate.py --gt gt.json --submission my_submission.json

    # Force validation-only even with a non-redacted GT
    python dataset/test/FETV/evaluate.py --gt dataset/test/FETV/gt.json --submission eval_FETV/output_fetv/submission_fetv_20260710_082206.json --validate

    # Full scoring (GT has real answers)
    python evaluate.py --gt gt.json --submission my_submission.json

    # Also write per-field CSV report
    python evaluate.py --gt gt.json --submission my_submission.json --out metrics.csv

Submission JSON format
----------------------
A flat JSON array, one object per clip::

    [
      {
        "clip_name": "001_000.mp4",
        "answer_date": "2026-01-01",
        "answer_time": "12:34:56",
        "answer_violation_type": "wrong_way",
        "answer_violator_type": "car",
        "answer_color": "light",
        "answer_initial_position": "Top-Left",
        "answer_initial_lane": "1",
        "answer_final_position": "Middle-Right",
        "answer_final_lane": "2",
        "answer_intersection_type": "T-intersection",
        "answer_weather": "clear",
        "answer_light": "daylight",
        "answer_description": "A car drives the wrong way through the junction."
      }
    ]

Ground-truth file
-----------------
Same JSON array schema as the submission (clip_name + answer_* fields).
A field value of "" or null marks it as redacted (validation-only mode).

CIDEr normalisation
-------------------
CIDEr scores from pycocoevalcap are in [0, ~10]. We normalise with
    CIDErnorm = min(raw_CIDEr / 10.0, 1.0)
Pass ``--cider-scale`` to override the divisor (e.g. ``--cider-scale 4``
if your CIDEr variant tops out at 4).
"""

import argparse
import json
import logging
import re
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Field definitions
# ---------------------------------------------------------------------------

CATEGORICAL_FIELDS = [
    "violation_type",
    "violator_type",
    "color",
    "initial_position",
    "final_position",
    "initial_lane",
    "final_lane",
    "intersection_type",
    "weather",
    "light",
]

DATE_FIELD = "date"
TIME_FIELD = "time"
CAPTION_FIELD = "description"

ALL_ANSWER_FIELDS = (
    [DATE_FIELD, TIME_FIELD]
    + CATEGORICAL_FIELDS
    + [CAPTION_FIELD]
)

# All structured fields that feed into MacroF1
STRUCTURED_FIELDS = [DATE_FIELD, TIME_FIELD] + CATEGORICAL_FIELDS

ALLOWED_VALUES = {
    "violation_type": {
        "wrong_way", "uturn", "jaywalking", "red_light",
        "lane_use_control", "lane_discipline", "no_violation",
    },
    "violator_type": {"car", "motorcycle", "pedestrian", "bus", "truck", "na"},
    "color": {"dark", "light", "red", "green", "yellow", "blue", "mixed", "na"},
    "initial_position": {
        "Top-Left", "Top-Center", "Top-Right",
        "Middle-Left", "Middle-Center", "Middle-Right",
        "Bottom-Left", "Bottom-Center", "Bottom-Right", "na",
    },
    "final_position": {
        "Top-Left", "Top-Center", "Top-Right",
        "Middle-Left", "Middle-Center", "Middle-Right",
        "Bottom-Left", "Bottom-Center", "Bottom-Right", "na",
    },
    "initial_lane": {"1", "2", "3", "4", "na"},
    "final_lane": {"1", "2", "3", "4", "na"},
    "intersection_type": {"T-intersection", "four-way intersection"},
    "weather": {"clear", "rainy", "cloudy"},
    "light": {"daylight", "night"},
}

TIME_TOLERANCE_SEC = 7

EXPECTED_CLIPS = 200


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_time_sec(ts: str) -> float | None:
    """Parse HH:MM:SS (or MM:SS) to total seconds. Returns None on failure."""
    ts = str(ts).strip()
    for fmt in ("%H:%M:%S", "%M:%S"):
        try:
            t = datetime.strptime(ts, fmt)
            return t.hour * 3600 + t.minute * 60 + t.second
        except ValueError:
            pass
    return None


def _normalise_str(v) -> str:
    """Strip and return as string; treat None/NaN as empty string."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    return str(v).strip()


def _is_redacted(gt_df: pd.DataFrame) -> bool:
    """True when all structured GT answers are blank/null."""
    for field in STRUCTURED_FIELDS:
        col = f"answer_{field}"
        if col in gt_df.columns:
            non_empty = gt_df[col].astype(str).str.strip().replace("nan", "")
            if non_empty.ne("").any():
                return False
    return True


# ---------------------------------------------------------------------------
# BERTScore (lazy import)
# ---------------------------------------------------------------------------

_BERTSCORER = None


def _bertscore_f1(predictions: list[str], references: list[str]) -> float:
    global _BERTSCORER
    if _BERTSCORER is None:
        import bert_score
        _BERTSCORER = bert_score.BERTScorer(lang="en", rescale_with_baseline=True)
    _, _, f1 = _BERTSCORER.score(predictions, references)
    return float(f1.mean())


# ---------------------------------------------------------------------------
# CIDEr (lazy import via pycocoevalcap)
# ---------------------------------------------------------------------------

def _cider_score(predictions: list[str], references: list[str]) -> float:
    """
    Compute corpus-level CIDEr using pycocoevalcap.
    Falls back to nltk-based CIDEr if pycocoevalcap is unavailable.
    """
    try:
        from pycocoevalcap.cider.cider import Cider
        scorer = Cider()
        # gts / res use integer keys
        gts = {i: [r] for i, r in enumerate(references)}
        res = {i: [p] for i, p in enumerate(predictions)}
        score, _ = scorer.compute_score(gts, res)
        return float(score)
    except ImportError:
        pass
    # Fallback: use evaluate library (HuggingFace)
    try:
        import evaluate as hf_evaluate
        cider = hf_evaluate.load("cider")
        result = cider.compute(predictions=predictions, references=[[r] for r in references])
        return float(result["cider"])
    except Exception as exc:
        logger.error(
            f"Could not compute CIDEr: {exc}. "
            "Install pycocoevalcap or the HuggingFace evaluate library."
        )
        return float("nan")


# ---------------------------------------------------------------------------
# Per-field scorers
# ---------------------------------------------------------------------------

def _score_date(gt_series: pd.Series, pred_series: pd.Series) -> float:
    """Exact-match accuracy for date strings (YYYY-MM-DD)."""
    correct = (
        gt_series.astype(str).str.strip()
        == pred_series.astype(str).str.strip()
    ).sum()
    return correct / len(gt_series)


def _score_time(gt_series: pd.Series, pred_series: pd.Series) -> float:
    """Binary accuracy with TIME_TOLERANCE_SEC-second window."""
    n_correct = 0
    for gt_val, pred_val in zip(gt_series, pred_series):
        gt_sec = _parse_time_sec(gt_val)
        pred_sec = _parse_time_sec(pred_val)
        if gt_sec is None:
            continue  # skip unparseable GT
        if pred_sec is not None and abs(gt_sec - pred_sec) <= TIME_TOLERANCE_SEC:
            n_correct += 1
    return n_correct / len(gt_series)


def _score_categorical(
    gt_series: pd.Series,
    pred_series: pd.Series,
    field: str,
) -> float:
    """Macro-averaged F1 for a categorical field."""
    gt = gt_series.astype(str).str.strip().tolist()
    pred = pred_series.astype(str).str.strip().tolist()
    labels = sorted(ALLOWED_VALUES.get(field, set(gt) | set(pred)))
    return float(f1_score(gt, pred, labels=labels, average="macro", zero_division=0))


def score(gt_df: pd.DataFrame, sub_df: pd.DataFrame,
          cider_scale: float = 10.0) -> dict:
    """
    Compute all FETV metrics. Returns a flat dict with per-field scores,
    MacroF1, BERTScore, CIDErnorm, and the final S_FETV.
    """
    merged = gt_df.merge(sub_df, on="clip_name", suffixes=("_gt", "_pred"))

    per_field: dict[str, float] = {}

    # --- date ---
    per_field[DATE_FIELD] = _score_date(
        merged[f"answer_{DATE_FIELD}_gt"],
        merged[f"answer_{DATE_FIELD}_pred"],
    )

    # --- time ---
    per_field[TIME_FIELD] = _score_time(
        merged[f"answer_{TIME_FIELD}_gt"],
        merged[f"answer_{TIME_FIELD}_pred"],
    )

    # --- categorical ---
    for field in CATEGORICAL_FIELDS:
        per_field[field] = _score_categorical(
            merged[f"answer_{field}_gt"],
            merged[f"answer_{field}_pred"],
            field,
        )

    macro_f1 = float(np.mean(list(per_field.values())))

    # --- caption ---
    captions_gt = merged[f"answer_{CAPTION_FIELD}_gt"].astype(str).tolist()
    captions_pred = merged[f"answer_{CAPTION_FIELD}_pred"].astype(str).tolist()

    raw_cider = _cider_score(captions_pred, captions_gt)
    cider_norm = min(raw_cider / cider_scale, 1.0) if not np.isnan(raw_cider) else float("nan")

    bert = _bertscore_f1(captions_pred, captions_gt)

    # --- final score ---
    if np.isnan(cider_norm):
        s_fetv = float("nan")
    else:
        s_fetv = 0.25 * cider_norm + 0.25 * bert + 0.5 * macro_f1

    return {
        "per_field": per_field,
        "macro_f1": macro_f1,
        "cider_raw": raw_cider,
        "cider_norm": cider_norm,
        "bertscore_f1": bert,
        "S_FETV": s_fetv,
    }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _check_categorical(sub_df: pd.DataFrame, field: str) -> list[tuple[str, str]]:
    """Return list of (clip_name, reason) for out-of-vocabulary predictions."""
    allowed = ALLOWED_VALUES[field]
    col = f"answer_{field}"
    bad = []
    for _, row in sub_df.iterrows():
        val = _normalise_str(row.get(col))
        if not val:
            bad.append((row["clip_name"], f"{col}: empty"))
        elif val not in allowed:
            bad.append((row["clip_name"], f"{col}: unknown value {val!r}"))
    return bad


def _check_date(sub_df: pd.DataFrame) -> list[tuple[str, str]]:
    col = "answer_date"
    bad = []
    for _, row in sub_df.iterrows():
        val = _normalise_str(row.get(col))
        try:
            datetime.strptime(val, "%Y-%m-%d")
        except ValueError:
            bad.append((row["clip_name"], f"{col}: not YYYY-MM-DD ({val!r})"))
    return bad


def _check_time(sub_df: pd.DataFrame) -> list[tuple[str, str]]:
    col = "answer_time"
    bad = []
    for _, row in sub_df.iterrows():
        val = _normalise_str(row.get(col))
        if _parse_time_sec(val) is None:
            bad.append((row["clip_name"], f"{col}: not HH:MM:SS ({val!r})"))
    return bad


def validate(gt_df: pd.DataFrame, sub_df: pd.DataFrame,
             allow_missing: bool = False) -> dict:
    """Validate submission format. Prints a report and returns a summary dict."""
    required_cols = {"clip_name"} | {f"answer_{f}" for f in ALL_ANSWER_FIELDS}
    missing_cols = required_cols - set(sub_df.columns)
    if missing_cols:
        raise ValueError(f"submission is missing columns: {sorted(missing_cols)}")

    dup_mask = sub_df["clip_name"].duplicated()
    if dup_mask.any():
        dups = sub_df.loc[dup_mask, "clip_name"].tolist()[:5]
        raise ValueError(f"submission has duplicate clip_name values: {dups}")

    gt_clips = set(gt_df["clip_name"])
    sub_clips = set(sub_df["clip_name"])
    missing_clips = gt_clips - sub_clips
    extra_clips = sub_clips - gt_clips

    print(f"Submission: {len(sub_df)} clips  (expected {EXPECTED_CLIPS})")
    print(f"GT:         {len(gt_df)} clips")
    print()
    print("Coverage:")
    if missing_clips:
        msg = f"{len(missing_clips)} GT clip(s) have no submitted prediction"
        if allow_missing:
            print(f"  ! {msg} (--allow-missing set; will score remainder)")
        else:
            for c in sorted(missing_clips)[:5]:
                print(f"      missing: {c}")
            raise ValueError(f"{msg}; pass --allow-missing to score the remainder.")
    else:
        print("  ok   every GT clip has a submitted prediction")
    if extra_clips:
        print(f"  warn {len(extra_clips)} submitted clip(s) not in GT (ignored)")

    print()
    print("Per-field format validation:")
    summary: dict[str, dict] = {}
    total_bad = 0

    def _report(field, bad_list):
        nonlocal total_bad
        marker = "ok  " if not bad_list else "warn"
        n_bad = len(bad_list)
        n_ok = len(sub_df) - n_bad
        print(f"  {marker} {field:26} {n_ok}/{len(sub_df)} valid")
        for clip, reason in bad_list[:3]:
            print(f"           - {clip}: {reason}")
        if n_bad > 3:
            print(f"           ... and {n_bad - 3} more")
        summary[field] = {"total": len(sub_df), "valid": n_ok, "bad_clips": [b[0] for b in bad_list]}
        total_bad += n_bad

    _report(DATE_FIELD, _check_date(sub_df))
    _report(TIME_FIELD, _check_time(sub_df))
    for field in CATEGORICAL_FIELDS:
        _report(field, _check_categorical(sub_df, field))

    # description: just check non-empty
    desc_bad = [
        (row["clip_name"], "answer_description: empty")
        for _, row in sub_df.iterrows()
        if not _normalise_str(row.get("answer_description"))
    ]
    _report(CAPTION_FIELD, desc_bad)

    print()
    if total_bad:
        print(
            f"Submission validates with {total_bad} field value(s) that may not "
            "score correctly."
        )
    else:
        print("Submission validates cleanly — all fields are well-formed.")

    return {
        "total_clips": len(gt_df),
        "submitted_clips": len(sub_df),
        "missing_clips": len(missing_clips),
        "extra_clips": len(extra_clips),
        "fields": summary,
    }


# ---------------------------------------------------------------------------
# GT loading
# ---------------------------------------------------------------------------

def _load_json(path: str) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a JSON array at the top level")
    df = pd.DataFrame(data)
    if "clip_name" not in df.columns:
        raise ValueError(f"{path}: missing required 'clip_name' field")
    return df


# ---------------------------------------------------------------------------
# Top-level evaluate
# ---------------------------------------------------------------------------

def evaluate(gt_path: str, submission_path: str,
             allow_missing: bool = False,
             validate_only: bool = False,
             cider_scale: float = 10.0) -> dict:
    gt_df = _load_json(gt_path)
    sub_df = _load_json(submission_path)

    redacted = _is_redacted(gt_df)
    do_score = (not validate_only) and (not redacted)

    summary = validate(gt_df, sub_df, allow_missing=allow_missing)

    if do_score:
        metrics = score(gt_df, sub_df, cider_scale=cider_scale)
        return {"mode": "score", "metrics": metrics, "validation": summary}

    reason = "validate-only flag" if validate_only else "GT answers are redacted"
    return {"mode": "validate", "reason": reason, "validation": summary}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(
        description="FETV evaluator + submission validator (Track 7).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--gt", required=True, metavar="JSON",
                   help="Path to GT JSON (same array schema as submission).")
    p.add_argument("--submission", required=True, metavar="JSON",
                   help="Path to submission JSON.")
    p.add_argument("--validate", action="store_true",
                   help="Run only format validation, even if GT has real answers.")
    p.add_argument("--allow-missing", action="store_true",
                   help="Score remainder when some GT clips have no submitted prediction.")
    p.add_argument("--cider-scale", type=float, default=10.0, metavar="FLOAT",
                   help="Divisor used to normalise raw CIDEr to [0,1] (default: 10.0).")
    p.add_argument("--out", default=None, metavar="CSV",
                   help="Optional CSV path to write the per-field metrics table.")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    result = evaluate(
        args.gt, args.submission,
        allow_missing=args.allow_missing,
        validate_only=args.validate,
        cider_scale=args.cider_scale,
    )

    print()
    if result["mode"] == "score":
        m = result["metrics"]
        print("=== Per-field scores ===")
        for field, val in m["per_field"].items():
            print(f"  {field:26} {val:.4f}")
        print()
        print(f"  {'MacroF1':26} {m['macro_f1']:.4f}")
        print(f"  {'CIDEr (raw)':26} {m['cider_raw']:.4f}")
        print(f"  {'CIDEr (norm)':26} {m['cider_norm']:.4f}")
        print(f"  {'BERTScore F1':26} {m['bertscore_f1']:.4f}")
        print()
        print(f"  S_FETV = 0.25·CIDErnorm + 0.25·BERTScore + 0.5·MacroF1")
        print(f"         = 0.25·{m['cider_norm']:.4f} + 0.25·{m['bertscore_f1']:.4f}"
              f" + 0.5·{m['macro_f1']:.4f}")
        print(f"         = {m['S_FETV']:.4f}")
        if args.out:
            flat = {**m["per_field"],
                    "macro_f1": m["macro_f1"],
                    "cider_raw": m["cider_raw"],
                    "cider_norm": m["cider_norm"],
                    "bertscore_f1": m["bertscore_f1"],
                    "S_FETV": m["S_FETV"]}
            pd.DataFrame([flat]).to_csv(args.out, index=False)
            print(f"\nwrote {args.out}", file=sys.stderr)
    else:
        print(f"=== Validation-only mode ({result['reason']}) ===")
        print("No scores computed. Submit to the evaluation server for leaderboard metrics.")


if __name__ == "__main__":
    main()
