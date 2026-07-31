#!/usr/bin/env python3
"""Force every mcq_openended answer to agree with its mcq twin (paired by
video_id) and use the chosen option's text as the answer/explanation.

Offline: operates directly on a predictions jsonl, no model / GPU needed.

Usage:
  python eval/harmonize_oe_to_mcq.py \
      --predictions /output/dp0_lowconf_fix_mcq_probe/predictions_mcq_fixed.jsonl \
      -o /output/dp0_lowconf_fix_mcq_probe/predictions_oe_harmonized.jsonl
"""
import argparse
import difflib
import json
import re
from pathlib import Path

_OPT_LINE_RE = re.compile(r"^\s*\(?([A-D])[\).:]\s+(.*\S)\s*$")


def parse_mcq_options(question):
    opts = {}
    for line in question.splitlines():
        m = _OPT_LINE_RE.match(line)
        if m:
            opts[m.group(1)] = m.group(2).strip()
    return opts


def _norm_opt(text):
    return re.sub(r"\s+", " ", (text or "").strip()).rstrip(".").lower()


def rec_letter(rec, opts):
    letter = (rec.get("label") or "").strip().upper()[:1]
    if letter in opts:
        return letter
    m = re.match(r"\s*\(?([A-D])\b", rec.get("prediction") or "")
    return m.group(1) if m and m.group(1) in opts else None


def align_letter(letter, src_opts, dst_opts, min_score):
    """Best counterpart of src_opts[letter] in dst_opts by text similarity.
    None if below min_score or ambiguous (runner-up within 0.05)."""
    if letter not in src_opts:
        return None, 0.0
    src = _norm_opt(src_opts[letter])
    scored = sorted(((difflib.SequenceMatcher(None, src, _norm_opt(t)).ratio(), L)
                     for L, t in dst_opts.items()), reverse=True)
    if not scored:
        return None, 0.0
    best_s, best_L = scored[0]
    if best_s < min_score or (len(scored) >= 2 and best_s - scored[1][0] < 0.05):
        return None, best_s
    return best_L, best_s


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--align-min-score", type=float, default=0.5)
    ap.add_argument("-o", "--output", required=True)
    args = ap.parse_args()

    records = [json.loads(l) for l in open(args.predictions) if l.strip()]
    mcq_by, oe_by = {}, {}
    for r in records:
        if r["task_type"] == "mcq":
            mcq_by[r["video_id"]] = r
        elif r["task_type"] == "mcq_openended":
            oe_by[r["video_id"]] = r
    vids = sorted(set(mcq_by) & set(oe_by))
    print(f"{len(vids)} mcq/mcq_openended pairs")

    n_flip = n_text = n_skip = 0
    for vid in vids:
        m, o = mcq_by[vid], oe_by[vid]
        mo, oo = parse_mcq_options(m["question"]), parse_mcq_options(o["question"])
        lm = rec_letter(m, mo)
        if lm is None:
            print(f"  SKIP {vid}: cannot resolve mcq letter")
            n_skip += 1
            continue
        aligned, sc = align_letter(lm, mo, oo, args.align_min_score)
        if aligned is None:
            print(f"  SKIP {vid}: mcq={lm} '{mo[lm][:60]}...' has no confident "
                  f"counterpart in oe options (best={sc:.2f})")
            n_skip += 1
            continue
        lo = rec_letter(o, oo)
        new_pred = f"{aligned}) {oo[aligned]}"
        if lo != aligned:
            print(f"  FLIP {vid}: oe {lo} -> {aligned} (mcq={lm}, align={sc:.2f})")
            n_flip += 1
        elif o["prediction"].strip() != new_pred:
            n_text += 1
        else:
            continue
        o["oe_harmonize"] = {"old_label": lo, "old_prediction": o["prediction"],
                             "mcq_letter": lm, "align_score": round(sc, 3)}
        o["label"] = aligned
        o["prediction"] = new_pred

    print(f"\nflipped choice: {n_flip}; rewrote text only: {n_text}; "
          f"skipped (unalignable): {n_skip}; unchanged: {len(vids) - n_flip - n_text - n_skip}")
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(records)} records -> {out}")


if __name__ == "__main__":
    main()
