#!/usr/bin/env python3
"""Reconsider MCQ and MCQ-openended answers after BCQ correction.

The pass injects BCQ facts as elimination constraints, reconciles paired option
sets through fresh votes, and can run an option-free perceptual probe for
suspicious fault or sequence questions.
"""
import argparse
import difflib
import json
import os
import re
import subprocess
import sys
import zlib
from collections import Counter, defaultdict
from pathlib import Path

from eval_aicity import extract_mcq_letter, extract_answer
from fix_lowconf_rag_bcq import (
    load_model_and_processor,
    load_rag_index,
    probe_video_facts,
    reask,
    _bcq_stem,
    _PROBE_ELIGIBLE_RE,
)


def vid_shard(vid, num_shards):
    """Stable shard owner for a video (crc32, not Python hash(): must agree across
    processes and runs so every stage of a video runs in the same worker and the
    parent knows which shard's output to take when merging)."""
    return zlib.crc32(vid.encode()) % num_shards


def _shard_vids(vids, args, stage):
    n = getattr(args, "num_shards", 1) or 1
    if n <= 1:
        return vids
    mine = [v for v in vids if vid_shard(v, n) == args.shard_index]
    print(f"  [{stage}] shard {args.shard_index}/{n}: {len(mine)}/{len(vids)} videos")
    return mine


def _strip_leading_mcq_label(text):
    """Drop a redundant leading option letter ("C.", "C)", "(C):", ...) since the
    caller re-prepends the canonical "<letter>. " itself (without this, the
    committed prediction would read e.g. "C. C) The gray sedan ...")."""
    m = re.match(r"^\(?([A-Da-d])\)?[.,:)]?\s+", text)
    return text[m.end():].strip() if m else text


_OPT_LINE_RE = re.compile(r"^\s*\(?([A-D])[\).:]\s+(.*\S)\s*$")


def parse_mcq_options(question):
    """{letter: option_text} for the A-D lines of an MCQ question block."""
    opts = {}
    for line in question.splitlines():
        m = _OPT_LINE_RE.match(line)
        if m:
            opts[m.group(1)] = m.group(2).strip()
    return opts


def _norm_opt(text):
    return re.sub(r"\s+", " ", (text or "").strip()).rstrip(".").lower()


def rec_letter(rec, opts=None):
    """The chosen option letter for a voted mcq/mcq_openended record (in the record's
    OWN option numbering), or None if it can't be resolved."""
    opts = parse_mcq_options(rec["question"]) if opts is None else opts
    letter = (rec.get("label") or "").strip().upper()[:1]
    if letter not in opts:
        letter = extract_mcq_letter(rec.get("raw_output") or rec.get("prediction", ""))
    return letter if letter in opts else None


def align_letter(letter, src_opts, dst_opts, min_score):
    """Map option `letter` (numbering of src_opts) to its best counterpart letter in
    dst_opts by text similarity, because the mcq and mcq_openended variants of the same
    question paraphrase the options (different colors/wording) and reorder them.
    Returns (dst_letter, score); dst_letter is None if the best match is below
    `min_score` or ambiguous (runner-up within 0.05), so unrelated option sets are not
    force-mapped."""
    if letter not in src_opts:
        return None, 0.0
    src = _norm_opt(src_opts[letter])
    scored = sorted(((difflib.SequenceMatcher(None, src, _norm_opt(t)).ratio(), L)
                     for L, t in dst_opts.items()), reverse=True)
    if not scored:
        return None, 0.0
    best_s, best_L = scored[0]
    if best_s < min_score:
        return None, best_s
    if len(scored) >= 2 and best_s - scored[1][0] < 0.05:
        return None, best_s
    return best_L, best_s


def set_choice(rec, letter, raw=None, opts=None):
    """Write `letter` (rec's own numbering) as rec's answer. Returns True on success."""
    opts = parse_mcq_options(rec["question"]) if opts is None else opts
    if letter not in opts:
        return False
    rec["label"] = letter
    if rec["task_type"] == "mcq_openended":
        # Prefer the model's own freshly-reconsidered explanation (from `raw`) so the
        # text is grounded in this specific re-ask, not just a copy of the option.
        # Fall back to the option's own text only when no raw output is available
        # (e.g. the letter came from the OTHER twin via alignment, so there's no
        # explanation of our own to use).
        expl = None
        if raw is not None:
            expl = _strip_leading_mcq_label(extract_answer(raw).strip())
            # Guard against the reconsidered raw actually having argued for a
            # DIFFERENT letter than the one we're committing to here (e.g. it was
            # accepted via plurality/pooled-vote logic, not because this exact raw
            # picked `letter`) -- in that case the text would contradict the label.
            if extract_mcq_letter(raw) != letter:
                expl = None
            # A letter-only answer ("C") strips to nothing useful; fall back.
            elif len(expl) <= 1:
                expl = None
        if expl:
            rec["prediction"] = f"{letter}. {expl}"
        else:
            rec["prediction"] = f"{letter}. {opts[letter]}"
    else:
        rec["prediction"] = letter
    return True

def unify_mcq_oe(records, rag_index, model, processor, args, mcq_fields, maps, changed):
    """Cross-check each mcq against its mcq_openended twin (paired by video_id; each
    video has exactly one of each). The two variants paraphrase and reorder their
    options, so answers are compared by aligning options by similarity, not by letter:
      - already the same option, both confidently (>= --recheck-agreement) -> leave alone
      - already the same option, but either side is low-confidence -> still reconsider
        both, since agreeing with itself is not the same as being correct (a hard
        question can get the same wrong answer from both variants).
      - different option        -> reconsider both independently (each via a
        --reconsider-samples-sample vote; CONFIRMING the current letter needs only a
        plurality, but CHANGING to a new letter needs >= --reconsider-min-votes; an
        inconclusive vote on EITHER side leaves the whole pair untouched), and
          * if they now pick the aligned same option -> unify the pair to it
          * if they still differ -> unify to the winner of the POOLED fresh vote
            counts (oe letters mapped into mcq letter space by alignment); on a
            pooled tie fall back to the strictly-higher stale vote_agreement; if
            that also ties the pair is left unchanged.
    If the two option sets can't be aligned confidently (--align-min-score), the pair
    is left as-is rather than force-unified."""
    mcq_by, oe_by = {}, {}
    for r in records:
        if r["task_type"] == "mcq":
            mcq_by[r["video_id"]] = r
        elif r["task_type"] == "mcq_openended":
            oe_by[r["video_id"]] = r
    vids = sorted(set(mcq_by) & set(oe_by))
    thr = args.align_min_score
    k = args.reconsider_samples
    need = args.reconsider_min_votes
    print(f"\n[mcq<->mcq_openended] {len(vids)} paired videos "
          f"(mcq={len(mcq_by)}, mcq_openended={len(oe_by)}); align-min-score={thr}; "
          f"recheck-agreement={args.recheck_agreement}; "
          f"reconsider-vote={need}/{k} @ T={args.reconsider_temperature}")
    if getattr(args, "probe_fault", False):
        # The perception-first stage handles suspicious fault and sequence pairs
        # with an option-free probe, so leave those pairs to that stage.
        handoff = set(probe_pair_vids(records, args)[0])
        n_before = len(vids)
        vids = [v for v in vids if v not in handoff]
        print(f"  handing {n_before - len(vids)} suspicious fault/sequence pairs "
              f"to the probe stage (--probe-fault)")
    vids = _shard_vids(vids, args, "unify")

    def reask_letter(rec, opts, cur):
        """Run a k-sample reconsideration vote with asymmetric acceptance:
          - CONFIRMING the current letter `cur` needs only to tie-or-win the fresh
            plurality (the original 5-vote already backs it);
                    - CHANGING to a different letter needs >= --reconsider-min-votes of the
                        --reconsider-samples fresh samples.
          - anything else is inconclusive -> None (caller leaves the pair alone).
        Returns (letter_or_None, raw_of_winning_letter, counts_dict_or_None-if-no-RAG)."""
        counts = Counter()
        raw_by = {}
        for i in range(k):
            temp = 0.0 if i == 0 else args.reconsider_temperature
            raw = reask(rec, rag_index, model, processor, args, mcq_fields, maps,
                        vote_hint=args.vote_hint, fault_hint=args.fault_hint, temperature=temp)
            if raw is None:
                return None, None, None  # no RAG match
            letter = extract_mcq_letter(raw)
            if letter in opts:
                counts[letter] += 1
                raw_by.setdefault(letter, raw)
        if not counts:
            return None, None, {}
        best, n = counts.most_common(1)[0]
        if cur and counts.get(cur, 0) == n:
            return cur, raw_by[cur], dict(counts)  # current letter confirmed (plurality)
        if n >= need:
            return best, raw_by[best], dict(counts)  # supermajority -> change accepted
        return None, None, dict(counts)  # inconclusive -> keep status quo

    def tag(rec, letter):
        """'{task_type}[{item_index}]={letter}' -- unambiguous id for log lines."""
        return f"{rec['task_type']}[{rec['item_index']}]={letter}"

    for vid in vids:
        m, o = mcq_by[vid], oe_by[vid]
        name = vid.split("/")[-1]
        mo, oo = parse_mcq_options(m["question"]), parse_mcq_options(o["question"])
        old = {r["item_index"]: (r.get("label"), r["prediction"]) for r in (m, o)}
        lm, lo = rec_letter(m, mo), rec_letter(o, oo)
        aligned, _ = align_letter(lm, mo, oo, thr) if lm else (None, 0.0)
        agr_m0 = m.get("vote_agreement") or 0.0
        agr_o0 = o.get("vote_agreement") or 0.0
        if lm and lo and aligned == lo:
            if min(agr_m0, agr_o0) >= args.recheck_agreement:
                continue  # already consistent AND both confident
            print(f"  {name}: already agree ({tag(m, lm)} ~ {tag(o, lo)}) but "
                  f"low-confidence (mcq_agr={agr_m0}, oe_agr={agr_o0}) -- rechecking both")

        rlm, rawm, cm = reask_letter(m, mo, lm)
        rlo, rawo, co = reask_letter(o, oo, lo)
        r_aligned, _ = align_letter(rlm, mo, oo, thr) if rlm else (None, 0.0)
        print(f"  {name}: reconsider vote -> {tag(m, rlm)} {cm} (was {lm}), "
              f"{tag(o, rlo)} {co} (was {lo}) "
              f"[None = inconclusive: neither confirmed nor {need}/{k} for a change]")
        if rlm is None or rlo is None:
            # Without conclusive fresh evidence on both sides, leave the pair
            # unchanged rather than forcing alignment from the original letters.
            print(f"  {name}: inconclusive reconsider "
                  f"({'mcq' if rlm is None else 'oe'} side) -- pair left unchanged")
            continue
        if r_aligned == rlo:
            set_choice(m, rlm, rawm, mo)
            set_choice(o, rlo, rawo, oo)
            src = "reconsider-agree"
            outcome = f"BOTH reconsidered to the same option -> {tag(m, rlm)}, {tag(o, rlo)}"
        else:
            # If both fresh votes are conclusive but still disagree, use pooled
            # counts after mapping OE letters into MCQ option space.
            pooled = Counter(cm or {})
            pool_ok = True
            for L, n in (co or {}).items():
                mapped, _ = align_letter(L, oo, mo, thr)
                if mapped is None:
                    pool_ok = False
                    break
                pooled[mapped] += n
            best = None
            if pool_ok and pooled:
                ranked = pooled.most_common()
                if len(ranked) == 1 or ranked[0][1] > ranked[1][1]:
                    best = ranked[0][0]
            agr_m = m.get("vote_agreement") or 0.0
            agr_o = o.get("vote_agreement") or 0.0
            if best is not None:
                tgt_oe, sc = align_letter(best, mo, oo, thr)
                if tgt_oe is None:
                    print(f"  {name}: pooled fresh vote picked mcq={best} but it doesn't "
                          f"align to an oe option (score={sc:.2f} < {thr}) -- left unchanged")
                    continue
                set_choice(m, best, rawm if best == rlm else None, mo)
                set_choice(o, tgt_oe, rawo if tgt_oe == rlo else None, oo)
                src = f"pooled-fresh:{dict(pooled)}"
                outcome = (f"still disagree -> pooled fresh votes {dict(pooled)} pick "
                           f"{tag(m, best)} / {tag(o, tgt_oe)} (align={sc:.2f})")
            else:
                # On a pooled tie or unmappable options, use stale agreement; if
                # that also ties, leave the pair unchanged.
                if agr_m == agr_o:
                    print(f"  {name}: still disagree, pooled fresh vote tied "
                          f"({dict(pooled)}) and vote_agreement tied ({agr_m}) "
                          f"-- pair left unchanged")
                    continue
                if agr_m > agr_o:
                    winner, wl, wraw, wopts, loser, lopts = m, rlm, rawm, mo, o, oo
                else:
                    winner, wl, wraw, wopts, loser, lopts = o, rlo, rawo, oo, m, mo
                set_choice(winner, wl, wraw, wopts)  # settle winner on its own pick
                tgt, sc = align_letter(wl, wopts, lopts, thr)
                if tgt is None:
                    print(f"  {name}: still disagree and {loser['task_type']}'s options don't "
                          f"align with {tag(winner, wl)} (score={sc:.2f} < {thr}) -- left unchanged")
                    continue
                set_choice(loser, tgt, None, lopts)
                src = f"votes:{winner['task_type']}(mcq_agr={agr_m},oe_agr={agr_o},align={sc:.2f})"
                outcome = (f"still disagree, pooled fresh vote tied ({dict(pooled)}) -> kept "
                           f"higher-agreement {tag(winner, wl)} (mcq_agr={agr_m} vs "
                           f"oe_agr={agr_o}), mapped onto {tag(loser, tgt)} (align={sc:.2f})")

        changed_here = []
        for rec in (m, o):
            old_label, old_pred = old[rec["item_index"]]
            new_label = rec.get("label")
            if new_label != old_label or rec["prediction"] != old_pred:
                rec["mcq_unify"] = {"src": src, "chosen": outcome,
                                    "old_label": old_label, "old_prediction": old_pred}
                changed.append(rec["item_index"])
                changed_here.append(f"{rec['task_type']}[{rec['item_index']}]: {old_label} -> {new_label}")
        if changed_here:
            print(f"  {name}: UNIFIED ({outcome}) -- " + "; ".join(changed_here))
        else:
            print(f"  {name}: {outcome} -- both already matched this option, no change")


def probe_pair_vids(records, args):
    """The set of video_ids the probe stage will handle: fault/sequence pairs
    (fix_lowconf_rag_bcq._PROBE_ELIGIBLE_RE) that are suspicious (either twin's
    original vote_agreement <= --probe-suspect-agreement, or the twins' current
    answers disagree under alignment). Shared by unify_mcq_oe (which must SKIP
    these when --probe-fault is on) and by probe_fault_stage itself."""
    mcq_by, oe_by = {}, {}
    for r in records:
        if r["task_type"] == "mcq":
            mcq_by[r["video_id"]] = r
        elif r["task_type"] == "mcq_openended":
            oe_by[r["video_id"]] = r
    thr = args.align_min_score
    max_agr = args.probe_suspect_agreement

    def suspicious(m, o):
        agr = min(m.get("vote_agreement") or 0.0, o.get("vote_agreement") or 0.0)
        if agr <= max_agr:
            return True
        lm = rec_letter(m)
        aligned, _ = align_letter(lm, parse_mcq_options(m["question"]),
                                  parse_mcq_options(o["question"]), thr) if lm else (None, 0.0)
        return aligned != rec_letter(o)

    elig = [v for v in sorted(set(mcq_by) & set(oe_by))
            if _PROBE_ELIGIBLE_RE.search(mcq_by[v]["question"])
            or _PROBE_ELIGIBLE_RE.search(oe_by[v]["question"])]
    return [v for v in elig if suspicious(mcq_by[v], oe_by[v])], len(elig)


def probe_fault_stage(records, rag_index, model, processor, args, mcq_fields, maps, changed):
    """Perception-first re-ask for fault/sequence mcq pairs (including pairs that
    already AGREE -- the entrenched errors are exactly the agreeing-wrong ones that
    unify_mcq_oe skips). For each eligible video:

      1. probe the raw perceptual facts (PROBE_QUESTIONS: who hits whom, from where,
         entry order, signal color per approach) with NO options shown;
      2. re-ask BOTH twins with the probed observations injected (old vote hint
         suppressed -- it anchors the model back onto the entrenched majority);
      3. change the pair only if both sides reach a >= --probe-min-votes
         supermajority of --probe-samples fresh samples AND land on the same
         aligned option. Anything less keeps the status quo.

    To keep runtime down, only SUSPICIOUS pairs are probed (GT-free filter): either
    twin's original vote_agreement <= --probe-suspect-agreement, or the twins'
    current answers still disagree under alignment. Confident agreeing pairs
    (both 1.0, aligned) are skipped; set --probe-suspect-agreement 1.0 to probe
    every eligible pair regardless.

    The change gate is 2x unanimity across two paraphrased option sets on top of a
    changed perception -- deliberately the strictest gate in the pipeline, since it
    is allowed to overturn currently-agreeing answers."""
    mcq_by, oe_by = {}, {}
    for r in records:
        if r["task_type"] == "mcq":
            mcq_by[r["video_id"]] = r
        elif r["task_type"] == "mcq_openended":
            oe_by[r["video_id"]] = r
    thr = args.align_min_score
    vids, n_elig = probe_pair_vids(records, args)
    k = args.probe_samples
    need = args.probe_min_votes if args.probe_min_votes is not None else max(k - 1, 1)
    print(f"\n[probe-fault] {len(vids)} suspicious of {n_elig} eligible fault/sequence "
          f"videos (agr<={args.probe_suspect_agreement} or twins disagree); "
          f"gate = {need}/{k} supermajority on BOTH twins + aligned agreement")
    vids = _shard_vids(vids, args, "probe")

    def probe_vote(rec, opts, facts):
        counts = Counter()
        raw_by = {}
        for i in range(k):
            temp = 0.0 if i == 0 else args.reconsider_temperature
            raw = reask(rec, rag_index, model, processor, args, mcq_fields, maps,
                        vote_hint=False, fault_hint=False, temperature=temp,
                        probe_facts=facts)
            if raw is None:
                return None, None
            letter = extract_mcq_letter(raw)
            if letter in opts:
                counts[letter] += 1
                raw_by.setdefault(letter, raw)
        if not counts:
            return None, None
        best, n = counts.most_common(1)[0]
        print(f"    {rec['task_type']}[{rec['item_index']}]: probe vote {dict(counts)}")
        if n < need:  # per-twin supermajority; the pair-level AND is the real gate
            return None, None
        return best, raw_by[best]

    for vid in vids:
        m, o = mcq_by[vid], oe_by[vid]
        name = vid.split("/")[-1]
        mo, oo = parse_mcq_options(m["question"]), parse_mcq_options(o["question"])
        lm, lo = rec_letter(m, mo), rec_letter(o, oo)
        key = (vid.strip("/"), m["question"].strip())
        rag_entry = rag_index.get(key) or {}
        facts = probe_video_facts(vid, model, processor, args,
                                  frame_ranges=rag_entry.get("relevant_frame_ranges"))
        if facts is None:
            print(f"  {name}: probe inference failed -- skipped")
            continue
        print(f"  {name}: probed facts:\n    " + facts.replace("\n", "\n    "))
        pm, rawm = probe_vote(m, mo, facts)
        po, rawo = probe_vote(o, oo, facts)
        if pm is None or po is None:
            print(f"  {name}: not unanimous on both twins -- left unchanged "
                  f"(mcq={lm}, oe={lo})")
            continue
        p_aligned, sc = align_letter(pm, mo, oo, thr)
        if p_aligned != po:
            print(f"  {name}: unanimous but twins disagree "
                  f"(mcq={pm} -> aligns to {p_aligned}, oe={po}) -- left unchanged")
            continue
        if pm == lm and po == lo:
            print(f"  {name}: unanimously re-confirmed current answers "
                  f"(mcq={lm}, oe={lo}) -- no change")
            continue
        old = {r["item_index"]: (r.get("label"), r["prediction"]) for r in (m, o)}
        set_choice(m, pm, rawm, mo)
        set_choice(o, po, rawo, oo)
        for rec in (m, o):
            old_label, old_pred = old[rec["item_index"]]
            if rec.get("label") != old_label or rec["prediction"] != old_pred:
                rec["probe_fix"] = {"src": f"probe-unanimous({k}/{k}x2,align={sc:.2f})",
                                    "old_label": old_label, "old_prediction": old_pred}
                changed.append(rec["item_index"])
        print(f"  {name}: PROBE-FLIPPED mcq {lm} -> {pm}, oe {lo} -> {po} "
              f"(unanimous {k}/{k} on both, align={sc:.2f})")


def _strip_arg(argv, name, has_value=True):
    """Remove `name` (and its value, both "--x v" and "--x=v" forms) from argv."""
    out, skip = [], False
    for a in argv:
        if skip:
            skip = False
            continue
        if a == name:
            skip = has_value
            continue
        if a.startswith(name + "="):
            continue
        out.append(a)
    return out


def launch_shards(args):
    """Parent mode: spawn one worker per GPU (CUDA_VISIBLE_DEVICES pinned, videos
    partitioned by stable crc32 hash so unify of a pair stays in one worker), then
    merge the shard outputs -- each record is taken from the shard that OWNS its
    video, so workers never conflict."""
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    n = len(gpus)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    child_argv = _strip_arg(sys.argv[1:], "--gpus")
    procs = []
    for i, gpu in enumerate(gpus):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
        log_path = args.output_dir / f"shard{i}.log"
        cmd = [sys.executable, "-u", str(Path(__file__).resolve())] + child_argv + [
            "--shard-index", str(i), "--num-shards", str(n)]
        print(f"[shard {i}] GPU {gpu} -> {log_path}")
        procs.append((i, subprocess.Popen(cmd, env=env, stdout=open(log_path, "w"),
                                          stderr=subprocess.STDOUT)))
    failed = []
    for i, p in procs:
        if p.wait() != 0:
            failed.append(i)
    for i, _ in procs:
        print(f"\n{'='*20} shard {i} log {'='*20}")
        print((args.output_dir / f"shard{i}.log").read_text())
    if failed:
        sys.exit(f"shards {failed} FAILED -- not merging (see logs above)")

    # ---- merge: take each record from the shard that owns its video ----
    base = [json.loads(l) for l in open(args.predictions)]
    shard_recs = []
    for i in range(n):
        fp = args.output_dir / f"predictions_mcq_fixed.shard{i}.jsonl"
        shard_recs.append({r["item_index"]: r for r in map(json.loads, open(fp))})
    merged, changed = [], 0
    for r in base:
        owner = vid_shard(r["video_id"], n)
        mr = shard_recs[owner].get(r["item_index"], r)
        if mr.get("label") != r.get("label") or mr["prediction"] != r["prediction"]:
            changed += 1
        merged.append(mr)
    out = args.output_dir / "predictions_mcq_fixed.jsonl"
    with open(out, "w") as f:
        for r in merged:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nMerged {n} shards, {changed} items changed vs input")
    print(f"Wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--base-model", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--lora", action="store_true", default=False)
    ap.add_argument("--sft-adapter-dir", default=None,
                    help="SFT LoRA adapter dir to apply+merge BEFORE the --model-dir "
                         "adapter (required for GRPO checkpoints trained on a merged "
                         "SFT base).")
    ap.add_argument("--predictions", required=True,
                    help="The predictions_lowconf_fixed.jsonl written by "
                         "fix_lowconf_rag_bcq.py (already has corrected bcq/bcq_openended).")
    ap.add_argument("--descriptions", type=Path, default=None,
                    help="Full-task jsonl for scene_description / temporal_description / "
                         "causal_linkage / video_summarization / open_qa context.")
    ap.add_argument("--rag-dir", type=Path, required=True)
    ap.add_argument("--video-dir", type=Path, required=True)
    ap.add_argument("--max-frames", type=int, default=100)
    ap.add_argument("--ctx-fields", default="scene,temporal,bcq",
                    help="Comma-separated context fields for the reconsider (any of "
                         "scene,temporal,causal,summary,open_qa,bcq). 'bcq' uses the "
                         "already-fixed bcq answers of the same video. Empty = no context.")
    ap.add_argument("--unify-mcq-oe", action=argparse.BooleanOptionalAction, default=True,
                    help="Cross-check each mcq against its mcq_openended twin and unify "
                         "them (see unify_mcq_oe above).")
    ap.add_argument("--probe-fault", action=argparse.BooleanOptionalAction, default=False,
                    help="After unify, run the perception-first probe stage on "
                         "fault/sequence mcq pairs (probe raw facts with no options, "
                         "then re-ask both twins; change only on unanimous x2 aligned "
                         "agreement). Can overturn currently-AGREEING pairs, which "
                         "unify never touches.")
    ap.add_argument("--probe-suspect-agreement", type=float, default=0.8,
                    help="Probe only pairs where either twin's original vote_agreement "
                         "is <= this, or the twins' answers still disagree (GT-free "
                         "suspicion filter; cuts runtime ~3x). 1.0 = probe all eligible.")
    ap.add_argument("--probe-samples", type=int, default=5,
                    help="Fresh samples per twin in the probe stage; a change needs "
                         "--probe-min-votes on both twins.")
    ap.add_argument("--probe-min-votes", type=int, default=None,
                    help="Per-twin supermajority needed to flip a pair in the probe "
                         "stage (default: probe-samples - 1).")
    ap.add_argument("--align-min-score", type=float, default=0.5,
                    help="Min difflib ratio to consider an mcq option the counterpart of "
                         "an mcq_openended option; below this the pair is left as-is.")
    ap.add_argument("--vote-hint", action=argparse.BooleanOptionalAction, default=True,
                    help="Include the earlier 5-sample vote_counts breakdown as a soft "
                         "hint in the reconsider prompt.")
    ap.add_argument("--fault-hint", action=argparse.BooleanOptionalAction, default=False,
                    help="Inject a fault-attribution note for root-cause questions "
                         "(default OFF; found net-negative).")
    ap.add_argument("--recheck-agreement", type=float, default=0.0,
                    help="In --unify-mcq-oe, reconsider an already-agreeing pair anyway "
                         "if either side's vote_agreement is below this (default 0.0 = off).")
    ap.add_argument("--reconsider-samples", type=int, default=5,
                    help="In --unify-mcq-oe, how many samples to draw per reconsider "
                         "(first at temp=0, rest at --reconsider-temperature).")
    ap.add_argument("--reconsider-temperature", type=float, default=0.7,
                    help="Sampling temperature for the non-greedy reconsider samples.")
    ap.add_argument("--reconsider-min-votes", type=int, default=5,
                    help="Fresh votes required to CHANGE a letter (confirming the current "
                        "letter only needs a plurality; default is unanimous).")
    ap.add_argument("--gpus", default=None,
                    help="Comma-separated GPU ids (e.g. 0,1,2,3,4). When given, spawns "
                         "one worker per GPU, partitions videos across them (stable "
                         "hash), and merges the results. Omit for single-process.")
    ap.add_argument("--shard-index", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--num-shards", type=int, default=1, help=argparse.SUPPRESS)
    ap.add_argument("--output-dir", "-o", type=Path, required=True)
    args = ap.parse_args()

    if args.gpus and args.shard_index is None:
        launch_shards(args)
        return

    mcq_fields = [f.strip() for f in args.ctx_fields.split(",") if f.strip()]

    # ---- load the bcq-fixed predictions ----
    records = [json.loads(l) for l in open(args.predictions)]

    # ---- build the per-video description context maps ----
    desc_src = [json.loads(l) for l in open(args.descriptions)] if args.descriptions else records
    scene_by_vid = {r["video_id"]: r["prediction"] for r in desc_src if r["task_type"] == "scene_description"}
    temporal_by_vid = {r["video_id"]: r["prediction"] for r in desc_src if r["task_type"] == "temporal_description"}
    causal_by_vid = {r["video_id"]: r["prediction"] for r in desc_src if r["task_type"] == "causal_linkage"}
    summary_by_vid = {r["video_id"]: r["prediction"] for r in desc_src if r["task_type"] == "video_summarization"}
    open_qa_by_vid = {r["video_id"]: r["prediction"] for r in desc_src if r["task_type"] == "open_qa"}
    print(f"scene for {len(scene_by_vid)} vids, temporal for {len(temporal_by_vid)} vids, "
          f"causal for {len(causal_by_vid)} vids, summary for {len(summary_by_vid)} vids, "
          f"open_qa for {len(open_qa_by_vid)} vids")

    maps = {"scene": scene_by_vid, "temporal": temporal_by_vid, "causal": causal_by_vid,
            "summary": summary_by_vid, "open_qa": open_qa_by_vid, "bcq": {}}

    print("Loading RAG evidence index...")
    rag_index = load_rag_index(args.rag_dir)
    print(f"  {len(rag_index)} evidence entries")
    model, processor = load_model_and_processor(args.model_dir, args.base_model, args.lora,
                                                sft_adapter_dir=args.sft_adapter_dir)

    changed = []

    # ---- build bcq context from the (already-corrected) bcq answers ----
    bcq_ctx = defaultdict(list)
    for r in records:
        if r["task_type"] == "bcq":
            label = (r.get("label") or r["prediction"]).strip().split()[0].rstrip(".,").capitalize()
            bcq_ctx[r["video_id"]].append(f"- {_bcq_stem(r['question'])} -> {label}")
    maps["bcq"] = {v: "\n" + "\n".join(lines) for v, lines in bcq_ctx.items()}
    if "bcq" in mcq_fields:
        print(f"\nbcq facts built for {len(maps['bcq'])} videos (used as elimination "
              f"constraints in the mcq/mcq_openended reconsider)")

    # ---- mcq <-> mcq_openended cross-consistency unify ----
    if args.unify_mcq_oe:
        unify_mcq_oe(records, rag_index, model, processor, args, mcq_fields, maps, changed)

    # ---- perception-first probe for fault/sequence questions ----
    if args.probe_fault:
        probe_fault_stage(records, rag_index, model, processor, args, mcq_fields, maps, changed)

    print(f"\nChanged {len(changed)} items total")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    name = ("predictions_mcq_fixed.jsonl" if args.shard_index is None
            else f"predictions_mcq_fixed.shard{args.shard_index}.jsonl")
    out = args.output_dir / name
    with open(out, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
