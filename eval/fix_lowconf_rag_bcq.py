#!/usr/bin/env python3
"""RAG-aware reconsideration for bcq/bcq_openended violation pairs (the "one Yes +
one No per video" rule) of a RAG-trained model, driven by the vote metadata from
vote_rag_mcq_bcq.py.

For each same-label violation pair, re-ask the model with the RAG evidence plus
its own scene_description / temporal_description context for that video. If
independent reconsideration naturally yields one Yes + one No, use that.
Otherwise (with --force-bcq) try a joint re-ask that shows the model both
questions together and asks it to split them; if that also fails, fall back to
flipping the lower-vote_agreement member.

Usage:
  python eval/fix_lowconf_rag_bcq.py \
      --model-dir /output/model_checkpoint --lora \
      --predictions /output/vote_mcqbcq/predictions_voted.jsonl,/output/vote_mcqbcq_oe/predictions_voted.jsonl \
      --descriptions eval/aicity_test/predictions/rag_test_predictions.jsonl \
      --rag-dir /data/RAG_Stage2_test_new/tar_test \
      --video-dir /data --tasks bcq,bcq_openended \
      --force-bcq \
      -o /output/dp0_lowconf_fix
"""
import argparse
import glob
import json
import re
from collections import defaultdict
from pathlib import Path

from eval_aicity import run_inference, extract_mcq_letter, extract_bcq_label, extract_answer

RECONSIDER_MAX_NEW_TOKENS = 2048

SYS_PROMPT = (
    "You are a video question-answering assistant. Answer the question about the "
    "video.\n\nYou may also get a block of retrieved evidence with two kinds of context:\n"
    "1. Cross-question context: factual_information is usually reliable; "
    "potential_information is only a weak hint and can be wrong.\n"
    "2. Visual context: scene descriptions, summaries, captions, and tracked "
    "objects from an automated pipeline that may be noisy.\n\n"
    "Base your answer on what the video shows, and trust the video when the "
    "evidence conflicts with it. Reason step by step inside <think></think>, then "
    "give the final answer inside <answer></answer>."
)

RECONSIDER_NOTE = "\n\nLook at the video again carefully, using the retrieved evidence above, and answer."


def load_model_and_processor(model_dir, base_model, lora, sft_adapter_dir=None):
    """Load the model, optionally chaining the SFT adapter before the GRPO one.

    For GRPO checkpoints trained on a merged SFT base, pass `sft_adapter_dir`:
    base -> apply+merge SFT adapter -> apply+merge GRPO adapter (`model_dir`).
    """
    import torch
    from peft import PeftModel
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    if lora:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            base_model, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", device_map="auto")
        if sft_adapter_dir:
            print(f"Applying SFT LoRA adapter from {sft_adapter_dir} (merged first) ...")
            model = PeftModel.from_pretrained(model, sft_adapter_dir).merge_and_unload()
        model = PeftModel.from_pretrained(model, model_dir).merge_and_unload()
    else:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_dir, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", device_map="auto")
    model.eval()
    processor = AutoProcessor.from_pretrained(base_model, fix_mistral_regex=True)
    return model, processor


def load_rag_index(rag_dir):
    index = {}
    for fp in glob.glob(str(Path(rag_dir) / "**" / "*.json"), recursive=True):
        try:
            doc = json.load(open(fp))
        except (json.JSONDecodeError, OSError):
            continue
        if not (isinstance(doc, dict) and "results" in doc):
            continue
        vid = (doc.get("video_id") or "").strip("/")
        if not vid:
            continue
        for r in doc["results"]:
            q = (r.get("question") or "").strip()
            if q:
                index[(vid, q)] = r
    return index


# context field -> label shown to the model
CTX_LABELS = {
    "scene": "Scene description",
    "temporal": "Temporal description",
    "causal": "Causal linkage",
    "summary": "Video summarization",
    "open_qa": "Open QA",
    "bcq": "Related yes/no facts about this video",
}


def ctx_parts_for(vid, enabled, maps):
    """List of (label, text) context lines for this video, in `enabled` order."""
    parts = []
    for field in enabled:
        text = maps.get(field, {}).get(vid)
        if text:
            parts.append((CTX_LABELS[field], text))
    return parts


def format_vote_hint(rec):
    """Format the earlier 5-sample majority-vote breakdown (vote_counts, from
    vote_rag_mcq_bcq.py) as a soft hint for reconsideration: it shows how split the
    earlier independent sampling was, WITHOUT anchoring the model on a single "the
    old answer was X" framing (which just gets re-confirmed/rubber-stamped). A
    genuinely tied/split vote is a signal the question is hard, not that any one
    option is right, so it's framed as a hint to weigh, not a prior to defer to.
    Returns None if there's no usable vote data."""
    counts = rec.get("vote_counts")
    if not counts:
        return None
    total = sum(counts.values())
    if total <= 0:
        return None
    breakdown = ", ".join(f"{k}: {v}/{total}" for k, v in
                          sorted(counts.items(), key=lambda kv: -kv[1]))
    return ("Note: earlier, this exact question was answered independently multiple "
            f"times (via sampling); the votes were: {breakdown}. This shows how split "
            "the earlier attempts were -- it is a hint, not necessarily correct, so "
            "weigh it against the video and evidence and trust your own fresh analysis.")


_FAULT_ATTRIBUTION_RE = re.compile(
    r"root cause|at fault|primary traffic violation|committed the (?:primary )?violation|"
    r"responsible for the collision|which vehicle is primarily", re.IGNORECASE)

FAULT_ATTRIBUTION_HINT = (
    "Note: this question asks about legal fault at an intersection. Do NOT assume that "
    "a vehicle proceeding straight automatically has the right-of-way, or that a turning, "
    "entering, or lane-changing vehicle is automatically at fault. Instead, for each "
    "vehicle involved, identify its direction of travel and check the color of the traffic "
    "signal that specifically governs THAT vehicle's own direction at the moment it enters "
    "the intersection or lane (use the evidence/context above if it mentions signal states). "
    "The vehicle that proceeded against a red signal for its own direction, or failed to "
    "yield when its own signal did not grant the right-of-way, is at fault -- regardless of "
    "whether it was going straight, turning, merging, or changing lanes."
)


def is_fault_attribution_question(question):
    return bool(_FAULT_ATTRIBUTION_RE.search(question or ""))


# ---------------------------------------------------------------------------
# neutral perceptual probes for fault/sequence questions
# ---------------------------------------------------------------------------
# The persistent mcq errors share one signature: both twins agree on a wrong
# fault narrative, usually blaming the top-to-bottom vehicle when the vehicle
# entering from the side was the violator (or vice versa). Re-asking the mcq --
# even with elimination constraints -- just re-confirms the narrative, because
# the wrong perception is baked into how the model watches the video with the
# options in front of it. These probes ask for the underlying PERCEPTUAL facts
# with no options shown, so the answer cannot be pattern-matched to a narrative.

PROBE_QUESTIONS = [
    "Which two vehicles collide in this video? For EACH of the two, state: its color and "
    "type; the direction it enters the frame from (top / bottom / left / right); whether it "
    "is going straight, turning, or changing lanes; and which of the two entered the "
    "intersection or conflict area FIRST. Finally state which vehicle strikes which (whose "
    "front hits whose side or rear).",
    "Focus only on the traffic signals in this video. Just before the collision, what is "
    "the color of the traffic signal governing EACH of the two colliding vehicles' OWN "
    "direction of travel? Be careful: a signal visible in the frame may govern a different "
    "approach than the vehicle nearest to it -- match each signal to the approach it faces. "
    "If no signal governs a vehicle's approach, say so. Then state which vehicle (if "
    "either) entered against a red signal for its own direction, or failed to yield the "
    "right-of-way.",
]

_PROBE_ELIGIBLE_RE = re.compile(
    r"root cause|at fault|primarily|sequence of events|violation", re.IGNORECASE)


def probe_video_facts(vid, model, processor, args, frame_ranges=None):
    """Greedy-ask the neutral PROBE_QUESTIONS about the video (video only, no RAG
    evidence, so the probe is independent of possibly-wrong pipeline captions).
    Returns a formatted observations block, or None if inference fails."""
    video_abs = str(args.video_dir / vid)
    obs = []
    for q in PROBE_QUESTIONS:
        video_ele = {"type": "video", "video": video_abs,
                     "relevant_frame_ranges": frame_ranges or [],
                     "max_frames": args.max_frames}
        prompt = (q + "\n\nReason step by step inside <think></think>, then give your "
                      "final factual observations inside <answer></answer>.")
        messages = [{"role": "system", "content": SYS_PROMPT},
                    {"role": "user", "content": [video_ele, {"type": "text", "text": prompt}]}]
        raw = run_inference(messages, model, processor, RECONSIDER_MAX_NEW_TOKENS, temperature=0.0)
        if raw:
            obs.append(extract_answer(raw).strip())
    if not obs:
        return None
    return ("An independent close inspection of this video reported the following "
            "observations (verify them against the video; they may contain errors):\n"
            + "\n".join(f"- {o}" for o in obs))


def build_messages(video_abs, result, max_frames, ctx_parts, vote_hint=None, fault_hint=None,
                   bcq_facts=None, probe_facts=None):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from eval_aicity_rag_test import _video_fps
    from qwenvl.data.data_processor_rag import format_evidence_rag
    fps = _video_fps(video_abs)
    evidence = format_evidence_rag(result, fps)
    ctx = ""
    if ctx_parts:
        ctx = "For extra context, here is an earlier analysis of this video:\n"
        for label, text in ctx_parts:
            ctx += f"{label}: \"{text}\"\n"
        ctx += "\n"
    if bcq_facts:
        # The bcq answers of this video are verified-reliable, so use them as ACTIVE
        # elimination constraints, not passive context: elimination is a different
        # operation than selection and can dislodge a consistent wrong prior.
        ctx += ("The following yes/no facts about this video have been independently "
                "verified and are established:" + bcq_facts + "\n\n"
                "Before answering, check each option against these established facts and "
                "ELIMINATE every option that contradicts any of them; then choose the best "
                "remaining option based on what the video shows.\n\n")
    if probe_facts:
        ctx += probe_facts + "\n\n"
    if vote_hint:
        ctx += vote_hint + "\n\n"
    if fault_hint:
        ctx += fault_hint + "\n\n"
    question = ctx + result["question"] + RECONSIDER_NOTE
    question_text = (evidence + "\n\n" + question) if evidence else question
    video_ele = {"type": "video", "video": video_abs,
                 "relevant_frame_ranges": result.get("relevant_frame_ranges") or [],
                 "max_frames": max_frames}
    return [{"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": [video_ele, {"type": "text", "text": question_text}]}]


def reask(rec, rag_index, model, processor, args, enabled, maps, vote_hint=False, fault_hint=False,
          temperature=0.0, probe_facts=None):
    vid = rec["video_id"]
    key = (vid.strip("/"), rec["question"].strip())
    result = rag_index.get(key)
    if result is None:
        return None
    hint = format_vote_hint(rec) if vote_hint else None
    fhint = FAULT_ATTRIBUTION_HINT if (fault_hint and is_fault_attribution_question(rec["question"])) else None
    # For mcq/mcq_openended, pull the (verified) bcq facts out of the passive context
    # and pass them as active elimination constraints instead.
    bcq_facts = None
    enabled_ctx = enabled
    if rec["task_type"] in ("mcq", "mcq_openended") and "bcq" in enabled:
        bcq_facts = maps.get("bcq", {}).get(vid)
        enabled_ctx = [f for f in enabled if f != "bcq"]
    messages = build_messages(str(args.video_dir / vid), result, args.max_frames,
                              ctx_parts_for(vid, enabled_ctx, maps), vote_hint=hint, fault_hint=fhint,
                              bcq_facts=bcq_facts, probe_facts=probe_facts)
    return run_inference(messages, model, processor, RECONSIDER_MAX_NEW_TOKENS, temperature=temperature)


_JOINT_BCQ_RE = re.compile(r"Q1\s*:\s*(Yes|No).*?Q2\s*:\s*(Yes|No)", re.IGNORECASE | re.DOTALL)


def _bcq_stem(q):
    return q.split("Answer with")[0].strip()


def force_bcq_joint(a, b, rag_index, model, processor, args, enabled, maps):
    """Last resort before the blind vote_agreement tie-break: show the model BOTH
    paired questions together (with RAG evidence for each + shared extra context)
    and ask it to directly decide which one is Yes and which is No, since exactly
    one must be. This gives the model real signal to split the pair instead of an
    arbitrary flip when vote_agreement is tied (very common, e.g. 1.0 vs 1.0).

    Returns (label_a, label_b, raw) with label_a != label_b, or None if the model
    couldn't produce a clean, split answer (caller should fall back)."""
    vid = a["video_id"]
    key_a = (vid.strip("/"), a["question"].strip())
    key_b = (vid.strip("/"), b["question"].strip())
    result_a = rag_index.get(key_a)
    result_b = rag_index.get(key_b)
    if result_a is None and result_b is None:
        return None

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from eval_aicity_rag_test import _video_fps
    from qwenvl.data.data_processor_rag import format_evidence_rag

    video_abs = str(args.video_dir / vid)
    fps = _video_fps(video_abs)
    evidence_bits = []
    if result_a is not None:
        ev = format_evidence_rag(result_a, fps)
        if ev:
            evidence_bits.append(f"Evidence for Q1 (\"{_bcq_stem(a['question'])}\"):\n{ev}")
    if result_b is not None:
        ev = format_evidence_rag(result_b, fps)
        if ev:
            evidence_bits.append(f"Evidence for Q2 (\"{_bcq_stem(b['question'])}\"):\n{ev}")
    evidence = "\n\n".join(evidence_bits)

    ctx = ""
    ctx_parts = ctx_parts_for(vid, enabled, maps)
    if ctx_parts:
        ctx = "For extra context, here is an earlier analysis of this video:\n"
        for label, text in ctx_parts:
            ctx += f"{label}: \"{text}\"\n"
        ctx += "\n"

    prompt = (
        f"{ctx}You will be asked two related yes/no questions about this video:\n"
        f"Q1: {_bcq_stem(a['question'])}\n"
        f"Q2: {_bcq_stem(b['question'])}\n\n"
        "Exactly one of these two questions has the answer \"Yes\" and the other has "
        "the answer \"No\" — they describe mutually exclusive aspects of the same event. "
        "Look at the video again carefully, using the retrieved evidence above, and "
        "decide which is which.\n\n"
        "Reason step by step inside <think></think>, then give your final answer inside "
        "<answer></answer> in EXACTLY this format on one line: \"Q1: Yes/No, Q2: Yes/No\"."
    )
    question_text = (evidence + "\n\n" + prompt) if evidence else prompt
    video_ele = {"type": "video", "video": video_abs, "relevant_frame_ranges": [], "max_frames": args.max_frames}
    messages = [{"role": "system", "content": SYS_PROMPT},
                {"role": "user", "content": [video_ele, {"type": "text", "text": question_text}]}]
    raw = run_inference(messages, model, processor, RECONSIDER_MAX_NEW_TOKENS, temperature=0.0)
    if raw is None:
        return None
    ans = extract_answer(raw) or raw
    m = _JOINT_BCQ_RE.search(ans)
    if not m:
        return None
    la, lb = m.group(1).capitalize(), m.group(2).capitalize()
    if la == lb:
        return None
    return la, lb, raw


def _strip_leading_label(text):
    """Drop a redundant leading "Yes"/"No" token (with trailing punctuation) since
    the caller re-prepends the (possibly different, flipped) canonical label."""
    m = re.match(r"^(yes|no)\b[.,:]?\s*", text, re.IGNORECASE)
    return text[m.end():].strip() if m else text


def regenerate_bcq_openended_explanation(rec, label, rag_index, model, processor, args, enabled, maps):
    """For bcq_openended, once the Yes/No label has been flipped (force-joint or
    force-fallback), the old explanation text was reasoned toward the OLD label and
    would contradict the new one. Re-ask the model for a fresh explanation, giving
    the flipped label as a settled prior fact, so the explanation is regrounded in
    the video/evidence and stays consistent with the new answer.

    Returns the new "Label. explanation" prediction string, or None if there's no
    RAG match / inference fails (caller should fall back to the old behavior)."""
    vid = rec["video_id"]
    key = (vid.strip("/"), rec["question"].strip())
    result = rag_index.get(key)
    if result is None:
        return None
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from eval_aicity_rag_test import _video_fps
    from qwenvl.data.data_processor_rag import format_evidence_rag
    video_abs = str(args.video_dir / vid)
    fps = _video_fps(video_abs)
    evidence = format_evidence_rag(result, fps)
    ctx = ""
    ctx_parts = ctx_parts_for(vid, enabled, maps)
    if ctx_parts:
        ctx = "For extra context, here is an earlier analysis of this video:\n"
        for lbl, text in ctx_parts:
            ctx += f"{lbl}: \"{text}\"\n"
        ctx += "\n"
    prompt = (
        f"{ctx}Question: {_bcq_stem(rec['question'])}\n\n"
        f"The correct answer to this question has been determined to be \"{label}\". "
        "Look at the video again carefully, using the retrieved evidence above, and "
        "write a brief explanation that supports this answer, grounded in what the "
        "video actually shows.\n\n"
        "Reason step by step inside <think></think>, then give the final answer inside "
        f"<answer></answer> in EXACTLY this format: \"{label}. <brief explanation>\"."
    )
    question_text = (evidence + "\n\n" + prompt) if evidence else prompt
    video_ele = {"type": "video", "video": video_abs,
                 "relevant_frame_ranges": result.get("relevant_frame_ranges") or [],
                 "max_frames": args.max_frames}
    messages = [{"role": "system", "content": SYS_PROMPT},
                {"role": "user", "content": [video_ele, {"type": "text", "text": question_text}]}]
    raw = run_inference(messages, model, processor, RECONSIDER_MAX_NEW_TOKENS, temperature=0.0)
    if raw is None:
        return None
    expl = _strip_leading_label(extract_answer(raw).strip())
    return f"{label}. {expl}".strip()


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
                    help="Comma-separated vote jsonl file(s) (mcqbcq and/or oe).")
    ap.add_argument("--descriptions", type=Path, default=None,
                    help="Full-task jsonl for scene_description / temporal_description / "
                         "causal_linkage / video_summarization / open_qa context.")
    ap.add_argument("--rag-dir", type=Path, required=True)
    ap.add_argument("--video-dir", type=Path, required=True)
    ap.add_argument("--max-frames", type=int, default=100)
    ap.add_argument("--tasks", default="bcq,bcq_openended")
    ap.add_argument("--force-bcq", action="store_true", default=True,
                    help="For bcq/bcq_openended violation pairs that don't split after "
                         "reconsideration, FORCE one Yes + one No by flipping the "
                         "lower-vote_agreement member.")
    ap.add_argument("--bcq-ctx-fields", default="scene,temporal",
                    help="Comma-separated context fields for the reconsider prompt "
                         "(any of scene,temporal,causal,summary,open_qa).")
    ap.add_argument("--output-dir", "-o", type=Path, required=True)
    args = ap.parse_args()
    bcq_fields = [f.strip() for f in args.bcq_ctx_fields.split(",") if f.strip()]

    tasks = {t.strip() for t in args.tasks.split(",") if t.strip()}
    records = []
    for p in args.predictions.split(","):
        records += [json.loads(l) for l in open(p.strip())]

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
            "summary": summary_by_vid, "open_qa": open_qa_by_vid}

    print("Loading RAG evidence index...")
    rag_index = load_rag_index(args.rag_dir)
    print(f"  {len(rag_index)} evidence entries")
    model, processor = load_model_and_processor(args.model_dir, args.base_model, args.lora,
                                                sft_adapter_dir=args.sft_adapter_dir)

    changed = []

    def bcq_stem(q):
        return q.split("Answer with")[0].strip()

    # ---- bcq / bcq_openended FIRST: same-label pair reconsider (+ optional force) ----
    for task in ("bcq", "bcq_openended"):
        if task not in tasks:
            continue
        byvid = defaultdict(list)
        for r in records:
            if r["task_type"] == task:
                byvid[r["video_id"]].append(r)
        pairs = [rs for rs in byvid.values() if len(rs) == 2
                 and (rs[0].get("label") or "").capitalize() == (rs[1].get("label") or "").capitalize()
                 and (rs[0].get("label") or "").capitalize() in ("Yes", "No")]
        print(f"\n[{task}] {len(pairs)} same-label violation pairs")
        for a, b in pairs:
            vid = a["video_id"]
            outs = []
            for rec in (a, b):
                raw = reask(rec, rag_index, model, processor, args, bcq_fields, maps)
                outs.append((extract_bcq_label(raw), raw) if raw is not None else None)
            if outs[0] is None or outs[1] is None:
                print(f"  {vid.split('/')[-1]}: no RAG match, skip")
                continue
            (la, ra), (lb, rb) = outs
            if la in ("Yes", "No") and lb in ("Yes", "No") and la != lb:
                for rec, nl, raw in ((a, la, ra), (b, lb, rb)):
                    old_pred = rec["prediction"]
                    rec["prediction"] = (extract_answer(raw).strip() if task == "bcq_openended" else nl)
                    rec["label"] = nl
                    rec["lowconf_fix"] = {"src": "reconsider", "old_prediction": old_pred}
                    changed.append(rec["item_index"])
                    print(f"  {vid.split('/')[-1]} [{rec['item_index']}]: {old_pred[:25]!r} -> {nl} (reconsider)")
            elif args.force_bcq:
                # No natural split -> first try a JOINT re-ask that shows the model
                # both questions together and asks it to directly decide which is
                # Yes and which is No (real signal, instead of an arbitrary flip).
                joint = force_bcq_joint(a, b, rag_index, model, processor, args, bcq_fields, maps)
                if joint is not None:
                    la2, lb2, raw_joint = joint
                    cur = (a.get("label") or "").capitalize()  # both same, pre-flip
                    flipped, kept = (a, b) if la2 != cur else (b, a)
                    agr_flip = flipped.get("vote_agreement") or 0
                    agr_kept = kept.get("vote_agreement") or 0
                    if agr_flip > agr_kept:
                        # The joint re-ask flipped the item that was ALREADY the more
                        # confident one (higher voIte_agreement) -- that contradicts the
                        # confidence signal, so distrust this joint split and fall through
                        # to the vote-based heuristic below instead of trusting it blindly.
                        print(f"  {vid.split('/')[-1]}: FORCE-JOINT flipped the higher-agreement "
                              f"item ({flipped['item_index']}: {agr_flip} > {agr_kept}), discarding joint result")
                        joint = None
                if joint is not None:
                    la2, lb2, raw_joint = joint
                    for rec, nl in ((a, la2), (b, lb2)):
                        old_pred = rec["prediction"]
                        if task == "bcq_openended":
                            # The joint raw only has one shared Q1/Q2 answer line, not a
                            # natural per-question explanation -- regenerate a fresh one
                            # grounded in the (now decided) flipped label.
                            new_pred = regenerate_bcq_openended_explanation(
                                rec, nl, rag_index, model, processor, args, bcq_fields, maps)
                            if new_pred is None:
                                expl = extract_answer(raw_joint).strip()
                                new_pred = f"{nl}. {expl}".strip()
                            rec["prediction"] = new_pred
                        else:
                            rec["prediction"] = nl
                        rec["label"] = nl
                        rec["lowconf_fix"] = {"src": "force-joint", "old_prediction": old_pred}
                        changed.append(rec["item_index"])
                        print(f"  {vid.split('/')[-1]} [{rec['item_index']}]: FORCE-JOINT -> {nl}")
                else:
                    # Fallback: the old heuristic, flipping the member with the lower
                    # vote_agreement to the opposite of the other's label. Only reached
                    # when the joint re-ask couldn't produce a clean split (e.g. no RAG
                    # match, or the model still refused to split even head-to-head).
                    cur = (a.get("label") or "").capitalize()  # both same
                    keep, flip = (a, b) if (a.get("vote_agreement") or 0) >= (b.get("vote_agreement") or 0) else (b, a)
                    new_label = "No" if cur == "Yes" else "Yes"
                    raw_flip = rb if flip is b else ra
                    old_pred = flip["prediction"]
                    if task == "bcq_openended":
                        # The old explanation was reasoned toward the PRE-flip label and
                        # would contradict new_label -- regenerate a fresh one grounded in
                        # the flipped answer instead of just re-prefixing stale text.
                        new_pred = regenerate_bcq_openended_explanation(
                            flip, new_label, rag_index, model, processor, args, bcq_fields, maps)
                        if new_pred is None:
                            expl = _strip_leading_label(extract_answer(raw_flip).strip())
                            new_pred = f"{new_label}. {expl}".strip()
                        flip["prediction"] = new_pred
                    else:
                        flip["prediction"] = new_label
                    flip["label"] = new_label
                    flip["lowconf_fix"] = {"src": "force-fallback", "old_prediction": old_pred,
                                           "flipped_lower_agreement": flip.get("vote_agreement"),
                                           "kept_agreement": keep.get("vote_agreement")}
                    changed.append(flip["item_index"])
                    print(f"  {vid.split('/')[-1]} [{flip['item_index']}]: FORCE-FALLBACK {cur}->{new_label} "
                          f"(flipped lower agree={flip.get('vote_agreement')}, kept={keep.get('vote_agreement')})")
            else:
                print(f"  {vid.split('/')[-1]}: no split, unchanged (reconsidered A={la} B={lb})")

    print(f"\nChanged {len(changed)} items total")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / "predictions_lowconf_fixed.jsonl"
    with open(out, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
