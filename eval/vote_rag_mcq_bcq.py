#!/usr/bin/env python3
"""5-sample majority voting for mcq+bcq on a RAG-trained model (e.g. dp0 /
dp0_new_grpo), using RAG-formatted messages (video + retrieved evidence +
question) instead of the background-prompt format.

This exists because those models' baseline eval (eval_aicity_rag_test.py) is
a single greedy pass with no confidence signal, unlike the background-LoRA
pipeline's eval_aicity_vote.py. This script produces the same kind of
`vote_agreement`/`vote_stable` metadata so the downstream bcq/mcq fix
scripts (fix_bcq_pair_violations_rag.py, and an mcq analog) have a
confidence signal to work with.

Same voting design as eval_aicity_vote.py's run_voted_closed: k=5 samples
(1 greedy + 4 sampled), majority vote; if unstable, one "informed tie-break"
pass showing the model its own prior answers and asking it to commit to one
final answer (instead of blindly re-asking or forcing more frames).

Usage (single GPU):
    python eval/vote_rag_mcq_bcq.py \
      --model-dir /output/model_checkpoint --lora \
      --test-json data/dataset/test/tar_test/test.json \
      --rag-dir /data/RAG_Stage2_test_new/tar_test \
      --video-dir /data \
      -o /output/vote_mcqbcq

Usage (multi-GPU, one shard process per GPU, then merge):
    for rank in 0 1 2 3 4; do
      CUDA_VISIBLE_DEVICES=$rank python eval/vote_rag_mcq_bcq.py \
        --model-dir /output/model_checkpoint --lora \
        --test-json data/dataset/test/tar_test/test.json \
        --rag-dir /data/RAG_Stage2_test_new/tar_test \
        --video-dir /data \
        --tasks mcq_openended,bcq_openended \
        -o /output/vote_mcqbcq_oe --shard-rank $rank --shard-size 5 &
    done
    wait
    python eval/vote_rag_mcq_bcq.py -o /output/vote_mcqbcq_oe --shard-size 5 --merge-shards
"""
import argparse
import glob
import json
import sys
import time
from collections import Counter
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

_THIS_DIR = Path(__file__).resolve().parent
_TRAIN_DIR = _THIS_DIR.parent / "train"
for _p in (_THIS_DIR, _TRAIN_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from eval_aicity import run_inference, extract_mcq_letter, extract_bcq_label, extract_answer  # noqa: E402
from qwenvl.data.data_processor_rag import format_evidence_rag, _video_fps  # noqa: E402

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


# ---------------------------------------------------------------------------
# Model / data loading
# ---------------------------------------------------------------------------

def load_model_and_processor(model_dir, base_model, lora):
    if lora:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            base_model, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2",
            device_map="auto",
        )
        model = PeftModel.from_pretrained(model, model_dir)
        model = model.merge_and_unload()
    else:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_dir, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2",
            device_map="auto",
        )
    model.eval()
    processor = AutoProcessor.from_pretrained(base_model, fix_mistral_regex=True)
    return model, processor


def load_rag_index(rag_dir):
    index = {}
    files = glob.glob(str(Path(rag_dir) / "**" / "*.json"), recursive=True)
    for fp in files:
        try:
            doc = json.load(open(fp))
        except (json.JSONDecodeError, OSError):
            continue
        if not (isinstance(doc, dict) and "results" in doc):
            continue
        video_id = (doc.get("video_id") or "").strip("/")
        if not video_id:
            continue
        for r in doc["results"]:
            q = (r.get("question") or "").strip()
            if q:
                index[(video_id, q)] = r
    return index


def build_messages(video_abs, result, max_frames, question_suffix=""):
    fps = _video_fps(video_abs)
    evidence = format_evidence_rag(result, fps)
    question = result["question"] + question_suffix
    question_text = (evidence + "\n\n" + question) if evidence else question
    video_ele = {
        "type": "video", "video": video_abs,
        "relevant_frame_ranges": result.get("relevant_frame_ranges") or [],
        "max_frames": max_frames,
    }
    return [
        {"role": "system", "content": SYS_PROMPT},
        {"role": "user", "content": [video_ele, {"type": "text", "text": question_text}]},
    ]


# ---------------------------------------------------------------------------
# Voting
# ---------------------------------------------------------------------------

def discrete_label(task, raw):
    if task in ("mcq", "mcq_openended"):
        lab = extract_mcq_letter(raw)
        return lab if lab in ("A", "B", "C", "D") else None
    if task in ("bcq", "bcq_openended"):
        lab = extract_bcq_label(raw)
        return lab if lab in ("Yes", "No") else None
    return None


def vote(labels):
    valid = [lab for lab in labels if lab is not None]
    if not valid:
        return None, 0, 0, {}
    counts = Counter(valid)
    label, count = counts.most_common(1)[0]
    return label, count, len(valid), dict(counts)


def sample_and_vote(video_abs, result, task, model, processor, args):
    """Run k samples (1 greedy + k-1 sampled at --vote-temperature) and take a
    majority vote. If the votes aren't stable enough (see --vote-agreement),
    fall back to one "informed tie-break" pass that shows the model its own
    prior answers and asks it to commit to a single final answer.

    Returns (raw_output, label, vote_counts, agreement, stable, source).
    """
    messages = build_messages(video_abs, result, args.max_frames)
    raws, labels = [], []
    for k in range(max(1, args.vote_samples)):
        temp = 0.0 if k == 0 else args.vote_temperature
        raw = run_inference(messages, model, processor, args.max_new_tokens, temperature=temp)
        raws.append(raw)
        labels.append(discrete_label(task, raw))

    label, count, n_valid, votes = vote(labels)
    agreement = (count / n_valid) if n_valid else 0.0
    if n_valid and agreement >= args.vote_agreement:
        raw = next(r for r, lab in zip(raws, labels) if lab == label)
        return raw, label, votes, agreement, True, "vote"

    tally = "\n".join(f"- {lbl}: chosen {c} time(s)" for lbl, c in votes.items()) or "- (no clear answer extracted)"
    note = ("\n\nYou were asked this exact question multiple times and gave different "
            "answers:\n" + tally + "\nLook at the video again carefully and give ONE "
            "final, definitive answer.")
    tb_messages = build_messages(video_abs, result, args.max_frames, note)
    raw = run_inference(tb_messages, model, processor, args.max_new_tokens, temperature=0.0)
    label = discrete_label(task, raw) or label
    return raw or raws[0], label, votes, agreement, False, "tiebreak"


def process_item(item, rag_index, model, processor, args):
    """Run vote-and-tiebreak for one test item. Returns a result record, or
    None if there's no matching RAG evidence for this (video_id, question).
    """
    vid = item["video_id"]
    result = rag_index.get((vid.strip("/"), item["question"].strip()))
    if result is None:
        return None

    video_abs = str(args.video_dir / vid)
    raw, label, votes, agreement, stable, source = sample_and_vote(
        video_abs, result, item["task_type"], model, processor, args)

    return {
        "item_index": item["item_index"], "task_type": item["task_type"], "video_id": vid,
        "question": item["question"], "raw_output": raw,
        "prediction": extract_answer(raw), "label": label,
        "vote_counts": votes, "vote_agreement": round(agreement, 3),
        "vote_stable": stable, "vote_source": source,
    }


# ---------------------------------------------------------------------------
# Shard merging / resume support
# ---------------------------------------------------------------------------

def merge_shards(output_dir: Path, shard_size: int) -> Path:
    pred_dir = output_dir / "predictions"
    merged = []
    for rank in range(shard_size):
        shard_path = pred_dir / f"predictions_voted_shard{rank}.jsonl"
        if not shard_path.exists():
            raise FileNotFoundError(f"Missing shard prediction file: {shard_path}")
        with open(shard_path) as f:
            merged.extend(json.loads(line) for line in f if line.strip())
    merged_path = output_dir / "predictions_voted.jsonl"
    with open(merged_path, "w") as f:
        for r in merged:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Merged {len(merged)} predictions from {shard_size} shard(s) -> {merged_path}")
    return merged_path


def load_done_ids(path: Path) -> set:
    """Return item_indexes already written to `path` (resume support)."""
    done = set()
    if path.exists():
        with open(path) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["item_index"])
                except (json.JSONDecodeError, KeyError):
                    pass
    return done


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=None,
                     help="Required unless --merge-shards is given.")
    ap.add_argument("--base-model", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--lora", action="store_true", default=False)
    ap.add_argument("--test-json", type=Path, default=None,
                     help="Required unless --merge-shards is given.")
    ap.add_argument("--rag-dir", type=Path, default=None,
                     help="Required unless --merge-shards is given.")
    ap.add_argument("--video-dir", type=Path, default=None,
                     help="Required unless --merge-shards is given.")
    ap.add_argument("--max-frames", type=int, default=100)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--vote-samples", type=int, default=5)
    ap.add_argument("--vote-temperature", type=float, default=0.7)
    ap.add_argument("--vote-agreement", type=float, default=0.6)
    ap.add_argument("--tasks", default="mcq,bcq",
                     help="Comma-separated task_types to vote on, e.g. 'mcq_openended,bcq_openended'.")
    ap.add_argument("--output-dir", "-o", type=Path, required=True)
    ap.add_argument("--shard-rank", type=int, default=0,
                     help="This process's shard index (0-based). Pin it to a single GPU with "
                          "CUDA_VISIBLE_DEVICES=<rank>.")
    ap.add_argument("--shard-size", type=int, default=1,
                     help="Total number of shards (= number of GPUs used).")
    ap.add_argument("--merge-shards", action="store_true", default=False,
                     help="Skip inference; just merge predictions_voted_shard{0..shard_size-1}.jsonl "
                          "into predictions_voted.jsonl.")
    args = ap.parse_args()

    if not args.merge_shards:
        missing = [n for n, v in (
            ("--model-dir", args.model_dir), ("--test-json", args.test_json),
            ("--rag-dir", args.rag_dir), ("--video-dir", args.video_dir),
        ) if v is None]
        if missing:
            ap.error(f"the following arguments are required: {', '.join(missing)}")
    return args


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.merge_shards:
        merge_shards(args.output_dir, args.shard_size)
        return

    task_set = {t.strip() for t in args.tasks.split(",") if t.strip()}
    test = json.load(open(args.test_json))
    items = test["items"] if isinstance(test, dict) else test
    all_targets = [it for it in items if it["task_type"] in task_set]
    targets = all_targets[args.shard_rank::args.shard_size]
    print(f"Voting on {len(targets)} items (of {len(all_targets)} total) "
          f"[shard {args.shard_rank + 1}/{args.shard_size}] ({sorted(task_set)})")

    rag_index = load_rag_index(args.rag_dir)
    print(f"RAG evidence index: {len(rag_index)} entries")

    model, processor = load_model_and_processor(args.model_dir, args.base_model, args.lora)

    if args.shard_size > 1:
        pred_dir = args.output_dir / "predictions"
        pred_dir.mkdir(parents=True, exist_ok=True)
        out_path = pred_dir / f"predictions_voted_shard{args.shard_rank}.jsonl"
    else:
        out_path = args.output_dir / "predictions_voted.jsonl"

    # Resume support: skip item_indexes already written by a prior partial run.
    done_ids = load_done_ids(out_path)
    if done_ids:
        print(f"[resume] {len(done_ids)} already-saved predictions found, skipping.")

    n_unmatched = 0
    n_done = 0
    start_time = time.time()
    with open(out_path, "a") as f_out:
        for i, item in enumerate(targets):
            if item["item_index"] in done_ids:
                continue

            record = process_item(item, rag_index, model, processor, args)
            if record is None:
                n_unmatched += 1
                continue

            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            f_out.flush()
            n_done += 1
            elapsed = time.time() - start_time
            rate = elapsed / n_done if n_done else 0.0
            eta = rate * (len(targets) - (i + 1))
            print(f"[shard {args.shard_rank}] {i + 1}/{len(targets)} "
                  f"(done={n_done} unmatched={n_unmatched}) "
                  f"elapsed={elapsed / 60:.1f}m eta={eta / 60:.1f}m", flush=True)

    print(f"\nWrote predictions to {out_path} (unmatched: {n_unmatched}/{len(targets)})")


if __name__ == "__main__":
    main()
