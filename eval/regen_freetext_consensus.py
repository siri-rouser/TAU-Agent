#!/usr/bin/env python3
"""Consensus-reranked, cross-task-context regeneration for free-text TAR answers.

Merges `eval_consensus_rerank.py` (medoid reranking over K sampled candidates)
with `regen_freetext_context.py` (re-asking each free-text question using the
model's OWN earlier answers to all four free-text questions of the same video
as extra context). For each target item this script:

  1. builds a "cross-task context" block from an ORIGINAL predictions snapshot
     (all four free-text answers for that video, including this task's own
     previous answer) — exactly like regen_freetext_context.py;
  2. samples K candidates (1 greedy + K-1 at --temperature) for the re-asked
     question with that context block + RAG evidence attached;
  3. picks the medoid (highest mean pairwise similarity to the other
     candidates) as the final prediction — exactly like eval_consensus_rerank.py.

Context always comes from the ORIGINAL snapshot (not from other shards or
regenerated results), so shard order and regeneration order do not matter and
there is no cascading.

Free-text tasks: open_qa, causal_linkage, temporal_description, video_summarization

Similarity scorers:
  --scorer bertscore  (default) roberta-large BERTScore F1, matches the official
                      metric; pairs are batched in one scorer call per item.
  --scorer minilm     sentence-transformers all-MiniLM-L6-v2 cosine; much
                      faster, slightly less aligned with the metric.

Usage (single GPU):
  python eval/regen_freetext_consensus.py \
      --model-dir /output/model_checkpoint --lora \
      --predictions eval/aicity_test/predictions/rag_test_predictions.jsonl \
      --rag-dir /data/RAG_Stage2_test_new/tar_test \
      --video-dir /data \
      --num-samples 5 --temperature 0.7 \
      -o /output/freetext_consensus

Usage (multi-GPU, one shard process per GPU, then merge):
  for rank in 0 1 2 3 4; do
    CUDA_VISIBLE_DEVICES=$rank python eval/regen_freetext_consensus.py \
        --model-dir /output/model_checkpoint --lora \
        --predictions eval/aicity_test/predictions/rag_test_predictions.jsonl \
        --rag-dir /data/RAG_Stage2_test_new/tar_test \
        --video-dir /data \
        --num-samples 5 --temperature 0.7 \
        -o /output/freetext_consensus --shard-rank $rank --shard-size 5 &
  done
  wait
  python eval/regen_freetext_consensus.py \
      --predictions eval/aicity_test/predictions/rag_test_predictions.jsonl \
      -o /output/freetext_consensus --shard-size 5 --merge-shards
"""
import argparse
import csv
import glob
import json
from pathlib import Path
import sys

_THIS_DIR = Path(__file__).resolve().parent
_TRAIN_DIR = _THIS_DIR.parent / "train"
for _p in (_THIS_DIR, _TRAIN_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from eval_aicity import run_inference, extract_answer  # noqa: E402
from eval_aicity_rag_test import (  # noqa: E402
    BACKGROUND_SYS,
    BACKGROUND_SYS_VIDEO_SUM,
    load_model_and_processor,
)

FREETEXT_TASKS = ("open_qa", "causal_linkage", "temporal_description", "video_summarization")

# Human-readable labels used when injecting a sibling answer as context.
TASK_LABELS = {
    "open_qa": "Open question answering",
    "causal_linkage": "Causal linkage between events",
    "temporal_description": "Temporal description of what happens over time",
    "video_summarization": "Video summary",
}

RECONSIDER_NOTE = (
    "\n\nLook at the video again carefully, using the retrieved evidence and the "
    "earlier analysis above, and answer the question."
)


# ---------------------------------------------------------------------------
# Similarity scorers (medoid picking), same as eval_consensus_rerank.py
# ---------------------------------------------------------------------------
class BertScoreSim:
    def __init__(self):
        from bert_score import BERTScorer
        self.scorer = BERTScorer(lang="en", rescale_with_baseline=True)

    def pairwise(self, cands: list[str]) -> list[list[float]]:
        """Full K x K similarity matrix (diagonal = 1)."""
        k = len(cands)
        pairs = [(i, j) for i in range(k) for j in range(k) if i != j]
        preds = [cands[i] for i, _ in pairs]
        refs = [cands[j] for _, j in pairs]
        _, _, f1 = self.scorer.score(preds, refs)
        f1 = f1.tolist()
        mat = [[1.0] * k for _ in range(k)]
        for (i, j), s in zip(pairs, f1):
            mat[i][j] = float(s)
        return mat


class MiniLMSim:
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

    def pairwise(self, cands: list[str]) -> list[list[float]]:
        import numpy as np
        emb = self.model.encode(cands, normalize_embeddings=True)
        return (np.asarray(emb) @ np.asarray(emb).T).tolist()


def pick_medoid(cands: list[str], sim) -> tuple[int, list[float]]:
    """Index of the candidate with the highest mean similarity to the others.

    Candidate 0 is the greedy sample; ties resolve to the lowest index, so
    greedy wins ties. Duplicate answers naturally reinforce each other.
    """
    k = len(cands)
    if k == 1:
        return 0, [1.0]
    mat = sim.pairwise(cands)
    means = [sum(mat[i][j] for j in range(k) if j != i) / (k - 1) for i in range(k)]
    best = max(range(k), key=lambda i: (means[i], -i))
    return best, means


# ---------------------------------------------------------------------------
# Cross-task context + RAG evidence message building (from regen_freetext_context.py)
# ---------------------------------------------------------------------------
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


def build_context_block(task_type, vid, context_by_task):
    """Assemble all four free-text answers, including this task's own previous one."""
    lines = []
    for other in FREETEXT_TASKS:
        ans = (context_by_task.get(other, {}).get(vid) or "").strip()
        suffix = " (your previous answer to this question)" if other == task_type else ""
        lines.append(f"- {TASK_LABELS[other]}{suffix}: \"{ans or '(not available)'}\"")
    if not lines:
        return ""
    return ("For extra context, here is this model's earlier analysis of this same "
            "video (it may be noisy, trust the video):\n"
            + "\n".join(lines) + "\n\n")


def build_messages(video_abs, result, max_frames, context_block, task_type):
    from eval_aicity_rag_test import _video_fps
    from qwenvl.data.data_processor_rag import (
        format_evidence_rag, RAG_BASE_FPS, RAG_DENSE_MULT)
    fps = _video_fps(video_abs)
    evidence = format_evidence_rag(result, fps)
    question = context_block + result["question"] + RECONSIDER_NOTE
    question_text = (evidence + "\n\n" + question) if evidence else question
    video_ele = {
        "type": "video",
        "video": video_abs,
        "relevant_frame_ranges": result.get("relevant_frame_ranges") or [],
        "slowfast_base_fps": RAG_BASE_FPS,
        "slowfast_dense_mult": RAG_DENSE_MULT,
        "max_frames": max_frames,
    }
    sys_prompt = BACKGROUND_SYS_VIDEO_SUM if task_type == "video_summarization" else BACKGROUND_SYS
    return [{"role": "system", "content": sys_prompt},
            {"role": "user", "content": [video_ele, {"type": "text", "text": question_text}]}]


def sample_candidates(rec, rag_index, model, processor, args, context_by_task):
    """Build the context-aware messages, then sample K candidate answers.

    Returns (candidates, raws) — extracted answers and their raw outputs (both
    possibly shorter than num_samples if some samples failed/were empty) — or
    None if the item has no matching RAG evidence.
    """
    import torch
    vid = rec["video_id"]
    key = (vid.strip("/"), rec["question"].strip())
    result = rag_index.get(key)
    if result is None:
        return None
    context_block = build_context_block(rec["task_type"], vid, context_by_task)
    messages = build_messages(str(args.video_dir / vid), result, args.max_frames,
                              context_block, rec["task_type"])

    cands, raws = [], []
    for s in range(args.num_samples):
        temp = 0.0 if s == 0 else args.temperature
        try:
            raw = run_inference(messages, model, processor,
                                args.max_new_tokens, temperature=temp)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"[WARN] OOM on {rec['item_index']} sample {s}")
            raw = ""
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] {rec['item_index']} sample {s}: {type(exc).__name__}: {exc}")
            raw = ""
        ans = extract_answer(raw).strip()
        if ans:
            cands.append(ans)
            raws.append(raw)
    return cands, raws


# ---------------------------------------------------------------------------
# Output helpers (from regen_freetext_context.py)
# ---------------------------------------------------------------------------
def write_outputs(output_dir: Path, records: list):
    """Write the final merged jsonl + submission CSV over ALL records."""
    out = output_dir / "predictions_freetext_consensus.jsonl"
    with open(out, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {out}")

    csv_out = output_dir / "submission_freetext_consensus.csv"
    with open(csv_out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["item_index", "prediction"])
        for r in records:
            writer.writerow([r["item_index"], r.get("prediction", "")])
    print(f"Wrote {csv_out} ({len(records)} items)")


def merge_shard_files(records: list, shard_paths: list) -> list:
    """Overlay the regenerated items from one or more shard jsonl files onto the
    original `records` (by item_index); items missing from every shard keep their
    original value."""
    changed_by_id = {}
    for shard_path in shard_paths:
        if not shard_path.exists():
            raise FileNotFoundError(f"Missing shard prediction file: {shard_path}")
        with open(shard_path) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    changed_by_id[r["item_index"]] = r
    print(f"Merged {len(changed_by_id)} regenerated items from {len(shard_paths)} shard(s)")
    return [changed_by_id.get(r["item_index"], r) for r in records]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", default=None,
                    help="Required unless --merge-shards is given.")
    ap.add_argument("--base-model", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--lora", action="store_true", default=False)
    ap.add_argument("--sft-adapter-dir", default=None,
                    help="SFT LoRA adapter dir to apply+merge BEFORE the --model-dir "
                         "adapter (required for GRPO checkpoints trained on a merged "
                         "SFT base).")
    ap.add_argument("--predictions", required=True,
                    help="Comma-separated full-task jsonl file(s) holding the original "
                         "free-text predictions (e.g. rag_test_predictions.jsonl).")
    ap.add_argument("--rag-dir", type=Path, default=None,
                    help="Required unless --merge-shards is given.")
    ap.add_argument("--video-dir", type=Path, default=None,
                    help="Required unless --merge-shards is given.")
    ap.add_argument("--max-frames", type=int, default=100)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--tasks", default=",".join(FREETEXT_TASKS),
                    help="Which free-text tasks to regenerate (comma-separated).")
    ap.add_argument("--num-samples", "-k", type=int, default=5,
                    help="Candidates per item: 1 greedy + (k-1) sampled.")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--scorer", choices=["bertscore", "minilm"], default="bertscore")
    ap.add_argument("--output-dir", "-o", type=Path, required=True)
    ap.add_argument("--shard-rank", type=int, default=0,
                    help="This process's shard index (0-based). Pin it to a single GPU "
                         "with CUDA_VISIBLE_DEVICES=<rank>.")
    ap.add_argument("--shard-size", type=int, default=1,
                    help="Total number of shards (= number of GPUs used).")
    ap.add_argument("--merge-shards", action="store_true", default=False,
                    help="Skip inference; just merge "
                         "predictions_freetext_consensus_shard{0..shard_size-1}.jsonl into "
                         "predictions_freetext_consensus.jsonl and rewrite the submission CSV.")
    args = ap.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    unknown = set(tasks) - set(FREETEXT_TASKS)
    if unknown:
        ap.error(f"--tasks contains non free-text task(s): {sorted(unknown)}. "
                 f"Allowed: {list(FREETEXT_TASKS)}")

    records = []
    for p in args.predictions.split(","):
        records += [json.loads(l) for l in open(p.strip())]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = args.output_dir / "predictions"

    # --- merge mode: combine shard outputs, overlay onto the original records,
    # and write the final jsonl + submission CSV. No model/RAG needed. --------
    if args.merge_shards:
        shard_paths = [pred_dir / f"predictions_freetext_consensus_shard{rank}.jsonl"
                      for rank in range(args.shard_size)]
        merged_records = merge_shard_files(records, shard_paths)
        write_outputs(args.output_dir, merged_records)
        return

    missing = [n for n, v in (
        ("--model-dir", args.model_dir), ("--rag-dir", args.rag_dir), ("--video-dir", args.video_dir),
    ) if v is None]
    if missing:
        ap.error(f"the following arguments are required: {', '.join(missing)}")

    # Original-snapshot context maps: task_type -> {video_id -> prediction}.
    context_by_task = {t: {} for t in FREETEXT_TASKS}
    for r in records:
        if r["task_type"] in context_by_task:
            context_by_task[r["task_type"]][r["video_id"]] = r.get("prediction", "")
    for t in FREETEXT_TASKS:
        print(f"context '{t}': {len(context_by_task[t])} vids")

    print("Loading RAG evidence index...")
    rag_index = load_rag_index(args.rag_dir)
    print(f"  {len(rag_index)} evidence entries")

    sim = BertScoreSim() if args.scorer == "bertscore" else MiniLMSim()
    model, processor = load_model_and_processor(
        args.model_dir, args.base_model, args.lora, sft_adapter_dir=args.sft_adapter_dir)

    # Flatten targets across the requested tasks (grouped by task, in --tasks
    # order), then shard across GPUs so each process gets a roughly equal
    # slice of the work. Context blocks always come from the ORIGINAL
    # snapshot above, so shards never depend on each other's results.
    all_targets = [r for task in tasks for r in records if r["task_type"] == task]
    shard_targets = all_targets[args.shard_rank::args.shard_size]
    print(f"Regenerating {len(shard_targets)} of {len(all_targets)} item(s) "
          f"[shard {args.shard_rank + 1}/{args.shard_size}]")

    pred_dir.mkdir(parents=True, exist_ok=True)
    out_path = pred_dir / f"predictions_freetext_consensus_shard{args.shard_rank}.jsonl"

    # Resume support: skip item_indexes already written by a prior partial run.
    done_ids = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["item_index"])
                except (json.JSONDecodeError, KeyError):
                    pass
        if done_ids:
            print(f"[resume] {len(done_ids)} already-regenerated items found, skipping.")

    n_changed = 0
    n_failed = 0
    with open(out_path, "a") as f_out:
        for i, r in enumerate(shard_targets):
            if r["item_index"] in done_ids:
                continue
            sampled = sample_candidates(r, rag_index, model, processor, args, context_by_task)
            if sampled is None:
                print(f"  {r['video_id'].split('/')[-1]}: no RAG match, skip")
                continue
            cands, raws = sampled

            if not cands:
                print(f"  {r['video_id'].split('/')[-1]}: all samples empty, keep original")
                n_failed += 1
                continue
            elif len(set(cands)) == 1:
                prediction, best_idx, means = cands[0], 0, [1.0] * len(cands)
            else:
                try:
                    best_idx, means = pick_medoid(cands, sim)
                except Exception as exc:  # noqa: BLE001
                    print(f"[WARN] scorer failed on {r['item_index']}: {exc}; using greedy")
                    best_idx, means = 0, []
                prediction = cands[best_idx]

            r["regen_context_fix"] = {"old_prediction": r.get("prediction"),
                                      "old_raw_output": r.get("raw_output")}
            r["prediction"] = prediction
            r["raw_output"] = raws[best_idx]
            r["consensus"] = {
                "num_candidates": len(cands),
                "picked": best_idx,
                "picked_is_greedy": best_idx == 0,
                "mean_sims": [round(m, 4) for m in means],
                "candidates": cands,
            }
            f_out.write(json.dumps(r, ensure_ascii=False) + "\n")
            f_out.flush()
            n_changed += 1
            print(f"  {r['video_id'].split('/')[-1]} [{r['item_index']}]: regenerated "
                  f"({len(cands)} cands, picked {best_idx}, {len(prediction)} chars)")

            if (i + 1) % 10 == 0 or (i + 1) == len(shard_targets):
                print(f"[shard {args.shard_rank}] {i + 1}/{len(shard_targets)} done "
                      f"(changed: {n_changed}, failed: {n_failed})")

    print(f"\nRegenerated {n_changed} items in this shard")
    if args.shard_size == 1:
        # Single-process convenience: merge immediately so the caller gets the
        # final jsonl + CSV without a separate --merge-shards invocation.
        merged_records = merge_shard_files(records, [out_path])
        write_outputs(args.output_dir, merged_records)
    else:
        print(f"Shard {args.shard_rank} finished. Wrote {out_path}\n"
              f"Once all {args.shard_size} shards finish, run with --merge-shards to "
              f"produce the final jsonl + submission CSV.")


if __name__ == "__main__":
    main()
