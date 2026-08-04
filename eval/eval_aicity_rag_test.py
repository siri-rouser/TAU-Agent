#!/usr/bin/env python3
"""Run RAG-augmented inference and write an ``item_index,prediction`` CSV.

The input mirrors RAG training: slow-fast video frames, retrieved evidence, and
the question. Training-time evidence and sampling helpers are reused for parity.
RAG results are matched to official items by ``(video_id, question)`` so items
without retrieved evidence are still represented in the submission.
"""

import argparse
import csv
import glob
import json
import os
import sys
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

BACKGROUND_SYS = (
    "You are a video question-answering assistant. Answer the question about the "
    "video.\n\n"
    "You may also get a block of retrieved evidence with two kinds of context:\n"
    "1. Cross-question context: factual_information is usually reliable; "
    "potential_information is only a weak hint (may have a confidence score) and "
    "can be wrong.\n"
    "2. Visual context: scene descriptions, summaries, captions, and tracked "
    "objects from an automated pipeline that may be noisy.\n\n"
    "Base your answer on what the video shows, and trust the video when the "
    "evidence conflicts with it. Reason step by step inside <think></think>, then "
    "give the final answer inside <answer></answer>."
)

BACKGROUND_SYS_VIDEO_SUM = (
    "You are an expert traffic accident investigator and video question-answering assistant. "
    "Your task is to provide a chronological and highly detailed summary of the events in the video.\n\n"
    
    "You may also get a block of retrieved evidence with two kinds of context:\n"
    "1. Cross-question context: factual_information is usually reliable; "
    "potential_information is only a weak hint and "
    "can be wrong.\n"
    "2. Visual context: scene descriptions, summaries, captions, and tracked "
    "objects from an automated pipeline that may be noisy.\n\n"
    
    "Base your analysis on what the video shows, and trust the video when the "
    "evidence conflicts with it. Reason step by step inside <think></think> tags. "
    "Then, provide your final summary inside <answer></answer> tags.\n\n"
    
    "To maximize accuracy, the summary inside your <answer> block MUST sequentially include the following elements:\n"
    "1. Initial State: Briefly describe the normal traffic flow or baseline scene before the anomaly occurs.\n"
    "2. The Incident & Timestamps: Describe the anomaly, stating the exact timestamps (e.g., 'At 00:05.20...'). "
    "Identify all involved vehicles by color and type (e.g., 'white SUV', 'black sedan', 'motorcycle') and describe "
    "their trajectory relative to the frame (e.g., 'traveling straight from the bottom toward the top', 'entering from the right').\n"
    "3. Collision Mechanics: Explicitly state the type of collision using exact terminology (e.g., 'T-bone collision', "
    "'rear-end collision', 'head-on collision', 'sideswipe', or 'pedestrian strike').\n"
    "4. Physical Aftermath: Detail the immediate physics of the crash. Use keywords like 'deflects', 'spins', 'rolls over', "
    "and explicitly state where the vehicles 'stall' (e.g., 'stalls in the center of the intersection'). Mention if riders/pedestrians "
    "are thrown to the ground.\n"
    "5. Secondary Reactions: Describe how the surrounding environment reacts. Mention if other vehicles are forced to stop/swerve, "
    "if bystanders/pedestrians gather to assist, or if it is a hit-and-run.\n"
    "6. Root Cause: Your final sentence MUST begin exactly with: 'The root cause of the incident was...' Follow this by "
    "identifying the specific traffic violation or error (e.g., 'failing to yield the right-of-way', 'running a red traffic light', "
    "'overspeeding', 'loss of vehicle control due to wet roads', 'jaywalking')."
)

BACKGROUND_SYS_SCENE_DESC = (
    "You are an expert traffic surveillance analyst. Your task is to provide a "
    "highly detailed, objective, and static description of the traffic scene. "
    "To maximize accuracy, your response MUST sequentially include the "
    "following elements, using precise terminology:\n\n"

    "1. View & Conditions: State the camera perspective (e.g., overhead, "
    "elevated CCTV, street-level, dashcam), time of day (daytime, nighttime), "
    "and weather/lighting conditions (clear, sunny, overcast, rainy, snowy, "
    "dry/wet road surface).\n"
    "2. Road Layout & Traffic Flow: Describe the type of road (e.g., "
    "multi-lane highway, 4-way intersection, T-junction). Explicitly state "
    "the number of lanes and the direction of traffic flow relative to the "
    "camera frame (e.g., \"flowing from top to bottom\", \"from left to "
    "right\"). Mention if there is a central median or divider.\n"
    "3. Infrastructure: Identify all visible traffic infrastructure, "
    "including traffic lights, signboards, poles, crosswalks/zebra "
    "crossings, sidewalks, and barricades.\n"
    "4. Surroundings & Stationary Objects: Describe the environment (e.g., "
    "trees, buildings, storefronts). Explicitly list any parked or "
    "stationary vehicles by color and type (e.g., \"a parked white SUV on "
    "the left\", \"parked motorcycles\"), and mention if pedestrians are "
    "present.\n"
    "5. Text Overlays: Identify any on-screen text, watermarks, dates, or "
    "running timers, and state their exact location on the frame (e.g., "
    "\"top-right corner\", \"bottom-left\").\n\n"

    "Do not describe moving events or accidents; focus strictly on the "
    "static layout and physical environment."
)

# Make sibling evaluation modules and the train package importable.
_THIS_DIR = Path(__file__).resolve().parent
_TRAIN_DIR = _THIS_DIR.parent / "train"
for _p in (_THIS_DIR, _TRAIN_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from eval_aicity import run_inference, extract_answer  # noqa: E402

# Reuse training-time evidence and slow-fast helpers for parity.
from qwenvl.data.data_processor_rag import (  # noqa: E402
    format_evidence_rag,
    _video_fps,
    RAG_BASE_FPS,
    RAG_DENSE_MULT,
)

# ---------------------------------------------------------------------------
# Loading: official items (for item_index) + RAG per-video results
# ---------------------------------------------------------------------------
def load_official_items(test_json: Path) -> list[dict]:
    with open(test_json) as f:
        data = json.load(f)
    items = data.get("items", [])
    if not items:
        raise ValueError(f"No items found in {test_json}")
    return items


def build_item_index(items: list[dict]) -> dict[tuple[str, str], dict]:
    """Map (video_id, stripped question) -> official item, for item_index lookup."""
    index: dict[tuple[str, str], dict] = {}
    for it in items:
        key = (it["video_id"], it["question"].strip())
        index[key] = it
    return index


def load_rag_results(rag_dir: Path) -> list[dict]:
    """Flatten the per-video RAG JSON tree into a list of result records.

    Each record: {video_id, video_rel, result}. `video_rel` is the path relative
    to the video root (matches official item video_id, e.g. 'tar_test/v=....mp4').
    """
    records = []
    files = sorted(glob.glob(os.path.join(str(rag_dir), "**", "*.json"), recursive=True))
    for fpath in files:
        try:
            doc = json.load(open(fpath))
        except (json.JSONDecodeError, OSError):
            print(f"[warn] skipping unreadable RAG JSON: {fpath}")
            continue
        if not (isinstance(doc, dict) and "results" in doc):
            continue
        video_id = doc.get("video_id", "")
        video_path = doc.get("video_path", "")
        # Prefer the explicit video_id; fall back to deriving from video_path.
        video_rel = video_id.strip("/")
        if not video_rel and "/data/" in video_path:
            video_rel = video_path.split("/data/", 1)[-1].strip("/")
        if not video_rel:
            continue
        for result in doc.get("results", []):
            if not result.get("question"):
                continue
            records.append({
                "video_id": video_id,
                "video_rel": video_rel,
                "result": result,
            })
    return records


# ---------------------------------------------------------------------------
# Gemini scene-description helper
# ---------------------------------------------------------------------------
def load_gemini_scene_description(video_rel: str, gemini_dir: Path) -> str:
    """Load the scene_description field from the per-clip Gemini caption JSON.

    The gemini_dir should point to the clip-level JSON files
    (e.g. /data/captions/gemini31/tar_test/).  The video_rel path is expected
    to look like 'tar_test/v=....mp4'; only the basename stem is used.
    Returns an empty string if the file is missing or has no scene_description.
    """
    stem = Path(video_rel).stem  # e.g. 'v=-3nwOfm1Pdk_0-00_0-16'
    json_path = gemini_dir / f"{stem}.json"
    try:
        with open(json_path) as f:
            data = json.load(f)
        return (data.get("scene_description") or "").strip()
    except (OSError, json.JSONDecodeError):
        return ""

# ---------------------------------------------------------------------------
# Input construction — mirrors preprocess_qwen_visual_rag (no evidence dropout)
# ---------------------------------------------------------------------------
def build_rag_messages(result: dict, video_abs: str, max_frames: int,
                       video_max_pixels: int, video_min_pixels: int,
                       gemini_scene_desc: str = "", task_type: str = "") -> list[dict]:
    # fps only needed to render evidence timestamps (cheap metadata read)
    fps = _video_fps(video_abs)

    raw_question = result["question"]
    # Prepend Gemini scene description before the question, matching the
    # training-time format used in scene_description.json (and other tasks).

    video_ele = {
        "type": "video",
        "video": video_abs,
        "relevant_frame_ranges": result.get("relevant_frame_ranges") or [],
        "slowfast_base_fps": RAG_BASE_FPS,
        "slowfast_dense_mult": RAG_DENSE_MULT,
        "max_frames": max_frames,
        "max_pixels": video_max_pixels,
        "min_pixels": video_min_pixels,
    }

    if gemini_scene_desc and task_type == "scene_description":
        question_text = (
            f"Here is reference scene description from gemini you can refer to : "
            f"{gemini_scene_desc}. Question: {raw_question}"
        )
        return [
            {"role": "system", "content": BACKGROUND_SYS_SCENE_DESC},
            {
                "role": "user",
                "content": [video_ele, {"type": "text", "text": question_text}],
            },
        ]

    else:
        evidence = format_evidence_rag(result, fps)
        # Inference: always include evidence (training applies dropout; eval does not).
        question_text = (evidence + "\n\n" + raw_question) if evidence else raw_question

        if task_type == "video_summarization":
            sys_prompt = BACKGROUND_SYS_VIDEO_SUM
        else:
            sys_prompt = BACKGROUND_SYS
        return [
            {"role": "system", "content": sys_prompt},
            {
                "role": "user",
                "content": [video_ele, {"type": "text", "text": question_text}],
            },
        ]

# ---------------------------------------------------------------------------
# Model loading (same approach as eval_aicity_official_test.py)
# ---------------------------------------------------------------------------
def load_model_and_processor(model_dir: str, base_model: str, lora: bool,
                             sft_adapter_dir: str | None = None):
    """Load the model, optionally chaining two LoRA adapters.

    When `sft_adapter_dir` is given (new merged-base GRPO checkpoints), the
    load order is: base -> apply+merge SFT adapter -> apply+merge GRPO adapter
    (`model_dir`). This mirrors training, where the GRPO LoRA was attached on
    top of the base with the SFT adapter already merged in.
    """
    if lora:
        print(f"Loading base model from {base_model} ...")
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
        if sft_adapter_dir:
            print(f"Applying SFT LoRA adapter from {sft_adapter_dir} (merged first) ...")
            model = PeftModel.from_pretrained(model, sft_adapter_dir)
            model = model.merge_and_unload()
        print(f"Applying LoRA adapter from {model_dir} ...")
        model = PeftModel.from_pretrained(model, model_dir)
        model = model.merge_and_unload()
    else:
        print(f"Loading model from {model_dir} ...")
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_dir,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
    model.eval()
    processor_source = base_model
    if os.path.exists(os.path.join(model_dir, "processor.json")):
        processor_source = model_dir
    processor = AutoProcessor.from_pretrained(processor_source, fix_mistral_regex=True)
    print(f"Processor loaded from {processor_source}")
    return model, processor


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------
def save_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_submission_csv(path: Path, items: list[dict],
                         predictions_by_id: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["item_index", "prediction"])
        for item in items:
            writer.writerow([item["item_index"], predictions_by_id.get(item["item_index"], "")])


# ---------------------------------------------------------------------------
# Sharded inference
# ---------------------------------------------------------------------------
def run_shard(
    records: list[dict],
    item_index_map: dict[tuple[str, str], dict],
    video_root: Path,
    model,
    processor,
    max_new_tokens: int,
    max_frames: int,
    video_max_pixels: int,
    video_min_pixels: int,
    shard_rank: int,
    shard_size: int,
    pred_save_path: Path,
    gemini_caption_dir: Path | None = None,
) -> dict:
    shard_records = records[shard_rank::shard_size]
    print(f"Loaded {len(shard_records)} RAG results for shard {shard_rank + 1}/{shard_size}")

    # Load already-done item_indexes so we can resume a partial run.
    done_ids: set[str] = set()
    pred_save_path.parent.mkdir(parents=True, exist_ok=True)
    if pred_save_path.exists():
        with open(pred_save_path) as _f:
            for _line in _f:
                try:
                    done_ids.add(json.loads(_line)["item_index"])
                except (json.JSONDecodeError, KeyError):
                    pass
        print(f"[resume] {len(done_ids)} already-saved predictions found, skipping.")

    out_records = []
    n_failed = 0
    n_unmatched = 0

    for i, rec in enumerate(shard_records):
        result = rec["result"]
        video_rel = rec["video_rel"]
        question = (result.get("question") or "").strip()

        matched = item_index_map.get((video_rel, question))
        if matched is None:
            # also try the raw (unstripped) video_id key form
            matched = item_index_map.get((rec["video_id"], question))
        if matched is None:
            n_unmatched += 1
            if n_unmatched <= 5:
                print(f"[WARN] no item_index for ({video_rel!r}, {question[:60]!r})")
            continue

        # Skip if already saved (resumable run).
        if matched["item_index"] in done_ids:
            continue

        video_abs = str(video_root / video_rel)
        gemini_scene_desc = ""
        task_type = matched.get("task_type", "")
        if gemini_caption_dir is not None and matched.get("task_type") == "scene_description":
            gemini_scene_desc = load_gemini_scene_description(video_rel, gemini_caption_dir)
        messages = build_rag_messages(
            result, video_abs, max_frames, video_max_pixels, video_min_pixels,
            gemini_scene_desc=gemini_scene_desc,task_type=task_type
        )

        try:
            raw_output = run_inference(messages, model, processor, max_new_tokens)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"[WARN] OOM on {matched['item_index']} ({video_rel})")
            raw_output = ""
            n_failed += 1
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Failed on {matched['item_index']} ({video_rel}): "
                  f"{type(exc).__name__}: {exc}")
            raw_output = ""
            n_failed += 1

        out_record = {
            "item_index": matched["item_index"],
            "task_type": matched.get("task_type", ""),
            "video_id": video_rel,
            "question": question,
            "raw_output": raw_output,
            "prediction": extract_answer(raw_output),
        }
        out_records.append(out_record)

        # Append immediately so progress is saved after every question.
        with open(pred_save_path, "a") as _f:
            _f.write(json.dumps(out_record, ensure_ascii=False) + "\n")

        if (i + 1) % 20 == 0 or (i + 1) == len(shard_records):
            print(f"[shard {shard_rank}] {i + 1}/{len(shard_records)} done "
                  f"(failed: {n_failed}, unmatched: {n_unmatched})")

    # NOTE: do NOT overwrite pred_save_path here. Records are already appended
    # incrementally above (one line per processed item, including on resume),
    # so re-saving only `out_records` (this run's newly-processed items) would
    # clobber previously-resumed entries still in the file from earlier runs.
    return {
        "n_results": len(shard_records),
        "n_written": len(out_records),
        "n_failed": n_failed,
        "n_unmatched": n_unmatched,
        "prediction_file": str(pred_save_path),
    }


def merge_shards(output_dir: Path, shard_size: int,
                 pred_stem: str = "rag_test_predictions") -> tuple[list[dict], dict]:
    pred_dir = output_dir / "predictions"
    merged_records = []
    for rank in range(shard_size):
        shard_path = pred_dir / f"{pred_stem}_shard{rank}.jsonl"
        if not shard_path.exists():
            raise FileNotFoundError(f"Missing shard prediction file: {shard_path}")
        with open(shard_path) as f:
            merged_records.extend(json.loads(line) for line in f if line.strip())

    merged_path = pred_dir / f"{pred_stem}.jsonl"
    save_jsonl(merged_path, merged_records)
    predictions_by_id = {r["item_index"]: r["prediction"] for r in merged_records}
    return merged_records, predictions_by_id


def summarize_records(records: list[dict], n_official: int) -> dict:
    task_counts: dict[str, int] = {}
    empty = 0
    for r in records:
        task_counts[r["task_type"]] = task_counts.get(r["task_type"], 0) + 1
        if not str(r["prediction"]).strip():
            empty += 1
    return {
        "n_official_items": n_official,
        "n_predicted": len(records),
        "n_missing_from_rag": n_official - len(records),
        "n_empty_predictions": empty,
        "task_counts": task_counts,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="RAG-augmented AI City Track 3 test inference -> submission CSV."
    )
    parser.add_argument("--model-dir", default=None,
                        help="Fine-tuned model dir (or LoRA adapter dir with --lora).")
    parser.add_argument("--base-model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--lora", action="store_true", default=False)
    parser.add_argument("--sft-adapter-dir", default=None,
                        help="SFT LoRA adapter dir to apply+merge BEFORE the --model-dir "
                             "adapter (required for GRPO checkpoints trained on a merged "
                             "SFT base).")
    parser.add_argument("--checkpoint", default=None,
                        help="Optional checkpoint-N subdir under --model-dir.")
    parser.add_argument("--rag-dir", type=Path, required=True,
                        help="Directory tree of per-video RAG JSONs (one per video).")
    parser.add_argument("--test-json", type=Path,
                        default=Path("data/dataset/test/tar_test/test.json"),
                        help="Official test JSON (provides item_index for every item).")
    parser.add_argument("--video-dir", type=Path,
                        default=Path("data/videos"),
                        help="Video root; video_id resolves under it "
                             "(e.g. <root>/tar_test/v=....mp4).")
    parser.add_argument("--output-dir", "-o", type=Path, required=True)
    parser.add_argument("--shard-rank", type=int, default=0)
    parser.add_argument("--shard-size", type=int, default=1)
    parser.add_argument("--merge-shards", action="store_true", default=False)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--max-frames", type=int, default=100,
                        help="Matches training --video_max_frames (default 100).")
    parser.add_argument("--video-max-pixels", type=int, default=256 * 28 * 28,
                        help="Matches training --video_max_pixels.")
    parser.add_argument("--video-min-pixels", type=int, default=24 * 28 * 28,
                        help="Matches training --video_min_pixels.")
    parser.add_argument("--gemini-caption-dir", type=Path, default=None,
                        help="Directory of per-clip Gemini caption JSONs "
                             "(e.g. /data/captions/gemini31/tar_test). "
                             "When provided, the scene_description field is "
                             "prepended to each question, matching training-time "
                             "format.")
    args = parser.parse_args()

    items = load_official_items(args.test_json)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pred_stem = "rag_test_predictions"

    # --- merge mode: combine shard outputs and write submission --------------
    if args.merge_shards:
        merged_records, predictions_by_id = merge_shards(args.output_dir, args.shard_size, pred_stem)
        ts = time.strftime("%Y%m%d_%H%M%S")
        submission_path = args.output_dir / f"submission_rag_test_{ts}.csv"
        write_submission_csv(submission_path, items, predictions_by_id)

        summary = summarize_records(merged_records, len(items))
        summary.update({
            "submission_csv": str(submission_path),
            "merged_prediction_file": str(args.output_dir / "predictions" / f"{pred_stem}.jsonl"),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
        with open(args.output_dir / f"rag_test_summary_{ts}.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Merged {summary['n_predicted']} predictions "
              f"({summary['n_missing_from_rag']} official items missing from RAG -> empty).")
        print(f"Submission CSV saved to {submission_path}")
        return

    if args.model_dir is None:
        raise ValueError("--model-dir is required for inference")

    model_dir = args.model_dir
    if args.checkpoint:
        model_dir = str(Path(args.model_dir) / f"checkpoint-{args.checkpoint}")

    index = build_item_index(items)
    records = load_rag_results(args.rag_dir)
    print(f"Official items: {len(items)}; RAG results: {len(records)}")
    print(f"Model: {model_dir}")
    print(f"Video root: {args.video_dir}")
    print(f"Sampling: max_frames={args.max_frames} "
          f"base_fps={RAG_BASE_FPS} dense_mult={RAG_DENSE_MULT}")
    if args.shard_size > 1:
        print(f"Shard: {args.shard_rank + 1}/{args.shard_size}")

    model, processor = load_model_and_processor(model_dir, args.base_model, args.lora,
                                                 sft_adapter_dir=args.sft_adapter_dir)

    if args.shard_size > 1:
        pred_path = args.output_dir / "predictions" / f"{pred_stem}_shard{args.shard_rank}.jsonl"
    else:
        pred_path = args.output_dir / "predictions" / f"{pred_stem}.jsonl"

    t0 = time.time()
    shard_summary = run_shard(
        records=records,
        item_index_map=index,
        video_root=args.video_dir,
        model=model,
        processor=processor,
        max_new_tokens=args.max_new_tokens,
        max_frames=args.max_frames,
        video_max_pixels=args.video_max_pixels,
        video_min_pixels=args.video_min_pixels,
        shard_rank=args.shard_rank,
        shard_size=args.shard_size,
        pred_save_path=pred_path,
        gemini_caption_dir=args.gemini_caption_dir,
    )
    shard_summary["elapsed_sec"] = round(time.time() - t0, 1)
    print(json.dumps(shard_summary, indent=2))

    if args.shard_size > 1:
        print(f"Shard {args.shard_rank} finished. "
              f"Re-run with --merge-shards after all shards complete.")
        return

    if pred_path.exists():
        with open(pred_path) as f:
            records_out = [json.loads(line) for line in f if line.strip()]
    else:
        print(f"[WARN] no predictions were written to {pred_path} "
              f"(0 RAG results matched official items?); submission will be all-empty.")
        records_out = []
    predictions_by_id = {r["item_index"]: r["prediction"] for r in records_out}

    ts = time.strftime("%Y%m%d_%H%M%S")
    submission_path = args.output_dir / f"submission_rag_test_{ts}.csv"
    write_submission_csv(submission_path, items, predictions_by_id)

    summary = summarize_records(records_out, len(items))
    summary.update({
        "submission_csv": str(submission_path),
        "prediction_file": str(pred_path),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    with open(args.output_dir / f"rag_test_summary_{ts}.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Submission CSV saved to {submission_path}")


if __name__ == "__main__":
    main()
