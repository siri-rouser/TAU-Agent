#!/usr/bin/env python3
"""Run RAG-augmented inference for FETV clips and write a submission JSON.

The script uses training-time slow-fast sampling and evidence formatting, then
coerces each response into the fixed FETV schema. Missing evidence or
unparseable responses receive defaults so every clip is included.
"""

import argparse
import glob
import json
import os
import re
import sys
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

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
# FETV submission schema (kept in sync with dataset/test/FETV/evaluate.py)
# ---------------------------------------------------------------------------
CATEGORICAL_FIELDS = [
    "violation_type", "violator_type", "color",
    "initial_position", "final_position",
    "initial_lane", "final_lane",
    "intersection_type", "weather", "light",
]
DATE_FIELD = "date"
TIME_FIELD = "time"
CAPTION_FIELD = "description"
# Order used when emitting each submission object.
ALL_FIELDS = [DATE_FIELD, TIME_FIELD] + CATEGORICAL_FIELDS + [CAPTION_FIELD]

ALLOWED_VALUES = {
    "violation_type": [
        "wrong_way", "uturn", "jaywalking", "red_light",
        "lane_use_control", "lane_discipline", "no_violation",
    ],
    "violator_type": ["car", "motorcycle", "pedestrian", "bus", "truck", "na"],
    "color": ["dark", "light", "red", "green", "yellow", "blue", "mixed", "na"],
    "initial_position": [
        "Top-Left", "Top-Center", "Top-Right",
        "Middle-Left", "Middle-Center", "Middle-Right",
        "Bottom-Left", "Bottom-Center", "Bottom-Right", "na",
    ],
    "final_position": [
        "Top-Left", "Top-Center", "Top-Right",
        "Middle-Left", "Middle-Center", "Middle-Right",
        "Bottom-Left", "Bottom-Center", "Bottom-Right", "na",
    ],
    "initial_lane": ["1", "2", "3", "4", "na"],
    "final_lane": ["1", "2", "3", "4", "na"],
    "intersection_type": ["T-intersection", "four-way intersection"],
    "weather": ["clear", "rainy", "cloudy"],
    "light": ["daylight", "night"],
}

# Safe defaults for missing / invalid values so the submission always validates.
FIELD_DEFAULTS = {
    "violation_type": "no_violation",
    "violator_type": "na",
    "color": "na",
    "initial_position": "na",
    "final_position": "na",
    "initial_lane": "na",
    "final_lane": "na",
    "intersection_type": "four-way intersection",
    "weather": "clear",
    "light": "daylight",
    DATE_FIELD: "2024-01-01",
    TIME_FIELD: "12:00:00",
    CAPTION_FIELD: "No notable event observed.",
}

# Case-insensitive lookup from any spelling to the canonical allowed value.
_CANON = {
    field: {v.lower(): v for v in values}
    for field, values in ALLOWED_VALUES.items()
}

# System prompt: defines the role, the task, the fisheye viewpoint, the strict
# label space, and the exact output contract for every FETV clip.
FETV_SYSTEM_PROMPT = (
    "You are a traffic-violation analyst for the AI City Challenge FishEye "
    "Traffic Violation (FETV) task. Each input is a short clip recorded by a "
    "fisheye camera mounted above a road intersection, so the view is wrapped "
    "and objects near the edges appear distorted. Retrieved evidence "
    "(scene description, per-segment captions, and tracked-object data) is "
    "provided before the question to help you localize and describe the primary "
    "road user and its behavior.\n\n"
    "Your job: identify the single most salient traffic violation (or confirm "
    "there is none), characterize the primary violator, and describe the event.\n\n"
    "Reason carefully and concisely inside <think></think>. Ground every claim "
    "in what is actually visible in the clip and in the retrieved evidence; do "
    "NOT invent details. When a field cannot be determined from the video, use "
    "\"na\" rather than guessing.\n\n"
    "After reasoning, output your final answer inside <answer></answer> as a "
    "SINGLE minified-or-pretty JSON object with EXACTLY these 13 keys and no "
    "others, and each value taken ONLY from the allowed set:\n"
    "{\n"
    '  "violation_type": one of [wrong_way, uturn, jaywalking, red_light, lane_use_control, lane_discipline, no_violation],\n'
    '  "violator_type": one of [car, motorcycle, pedestrian, bus, truck, na],\n'
    '  "color": dominant color of the violator, one of [dark, light, red, green, yellow, blue, mixed, na],\n'
    '  "initial_position": the violator\'s position in the frame when first relevant, one of [Top-Left, Top-Center, Top-Right, Middle-Left, Middle-Center, Middle-Right, Bottom-Left, Bottom-Center, Bottom-Right, na],\n'
    '  "final_position": the violator\'s position when last relevant, one of [Top-Left, Top-Center, Top-Right, Middle-Left, Middle-Center, Middle-Right, Bottom-Left, Bottom-Center, Bottom-Right, na],\n'
    '  "initial_lane": lane index the violator starts in, one of [1, 2, 3, 4, na],\n'
    '  "final_lane": lane index the violator ends in, one of [1, 2, 3, 4, na],\n'
    '  "intersection_type": one of [T-intersection, four-way intersection],\n'
    '  "weather": one of [clear, rainy, cloudy],\n'
    '  "light": one of [daylight, night],\n'
    '  "date": the clip date as "YYYY-MM-DD",\n'
    '  "time": the clip time as "HH:MM:SS",\n'
    '  "description": one or two sentences describing the violator, its motion, and the violation (or that no violation occurred)\n'
    "}\n\n"
    "Hard rules:\n"
    "- Emit EXACTLY these 13 keys, spelled exactly as above, all lowercase keys.\n"
    "- Every categorical value MUST be copied verbatim from its allowed list "
    "(case-sensitive); never invent new labels.\n"
    "- If there is no violation, set violation_type to \"no_violation\" and set "
    "any inapplicable field to \"na\".\n"
    "- Put NOTHING except the JSON object inside <answer></answer> \u2014 no prose, "
    "no markdown fences, no comments."
)

# Short, unified user question asked for every FETV clip. The retrieved evidence
# is prepended in front of this at inference time.
FETV_USER_QUESTION = (
    "Analyze this fisheye intersection clip and report the primary traffic "
    "violation and its details as the JSON object specified in the instructions."
)


# ---------------------------------------------------------------------------
# Loading RAG results + full clip list
# ---------------------------------------------------------------------------
def _clip_name_from_rel(video_rel: str) -> str:
    """Return the FETV clip_name (basename, e.g. '001_000.mp4') from a rel path."""
    return os.path.basename(video_rel.strip("/"))


def load_rag_results(rag_dir: Path) -> list[dict]:
    """Flatten the per-clip FETV RAG JSON tree into a list of result records.

    Each record: {clip_name, video_rel, result}.
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
        video_rel = video_id.strip("/")
        if not video_rel and "/data/" in video_path:
            video_rel = video_path.split("/data/", 1)[-1].strip("/")
        if not video_rel:
            continue
        for result in doc.get("results", []):
            if not result.get("question"):
                continue
            records.append({
                "clip_name": _clip_name_from_rel(video_rel),
                "video_rel": video_rel,
                "result": result,
            })
    return records


def enumerate_clip_names(clip_dir: Path) -> list[str]:
    """List every FETV clip filename under clip_dir (for full submission coverage)."""
    files = sorted(glob.glob(os.path.join(str(clip_dir), "**", "*.mp4"), recursive=True))
    return [os.path.basename(f) for f in files]


# ---------------------------------------------------------------------------
# Input construction — mirrors preprocess_qwen_visual_rag, NO system prompt
# ---------------------------------------------------------------------------
def build_fetv_messages(result: dict, video_abs: str, max_frames: int,
                        video_max_pixels: int, video_min_pixels: int) -> list[dict]:
    # fps only needed to render evidence timestamps (cheap metadata read)
    fps = _video_fps(video_abs)

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

    evidence = format_evidence_rag(result, fps)
    # Retrieved evidence goes in FRONT of the short unified question.
    question_text = (
        (evidence + "\n\n" + FETV_USER_QUESTION) if evidence else FETV_USER_QUESTION
    )

    # System prompt carries the task definition + output contract.
    return [
        {"role": "system", "content": FETV_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [video_ele, {"type": "text", "text": question_text}],
        },
    ]


# ---------------------------------------------------------------------------
# Output parsing / coercion into the FETV submission schema
# ---------------------------------------------------------------------------
def _extract_json_obj(text: str) -> dict | None:
    """Best-effort extraction of a single JSON object from model text."""
    if not text:
        return None
    candidates = []
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        candidates.append(fence.group(1))
    candidates.append(text)
    for cand in candidates:
        start, end = cand.find("{"), cand.rfind("}")
        if start == -1 or end == -1 or end <= start:
            continue
        snippet = cand[start:end + 1]
        try:
            obj = json.loads(snippet)
        except json.JSONDecodeError:
            cleaned = re.sub(r",\s*([}\]])", r"\1", snippet)  # drop trailing commas
            try:
                obj = json.loads(cleaned)
            except json.JSONDecodeError:
                continue
        if isinstance(obj, dict):
            return obj
    return None


def _coerce_categorical(field: str, value) -> str:
    if value is None:
        return FIELD_DEFAULTS[field]
    v = str(value).strip()
    if not v:
        return FIELD_DEFAULTS[field]
    canon = _CANON[field].get(v.lower())
    return canon if canon is not None else FIELD_DEFAULTS[field]


def _coerce_date(value) -> str:
    if value is None:
        return FIELD_DEFAULTS[DATE_FIELD]
    v = str(value).strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", v):
        return v
    return FIELD_DEFAULTS[DATE_FIELD]


def _coerce_time(value) -> str:
    if value is None:
        return FIELD_DEFAULTS[TIME_FIELD]
    v = str(value).strip()
    m = re.fullmatch(r"(\d{1,2}):(\d{2})(?::(\d{2}))?", v)
    if not m:
        return FIELD_DEFAULTS[TIME_FIELD]
    h, mm, ss = int(m.group(1)), int(m.group(2)), int(m.group(3) or 0)
    if h > 23 or mm > 59 or ss > 59:
        return FIELD_DEFAULTS[TIME_FIELD]
    return f"{h:02d}:{mm:02d}:{ss:02d}"


def parse_fetv_answer(raw_output: str) -> dict:
    """Parse a model output into the 12 structured fields + description.

    Returns a dict keyed by the bare field names (no ``answer_`` prefix).
    Missing or invalid values fall back to FIELD_DEFAULTS.
    """
    answer_text = extract_answer(raw_output)
    obj = _extract_json_obj(answer_text) or _extract_json_obj(raw_output) or {}
    # Allow the model to nest under an "answer"/"prediction" key.
    if isinstance(obj.get("answer"), dict):
        obj = obj["answer"]

    out = {}
    for field in CATEGORICAL_FIELDS:
        out[field] = _coerce_categorical(field, obj.get(field))
    out[DATE_FIELD] = _coerce_date(obj.get(DATE_FIELD))
    out[TIME_FIELD] = _coerce_time(obj.get(TIME_FIELD))

    desc = obj.get(CAPTION_FIELD)
    desc = str(desc).strip() if desc is not None else ""
    if not desc:
        # Fall back to any free-form answer text so the caption is non-empty.
        desc = answer_text.strip() or FIELD_DEFAULTS[CAPTION_FIELD]
    out[CAPTION_FIELD] = desc
    return out


def default_fields() -> dict:
    """Full default field dict for clips with no usable prediction."""
    return {f: FIELD_DEFAULTS[f] for f in ALL_FIELDS}


def to_submission_object(clip_name: str, fields: dict) -> dict:
    obj = {"clip_name": clip_name}
    for f in ALL_FIELDS:
        obj[f"answer_{f}"] = fields.get(f, FIELD_DEFAULTS[f])
    return obj


# ---------------------------------------------------------------------------
# Model loading (identical approach to eval_aicity_rag_test.py)
# ---------------------------------------------------------------------------
def load_model_and_processor(model_dir: str, base_model: str, lora: bool,
                             sft_adapter_dir: str | None = None):
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
    processor = AutoProcessor.from_pretrained(base_model, fix_mistral_regex=True)
    print(f"Processor loaded from {base_model}")
    return model, processor


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------
def save_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_submission_json(path: Path, clip_names: list[str],
                          fields_by_clip: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    submission = [
        to_submission_object(clip, fields_by_clip.get(clip, default_fields()))
        for clip in clip_names
    ]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(submission, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Sharded inference
# ---------------------------------------------------------------------------
def run_shard(
    records: list[dict],
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
) -> dict:
    shard_records = records[shard_rank::shard_size]
    print(f"Loaded {len(shard_records)} RAG results for shard {shard_rank + 1}/{shard_size}")

    # Load already-done clips so we can resume a partial run.
    done_clips: set[str] = set()
    pred_save_path.parent.mkdir(parents=True, exist_ok=True)
    if pred_save_path.exists():
        with open(pred_save_path) as _f:
            for _line in _f:
                try:
                    done_clips.add(json.loads(_line)["clip_name"])
                except (json.JSONDecodeError, KeyError):
                    pass
        print(f"[resume] {len(done_clips)} already-saved predictions found, skipping.")

    n_done = 0
    n_failed = 0

    for i, rec in enumerate(shard_records):
        clip_name = rec["clip_name"]
        if clip_name in done_clips:
            continue

        result = rec["result"]
        video_rel = rec["video_rel"]
        video_abs = str(video_root / video_rel)

        messages = build_fetv_messages(
            result, video_abs, max_frames, video_max_pixels, video_min_pixels,
        )

        try:
            raw_output = run_inference(messages, model, processor, max_new_tokens)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"[WARN] OOM on {clip_name} ({video_rel})")
            raw_output = ""
            n_failed += 1
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Failed on {clip_name} ({video_rel}): "
                  f"{type(exc).__name__}: {exc}")
            raw_output = ""
            n_failed += 1

        fields = parse_fetv_answer(raw_output) if raw_output else default_fields()
        out_record = {
            "clip_name": clip_name,
            "video_id": video_rel,
            "raw_output": raw_output,
            "fields": fields,
        }
        n_done += 1

        with open(pred_save_path, "a") as _f:
            _f.write(json.dumps(out_record, ensure_ascii=False) + "\n")

        if (i + 1) % 20 == 0 or (i + 1) == len(shard_records):
            print(f"[shard {shard_rank}] {i + 1}/{len(shard_records)} done "
                  f"(failed: {n_failed})")

    return {
        "n_results": len(shard_records),
        "n_written": n_done,
        "n_failed": n_failed,
        "prediction_file": str(pred_save_path),
    }


def merge_shards(output_dir: Path, shard_size: int,
                 pred_stem: str = "fetv_predictions") -> dict[str, dict]:
    pred_dir = output_dir / "predictions"
    fields_by_clip: dict[str, dict] = {}
    for rank in range(shard_size):
        shard_path = pred_dir / f"{pred_stem}_shard{rank}.jsonl"
        if not shard_path.exists():
            raise FileNotFoundError(f"Missing shard prediction file: {shard_path}")
        with open(shard_path) as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                fields_by_clip[rec["clip_name"]] = rec["fields"]
    # Also persist a merged jsonl for reference.
    merged = [{"clip_name": c, "fields": fields_by_clip[c]} for c in sorted(fields_by_clip)]
    save_jsonl(pred_dir / f"{pred_stem}.jsonl", merged)
    return fields_by_clip


def summarize(clip_names: list[str], fields_by_clip: dict[str, dict]) -> dict:
    covered = sum(1 for c in clip_names if c in fields_by_clip)
    vt_counts: dict[str, int] = {}
    for c in clip_names:
        fields = fields_by_clip.get(c, default_fields())
        vt = fields.get("violation_type", "na")
        vt_counts[vt] = vt_counts.get(vt, 0) + 1
    return {
        "n_clips": len(clip_names),
        "n_predicted": covered,
        "n_defaulted": len(clip_names) - covered,
        "violation_type_counts": vt_counts,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="RAG-augmented FETV (Track 7) inference -> submission JSON."
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
                        help="Directory tree of per-clip FETV RAG JSONs "
                             "(e.g. /data/FETV_rag/fetv).")
    parser.add_argument("--clip-dir", type=Path, default=Path("/data/FETV"),
                        help="Directory of FETV .mp4 clips; used to guarantee the "
                             "submission covers every clip (default: %(default)s).")
    parser.add_argument("--video-dir", type=Path, default=Path("/data"),
                        help="Video root; each clip resolves under it via its "
                             "relative video_id (default: %(default)s).")
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
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pred_stem = "fetv_predictions"

    # Full clip list for submission coverage. Fall back to RAG records if the
    # clip directory is unavailable.
    clip_names = enumerate_clip_names(args.clip_dir)
    records = load_rag_results(args.rag_dir)
    if not clip_names:
        clip_names = sorted({r["clip_name"] for r in records})
    print(f"FETV clips: {len(clip_names)}; RAG results: {len(records)}")

    # --- merge mode: combine shard outputs and write submission --------------
    if args.merge_shards:
        fields_by_clip = merge_shards(args.output_dir, args.shard_size, pred_stem)
        ts = time.strftime("%Y%m%d_%H%M%S")
        submission_path = args.output_dir / f"submission_fetv_{ts}.json"
        write_submission_json(submission_path, clip_names, fields_by_clip)

        summary = summarize(clip_names, fields_by_clip)
        summary.update({
            "submission_json": str(submission_path),
            "merged_prediction_file": str(args.output_dir / "predictions" / f"{pred_stem}.jsonl"),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
        with open(args.output_dir / f"fetv_summary_{ts}.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Merged {summary['n_predicted']} predictions "
              f"({summary['n_defaulted']} clips defaulted).")
        print(f"Submission JSON saved to {submission_path}")
        return

    if args.model_dir is None:
        raise ValueError("--model-dir is required for inference")

    model_dir = args.model_dir
    if args.checkpoint:
        model_dir = str(Path(args.model_dir) / f"checkpoint-{args.checkpoint}")

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
        pred_path = args.output_dir / "predictions" / f"{pred_stem}_shard0.jsonl"

    t0 = time.time()
    shard_summary = run_shard(
        records=records,
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
    )
    shard_summary["elapsed_sec"] = round(time.time() - t0, 1)
    print(json.dumps(shard_summary, indent=2))

    if args.shard_size > 1:
        print(f"Shard {args.shard_rank} finished. "
              f"Re-run with --merge-shards after all shards complete.")
        return

    # Single-process run: build the submission directly from this shard's file.
    fields_by_clip = merge_shards(args.output_dir, 1, pred_stem)
    ts = time.strftime("%Y%m%d_%H%M%S")
    submission_path = args.output_dir / f"submission_fetv_{ts}.json"
    write_submission_json(submission_path, clip_names, fields_by_clip)

    summary = summarize(clip_names, fields_by_clip)
    summary.update({
        "submission_json": str(submission_path),
        "prediction_file": str(pred_path),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    with open(args.output_dir / f"fetv_summary_{ts}.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Submission JSON saved to {submission_path}")


if __name__ == "__main__":
    main()
