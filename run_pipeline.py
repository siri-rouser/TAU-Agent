#!/usr/bin/env python3
"""
End-to-end TAU-Agent example: video + question(s) -> RAG evidence -> Qwen3-VL
inference -> printed answer.

This ties together the two halves of the project into one runnable example:

  1. Generate RAG evidence for the video/question(s), the same one-shot
     agentic retrieval used in `RAG_retriever/main.py` (captioning, then a
     ReAct agent that calls `caption_retrieval`/`free_text_tracking` tools).
  2. Format the model input block exactly as at SFT training time: video
     (slow-fast sampled) + retrieved evidence text + question, via the real
     training helper `format_evidence_rag` (train/qwenvl/data/data_processor_rag.py).
  3. Run inference with a fine-tuned Qwen3-VL model (LoRA adapter or full
     checkpoint).
  4. Print the raw model output and the extracted <answer> for each question.

Requires `MODELSELL_API_KEY` (or `OPENROUTER_API_KEY`) for the retrieval step,
and a fine-tuned Qwen3-VL checkpoint/LoRA adapter for the inference step.

Example:
    python run_pipeline.py \\
        --video-path data/videos/tar_test/example.mp4 \\
        --question "Does the gray sedan run the red light?" \\
        --model-dir /output/aicity_new_rag2_stage1_dp0_last_update --lora
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

# Make RAG_retriever/ and train/ importable regardless of cwd, so this script
# can be run from anywhere as `python run_pipeline.py ...`.
_THIS_DIR = Path(__file__).resolve().parent
_RAG_RETRIEVER_DIR = _THIS_DIR / "RAG_retriever"
_TRAIN_DIR = _THIS_DIR / "train"
for _p in (_RAG_RETRIEVER_DIR, _TRAIN_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from main import preprocess, RAG_Retriever, _enrich_result  # noqa: E402
from qwenvl.data.data_processor_rag import (  # noqa: E402
    format_evidence_rag,
    _video_fps,
    RAG_BASE_FPS,
    RAG_DENSE_MULT,
)

# ---------------------------------------------------------------------------
# System prompts (mirrors eval/eval_aicity_rag_test.py, used at training time)
# ---------------------------------------------------------------------------

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

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)


def extract_answer(raw: str) -> str:
    """Return text inside <answer>...</answer>, or the full stripped output."""
    m = _ANSWER_RE.search(raw)
    return m.group(1).strip() if m else raw.strip()


# ---------------------------------------------------------------------------
# Step 1: generate RAG evidence (same agentic retrieval as RAG_retriever/main.py)
# ---------------------------------------------------------------------------

def generate_rag_evidence(video_path, questions, api_key, captioning_agent):
    """Caption the video (if needed), run the ReAct retriever per question, and
    enrich each result with resolved captions/track data. Returns a list of
    enriched entries (one per question), each usable directly by
    `format_evidence_rag`.
    """
    preprocess(video_path_list=[video_path], captioning_agent=captioning_agent, api_key=api_key)

    entries = []
    for question in questions:
        retrieval_result = RAG_Retriever(
            video_path, question, api_key, captioning_agent, main_agent="gpt-5.4",
        )
        entry = {"video_path": video_path, "question": question, "retrieval_result": retrieval_result}
        entries.append(_enrich_result(entry, captioning_agent))
    return entries


# ---------------------------------------------------------------------------
# Step 2: format the model input block (mirrors eval/eval_aicity_rag_test.py's
# build_rag_messages / train/qwenvl/data/data_processor_rag.py's training format)
# ---------------------------------------------------------------------------

def build_messages(entry, video_abs, max_frames, video_max_pixels, video_min_pixels):
    fps = _video_fps(video_abs)
    evidence = format_evidence_rag(entry, fps)
    question_text = (evidence + "\n\n" + entry["question"]) if evidence else entry["question"]

    video_ele = {
        "type": "video",
        "video": video_abs,
        "relevant_frame_ranges": entry.get("relevant_frame_ranges") or [],
        "slowfast_base_fps": RAG_BASE_FPS,
        "slowfast_dense_mult": RAG_DENSE_MULT,
        "max_frames": max_frames,
        "max_pixels": video_max_pixels,
        "min_pixels": video_min_pixels,
    }
    return [
        {"role": "system", "content": BACKGROUND_SYS},
        {"role": "user", "content": [video_ele, {"type": "text", "text": question_text}]},
    ]


# ---------------------------------------------------------------------------
# Step 3: load the fine-tuned model and run inference
# ---------------------------------------------------------------------------

def load_model_and_processor(model_dir, base_model, lora, sft_adapter_dir=None):
    """Load the model, optionally chaining two LoRA adapters.

    When `sft_adapter_dir` is given (merged-base GRPO checkpoints), the load
    order is: base -> apply+merge SFT adapter -> apply+merge GRPO adapter
    (`model_dir`), mirroring how the GRPO LoRA was trained on top of an
    already-merged SFT base.
    """
    if lora:
        print(f"Loading base model from {base_model} ...")
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            base_model, torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2", device_map="auto",
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
            model_dir, torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2", device_map="auto",
        )
    model.eval()
    processor_source = base_model
    if os.path.exists(os.path.join(model_dir, "processor.json")):
        processor_source = model_dir
    processor = AutoProcessor.from_pretrained(processor_source, fix_mistral_regex=True)
    print(f"Processor loaded from {processor_source}")
    return model, processor


def run_inference(messages, model, processor, max_new_tokens=2048, temperature=0.0):
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos, video_kwargs = process_vision_info(
        messages, image_patch_size=16, return_video_kwargs=True, return_video_metadata=True,
    )
    if videos is not None:
        videos, video_metadatas = zip(*videos)
        videos, video_metadatas = list(videos), list(video_metadatas)
    else:
        video_metadatas = None

    inputs = processor(
        text=text, images=images, videos=videos, video_metadata=video_metadatas,
        return_tensors="pt", do_resize=False, **video_kwargs,
    )
    inputs = inputs.to(model.device)

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs, max_new_tokens=max_new_tokens, temperature=temperature,
            do_sample=temperature > 0, num_beams=1,
        )
    trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    return processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="End-to-end example: RAG evidence -> formatted input -> Qwen3-VL inference.")
    parser.add_argument("--video-path", required=True,
                         help="Path to the input video, e.g. data/videos/tar_test/example.mp4")
    parser.add_argument("--question", required=True, nargs="+",
                         help="One or more questions to ask about the video (space-separated, quote each one).")
    parser.add_argument("--captioning-agent", default="Gemini35Flash",
                         choices=["Gemini31pro", "Gemini35Flash"],
                         help="Caption model used as RAG evidence (default: Gemini35Flash).")
    parser.add_argument("--model-dir", required=True,
                         help="Fine-tuned model dir (or LoRA adapter dir with --lora).")
    parser.add_argument("--base-model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--lora", action="store_true", default=False)
    parser.add_argument("--sft-adapter-dir", default=None,
                         help="SFT LoRA adapter dir to apply+merge BEFORE --model-dir "
                              "(only needed for GRPO checkpoints trained on a merged SFT base).")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--max-frames", type=int, default=100)
    parser.add_argument("--video-max-pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--video-min-pixels", type=int, default=24 * 28 * 28)
    parser.add_argument("--output-json", default=None,
                         help="Optional path to save the full results (evidence + raw output + "
                              "extracted answer) as JSON.")
    args = parser.parse_args()

    api_key = os.environ.get("MODELSELL_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: MODELSELL_API_KEY or OPENROUTER_API_KEY environment variable not set.")
        sys.exit(1)

    video_abs = str(Path(args.video_path).resolve())

    # --- Step 1: generate RAG evidence -------------------------------------
    print(f"[1/4] Generating RAG evidence for {args.video_path} ...")
    entries = generate_rag_evidence(args.video_path, args.question, api_key, args.captioning_agent)

    print(f"[2/4] Loading model from {args.model_dir} ...")
    model, processor = load_model_and_processor(
        args.model_dir, args.base_model, args.lora, sft_adapter_dir=args.sft_adapter_dir,
    )

    # --- Step 2+3: format each question's input block, then run inference --
    print("[3/4] Formatting inputs and running inference ...")
    output_records = []
    for entry in entries:
        messages = build_messages(entry, video_abs, args.max_frames, args.video_max_pixels, args.video_min_pixels)
        raw_output = run_inference(messages, model, processor, args.max_new_tokens)
        output_records.append({
            "video_path": args.video_path,
            "question": entry["question"],
            "raw_output": raw_output,
            "answer": extract_answer(raw_output),
        })

    # --- Step 4: print results ----------------------------------------------
    print("\n[4/4] Results:")
    for rec in output_records:
        print("=" * 80)
        print(f"Q: {rec['question']}")
        print(f"A: {rec['answer']}")
        print("-" * 80)
        print(f"Raw output:\n{rec['raw_output']}")

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(output_records, f, indent=2)
        print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
