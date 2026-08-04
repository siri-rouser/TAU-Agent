#!/usr/bin/env python3
"""Shared inference and answer-parsing helpers for AICity evaluation scripts.

Task-to-metric mapping:
  mcq, mcq_openended          → F1 (macro), AP (macro-averaged precision)
  bcq, bcq_openended          → F1 (binary), AP (binary precision)
  temporal_localization        → mean IoU
  open_qa, causal_linkage,
  scene_description,
  temporal_description,
    video_summarization          → BLEU, ROUGE-L, METEOR
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from peft import PeftModel

# ---------------------------------------------------------------------------
# Task groupings
# ---------------------------------------------------------------------------

MCQ_TASKS      = ["mcq", "mcq_openended"]
BCQ_TASKS      = ["bcq", "bcq_openended"]
TEMPORAL_TASKS = ["temporal_localization"]
TEXT_TASKS     = [
    "open_qa",
    "causal_linkage",
    "scene_description",
    "temporal_description",
    "video_summarization",
]
ALL_TASKS = MCQ_TASKS + BCQ_TASKS + TEMPORAL_TASKS + TEXT_TASKS

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(messages, model, processor, max_new_tokens: int = 2048, temperature: float = 0.0) -> str:
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    images, videos, video_kwargs = process_vision_info(
        messages,
        image_patch_size=16,
        return_video_kwargs=True,
        return_video_metadata=True,
    )
    if videos is not None:
        videos, video_metadatas = zip(*videos)
        videos, video_metadatas = list(videos), list(video_metadatas)
    else:
        video_metadatas = None

    inputs = processor(
        text=text,
        images=images,
        videos=videos,
        video_metadata=video_metadatas,
        return_tensors="pt",
        do_resize=False,
        **video_kwargs,
    )
    inputs = inputs.to(model.device)

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            num_beams=1,
        )
    trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    # skip_special_tokens=True removes registered special tokens (e.g. <|im_end|>,
    # <think>, </think>) but NOT <answer>/<answer> which are plain text tokens added
    # during fine-tuning supervision — those tags will still appear in the output
    # and are parsed by extract_answer() below.
    return processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)


def extract_answer(raw: str) -> str:
    """Return text inside <answer>…</answer>, or the full stripped output."""
    m = _ANSWER_RE.search(raw)
    return m.group(1).strip() if m else raw.strip()

def extract_mcq_letter(raw: str) -> str:
    """Extract first A/B/C/D from model output."""
    text = extract_answer(raw)
    m = re.search(r"\b([A-Da-d])\b", text)
    if m:
        return m.group(1).upper()
    # fallback: first character
    return text[:1].upper() if text else "X"


def extract_bcq_label(raw: str) -> str:
    """Extract Yes / No from model output."""
    text = extract_answer(raw).lower()
    if re.search(r"\byes\b", text):
        return "Yes"
    if re.search(r"\bno\b", text):
        return "No"
    return "X"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    pass

if __name__ == "__main__":
    main()
