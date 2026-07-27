import json
import os
import random
import logging
import re
import time
import itertools
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List, Tuple, Any
from collections.abc import Sequence
from pathlib import Path

import glob
import numpy as np
import torch
from torch.utils.data import Dataset

import transformers
from qwen_vl_utils import process_vision_info

from . import data_list
from .rope2d import get_rope_index_25, get_rope_index_2, get_rope_index_3

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = 151655
VIDEO_TOKEN_INDEX = 151656
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"

local_rank = 0

def is_global_rank_0():
    # If not a torchrun launch: treat as rank 0
    return int(os.environ.get("RANK", 0)) == 0

def rank0_print(*args, **kwargs):
    if is_global_rank_0():
        print(*args, **kwargs)


def read_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def _make_abs_paths(base: Path, files: str) -> str:
    return f"{(base / files).resolve()}"


def update_processor_pixels(processor, data_args):
    logger = logging.getLogger(__name__)

    # --- Image Processor ---
    ip = processor.image_processor
    rank0_print("=== BEFORE IMAGE PROCESSOR PARAMETERS ===")
    rank0_print(f"Image min_pixels: {getattr(ip, 'min_pixels', 'N/A')}")
    rank0_print(f"Image max_pixels: {getattr(ip, 'max_pixels', 'N/A')}")
    rank0_print(f"ip.size: {ip.size}")
    rank0_print(f"Image size (shortest_edge): {ip.size.get('shortest_edge', 'N/A')}")
    rank0_print(f"Image size (longest_edge):  {ip.size.get('longest_edge', 'N/A')}")

    if hasattr(ip, "min_pixels") and hasattr(ip, "max_pixels"):
        ip.min_pixels = data_args.min_pixels
        ip.max_pixels = data_args.max_pixels
        rank0_print(f"✅ Updated image_processor min_pixels to {data_args.min_pixels}")
        rank0_print(f"✅ Updated image_processor max_pixels to {data_args.max_pixels}")

    if hasattr(ip, "size") and isinstance(ip.size, dict):
        ip.size["shortest_edge"] = data_args.min_pixels
        ip.size["longest_edge"] = data_args.max_pixels
        rank0_print(
            f"✅ Updated image_processor size['shortest_edge'] to {data_args.min_pixels}"
        )
        rank0_print(
            f"✅ Updated image_processor size['longest_edge'] to {data_args.max_pixels}"
        )

    rank0_print("=== AFTER IMAGE PROCESSOR PARAMETERS ===")
    rank0_print(f"Image min_pixels: {getattr(ip, 'min_pixels', 'N/A')}")
    rank0_print(f"Image max_pixels: {getattr(ip, 'max_pixels', 'N/A')}")
    rank0_print(f"Image size (shortest_edge): {ip.size.get('shortest_edge', 'N/A')}")
    rank0_print(f"Image size (longest_edge):  {ip.size.get('longest_edge', 'N/A')}")

    # --- Video Processor ---
    if hasattr(processor, "video_processor") and processor.video_processor is not None:
        vp = processor.video_processor
        rank0_print("\n=== BEFORE VIDEO PROCESSOR PARAMETERS ===")
        rank0_print(f"Video min_pixels: {getattr(vp, 'min_pixels', 'N/A')}")
        rank0_print(f"Video max_pixels: {getattr(vp, 'max_pixels', 'N/A')}")
        rank0_print(f"Video min_frames: {getattr(vp, 'min_frames', 'N/A')}")
        rank0_print(f"Video max_frames: {getattr(vp, 'max_frames', 'N/A')}")
        rank0_print(f"Video fps: {getattr(vp, 'fps', 'N/A')}")
        rank0_print(
            f"Video size (shortest_edge): {vp.size.get('shortest_edge', 'N/A')}"
        )
        rank0_print(f"Video size (longest_edge):  {vp.size.get('longest_edge', 'N/A')}")

        if hasattr(vp, "min_pixels") and hasattr(vp, "max_pixels"):
            vp.min_pixels = data_args.video_min_pixels
            vp.max_pixels = data_args.video_max_pixels
            rank0_print(
                f"✅ Updated Qwen2-VL video_processor min_pixels to {data_args.video_min_pixels}"
            )
            rank0_print(
                f"✅ Updated Qwen2-VL video_processor max_pixels to {data_args.video_max_pixels}"
            )

        if hasattr(vp, "min_frames") and hasattr(vp, "max_frames"):
            vp.min_frames = data_args.video_min_frames
            vp.max_frames = data_args.video_max_frames
            rank0_print(
                f"✅ Updated video_processor min_frames to {data_args.video_min_frames}"
            )
            rank0_print(
                f"✅ Updated video_processor max_frames to {data_args.video_max_frames}"
            )

        if hasattr(vp, "fps"):
            vp.fps = data_args.video_fps
            rank0_print(f"✅ Updated video_processor fps to {data_args.video_fps}")

        if hasattr(vp, "size") and isinstance(vp.size, dict):
            vp.size["shortest_edge"] = data_args.video_min_pixels
            vp.size["longest_edge"] = data_args.video_max_pixels
            rank0_print(
                f"✅ Updated Video size (shortest_edge): {vp.size.get('shortest_edge', 'N/A')}"
            )
            rank0_print(
                f"✅ Updated Video size (longest_edge):  {vp.size.get('longest_edge', 'N/A')}"
            )

        rank0_print("=== AFTER VIDEO PROCESSOR PARAMETERS ===")
        rank0_print(f"Video min_pixels: {getattr(vp, 'min_pixels', 'N/A')}")
        rank0_print(f"Video max_pixels: {getattr(vp, 'max_pixels', 'N/A')}")
        rank0_print(f"Video min_frames: {getattr(vp, 'min_frames', 'N/A')}")
        rank0_print(f"Video max_frames: {getattr(vp, 'max_frames', 'N/A')}")
        rank0_print(f"Video fps: {getattr(vp, 'fps', 'N/A')}")
        rank0_print(
            f"Video size (shortest_edge): {vp.size.get('shortest_edge', 'N/A')}"
        )
        rank0_print(f"Video size (longest_edge):  {vp.size.get('longest_edge', 'N/A')}")

    return processor


def _build_messages(item: Dict[str, Any], base_path: Path, system_prompt: str = None) -> List[Dict[str, Any]]:
    # Extract and normalize images and videos
    images = item.get("image") or []
    if isinstance(images, str):
        images = [images]

    videos = item.get("video") or []
    if isinstance(videos, str):
        videos = [videos]

    # Build media pools with absolute paths
    image_pool = [
        {"type": "image", "image": _make_abs_paths(base_path, img)} for img in images
    ]
    video_pool = [
        {"type": "video", "video": _make_abs_paths(base_path, vid)} for vid in videos
    ]
    

    messages = []
    if system_prompt:
        messages.append({
            "role": "system", 
            "content": [{"type": "text", "text": system_prompt}]
        })
    
    for turn in item["conversations"]:
        role = "user" if turn["from"] == "human" else "assistant"
        text: str = turn["value"]

        if role == "user":
            content = []
            # Split text by <image> or <video> placeholders while keeping delimiters
            text_parts = re.split(r"(<image>|<video>)", text)

            for seg in text_parts:
                if seg == "<image>":
                    if not image_pool:
                        raise ValueError(
                            "Number of <image> placeholders exceeds the number of provided images"
                        )
                    content.append(image_pool.pop(0))
                elif seg == "<video>":
                    if not video_pool:
                        raise ValueError(
                            "Number of <video> placeholders exceeds the number of provided videos"
                        )
                    content.append(video_pool.pop(0))
                elif seg.strip():
                    content.append({"type": "text", "text": seg.strip()})

            messages.append({"role": role, "content": content})
        else:
            # Assistant messages contain only text
            messages.append({"role": role, "content": [{"type": "text", "text": text}]})

    # Check for unused media files
    if image_pool:
        raise ValueError(
            f"{len(image_pool)} image(s) remain unused (not consumed by placeholders)"
        )
    if video_pool:
        raise ValueError(
            f"{len(video_pool)} video(s) remain unused (not consumed by placeholders)"
        )

    return messages


# ===========================================================================
# RAG-specific data path: per-video JSON with retrieved evidence + slow-fast
# video sampling. Model input becomes:
#     <video frames (slow-fast sampled)> + <rag evidence> + <question>
# ===========================================================================

# Slow-fast sampling config (overridable via env).
RAG_BASE_FPS         = float(os.environ.get("AICITY_BASE_FPS", "2.0"))   # frames/sec outside relevant regions
RAG_DENSE_MULT       = float(os.environ.get("AICITY_DENSE_MULT", "2.0")) # relevant regions sampled this much denser
RAG_EVIDENCE_DROPOUT = float(os.environ.get("AICITY_EVIDENCE_DROPOUT", "0.25"))
RAG_MAX_CAPS         = int(os.environ.get("AICITY_RAG_MAX_CAPS", "5"))
RAG_MAX_TRACKS       = int(os.environ.get("AICITY_RAG_MAX_TRACKS", "5"))
# Set AICITY_RAG_INCLUDE_TRACKS=0 to drop the "Tracked objects:" (object tracking
# + bounding boxes) block from the RAG evidence. Default 1 = keep current behavior.
RAG_IS_STAGE2        = int(os.environ.get("AICITY_RAG_IS_STAGE2", "1"))  # stage2: only keep relevant tracks
RAG_INCLUDE_TRACKS   = int(os.environ.get("AICITY_RAG_INCLUDE_TRACKS", "1"))
RAG_MAX_OBS          = int(os.environ.get("AICITY_RAG_MAX_OBS", "20"))
RAG_OBS_FPS          = float(os.environ.get("AICITY_RAG_OBS_FPS", "1.0"))  # sample track observations at ~1 fps
RAG_RANDOM_SEED      = int(os.environ.get("AICITY_RAG_SEED", "42"))
FRAME_FACTOR         = 2  # Qwen requires nframes divisible by 2
RAG_RNG              = random.Random(RAG_RANDOM_SEED)


def _sec(frame, fps):
    return frame / fps if fps else None


def _subsample_even(items, n):
    """Evenly sample up to `n` items, always including the first and last.

    If len(items) <= n, returns items unchanged. Otherwise picks `n` indices
    spread uniformly across [0, len-1] (endpoints inclusive).
    """
    L = len(items)
    if n <= 0 or L <= n:
        return items
    if n == 1:
        return [items[0]]
    idxs = sorted({round(i * (L - 1) / (n - 1)) for i in range(n)})
    return [items[i] for i in idxs]


def _subsample_by_time(obs, fps, target_fps, max_n):
    """Sample observations at ~`target_fps` (spacing by frame/fps seconds),
    always keeping the first and last, then cap to `max_n` via even subsampling.

    Falls back to index-based even sampling when frame/fps info is unavailable.
    """
    if len(obs) <= 1:
        return obs
    if fps and target_fps > 0:
        min_gap = fps / target_fps  # min frames between kept observations
        kept = [obs[0]]
        last_frame = obs[0].get("frame")
        for o in obs[1:-1]:
            frame = o.get("frame")
            if frame is None or last_frame is None:
                continue
            if frame - last_frame >= min_gap:
                kept.append(o)
                last_frame = frame
        kept.append(obs[-1])
    else:
        kept = obs
    return _subsample_even(kept, max_n)


def _fmt_time(a, b, fps):
    sa, sb = _sec(a, fps), _sec(b, fps)
    return f"{sa:.0f}s-{sb:.0f}s" if sa is not None else None


BACKGROUND_SYS = (
    "You are a video question-answering assistant. Answer the question about the "
    "video.\n\n"
    "You may also get a block of retrieved evidence with two kinds of context:\n"
    "1. Cross-question context: factual_information is usually reliable; "
    "potential_information is only a weak hint and "
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

# Map annotation filename stems to a specialist system prompt.
# Any stem not listed here falls back to the generic BACKGROUND_SYS.
_TASK_SYSTEM_PROMPT_MAP = {
    "video_summarization": BACKGROUND_SYS_VIDEO_SUM,
}


def _pick_system_prompt(source: dict) -> str:
    """Return the task-appropriate background system prompt.

    Resolves the task by inspecting the annotation path stem stored in
    ``source['_src_annotation']``, so no extra field is required on each item.
    """
    src = source.get("_src_annotation", "")
    stem = Path(src).stem if src else ""
    return _TASK_SYSTEM_PROMPT_MAP.get(stem, BACKGROUND_SYS)


def format_evidence_rag(result, fps, max_caps=RAG_MAX_CAPS, max_tracks=RAG_MAX_TRACKS, max_obs=RAG_MAX_OBS):
    """Build the evidence text block from a RAG `result` entry (None if empty).

    Evidence = video_summary + top-k timestamped captions (by importance) +
    top-k tracked objects (by importance). Frame numbers come from the caption
    KEY range and from real track observations; converted to seconds via `fps`.
    """

    lines = []
    if RAG_IS_STAGE2:
        stage2_factual = result.get("stage2_factual") or []
        stage2_potential = result.get("stage2_potential") or []

        ctx_lines = []
        if stage2_factual:
            ctx_lines.append("factual_information (usually reliable):")
            for it in stage2_factual:
                if not isinstance(it, dict):
                    continue
                content = (it.get("content") or "").strip()
                if content:
                    ctx_lines.append(f"  - {content}")
        if stage2_potential:
            ctx_lines.append("potential_information (weak hint, may be wrong):")
            for it in stage2_potential:
                if not isinstance(it, dict):
                    continue
                content = (it.get("content") or "").strip()
                if not content:
                    continue
                ctx_lines.append(f"  - {content}")

        if ctx_lines:
            lines.append("[Cross-question context]")
            lines.extend(ctx_lines)

    lines.append("[Visual context]")
    scene_desc = (result.get("scene_description") or "").strip()
    if scene_desc:
        lines.append(f"Scene Description: {scene_desc}")

    summ = (result.get("video_summary") or "").strip()
    if summ:
        lines.append(f"Video Summary: {summ}")

    # captions: sort by importance desc, time from the caption KEY (real frame range)
    caps = []
    for _sid, v in (result.get("segment_captions") or {}).items():
        if not isinstance(v, dict):
            continue
        key = v.get("key", "")
        cap = (v.get("caption") or "").strip()
        if "_" not in key or not cap:
            continue
        a, b = key.split("_")[:2]
        try:
            t = _fmt_time(int(a), int(b), fps)
        except ValueError:
            t = None
        caps.append((v.get("importance") or 0.0, t, cap))
    caps.sort(key=lambda x: -x[0])
    if caps:
        lines.append("Relevant captions (most relevant first):")
        for _imp, t, cap in caps[:max_caps]:
            lines.append(f"  - {t + ': ' if t else ''}{cap}")

    # tracks: only those with real observations -> list ALL observations
    # (timestamp + bounding box) so the model sees the full trajectory.
    # Disabled entirely when AICITY_RAG_INCLUDE_TRACKS=0 (object-tracking ablation).
    trk = []
    if RAG_INCLUDE_TRACKS:
        for tdat in (result.get("relevant_tracks_data") or []):
            obs = tdat.get("observations")
            if not obs:
                continue
            trk.append((tdat.get("importance") or 0.0, tdat.get("category", "object"), obs))
    trk.sort(key=lambda x: -x[0])
    if trk:
        lines.append("Tracked objects:")
        for _imp, cat, obs in trk[:max_tracks]:
            lines.append(f"  - {cat}:")
            for o in _subsample_by_time(obs, fps, RAG_OBS_FPS, max_obs):
                frame = o.get("frame")
                sec = _sec(frame, fps) if frame is not None else None
                box = o.get("box_xyxy")
                box_str = ""
                if box:
                    box_str = " box=[" + ",".join(f"{int(c)}" for c in box) + "]"
                time_str = f"{sec:.1f}s" if sec is not None else str(frame)
                lines.append(f"      {time_str}{box_str}")

    if not lines:
        return None
    return "[Retrieved evidence]\n" + "\n".join(lines)


def _video_fps(video_path):
    """Lightweight fps probe (metadata only, no full decode) for evidence timestamps."""
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if fps and fps > 0:
            return float(fps)
    except Exception:
        pass
    try:
        import decord
        return float(decord.VideoReader(video_path).get_avg_fps() or RAG_BASE_FPS)
    except Exception:
        return 30.0


def preprocess_qwen_visual_rag(source, processor, data_args) -> Dict:
    """
    RAG variant of preprocess_qwen_visual.
    """
    result   = source["_rag_result"]
    data_path = Path(source.get("data_path", ""))
    video_rel = source["video"]
    video_abs = str((data_path / video_rel).resolve())
    system_prompt = source.get("system_prompt") or source.get("sys_prompt") or None

    # fps only needed to render evidence timestamps (cheap metadata read)
    fps = _video_fps(video_abs)

    # evidence block (+ dropout) prepended to the question
    raw_question = result["question"]
    evidence = format_evidence_rag(result, fps)
    if evidence and RAG_RNG.random() >= RAG_EVIDENCE_DROPOUT:
        question_text = evidence + "\n\n" + raw_question
        system_prompt = _pick_system_prompt(source) if not system_prompt else system_prompt
    else:
        question_text = raw_question

    # training target: <think>reasoning</think>\n<answer>answer</answer>
    reasoning = result.get("gt_reasoning", "")
    answer = result.get("gt_answer", "")
    target_text = f"<think>{reasoning}</think>\n<answer>{answer}</answer>"

    # video element: path + slow-fast config consumed by vision_process readers
    video_ele = {
        "type": "video",
        "video": video_abs,
        "relevant_frame_ranges": result.get("relevant_frame_ranges") or [],
        "slowfast_base_fps": RAG_BASE_FPS, # 2 FPS
        "slowfast_dense_mult": RAG_DENSE_MULT, # 4 FPS(2*2)
        "max_frames": int(getattr(data_args, "video_max_frames", 100) or 100),
    }

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
    messages.append({
        "role": "user",
        "content": [video_ele, {"type": "text", "text": question_text}],
    })
    messages.append({"role": "assistant", "content": [{"type": "text", "text": target_text}]})

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False,
    )

    images, videos, video_kwargs = process_vision_info(
        messages,
        image_patch_size=16,
        return_video_kwargs=True,
        return_video_metadata=True,
    )
    if videos is not None:
        videos, video_metadatas = zip(*videos)
        videos = list(videos)
        video_metadatas = list(video_metadatas)
    else:
        video_metadatas = None

    inputs = processor(
        text=[text],
        images=images,
        videos=videos,
        video_metadata=video_metadatas,
        return_tensors="pt",
        do_resize=False,
        **video_kwargs,
    )

    input_ids = inputs["input_ids"]
    if isinstance(input_ids, list):
        input_ids = torch.tensor(input_ids).unsqueeze(0)

    labels = torch.full_like(input_ids, IGNORE_INDEX)
    input_ids_flat = input_ids[0].tolist()
    L = len(input_ids_flat)
    pos = 0
    while pos < L:
        if input_ids_flat[pos] == 77091:
            ans_start = pos + 2
            ans_end = ans_start
            while ans_end < L and input_ids_flat[ans_end] != 151645:
                ans_end += 1
            if ans_end < L:
                labels[0, ans_start: ans_end + 2] = input_ids[0, ans_start: ans_end + 2]
                pos = ans_end
        pos += 1

    inputs["labels"] = labels
    return inputs


def load_rag_annotations(ann_dir, data_path):
    """Glob a RAG task directory (one JSON per video) into a flat list of items.

    Each result inside a file's `results` list becomes one training item:
        { "video": <rel path>, "data_path": ..., "_rag_result": <result>, "_rag_item": True }
    """
    items = []
    files = sorted(glob.glob(os.path.join(ann_dir, "**", "*.json"), recursive=True))
    for f in files:
        try:
            doc = json.load(open(f))
        except Exception:
            continue
        if not (isinstance(doc, dict) and "results" in doc):
            continue
        video_path = doc.get("video_path", "")
        video_rel = video_path.split("/data/", 1)[-1].strip("/") if "/data/" in video_path \
            else (doc.get("video_id", "") or video_path).strip("/")
        if not video_rel:
            continue
        for result in doc["results"]:
            if not result.get("question") or not result.get("gt_answer"):
                continue
            items.append({
                "video": video_rel,
                "data_path": data_path,
                "_rag_result": result,
                "_rag_item": True,
            })
    return items


def load_rag_split_annotations(split_file, rag_dir, data_path):
    """Build RAG training items for the (video, question) pairs in a split file.

    `split_file` is a single tao-vl-reason-v1.0 JSON (e.g. train_sft_70split/mcq.json
    or val_10split/mcq.json) that defines WHICH (video_id, question) pairs belong to
    this split, plus the curated ground-truth answer/reasoning. `rag_dir` is the
    per-video RAG evidence tree for the same task (e.g. /data/aicity_train/mcq),
    where each JSON holds a `results` list with the retrieved evidence.

    The split file is the source of truth for membership and the training target;
    the RAG tree supplies the retrieved evidence. Each matched pair becomes one
    RAG item identical in shape to `load_rag_annotations` output.
    """
    try:
        doc = json.load(open(split_file))
    except Exception:
        rank0_print(f"[rag_split] WARNING: could not read split file {split_file}")
        return []
    split_items = doc.get("items", []) if isinstance(doc, dict) else []

    # (video_id, stripped question) -> (answer, reasoning)
    split_map = {}
    for it in split_items:
        vid = (it.get("video_id") or "").strip("/")
        q = (it.get("question") or "").strip()
        if not vid or not q:
            continue
        split_map[(vid, q)] = (it.get("answer", ""), it.get("reasoning", ""))

    items = []
    matched = set()
    files = sorted(glob.glob(os.path.join(rag_dir, "**", "*.json"), recursive=True))
    for f in files:
        try:
            rdoc = json.load(open(f))
        except Exception:
            continue
        if not (isinstance(rdoc, dict) and "results" in rdoc):
            continue
        video_path = rdoc.get("video_path", "")
        video_id = rdoc.get("video_id", "")
        video_rel = video_id.strip("/")
        if not video_rel and "/data/" in video_path:
            video_rel = video_path.split("/data/", 1)[-1].strip("/")
        if not video_rel:
            continue
        for result in rdoc["results"]:
            q = (result.get("question") or "").strip()
            if not q:
                continue
            key = (video_rel, q)
            if key not in split_map:
                continue
            answer, reasoning = split_map[key]
            # Use the curated split answer/reasoning as the training target,
            # falling back to the RAG result's own ground truth if missing.
            result = dict(result)
            result["gt_answer"] = answer or result.get("gt_answer", "")
            result["gt_reasoning"] = reasoning or result.get("gt_reasoning", "")
            if not result["gt_answer"]:
                continue
            items.append({
                "video": video_rel,
                "data_path": data_path,
                "_rag_result": result,
                "_rag_item": True,
            })
            matched.add(key)

    missing = len(split_map) - len(matched)
    rank0_print(
        f"[rag_split] {os.path.basename(split_file)}: matched {len(matched)}/{len(split_map)} "
        f"split items to RAG evidence ({missing} unmatched)"
    )
    return items


def preprocess_qwen_visual(
    sources,
    processor,
) -> Dict:
    if len(sources) != 1:
        raise ValueError(f"Expected 1 source, got {len(sources)}")

    source = sources[0]
    base_path = Path(source.get("data_path", ""))
    system_prompt = source.get("system_prompt") or source.get("sys_prompt") or None
    messages = _build_messages(source, base_path, system_prompt=system_prompt)

    # # Original term
    # full_result = processor.apply_chat_template(
    #     messages, tokenize=True, return_dict=True, return_tensors="pt"
    # )

    # More controliable video input
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,  # for SFT training usually False
    )

    images, videos, video_kwargs = process_vision_info(
        messages,
        image_patch_size=16,
        return_video_kwargs=True,
        return_video_metadata=True,
    )

    # Split videos and metadata if present
    if videos is not None:
        videos, video_metadatas = zip(*videos)
        videos = list(videos)
        video_metadatas = list(video_metadatas)
    else:
        video_metadatas = None

    # 4. Final processor call to get input_ids + vision tensors
    inputs = processor(
        text=[text],               # batch of size 1
        images=images,
        videos=videos,
        video_metadata=video_metadatas,
        return_tensors="pt",
        do_resize=False,           # IMPORTANT: already resized in process_vision_info
        **video_kwargs,
    )

    input_ids = inputs["input_ids"]

    # input_ids = full_result["input_ids"]
    
    if isinstance(input_ids, list):
        input_ids = torch.tensor(input_ids).unsqueeze(0)

    labels = torch.full_like(input_ids, IGNORE_INDEX)

    input_ids_flat = input_ids[0].tolist()
    L = len(input_ids_flat)
    pos = 0
    while pos < L:
        if input_ids_flat[pos] == 77091:
            ans_start = pos + 2
            ans_end = ans_start
            while ans_end < L and input_ids_flat[ans_end] != 151645:
                ans_end += 1
            if ans_end < L:
                labels[0, ans_start : ans_end + 2] = input_ids[
                    0, ans_start : ans_end + 2
                ]
                pos = ans_end
        pos += 1

    inputs["labels"] = labels
    # full_result["labels"] = labels
    # full_result["input_ids"] = input_ids
    return inputs

class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, processor, data_args):
        super(LazySupervisedDataset, self).__init__()

        dataset = data_args.dataset_use.split(",")
        dataset_list = data_list(dataset)
        rank0_print(f"Loading datasets: {dataset_list}")
        self.video_max_total_pixels = getattr(
            data_args, "video_max_total_pixels", 1664 * 28 * 28
        )
        self.video_min_total_pixels = getattr(
            data_args, "video_min_total_pixels", 320 * 180 * 3
        )
        self.model_type = data_args.model_type
        if data_args.model_type == "qwen3vl":
            self.get_rope_index = get_rope_index_3
        elif data_args.model_type == "qwen2.5vl":
            self.get_rope_index = get_rope_index_25
        elif data_args.model_type == "qwen2vl":
            self.get_rope_index = get_rope_index_2
        else:
            raise ValueError(f"model_type: {data_args.model_type} not supported")

        list_data_dict = []

        for data in dataset_list:
            # RAG datasets: annotation_path is a DIRECTORY of per-video JSONs.
            if data.get("is_rag"):
                annotations = load_rag_annotations(data["annotation_path"], data["data_path"])
                sampling_rate = data.get("sampling_rate", 1.0)
                if sampling_rate < 1.0:
                    annotations = RAG_RNG.sample(annotations, int(len(annotations) * sampling_rate))
                    rank0_print(f"sampling {len(annotations)} examples from RAG dataset {data['annotation_path']}")
                else:
                    rank0_print(f"RAG dataset: {data['annotation_path']} ({len(annotations)} items)")
                for ann in annotations:
                    ann["_src_annotation"] = data["annotation_path"]
                    ann["_src_data_path"] = data["data_path"]
                    ann["_src_dataset_name"] = data.get("name", os.path.basename(data["annotation_path"]))
                list_data_dict += annotations
                continue

            # RAG-split datasets: annotation_path is a single tao-vl-reason FILE
            # that defines membership + ground truth; rag_dir is the per-video RAG
            # evidence tree. Join them into RAG items restricted to this split.
            if data.get("is_rag_split"):
                annotations = load_rag_split_annotations(
                    data["annotation_path"], data["rag_dir"], data["data_path"]
                )
                sampling_rate = data.get("sampling_rate", 1.0)
                if sampling_rate < 1.0:
                    annotations = RAG_RNG.sample(annotations, int(len(annotations) * sampling_rate))
                    rank0_print(f"sampling {len(annotations)} examples from RAG-split dataset {data['annotation_path']}")
                else:
                    rank0_print(f"RAG-split dataset: {data['annotation_path']} ({len(annotations)} items)")
                for ann in annotations:
                    ann["_src_annotation"] = data["annotation_path"]
                    ann["_src_data_path"] = data["data_path"]
                    ann["_src_dataset_name"] = data.get("name", os.path.basename(data["annotation_path"]))
                list_data_dict += annotations
                continue

            file_format = data["annotation_path"].split(".")[-1]
            if file_format == "jsonl":
                annotations = read_jsonl(data["annotation_path"])
            else:
                annotations = json.load(open(data["annotation_path"], "r"))

            # Handle tao-vl-reason-v1.0 format (nvidia/PhysicalAI-Traffic-Anomaly-Reasoning)
            if isinstance(annotations, dict) and annotations.get("format") == "tao-vl-reason-v1.0":
                raw_items = annotations.get("items", [])
                # Optional dataset-level system prompt applied to every item.
                dataset_sys_prompt = annotations.get("system_prompt")
                converted = []
                for item in raw_items:
                    reasoning = item.get("reasoning", "")
                    answer = item.get("answer", "")
                    response = f"<think>{reasoning}</think>\n<answer>{answer}</answer>"
                    conv_item = {
                        "video": item["video_id"],
                        "conversations": [
                            {"from": "human", "value": f"<video>\n{item['question']}"},
                            {"from": "gpt",   "value": response},
                        ],
                    }
                    # Per-item system prompt takes precedence over the dataset one.
                    sys_prompt = item.get("system_prompt") or dataset_sys_prompt
                    if sys_prompt:
                        conv_item["system_prompt"] = sys_prompt
                    converted.append(conv_item)
                annotations = converted

            sampling_rate = data.get("sampling_rate", 1.0)
            if sampling_rate < 1.0:
                annotations = RAG_RNG.sample(
                    annotations, int(len(annotations) * sampling_rate)
                )
                rank0_print(f"sampling {len(annotations)} examples from dataset {data}")
            else:
                rank0_print(f"dataset name: {data}")
            for ann in annotations:
                if isinstance(ann, list):
                    for sub_ann in ann:
                        sub_ann["data_path"] = data["data_path"]
                else:
                    ann["data_path"] = data["data_path"]

                ann["_src_annotation"] = data["annotation_path"]
                ann["_src_data_path"] = data["data_path"]
                ann["_src_dataset_name"] = data.get("name", os.path.basename(data["annotation_path"]))
                
            list_data_dict += annotations

        rank0_print(f"Total training samples: {len(list_data_dict)}")

        RAG_RNG.shuffle(list_data_dict)  # Fixed-seed shuffle for reproducibility

        rank0_print("Formatting inputs...Skip in lazy mode")
        # The update_processor_pixels is no need now
        processor = update_processor_pixels(processor, data_args) # get updated image/video processor
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.data_args = data_args
        self.merge_size = getattr(processor.image_processor, "merge_size", 2)
        self.list_data_dict = list_data_dict

        if data_args.data_packing:
            self.item_fn = self._get_packed_item
        else:
            self.item_fn = self._get_item


    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            if sample.get("_rag_item"):
                # rough proxy: question + reasoning + answer word counts (+video)
                r = sample["_rag_result"]
                words = len((r.get("question", "") + " " + r.get("gt_reasoning", "")
                             + " " + r.get("gt_answer", "")).split())
                length_list.append(words + 128)
                continue
            img_tokens = 128 if "image" in sample else 0
            length_list.append(
                sum(len(conv["value"].split()) for conv in sample["conversations"])
                + img_tokens
            )
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            if sample.get("_rag_item"):
                r = sample["_rag_result"]
                cur_len = len((r.get("question", "") + " " + r.get("gt_reasoning", "")
                               + " " + r.get("gt_answer", "")).split())
                length_list.append(cur_len)  # always multimodal (has video)
                continue
            cur_len = sum(
                len(conv["value"].split()) for conv in sample["conversations"]
            )
            cur_len = (
                cur_len if ("image" in sample) or ("video" in sample) else -cur_len
            )
            length_list.append(cur_len)
        return length_list

    @property
    def pre_calculated_length(self):
        if "num_tokens" in self.list_data_dict[0]:
            length_list = [sample["num_tokens"] for sample in self.list_data_dict]
            return np.array(length_list)
        else:
            print("No pre-calculated length available.")
            return np.array([1] * len(self.list_data_dict))

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        num_base_retries = 3
        num_final_retries = 30

        # try the current sample first
        for attempt_idx in range(num_base_retries):
            try:
                sources = self.list_data_dict[i]
                if isinstance(sources, dict):
                    sources = [sources]
                if not hasattr(self, '_debug_count'):
                    self._debug_count = 0
                if self._debug_count < 3:
                    print(f"[DEBUG LazySupervisedDataset] i={i}, sources={sources}", flush=True)
                    self._debug_count += 1
                sample = self.item_fn(sources)
                return sample
            except Exception as e:
                # sleep 1s in case it is a cloud disk issue
                print(f"[Try #{attempt_idx}] Failed to fetch sample {i}. Exception:", e)
                time.sleep(1)

        # try other samples, in case it is file corruption issue
        for attempt_idx in range(num_base_retries):
            try:
                next_index = min(i + 1, len(self.list_data_dict) - 1)
                sources = self.list_data_dict[next_index]
                if isinstance(sources, dict):
                    sources = [sources]

                sample = self.item_fn(sources)
                return sample
            except Exception as e:
                # no need to sleep
                print(
                    f"[Try other #{attempt_idx}] Failed to fetch sample {next_index}. Exception:",
                    e,
                )
                pass

        try:
            sources = self.list_data_dict[i]
            if isinstance(sources, dict):
                sources = [sources]
            sample = self.item_fn(sources)
            return sample
        except Exception as e:
            raise e

    def _get_item(self, sources) -> Dict[str, torch.Tensor]:
        # RAG items carry raw retrieval results + need slow-fast video sampling.
        source0 = sources[0] if isinstance(sources, (list, tuple)) else sources
        if isinstance(source0, dict) and source0.get("_rag_item"):
            data_dict = preprocess_qwen_visual_rag(
                source0,
                self.processor,
                self.data_args,
            )
        else:
            data_dict = preprocess_qwen_visual(
                sources, # source is one item in self.list_data_dict which contains 
                self.processor,
            )

        seq_len = data_dict["input_ids"][0].size(0)

        if "image_grid_thw" in data_dict:
            grid_thw = data_dict.get("image_grid_thw")
            if not isinstance(grid_thw, Sequence):
                grid_thw = [grid_thw]
        else:
            grid_thw = None

        if "video_grid_thw" in data_dict:
            video_grid_thw = data_dict.get("video_grid_thw")
            if not isinstance(video_grid_thw, Sequence):
                video_grid_thw = [video_grid_thw]
            second_per_grid_ts = [
                self.processor.video_processor.temporal_patch_size
                / self.processor.video_processor.fps
            ] * len(video_grid_thw)

        else:
            video_grid_thw = None
            second_per_grid_ts = None

        position_ids, _ = self.get_rope_index(
            self.merge_size,
            data_dict["input_ids"],
            image_grid_thw=torch.cat(grid_thw, dim=0) if grid_thw else None,
            video_grid_thw=(
                torch.cat(video_grid_thw, dim=0) if video_grid_thw else None
            ),
            second_per_grid_ts=second_per_grid_ts if second_per_grid_ts else None,
        )

        data_dict["position_ids"] = position_ids
        data_dict["attention_mask"] = [seq_len]
        return data_dict

    def _get_packed_item(self, sources) -> Dict[str, torch.Tensor]:

        if isinstance(sources, dict):
            if isinstance(source, dict):
                sources = [sources]
            assert len(sources) == 1
            return self._get_item(sources)

        if isinstance(sources, list):
            data_list = []
            new_data_dict = {}
            for source in sources:
                if isinstance(source, dict):
                    source = [source]
                assert (
                    len(source) == 1
                )
                data_list.append(self._get_item(source))

            input_ids = torch.cat([d["input_ids"] for d in data_list], dim=1)
            labels = torch.cat([d["labels"] for d in data_list], dim=1)
            position_ids = torch.cat([d["position_ids"] for d in data_list], dim=2)
            attention_mask = [
                d["attention_mask"][0] for d in data_list if "attention_mask" in d
            ]
            new_data_dict = {
                "input_ids": input_ids,
                "labels": labels,
                "position_ids": position_ids,
                "attention_mask": attention_mask if attention_mask else None,
            }

            if any("pixel_values" in d for d in data_list):
                new_data_dict.update(
                    {
                        "pixel_values": torch.cat(
                            [
                                d["pixel_values"]
                                for d in data_list
                                if "pixel_values" in d
                            ],
                            dim=0,
                        ),
                        "image_grid_thw": torch.cat(
                            [
                                d["image_grid_thw"]
                                for d in data_list
                                if "image_grid_thw" in d
                            ],
                            dim=0,
                        ),
                    }
                )

            if any("pixel_values_videos" in d for d in data_list):
                new_data_dict.update(
                    {
                        "pixel_values_videos": torch.cat(
                            [
                                d["pixel_values_videos"]
                                for d in data_list
                                if "pixel_values_videos" in d
                            ],
                            dim=0,
                        ),
                        "video_grid_thw": torch.cat(
                            [
                                d["video_grid_thw"]
                                for d in data_list
                                if "video_grid_thw" in d
                            ],
                            dim=0,
                        ),
                    }
                )
            return new_data_dict


def pad_and_cat(tensor_list):
    max_length = max(tensor.shape[2] for tensor in tensor_list)

    padded_tensors = []
    for tensor in tensor_list:
        pad_length = max_length - tensor.shape[2]
        padded_tensor = torch.nn.functional.pad(tensor, (0, pad_length), "constant", 1)
        padded_tensors.append(padded_tensor)

    stacked_tensor = torch.cat(padded_tensors, dim=1)

    return stacked_tensor


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, position_ids = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels", "position_ids")
        )
        input_ids = [ids.squeeze(0) for ids in input_ids]
        labels = [ids.squeeze(0) for ids in labels]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        position_ids = pad_and_cat(position_ids)
        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        labels = labels[:, : self.tokenizer.model_max_length]
        position_ids = position_ids[:, :, : self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
        images = list(
            instance["pixel_values"]
            for instance in instances
            if "pixel_values" in instance
        )
        videos = list(
            instance["pixel_values_videos"]
            for instance in instances
            if "pixel_values_videos" in instance
        )
        if len(images) != 0:
            concat_images = torch.cat([image for image in images], dim=0)
            grid_thw = [
                instance["image_grid_thw"]
                for instance in instances
                if "image_grid_thw" in instance
            ]
            grid_thw = torch.cat(grid_thw, dim=0)
        else:
            concat_images = None
            grid_thw = None

        if len(videos) != 0:
            concat_videos = torch.cat([video for video in videos], dim=0)
            video_grid_thw = [
                instance["video_grid_thw"]
                for instance in instances
                if "video_grid_thw" in instance
            ]
            video_grid_thw = torch.cat(video_grid_thw, dim=0)
        else:
            concat_videos = None
            video_grid_thw = None

        batch["pixel_values"] = concat_images
        batch["image_grid_thw"] = grid_thw
        batch["pixel_values_videos"] = concat_videos
        batch["video_grid_thw"] = video_grid_thw
        batch["position_ids"] = position_ids
        return batch


@dataclass
class FlattenedDataCollatorForSupervisedDataset(DataCollatorForSupervisedDataset):
    """Collate examples into packed sequence with multi-modal support."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, position_ids, attention_mask = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels", "position_ids", "attention_mask")
        )
        attention_mask = list(
            itertools.chain(
                *(
                    instance["attention_mask"]
                    for instance in instances
                    if "attention_mask" in instance
                )
            )
        )
        seq_lens = torch.tensor([0] + attention_mask, dtype=torch.int32)
        cumsum_seq_lens = torch.cumsum(seq_lens, dim=0, dtype=torch.int32)
        input_ids = torch.cat(input_ids, dim=1)
        labels = torch.cat(labels, dim=1)
        position_ids = torch.cat(position_ids, dim=2)

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=cumsum_seq_lens,
            position_ids=position_ids,
        )
        images = list(
            instance["pixel_values"]
            for instance in instances
            if "pixel_values" in instance
        )
        videos = list(
            instance["pixel_values_videos"]
            for instance in instances
            if "pixel_values_videos" in instance
        )
        if len(images) != 0:
            concat_images = torch.cat([image for image in images], dim=0)
            grid_thw = [
                instance["image_grid_thw"]
                for instance in instances
                if "image_grid_thw" in instance
            ]
            grid_thw = torch.cat(grid_thw, dim=0)
        else:
            concat_images = None
            grid_thw = None

        if len(videos) != 0:
            concat_videos = torch.cat([video for video in videos], dim=0)
            video_grid_thw = [
                instance["video_grid_thw"]
                for instance in instances
                if "video_grid_thw" in instance
            ]
            video_grid_thw = torch.cat(video_grid_thw, dim=0)
        else:
            concat_videos = None
            video_grid_thw = None

        batch["pixel_values"] = concat_images
        batch["image_grid_thw"] = grid_thw
        batch["pixel_values_videos"] = concat_videos
        batch["video_grid_thw"] = video_grid_thw

        return batch


def make_supervised_data_module_rag(processor, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(processor, data_args=data_args)

    if data_args.data_flatten or data_args.data_packing:
        data_collator = FlattenedDataCollatorForSupervisedDataset(processor.tokenizer)
        return dict(
            train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
        )
    data_collator = DataCollatorForSupervisedDataset(processor.tokenizer)
    return dict(
        train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
    )


# ===========================================================================
# GRPO-compatible RAG dataset
# Returns raw prompt text + ground-truth answer instead of tokenized tensors,
# so that QwenGRPOTrainer can generate completions and score them with a reward.
# ===========================================================================

# ── task-type lookup from annotation filename / directory stem ───────────────
# mcq_openended and bcq_openended use BERTScore (open-ended answers), so they
# fall through to the "bertscore" default along with all other free-text tasks.
_FILENAME_TASK_MAP = {
    "mcq":                   "mcq",
    "bcq":                   "mcq",
    "temporal_localization": "temporal",
}


def _task_type_from_path(annotation_path: str) -> str:
    """Derive GRPO reward task type from annotation filename or directory stem."""
    stem = Path(annotation_path).stem
    return _FILENAME_TASK_MAP.get(stem, "bertscore")


class GRPORagDataset(Dataset):
    """GRPO-compatible dataset that mirrors LazySupervisedDataset's data loading
    but returns un-tokenised prompt + ground-truth instead of fully tokenised tensors.

    Each __getitem__ returns::

        {
            "prompt":       str  – apply_chat_template output (add_generation_prompt=True)
            "assistant":    str  – target text (ground truth, for reward computation)
            "task_type":    str  – 'mcq' | 'temporal' | 'bertscore' (from filename stem)
            "images":       list – pre-processed images (or None)
            "videos":       list – pre-processed videos (with metadata for Qwen3VL)
            "video_kwargs": dict – extra video kwargs for the processor
        }
    """

    def __init__(self, processor, data_args):
        super().__init__()

        dataset = data_args.dataset_use.split(",")
        dataset_list = data_list(dataset)
        rank0_print(f"[GRPORagDataset] Loading datasets: {dataset_list}")

        list_data_dict = []

        for data in dataset_list:
            # RAG datasets: annotation_path is a directory of per-video JSONs
            if data.get("is_rag"):
                annotations = load_rag_annotations(data["annotation_path"], data["data_path"])
                sampling_rate = data.get("sampling_rate", 1.0)
                if sampling_rate < 1.0:
                    annotations = RAG_RNG.sample(annotations, int(len(annotations) * sampling_rate))
                    rank0_print(f"[GRPORagDataset] sampling {len(annotations)} from {data['annotation_path']}")
                else:
                    rank0_print(f"[GRPORagDataset] RAG: {data['annotation_path']} ({len(annotations)} items)")
                task_type = _task_type_from_path(data["annotation_path"])
                for ann in annotations:
                    ann["_src_annotation"] = data["annotation_path"]
                    ann["_src_data_path"] = data["data_path"]
                    ann["task_type"] = task_type
                list_data_dict += annotations
                continue

            # RAG-split datasets: single tao-vl-reason FILE for membership +
            # per-video RAG evidence tree (rag_dir), joined to this split only.
            if data.get("is_rag_split"):
                annotations = load_rag_split_annotations(
                    data["annotation_path"], data["rag_dir"], data["data_path"]
                )
                sampling_rate = data.get("sampling_rate", 1.0)
                if sampling_rate < 1.0:
                    annotations = RAG_RNG.sample(annotations, int(len(annotations) * sampling_rate))
                    rank0_print(f"[GRPORagDataset] sampling {len(annotations)} from {data['annotation_path']}")
                else:
                    rank0_print(f"[GRPORagDataset] RAG-split: {data['annotation_path']} ({len(annotations)} items)")
                task_type = _task_type_from_path(data["annotation_path"])
                for ann in annotations:
                    ann["_src_annotation"] = data["annotation_path"]
                    ann["_src_data_path"] = data["data_path"]
                    ann["task_type"] = task_type
                list_data_dict += annotations
                continue

            file_format = data["annotation_path"].split(".")[-1]
            if file_format == "jsonl":
                annotations = read_jsonl(data["annotation_path"])
            else:
                annotations = json.load(open(data["annotation_path"], "r"))

            # Handle tao-vl-reason-v1.0 format
            if isinstance(annotations, dict) and annotations.get("format") == "tao-vl-reason-v1.0":
                raw_items = annotations.get("items", [])
                # Optional dataset-level system prompt applied to every item.
                dataset_sys_prompt = annotations.get("system_prompt")
                converted = []
                for item in raw_items:
                    reasoning = item.get("reasoning", "")
                    answer = item.get("answer", "")
                    response = f"<think>{reasoning}</think>\n<answer>{answer}</answer>"
                    conv_item = {
                        "video": item["video_id"],
                        "conversations": [
                            {"from": "human", "value": f"<video>\n{item['question']}"},
                            {"from": "gpt",   "value": response},
                        ],
                    }
                    # Per-item system prompt takes precedence over the dataset one.
                    sys_prompt = item.get("system_prompt") or dataset_sys_prompt
                    if sys_prompt:
                        conv_item["system_prompt"] = sys_prompt
                    converted.append(conv_item)
                annotations = converted

            sampling_rate = data.get("sampling_rate", 1.0)
            if sampling_rate < 1.0:
                annotations = RAG_RNG.sample(annotations, int(len(annotations) * sampling_rate))
                rank0_print(f"[GRPORagDataset] sampling {len(annotations)} from {data}")
            else:
                rank0_print(f"[GRPORagDataset] dataset: {data}")

            task_type = _task_type_from_path(data["annotation_path"])
            for ann in annotations:
                if isinstance(ann, list):
                    for sub_ann in ann:
                        sub_ann["data_path"] = data["data_path"]
                        sub_ann["task_type"] = task_type
                else:
                    ann["data_path"] = data["data_path"]
                    ann["task_type"] = task_type
                ann["_src_annotation"] = data["annotation_path"]
                ann["_src_data_path"] = data["data_path"]

            list_data_dict += annotations

        rank0_print(f"[GRPORagDataset] Total samples: {len(list_data_dict)}")
        RAG_RNG.shuffle(list_data_dict)

        processor = update_processor_pixels(processor, data_args)
        self.processor = processor
        self.data_args = data_args
        self.list_data_dict = list_data_dict

        image_patch_size = getattr(self.processor.image_processor, "patch_size", None)
        self.image_patch_size = int(image_patch_size) if image_patch_size is not None else 16

        # Determine whether to return video metadata (required for Qwen3-VL)
        model_id = str(getattr(processor, "_name_or_path", "") or "").lower()
        processor_name = type(processor).__name__.lower()
        video_processor_name = type(getattr(processor, "video_processor", None)).__name__.lower()
        self.return_video_metadata = (
            "qwen3" in model_id
            or "qwen3" in processor_name
            or "qwen3" in video_processor_name
        )

    def __len__(self):
        return len(self.list_data_dict)

    # ------------------------------------------------------------------ #
    #  Per-item builders                                                    #
    # ------------------------------------------------------------------ #

    def _get_rag_item(self, source):
        """Build a GRPO sample from a RAG annotation item."""
        result = source["_rag_result"]
        data_path = Path(source.get("data_path", ""))
        video_rel = source["video"]
        video_abs = str((data_path / video_rel).resolve())
        system_prompt = source.get("system_prompt") or source.get("sys_prompt") or None

        # Evidence text (cheap fps probe, + dropout)
        fps = _video_fps(video_abs)
        raw_question = result["question"]
        evidence = format_evidence_rag(result, fps)
        if evidence and RAG_RNG.random() >= RAG_EVIDENCE_DROPOUT:
            question_text = evidence + "\n\n" + raw_question
            system_prompt = _pick_system_prompt(source) if not system_prompt else system_prompt    
        else:
            question_text = raw_question

        # Ground-truth target (used by the reward function, NOT fed to the model)
        reasoning = result.get("gt_reasoning", "")
        answer = result.get("gt_answer", "")
        assistant_text = f"<think>{reasoning}</think>\n<answer>{answer}</answer>"
        task_type = source.get("task_type", "bertscore")

        # Slow-fast video element
        video_ele = {
            "type": "video",
            "video": video_abs,
            "relevant_frame_ranges": result.get("relevant_frame_ranges") or [],
            "slowfast_base_fps": RAG_BASE_FPS,
            "slowfast_dense_mult": RAG_DENSE_MULT,
            "max_frames": int(getattr(self.data_args, "video_max_frames", 100) or 100),
        }

        prompt_messages = []
        if system_prompt:
            prompt_messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
        prompt_messages.append({
            "role": "user",
            "content": [video_ele, {"type": "text", "text": question_text}],
        })

        user_prompt = self.processor.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        images, videos, video_kwargs = process_vision_info(
            prompt_messages,
            image_patch_size=self.image_patch_size,
            return_video_kwargs=True,
            return_video_metadata=self.return_video_metadata,
        )

        return dict(
            prompt=user_prompt,
            assistant=assistant_text,
            task_type=task_type,
            images=images,
            videos=videos,
            video_kwargs=video_kwargs,
        )

    def _get_regular_item(self, source):
        """Build a GRPO sample from a standard conversations annotation."""
        base_path = Path(source.get("data_path", ""))
        system_prompt = source.get("system_prompt") or source.get("sys_prompt") or None

        conversations = source["conversations"]
        user_turn = conversations[0]
        assistant_turn = conversations[1]

        image_files = source.get("image") or []
        if isinstance(image_files, str):
            image_files = [image_files]
        image_files = [
            str((base_path / p).resolve()) if not os.path.isabs(p) else p
            for p in image_files
        ]

        video_files = source.get("video") or []
        if isinstance(video_files, str):
            video_files = [video_files]
        video_files = [
            str((base_path / p).resolve()) if not os.path.isabs(p) else p
            for p in video_files
        ]

        image_pool = [{"type": "image", "image": p} for p in image_files]
        video_pool = [{"type": "video", "video": p} for p in video_files]

        text = user_turn["value"]
        content = []
        for seg in re.split(r"(<image>|<video>)", text):
            if seg == DEFAULT_IMAGE_TOKEN:
                content.append(image_pool.pop(0))
            elif seg == DEFAULT_VIDEO_TOKEN:
                content.append(video_pool.pop(0))
            elif seg.strip():
                content.append({"type": "text", "text": seg.strip()})

        prompt_messages = []
        if system_prompt:
            prompt_messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
        prompt_messages.append({"role": "user", "content": content})

        user_prompt = self.processor.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        images, videos, video_kwargs = process_vision_info(
            prompt_messages,
            image_patch_size=self.image_patch_size,
            return_video_kwargs=True,
            return_video_metadata=self.return_video_metadata,
        )

        task_type = source.get("task_type", "bertscore")

        return dict(
            prompt=user_prompt,
            assistant=assistant_turn["value"],
            task_type=task_type,
            images=images,
            videos=videos,
            video_kwargs=video_kwargs,
        )

    def __getitem__(self, i):
        num_retries = 5
        for attempt in range(num_retries):
            try:
                source = self.list_data_dict[i]
                if i < 3 and attempt == 0:
                    print(f"[DEBUG GRPORagDataset] i={i}, source={source}", flush=True)
                if source.get("_rag_item"):
                    return self._get_rag_item(source)
                else:
                    return self._get_regular_item(source)
            except Exception as e:
                print(f"[GRPORagDataset] Retry {attempt} for sample {i}: {e}", flush=True)
                i = random.randint(0, len(self.list_data_dict) - 1)
        raise RuntimeError(f"[GRPORagDataset] Failed to load sample after {num_retries} retries")


def make_grpo_data_module_rag(processor, data_args) -> Dict:
    """Make a GRPO-compatible dataset using the RAG data processor."""
    dataset = GRPORagDataset(processor=processor, data_args=data_args)
    return dict(train_dataset=dataset, eval_dataset=None)


if __name__ == "__main__":
    pass
