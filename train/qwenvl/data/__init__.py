import os
import re

# Repo root, resolved relative to this file's own location (train/qwenvl/data/)
# so the defaults below work regardless of the caller's cwd (repo root, train/,
# inside Docker, or on bare metal) — no hardcoded /workspace or /data prefix
# required. Override any of the env vars below to point elsewhere (e.g. if
# your annotation/RAG trees live outside the repo).
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

# Video root — videos are typically NOT checked into the repo (mount/download
# them separately). Override with: export AICITY_VIDEO_DIR=/path/to/videos
_AICITY_VIDEO_DIR = os.environ.get("AICITY_VIDEO_DIR", os.path.join(_REPO_ROOT, "data", "videos"))

# AICity Challenge 2026 Track 3 training annotations (data/dataset/train/track3/).
# Override with: export AICITY_TRAIN_DIR=/path/to/track3
_AICITY_TRAIN_DIR = os.environ.get("AICITY_TRAIN_DIR", os.path.join(_REPO_ROOT, "data", "dataset", "train", "track3"))

# PSI-VQA training annotations (data/dataset/train/PSI/).
# Override with: export PSI_VQA_TRAIN_DIR=/path/to/PSI
_PSI_VQA_TRAIN_DIR = os.environ.get("PSI_VQA_TRAIN_DIR", os.path.join(_REPO_ROOT, "data", "dataset", "train", "PSI"))

# RAG per-video annotation tree, one directory per task (data/RAG_Info/train/{task}/).
# Override with: export AICITY_RAG_DIR=/path/to/RAG_Info/train
_AICITY_RAG_DIR = os.environ.get("AICITY_RAG_DIR", os.path.join(_REPO_ROOT, "data", "RAG_Info", "train"))


def _aicity_train(task: str) -> dict:
    return {
        "annotation_path": os.path.join(_AICITY_TRAIN_DIR, f"{task}.json"),
        "data_path": _AICITY_VIDEO_DIR,
    }


def _psi_vqa_rag(task: str) -> dict:
    # annotation_path is the plain PSI-VQA split file; rag_dir is the shared
    # per-video RAG evidence tree for the same task (joined at load time).
    return {
        "annotation_path": os.path.join(_PSI_VQA_TRAIN_DIR, f"{task}.json"),
        "data_path": _AICITY_VIDEO_DIR,
        "is_rag_split": True,
        "rag_dir": os.path.join(_AICITY_RAG_DIR, task),
    }


def _aicity_rag(task: str) -> dict:
    # annotation_path is the plain track3 train split file; rag_dir is the
    # per-video RAG evidence tree for the same task (joined at load time).
    return {
        "annotation_path": os.path.join(_AICITY_TRAIN_DIR, f"{task}.json"),
        "data_path": _AICITY_VIDEO_DIR,
        "is_rag_split": True,
        "rag_dir": os.path.join(_AICITY_RAG_DIR, task),
    }


data_dict = {
    # ---------------------------------------------------------------------------
    # PSI-VQA training annotations (data/dataset/train/PSI)
    # ---------------------------------------------------------------------------
    "psi_vqa_train_bcq":                  _psi_vqa_rag("bcq"),
    "psi_vqa_train_mcq":                  _psi_vqa_rag("mcq"),
    "psi_vqa_train_open_qa":              _psi_vqa_rag("open_qa"),
    "psi_vqa_train_temporal_localization":_psi_vqa_rag("temporal_localization"),
    # ---------------------------------------------------------------------------
    # RAG per-video tree (data/RAG_Info/train/{task}). "scene_description" has
    # no RAG evidence by design, so it falls back to the plain annotation file
    # (same as the old _aicity_*_plain helpers).
    # ---------------------------------------------------------------------------
    "aicity_rag_mcq":                  _aicity_rag("mcq"),
    "aicity_rag_mcq_openended":        _aicity_rag("mcq_openended"),
    "aicity_rag_bcq":                  _aicity_rag("bcq"),
    "aicity_rag_bcq_openended":        _aicity_rag("bcq_openended"),
    "aicity_rag_open_qa":              _aicity_rag("open_qa"),
    "aicity_rag_causal_linkage":       _aicity_rag("causal_linkage"),
    "aicity_rag_scene_description":    _aicity_train("scene_description"),
    "aicity_rag_temporal_description": _aicity_rag("temporal_description"),
    "aicity_rag_temporal_localization":_aicity_rag("temporal_localization"),
    "aicity_rag_video_summarization":  _aicity_rag("video_summarization"),
}


def parse_sampling_rate(dataset_name):
    match = re.search(r"%(\d+)$", dataset_name)
    if match:
        return int(match.group(1)) / 100.0
    return 1.0


def data_list(dataset_names):
    config_list = []
    for dataset_name in dataset_names:
        sampling_rate = parse_sampling_rate(dataset_name)
        dataset_name = re.sub(r"%(\d+)$", "", dataset_name)
        if dataset_name in data_dict.keys():
            config = data_dict[dataset_name].copy()
            config["sampling_rate"] = sampling_rate
            config_list.append(config)
        else:
            raise ValueError(f"do not find {dataset_name}")
    return config_list


if __name__ == "__main__":
    dataset_names = ["aicity_train_mcq"]
    configs = data_list(dataset_names)
    for config in configs:
        print(config)
