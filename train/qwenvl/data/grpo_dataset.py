import copy
import os
import re
import sys
import random
from typing import Any, Dict, List
import torch
import transformers
try:
    import ujson as json
except ModuleNotFoundError:  # teatime container lacks ujson; stdlib json is a drop-in here
    import json
from torch.utils.data import Dataset
from pathlib import Path
from qwen_vl_utils import process_vision_info

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from qwenvl.train.argument import DataArguments

DEFAULT_IM_START_TOKEN = "<|im_start|>"
DEFAULT_IM_END_TOKEN = "<|im_end|>"

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"

class GRPODataset(Dataset):
    """Dataset for DPO training"""

    def __init__(
        self,
        data_path: str | list,
        processor: transformers.ProcessorMixin,
        data_args: DataArguments,
        model_id,
        padding=True,
    ):
        super(GRPODataset, self).__init__()
        
        if isinstance(data_path, str):
            list_data_dict = json.load(open(data_path, "r"))
        elif isinstance(data_path, (list, tuple)) and len(data_path) > 0 and isinstance(data_path[0], str):
            # list of json paths
            list_data_dict = []
            for p in data_path:
                list_data_dict.extend(json.load(open(p, "r")))
        else:
            # already a list of samples (list[dict])
            list_data_dict = data_path

        self.model_id = model_id
        self.processor = processor
        self.list_data_dict = list_data_dict
        random.seed(42)
        random.shuffle(self.list_data_dict)
        self.data_args = data_args
        self.padding = padding
        self.image_min_pixel = 64*32*32
        self.image_max_pixel = 1024*32*32
        self.video_min_pixel = data_args.video_min_pixels
        self.video_max_pixel = data_args.video_max_pixels
        self.image_resized_w = None
        self.image_resized_h = None
        self.video_resized_w = None
        self.video_resized_h = None
        self.fps = data_args.fps
        self.nframes = data_args.nframes

        image_patch_size = getattr(self.processor.image_processor, "patch_size", None)
        if image_patch_size is not None:
            self.image_patch_size = int(image_patch_size)
        else:
            self.image_patch_size = 16 if "Qwen3" in self.model_id else 14

        model_name = str(model_id).lower()
        processor_name = type(self.processor).__name__.lower()
        video_processor_name = type(getattr(self.processor, "video_processor", None)).__name__.lower()
        self.return_video_metadata = (
            "qwen3" in model_name
            or "qwen3" in processor_name
            or "qwen3" in video_processor_name
        )

        self.processor.image_processor.do_resize = False

    def _make_abs_path(self, base_path: Path, media_path: str) -> str:
        if media_path.startswith("http://") or media_path.startswith("https://"):
            return media_path
        if os.path.isabs(media_path):
            return media_path
        return str((base_path / media_path).resolve())

    def _build_user_content(
        self,
        text: str,
        image_paths: List[str],
        video_paths: List[str],
    ) -> List[Dict[str, Any]]:
        image_pool = [{"type": "image", "image": p} for p in image_paths]
        video_pool = [{"type": "video", "video": p} for p in video_paths]

        content: List[Dict[str, Any]] = []
        text_parts = re.split(r"(<image>|<video>)", text)
        for segment in text_parts:
            if segment == DEFAULT_IMAGE_TOKEN:
                if not image_pool:
                    raise ValueError("More <image> placeholders than provided images")
                content.append(image_pool.pop(0))
            elif segment == DEFAULT_VIDEO_TOKEN:
                if not video_pool:
                    raise ValueError("More <video> placeholders than provided videos")
                content.append(video_pool.pop(0))
            elif segment:
                content.append({"type": "text", "text": segment})

        if image_pool:
            raise ValueError(
                f"{len(image_pool)} image(s) remain unused (not consumed by placeholders)"
            )
        if video_pool:
            raise ValueError(
                f"{len(video_pool)} video(s) remain unused (not consumed by placeholders)"
            )

        return content

    def __len__(self):
        return len(self.list_data_dict)
    
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        conversations = sources["conversations"]

        user_input = conversations[0]
        gpt_response = conversations[1]

        base_path = Path(sources.get("data_path", ""))

        image_files = sources.get("image") or []
        if isinstance(image_files, str):
            image_files = [image_files]
        image_files = [self._make_abs_path(base_path, p) for p in image_files]

        video_files = sources.get("video") or []
        if isinstance(video_files, str):
            video_files = [video_files]
        video_files = [self._make_abs_path(base_path, p) for p in video_files]

        system_prompt = sources.get("system_prompt") or sources.get("sys_prompt") or None

        prompt_messages = []
        if system_prompt:
            prompt_messages.append(
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_prompt}],
                }
            )

        user_content = self._build_user_content(user_input["value"], image_files, video_files)
        prompt_messages.append({"role": "user", "content": user_content})

        user_prompt = self.processor.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        assistant_prompt = gpt_response["value"]

        images, videos, video_kwargs = process_vision_info(
            prompt_messages,
            image_patch_size=self.image_patch_size,
            return_video_kwargs=True,
            return_video_metadata=self.return_video_metadata,
        )

        data_dict = dict(
            prompt=user_prompt,
            assistant=assistant_prompt,
            images=images,
            videos=videos,
            video_kwargs=video_kwargs,
        )

        return data_dict
    
def make_grpo_data_module(model_id, processor, data_args):
    """Make dataset and collator for supervised fine-tuning."""
    grpo_dataset = GRPODataset(
        data_path=data_args.dataset_use, processor=processor, data_args=data_args, model_id=model_id
    )

    return dict(train_dataset=grpo_dataset,
                eval_dataset=None)