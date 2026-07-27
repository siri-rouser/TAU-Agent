import transformers
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List
from trl import GRPOConfig

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen2.5-VL-3B-Instruct")
    tune_mm_llm: bool = field(default=False)
    tune_mm_mlp: bool = field(default=False)
    tune_mm_vision: bool = field(default=False)

@dataclass
class DataArguments:
    dataset_use: str = field(default="")
    video_path: str = field(default="")
    eval_dataset_use: str = field(default=None)
    data_flatten: bool = field(default=False)
    data_packing: bool = field(default=False)
    base_interval: int = field(default=2)
    max_pixels: int = field(default=28 * 28 * 256)
    min_pixels: int = field(default=28 * 28 * 32)
    video_max_frames: Optional[int] = field(default=256)
    video_min_frames: Optional[int] = field(default=4)
    video_max_pixels: int = field(default=256 * 28 * 28) 
    video_min_pixels: int = field(default=16 * 28 * 28)
    video_fps: float = 2.0
    nframes: Optional[int] = None
    fps: Optional[int] = field(default=2, metadata={"help": "Frames per second for video data."})

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    adaptor_dir: Optional[str] = field(default=None)
    lora_bias: str = field(default="none")
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=8192,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    mm_projector_lr: Optional[float] = None
    vision_tower_lr: Optional[float] = None

    ## Lora config
    lora_enable: bool = field(default=False)
    lora_r: int = field(default=64)
    lora_alpha: int = field(default=128)
    lora_dropout: float = field(default=0.0)

@dataclass
class GRPOTrainingArguments(GRPOConfig):
    cache_dir: Optional[str] = field(default=None)
    lora_bias: str = field(default="none")
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=8192,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    mm_projector_lr: Optional[float] = None
    vision_tower_lr: Optional[float] = None
    vision_lr: Optional[float] = None
    merger_lr: Optional[float] = None
    adaptor_dir: Optional[str] = field(default=None)

    ## Lora config
    lora_enable: bool = field(default=False)
    lora_r: int = field(default=128)
    lora_alpha: int = field(default=256)
    lora_dropout: float = field(default=0.0)

    ## GRPO config
    max_prompt_length: int = field(default=16384)
    max_completion_length: int = field(default=1024)
    num_generations: int = field(default=4)
    generation_batch_size: int = field(default=4)
    temperature: float = field(default=0.7)
    top_p: float = field(default=0.95)
    do_smape: bool=True
