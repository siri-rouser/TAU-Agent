

import os
import logging
import pathlib
import torch
import transformers
import sys
import torch.distributed as dist
from train_utils import get_peft_state_maybe_zero_3, get_peft_state_non_lora_maybe_zero_3, safe_save_model_for_hf_trainer
from pathlib import Path
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from trainer import replace_qwen2_vl_attention_class

from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMoeForConditionalGeneration
)
from qwenvl.data.data_processor_rag import make_supervised_data_module_rag
from qwenvl.train.argument import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers import AutoProcessor, Trainer

local_rank = None


def is_global_rank_0():
    # If not a torchrun launch: treat as rank 0
    return int(os.environ.get("RANK", 0)) == 0

def rank0_print(*args, **kwargs):
    if is_global_rank_0():
        print(*args, **kwargs)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def set_model(model_args, model):
    if model_args.tune_mm_vision:
        for n, p in model.visual.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_mlp:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_llm:
        for n, p in model.language_model.named_parameters():
            p.requires_grad = True
        model.lm_head.requires_grad = True
    else:
        for n, p in model.language_model.named_parameters():
            p.requires_grad = False
        model.lm_head.requires_grad = False


def train(attn_implementation="flash_attention_2"):
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank

    os.makedirs(training_args.output_dir, exist_ok=True)
    init_model_path = model_args.model_name_or_path
    resume_lora_path = None
    if training_args.adaptor_dir:
        adaptor_ckpt = get_last_checkpoint(training_args.adaptor_dir)
        adaptor_dir_path = pathlib.Path(training_args.adaptor_dir)

        if training_args.lora_enable:
            if adaptor_ckpt is not None and (pathlib.Path(adaptor_ckpt) / "adapter_config.json").exists():
                resume_lora_path = adaptor_ckpt
                rank0_print(f"Resume LoRA adapters from checkpoint: {resume_lora_path}")
            elif (adaptor_dir_path / "adapter_config.json").exists():
                resume_lora_path = training_args.adaptor_dir
                rank0_print(f"Resume LoRA adapters from dir: {resume_lora_path}")
            elif adaptor_ckpt is not None:
                init_model_path = adaptor_ckpt
                rank0_print(f"LoRA adapter config not found; initialize base model from checkpoint: {init_model_path}")
            elif (adaptor_dir_path / "config.json").exists():
                init_model_path = training_args.adaptor_dir
                rank0_print(f"LoRA adapter config not found; initialize base model from dir: {init_model_path}")
            else:
                rank0_print(
                    f"Warning: adaptor_dir set but no checkpoint/model found at {training_args.adaptor_dir}; "
                    f"fallback to {model_args.model_name_or_path}"
                )
        else:
            if adaptor_ckpt is not None:
                init_model_path = adaptor_ckpt
                rank0_print(f"Initialize model weights from adaptor checkpoint: {init_model_path}")
            elif (adaptor_dir_path / "config.json").exists():
                init_model_path = training_args.adaptor_dir
                rank0_print(f"Initialize model weights from adaptor dir: {init_model_path}")
            else:
                rank0_print(
                    f"Warning: adaptor_dir set but no checkpoint/model found at {training_args.adaptor_dir}; "
                    f"fallback to {model_args.model_name_or_path}"
                )

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        init_model_path,
        cache_dir=training_args.cache_dir,
        attn_implementation=attn_implementation,
        dtype=(torch.bfloat16 if training_args.bf16 else None),
    )
    data_args.model_type = "qwen3vl"

    rank0_print(f'the initlized model is {init_model_path} the class is {model.__class__.__name__}')
    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
    )

    if data_args.data_flatten or data_args.data_packing:
        replace_qwen2_vl_attention_class()
    model.config.use_cache = False

    if training_args.gradient_checkpointing: # technics to decrease the mem usage. 
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    # NOTE: for second time finetuning  change model = get_peft_model(model, lora_config) to model = PeftModel.from_pretrained()
    if training_args.lora_enable:
        rank0_print("LoRA enabled")

        for p in model.parameters():
            p.requires_grad = False

        lora_config = LoraConfig(
            r=training_args.lora_r or 64,
            lora_alpha=training_args.lora_alpha or 128,
            lora_dropout=training_args.lora_dropout or 0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],  # Qwen 的 attention 线性层 + MLP 层
            bias=training_args.lora_bias or "none",
            task_type=TaskType.CAUSAL_LM,
        )
        if resume_lora_path is not None:
            rank0_print(f"Loading existing LoRA adapters from {resume_lora_path}...")
            model = PeftModel.from_pretrained(model, resume_lora_path)
            for name, param in model.named_parameters():
                if "lora" in name.lower():
                    param.requires_grad = True
        else:
            rank0_print("Adding LoRA adapters...")
            model = get_peft_model(model, lora_config)

        model.print_trainable_parameters()
    else:
        set_model(model_args, model)

        if torch.distributed.get_rank() == 0:
            model.visual.print_trainable_parameters()
            model.model.print_trainable_parameters()

    data_module = make_supervised_data_module_rag(processor, data_args=data_args)
    trainer = Trainer(
        model=model, processing_class=tokenizer, args=training_args, **data_module
    )

    output_ckpt = get_last_checkpoint(training_args.output_dir)
    if output_ckpt is not None:
        logging.info(f"checkpoint found in output_dir, resume training from: {output_ckpt}")
        trainer.train(resume_from_checkpoint=output_ckpt)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )

        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters(), require_grad_only=True
        )

        if local_rank == 0 or local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            processor.save_pretrained(training_args.output_dir)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, "non_lora_state_dict.bin"))
        print("LoRA and non-LoRA model saved.")
    else:
        safe_save_model_for_hf_trainer(trainer, output_dir=training_args.output_dir)
        processor.save_pretrained(training_args.output_dir)
    
    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")