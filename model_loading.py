"""Shared model loading helpers for full-precision and 4-bit QLoRA runs."""

import os
from pathlib import Path

import torch
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, BitsAndBytesConfig


def _select(cfg, key, default=None):
    value = OmegaConf.select(cfg, key, default=default)
    return default if value is None else value


def uses_4bit(cfg):
    return bool(_select(cfg, "quantization.load_in_4bit", False))


def make_quantization_config(cfg):
    if not uses_4bit(cfg):
        return None

    compute_dtype_name = str(
        _select(cfg, "quantization.bnb_4bit_compute_dtype", "bfloat16")
    )
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=str(
            _select(cfg, "quantization.bnb_4bit_quant_type", "nf4")
        ),
        bnb_4bit_use_double_quant=bool(
            _select(cfg, "quantization.bnb_4bit_use_double_quant", True)
        ),
        bnb_4bit_compute_dtype=getattr(torch, compute_dtype_name),
    )


def _is_local_adapter(model_path):
    path = Path(str(model_path))
    return path.is_dir() and (path / "adapter_config.json").exists()


def load_causal_lm(model_path, cfg, model_cfg, config=None):
    """Load a base model or local PEFT adapter with optional 4-bit weights."""
    kwargs = {
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": True,
        "use_flash_attention_2": model_cfg["flash_attention2"] == "true",
    }
    if config is not None:
        kwargs["config"] = config

    quantization_config = make_quantization_config(cfg)
    if quantization_config is not None:
        kwargs["quantization_config"] = quantization_config
        kwargs["device_map"] = {"": int(os.environ.get("LOCAL_RANK", "0"))}

    if _is_local_adapter(model_path):
        from peft import AutoPeftModelForCausalLM

        return AutoPeftModelForCausalLM.from_pretrained(model_path, **kwargs)

    return AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
