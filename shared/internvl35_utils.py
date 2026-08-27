#!/usr/bin/env python3
"""Shared InternVL3.5-HF loading, prompt, target-alignment, and I/O helpers."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import transformers
from packaging.version import Version
from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor, AutoTokenizer


DEFAULT_MODEL_ID = "OpenGVLab/InternVL3_5-8B-HF"
CAPTION_PROMPT = (
    "Describe the image in one factual English sentence of no more than 20 words. "
    "Do not include information that is not clearly visible."
)


def require_internvl_transformers() -> None:
    installed = Version(transformers.__version__.split("+")[0])
    required = Version("4.52.1")
    if installed < required:
        raise RuntimeError(
            f"InternVL3.5 requires transformers>={required}; found {installed}. "
            "Install the pinned requirements-internvl.txt dependencies."
        )


def choose_cuda_dtype() -> torch.dtype:
    if not torch.cuda.is_available():
        raise RuntimeError("InternVL attribution requires a visible CUDA GPU.")
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def validate_internvl8b_config(config) -> None:
    """Reject accidental 4B or other checkpoints in the baked 8B experiment."""
    text_config = config.text_config
    actual = {
        "language_layers": int(text_config.num_hidden_layers),
        "language_hidden_size": int(text_config.hidden_size),
        "image_sequence_length": int(config.image_seq_length),
    }
    expected = {
        "language_layers": 36,
        "language_hidden_size": 4096,
        "image_sequence_length": 256,
    }
    if actual != expected:
        raise RuntimeError(
            "This experiment is baked for OpenGVLab/InternVL3_5-8B-HF; "
            f"expected {expected}, loaded {actual}."
        )


def load_internvl(model_id: str, *, gradients: bool = False):
    require_internvl_transformers()
    dtype = choose_cuda_dtype()
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    validate_internvl8b_config(config)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        config=config,
        torch_dtype=dtype,
        device_map="auto",
        attn_implementation="sdpa",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).eval()
    if not gradients:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    processor = AutoProcessor.from_pretrained(
        model_id, trust_remote_code=True, use_fast=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=True, use_fast=True
    )
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError(
            "The InternVL tokenizer must be fast so audited character spans can be "
            "mapped exactly to model tokens."
        )
    return model, processor, tokenizer, dtype


def image_messages(image: Any, prompt: str = CAPTION_PROMPT) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def prepare_inputs(processor, image: Any, device, dtype, prompt: str = CAPTION_PROMPT):
    inputs = processor.apply_chat_template(
        image_messages(image, prompt),
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    return inputs.to(device, dtype=dtype)


def align_audited_target(tokenizer, record: dict[str, Any]) -> dict[str, Any]:
    """Retokenize the audited caption and select its final target-overlap token."""
    caption = str(record["selected_coco_caption"])
    char_start, char_end = map(int, record["target_caption_char_span"])
    if not 0 <= char_start < char_end <= len(caption):
        raise ValueError(
            f"{record['image_path']}: invalid target character span "
            f"{[char_start, char_end]} for caption length {len(caption)}"
        )
    encoded = tokenizer(
        caption,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    caption_ids = [int(value) for value in encoded["input_ids"]]
    offsets = [tuple(map(int, pair)) for pair in encoded["offset_mapping"]]
    overlaps = [
        index
        for index, (start, end) in enumerate(offsets)
        if end > start and max(start, char_start) < min(end, char_end)
    ]
    if not overlaps:
        raise ValueError(
            f"{record['image_path']}: no InternVL token overlaps audited target "
            f"{caption[char_start:char_end]!r}"
        )
    target_index = overlaps[-1]
    target_id = caption_ids[target_index]
    return {
        "caption_ids": caption_ids,
        "target_index": target_index,
        "target_id": target_id,
        "target_token": tokenizer.convert_ids_to_tokens(target_id),
        "target_token_span": [overlaps[0], overlaps[-1] + 1],
    }


def teacher_forced_ids(inputs, alignment: dict[str, Any], device) -> torch.Tensor:
    prompt_ids = inputs["input_ids"][0].detach().cpu().tolist()
    return torch.tensor(
        [prompt_ids + alignment["caption_ids"]], dtype=torch.long, device=device
    )


def atomic_save_npy(path: str | Path, value: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        np.save(stream, value)
    os.replace(temporary, path)


def atomic_write_json(path: str | Path, value: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)


def npy_is_complete(path: str | Path, *, dimensions: int | None = 2) -> bool:
    try:
        array = np.load(path, mmap_mode="r")
        valid = array.size > 0 and (dimensions is None or array.ndim == dimensions)
        del array
        return bool(valid)
    except (OSError, EOFError, ValueError):
        return False


def curve_json_is_complete(path: str | Path, expected_steps: int = 64) -> bool:
    try:
        saved = json.loads(Path(path).read_text(encoding="utf-8"))
        lengths = [
            len(saved[key])
            for key in (
                "insertion_score",
                "deletion_score",
                "insertion_word_score",
                "deletion_word_score",
                "region_area",
            )
        ]
        return lengths == [expected_steps] * 5
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return False
