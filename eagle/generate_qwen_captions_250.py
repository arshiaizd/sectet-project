#!/usr/bin/env python3
"""Generate deterministic Qwen2.5-VL captions for a COCO subset.

Designed for ``torchrun`` data parallelism: every rank owns a strided group of
images and one complete model replica.  Per-image JSON files are atomic resume
markers.  Rank zero merges them in the exact order of the input manifest after
all ranks finish.

This stage only generates captions and token IDs.  It intentionally does not
choose or modify the COCO ground-truth target.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


CAPTION_PROMPT = (
    "Describe the image in one factual English sentence of no more than 20 "
    "words. Do not include information that is not clearly visible."
)

REQUIRED_OUTPUT_KEYS = {
    "image_path",
    "generate_sentence",
    "generated_ids",
    "output_word_id",
    "caption_prompt",
    "source_annotation_id",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--model-id", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--attention-implementation",
        choices=["auto", "eager", "flash_attention_2"],
        default="auto",
    )
    return parser.parse_args()


def atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def output_path(output_dir: Path, record: dict) -> Path:
    return output_dir / "json" / Path(record["image_path"]).with_suffix(".json")


def output_is_complete(path: Path, source: dict) -> bool:
    if not path.is_file():
        return False
    try:
        saved = load_json(path)
        return (
            isinstance(saved, dict)
            and REQUIRED_OUTPUT_KEYS.issubset(saved)
            and saved["image_path"] == source["image_path"]
            and int(saved["source_annotation_id"])
            == int(source["annotation_id"])
            and isinstance(saved["generated_ids"], list)
            and isinstance(saved["output_word_id"], list)
            and len(saved["generated_ids"]) > len(saved["output_word_id"]) > 0
            and isinstance(saved["generate_sentence"], str)
            and bool(saved["generate_sentence"].strip())
        )
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return False


def choose_attention(requested: str, device: torch.device) -> str:
    major, _ = torch.cuda.get_device_capability(device)
    flash_available = importlib.util.find_spec("flash_attn") is not None
    flash_supported = major >= 8 and flash_available
    if requested == "auto":
        return "flash_attention_2" if flash_supported else "eager"
    if requested == "flash_attention_2" and not flash_supported:
        raise RuntimeError(
            "FlashAttention-2 was requested but this GPU/software combination "
            "does not support it."
        )
    return requested


def validate_input(records: object, image_root: Path) -> list[dict]:
    if not isinstance(records, list) or not records:
        raise ValueError("input manifest must be a nonempty JSON list")
    required = {"image_path", "annotation_id", "segmentation", "select_category"}
    seen_images = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict) or not required.issubset(record):
            raise ValueError(f"input record {index} is missing required fields")
        image_path = record["image_path"]
        if image_path in seen_images:
            raise ValueError(f"duplicate image_path: {image_path}")
        seen_images.add(image_path)
        if not (image_root / image_path).is_file():
            raise FileNotFoundError(image_root / image_path)
    return records


def generate_one(
    source: dict,
    image_root: Path,
    model: Qwen2_5_VLForConditionalGeneration,
    processor: AutoProcessor,
    device: torch.device,
    rank: int,
    max_new_tokens: int,
    attention_implementation: str,
) -> dict:
    image_path = image_root / source["image_path"]
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": CAPTION_PROMPT},
            ],
        }
    ]
    rendered_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[rendered_text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)
    input_length = int(inputs["input_ids"].shape[1])

    started = time.perf_counter()
    with torch.inference_mode():
        sequences = model.generate(
            **inputs,
            do_sample=False,
            num_beams=1,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )
    runtime_seconds = time.perf_counter() - started

    full_ids = [int(value) for value in sequences[0].detach().cpu().tolist()]
    output_ids = full_ids[input_length:]
    sentence = processor.decode(
        output_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()
    if not sentence:
        raise RuntimeError(f"empty generated caption for {source['image_path']}")

    result = dict(source)
    result.update(
        {
            "caption_prompt": CAPTION_PROMPT,
            "generate_sentence": sentence,
            "generated_ids": full_ids,
            "output_word_id": output_ids,
            "source_annotation_id": int(source["annotation_id"]),
            "caption_runtime_seconds": runtime_seconds,
            "caption_worker_rank": rank,
            "caption_generation": {
                "model_id": str(model.config._name_or_path),
                "do_sample": False,
                "num_beams": 1,
                "max_new_tokens": max_new_tokens,
                "attention_implementation": attention_implementation,
            },
        }
    )
    return result


def merge_outputs(records: list[dict], output_dir: Path) -> Path:
    merged = []
    for source in records:
        path = output_path(output_dir, source)
        if not output_is_complete(path, source):
            raise RuntimeError(f"missing or invalid caption output: {path}")
        merged.append(load_json(path))
    merged_path = output_dir / "qwen25vl7b_captions_250.json"
    atomic_write_json(merged_path, merged)
    return merged_path


def main() -> None:
    args = parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for caption generation")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    if world_size > 1:
        dist.init_process_group(backend="nccl")

    records = validate_input(load_json(args.input), args.image_root)
    rank_records = records[rank::world_size]
    pending = [
        record
        for record in rank_records
        if not output_is_complete(output_path(args.output_dir, record), record)
    ]
    print(
        f"rank={rank}/{world_size} assigned={len(rank_records)} "
        f"pending={len(pending)} gpu={torch.cuda.get_device_name(device)}",
        flush=True,
    )

    if pending:
        attention = choose_attention(args.attention_implementation, device)
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        print(f"rank={rank} dtype={dtype} attention={attention}", flush=True)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_id,
            torch_dtype=dtype,
            device_map={"": device},
            attn_implementation=attention,
            local_files_only=True,
        )
        model.eval()
        processor = AutoProcessor.from_pretrained(
            args.model_id,
            use_fast=False,
            local_files_only=True,
        )

        for source in tqdm(pending, desc=f"rank {rank} captions"):
            result = generate_one(
                source,
                args.image_root,
                model,
                processor,
                device,
                rank,
                args.max_new_tokens,
                attention,
            )
            atomic_write_json(output_path(args.output_dir, source), result)
            tqdm.write(
                f"rank {rank} captioned {source['image_path']} in "
                f"{result['caption_runtime_seconds']:.2f}s: "
                f"{result['generate_sentence']}"
            )
        del model
        torch.cuda.empty_cache()

    if world_size > 1:
        dist.barrier()
    if rank == 0:
        merged_path = merge_outputs(records, args.output_dir)
        print(f"merged {len(records)} captions into {merged_path}", flush=True)
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
