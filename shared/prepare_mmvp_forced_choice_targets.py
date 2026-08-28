#!/usr/bin/env python3
"""Freeze exact-option MMVP predictions for one model and share them across methods."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-family", choices=("qwen", "internvl"), required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def stop_ids(model, tokenizer) -> list[int]:
    values: list[int] = []
    configured = model.generation_config.eos_token_id
    if configured is not None:
        values.extend(configured if isinstance(configured, list) else [configured])
    for value in (tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")):
        if isinstance(value, int) and value >= 0:
            values.append(value)
    result = list(dict.fromkeys(map(int, values)))
    if not result:
        raise RuntimeError("Could not determine generation stop token IDs")
    return result


def option_ids(tokenizer, options: list[str]) -> list[list[int]]:
    candidates = [
        [int(value) for value in tokenizer.encode(option, add_special_tokens=False)]
        for option in options
    ]
    if any(not candidate for candidate in candidates):
        raise ValueError(f"An option tokenized to an empty sequence: {options}")
    if len({tuple(candidate) for candidate in candidates}) != len(candidates):
        raise ValueError(f"Options tokenize identically: {options}")
    return candidates


def constrained_answer(model, tokenizer, inputs, options: list[str]):
    prompt_length = int(inputs["input_ids"].shape[1])
    candidates = option_ids(tokenizer, options)
    eos = stop_ids(model, tokenizer)

    def allowed(_batch_id: int, input_ids: torch.Tensor) -> list[int]:
        continuation = [int(value) for value in input_ids[prompt_length:].tolist()]
        result: set[int] = set()
        for candidate in candidates:
            if continuation == candidate:
                result.update(eos)
            elif len(continuation) < len(candidate) and candidate[: len(continuation)] == continuation:
                result.add(candidate[len(continuation)])
        if not result:
            raise RuntimeError(f"Constrained decoding left the option trie: {continuation}")
        return sorted(result)

    generated = model.generate(
        **inputs,
        do_sample=False,
        num_beams=1,
        max_new_tokens=max(map(len, candidates)) + 1,
        prefix_allowed_tokens_fn=allowed,
        eos_token_id=eos,
        pad_token_id=(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos[0]),
    )
    answer = [int(value) for value in generated[0, prompt_length:].tolist()]
    while answer and answer[-1] in eos:
        answer.pop()
    matches = [index for index, candidate in enumerate(candidates) if answer == candidate]
    if len(matches) != 1:
        raise RuntimeError(f"Generated IDs {answer} do not equal exactly one option in {options}")
    return matches[0], options[matches[0]], answer


def load_runtime(family: str, model_id: str):
    if family == "qwen":
        from qwen_vl_utils import process_vision_info

        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=dtype, device_map="auto", low_cpu_mem_usage=True
        ).eval()
        processor = AutoProcessor.from_pretrained(model_id, use_fast=False)

        def prepare(image: str, prompt: str):
            messages = [{"role": "user", "content": [
                {"type": "image", "image": image}, {"type": "text", "text": prompt}
            ]}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            images, videos = process_vision_info(messages)
            return processor(
                text=[text], images=images, videos=videos, padding=True, return_tensors="pt"
            ).to(model.device)

        return model, processor, processor.tokenizer, prepare

    from shared.internvl35_utils import load_internvl, prepare_inputs

    model, processor, tokenizer, dtype = load_internvl(model_id)

    def prepare(image: str, prompt: str):
        return prepare_inputs(processor, image, model.device, dtype, prompt)

    return model, processor, tokenizer, prepare


def is_current(records: Any, family: str, model_id: str, source_sha: str) -> bool:
    return (
        isinstance(records, list)
        and len(records) == 150
        and all(record.get("target_model_family") == family for record in records)
        and all(record.get("target_model_id") == model_id for record in records)
        and all(record.get("target_source_sha256") == source_sha for record in records)
        and all(record.get("predicted_answer_token_ids") for record in records)
    )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Preparing model predictions requires one visible CUDA GPU")
    source_bytes = args.input.read_bytes()
    source_sha = hashlib.sha256(source_bytes).hexdigest()
    source = json.loads(source_bytes)
    if len(source) != 150:
        raise ValueError(f"Expected MMVP-150 source, found {len(source)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    lock_path = args.output.with_suffix(args.output.suffix + ".lock")
    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        existing: list[dict[str, Any]] = []
        if args.output.is_file() and not args.force:
            try:
                candidate = json.loads(args.output.read_text(encoding="utf-8"))
                if is_current(candidate, args.model_family, args.model_id, source_sha):
                    print(f"MMVP target manifest already complete: {args.output}")
                    return
                if isinstance(candidate, list):
                    existing = candidate
            except (OSError, ValueError, json.JSONDecodeError):
                pass
        valid_prefix = 0
        if not args.force:
            for index, record in enumerate(existing):
                if (
                    index < len(source)
                    and record.get("dataset_sample_index") == index
                    and record.get("target_model_family") == args.model_family
                    and record.get("target_model_id") == args.model_id
                    and record.get("target_source_sha256") == source_sha
                    and record.get("predicted_answer_token_ids")
                ):
                    valid_prefix += 1
                else:
                    break
        output = existing[:valid_prefix]
        if valid_prefix == 150:
            print(f"MMVP target manifest already complete: {args.output}")
            return

        model, _, tokenizer, prepare = load_runtime(args.model_family, args.model_id)
        for index in range(valid_prefix, len(source)):
            record = dict(source[index])
            image_path = args.images_dir / record["image_filename"]
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            inputs = prepare(str(image_path), record["prompt"])
            prompt_ids = [int(value) for value in inputs["input_ids"][0].tolist()]
            predicted_index, predicted_text, answer_ids = constrained_answer(
                model, tokenizer, inputs, list(record["options"])
            )
            record.update(
                {
                    "image_path": record["image_filename"],
                    "predicted_option_index": predicted_index,
                    "predicted_option_text": predicted_text,
                    "prediction_correct": predicted_index == int(record["ground_truth_option_index"]),
                    "predicted_answer_token_ids": answer_ids,
                    "target_generated_indices": list(range(len(answer_ids))),
                    "target_generated_ids": answer_ids,
                    "output_word_id": answer_ids,
                    "generated_ids": prompt_ids + answer_ids,
                    "generate_sentence": predicted_text,
                    "target_model_family": args.model_family,
                    "target_model_id": args.model_id,
                    "target_source_sha256": source_sha,
                    "target_protocol": "all teacher-forced tokens of the model-selected exact option text",
                }
            )
            output.append(record)
            atomic_json(args.output, output)
            print(
                f"[{index + 1}/150] {record['image_filename']} prediction={predicted_text!r} "
                f"tokens={len(answer_ids)} correct={record['prediction_correct']}",
                flush=True,
            )
        if not is_current(output, args.model_family, args.model_id, source_sha):
            raise RuntimeError("Prepared target manifest failed final validation")
        print(f"Prepared shared MMVP targets: {args.output}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
