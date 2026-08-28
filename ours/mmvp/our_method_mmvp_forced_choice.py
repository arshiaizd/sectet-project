#!/usr/bin/env python3
"""Our activation-patching method on balanced forced-choice MMVP-150."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Sequence

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from our_method_mmvp import (
    DEFAULT_IMAGES,
    DEFAULT_MODEL,
    MAX_PIXELS,
    MIN_PIXELS,
    append_dict_rows_to_csv,
    ensure_parent_dir,
    model_input_device,
    prepare_mmvp_prompt_inputs,
    prepare_resume_csv,
    score_sample,
)


DEFAULT_MANIFEST = str(
    Path(__file__).resolve().parents[2]
    / "shared/mmvp_forced_choice_150_seed_20260828.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL)
    parser.add_argument("--images-dir", default=DEFAULT_IMAGES)
    parser.add_argument("--eval-list", default=DEFAULT_MANIFEST)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--begin", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1)
    parser.add_argument("--start-layer", type=int, default=0)
    parser.add_argument("--end-layer", type=int, default=-1)
    parser.add_argument("--neighbor-mode", choices=["center", "cross", "square"], default="square")
    parser.add_argument("--exclude-center", action="store_true")
    parser.add_argument("--baseline", choices=["black", "white"], default="white")
    parser.add_argument("--inject-mode", choices=["norm_preserve", "raw"], default="norm_preserve")
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def eos_token_ids(model, tokenizer) -> list[int]:
    values: list[int] = []
    configured = model.generation_config.eos_token_id
    if configured is not None:
        values.extend(configured if isinstance(configured, list) else [configured])
    if tokenizer.eos_token_id is not None:
        values.append(tokenizer.eos_token_id)
    values.extend(
        token_id
        for token_id in (
            tokenizer.convert_tokens_to_ids("<|im_end|>"),
            tokenizer.convert_tokens_to_ids("<|endoftext|>"),
        )
        if isinstance(token_id, int) and token_id >= 0
    )
    result = list(dict.fromkeys(int(value) for value in values))
    if not result:
        raise RuntimeError("could not determine Qwen generation stop token IDs")
    return result


def option_token_ids(tokenizer, options: Sequence[str]) -> list[list[int]]:
    result = [
        tokenizer.encode(str(option), add_special_tokens=False)
        for option in options
    ]
    if any(not sequence for sequence in result):
        raise ValueError(f"an option tokenized to an empty sequence: {options}")
    if len({tuple(sequence) for sequence in result}) != len(result):
        raise ValueError(f"options tokenize identically: {options}")
    return [[int(value) for value in sequence] for sequence in result]


@torch.inference_mode()
def generate_exact_option(processor, model, image: Image.Image, prompt: str, options: list[str]):
    device = model_input_device(model)
    batch = prepare_mmvp_prompt_inputs(processor, image, prompt, device)
    input_length = int(batch["input_ids"].shape[1])
    candidates = option_token_ids(processor.tokenizer, options)
    stop_ids = eos_token_ids(model, processor.tokenizer)

    def allowed_tokens(_batch_id: int, input_ids: torch.Tensor) -> list[int]:
        continuation = [int(value) for value in input_ids[input_length:].tolist()]
        allowed: set[int] = set()
        for sequence in candidates:
            if continuation == sequence:
                allowed.update(stop_ids)
            elif len(continuation) < len(sequence) and sequence[: len(continuation)] == continuation:
                allowed.add(sequence[len(continuation)])
        if not allowed:
            raise RuntimeError(f"constrained decoding left option trie: {continuation}")
        return sorted(allowed)

    generated = model.generate(
        **batch,
        do_sample=False,
        num_beams=1,
        max_new_tokens=max(len(sequence) for sequence in candidates) + 1,
        prefix_allowed_tokens_fn=allowed_tokens,
        eos_token_id=stop_ids,
        pad_token_id=processor.tokenizer.pad_token_id,
    )
    continuation = [int(value) for value in generated[0, input_length:].tolist()]
    while continuation and continuation[-1] in stop_ids:
        continuation.pop()
    matches = [index for index, sequence in enumerate(candidates) if continuation == sequence]
    if len(matches) != 1:
        decoded = processor.tokenizer.decode(continuation, skip_special_tokens=True)
        raise RuntimeError(
            f"constrained output did not match exactly one option: ids={continuation}, "
            f"decoded={decoded!r}, options={options}"
        )
    selected_index = matches[0]
    return selected_index, options[selected_index], candidates[selected_index]


def main() -> None:
    args = parse_args()
    if args.overwrite and args.resume:
        raise ValueError("--overwrite and --resume cannot be combined")
    ensure_parent_dir(args.output_csv)
    if args.overwrite and os.path.exists(args.output_csv):
        os.remove(args.output_csv)
    complete = prepare_resume_csv(args.output_csv) if args.resume else set()

    contents = json.loads(Path(args.eval_list).read_text(encoding="utf-8"))
    if len(contents) != 150:
        raise ValueError(f"expected forced-choice MMVP-150, found {len(contents)}")
    images_dir = Path(args.images_dir)
    missing = [item["image_filename"] for item in contents if not (images_dir / item["image_filename"]).is_file()]
    if missing:
        raise FileNotFoundError(f"missing {len(missing)} images, first: {missing[:5]}")

    processor = AutoProcessor.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        use_fast=False,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map="auto",
    ).eval()

    end = len(contents) if args.end < 0 else min(args.end, len(contents))
    include_center = not args.exclude_center
    for index in tqdm(range(args.begin, end), desc="forced-choice MMVP samples"):
        content = contents[index]
        if index in complete:
            print(f"[resume] sample={index} image={content['image_filename']}")
            continue
        image_path = images_dir / content["image_filename"]
        try:
            image = Image.open(image_path).convert("RGB")
            predicted_index, predicted_text, target_ids = generate_exact_option(
                processor, model, image, content["prompt"], content["options"]
            )
            adapted: dict[str, Any] = dict(content)
            adapted.update(
                {
                    "question_id": content["source_question_id"],
                    "question": content["prompt"],
                    "answer": content["ground_truth_option_text"],
                    "generate_sentence": predicted_text,
                    "selected_interpretation_token_id": list(range(len(target_ids))),
                    "selected_interpretation_token_word_id": target_ids,
                }
            )
            rows, summary = score_sample(
                processor,
                model,
                adapted,
                image_path,
                args.start_layer,
                args.end_layer,
                args.neighbor_mode,
                include_center,
                args.baseline,
                args.inject_mode,
                args.self_check and index == args.begin,
            )
        except Exception as error:
            print(
                f"[error] sample={index} image={content['image_filename']}: "
                f"{type(error).__name__}: {error}",
                flush=True,
            )
            continue

        correct = predicted_index == int(content["ground_truth_option_index"])
        for row in rows:
            row.update(
                {
                    "dataset_sample_index": index,
                    "mmvp_pair_index": content["mmvp_pair_index"],
                    "prompt_version": content["prompt_version"],
                    "allowed_outputs": json.dumps(content["allowed_outputs"], ensure_ascii=False),
                    "predicted_option_index": predicted_index,
                    "predicted_option_text": predicted_text,
                    "ground_truth_option_index": content["ground_truth_option_index"],
                    "ground_truth_option_text": content["ground_truth_option_text"],
                    "prediction_correct": correct,
                    "target_selection_protocol": "greedy decoding constrained to exact option texts",
                }
            )
        append_dict_rows_to_csv(args.output_csv, rows)
        print(
            f"[done] sample={index} image={summary['image_path']} "
            f"prediction={predicted_text!r} ground_truth={content['ground_truth_option_text']!r} "
            f"correct={correct} tokens={summary['num_target_tokens']}",
            flush=True,
        )
        torch.cuda.empty_cache()

    print(f"Saved forced-choice MMVP attribution CSV: {args.output_csv}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
