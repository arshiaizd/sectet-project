#!/usr/bin/env python3
"""Our vision-encoder activation-patching method on official MMVP-300.

This is a task adapter, not a modification of the COCO implementation.  The
patch site, neighborhood, baseline, injection rule, and ranking procedure are
reused from ``our_method_insertion_online_encoder.py``.  The only task-specific
change is the score: MMVP has arbitrary VQA answers, so we use the mean
teacher-forced probability of the exact generated answer tokens stored in
EAGLE's official Qwen2.5-VL-7B MMVP manifest.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


OURS_DIR = Path(__file__).resolve().parents[1]
if str(OURS_DIR) not in sys.path:
    sys.path.insert(0, str(OURS_DIR))

from our_method_insertion_online_encoder import (  # noqa: E402
    MAX_PIXELS,
    MIN_PIXELS,
    append_dict_rows_to_csv,
    build_teacher_forced_prefix_batch,
    collect_source_vision_block_inputs,
    compute_reverse_window_index,
    ensure_parent_dir,
    get_vision_module,
    infer_merged_vision_grid,
    make_baseline_canvas,
    merged_patches_to_raw_positions,
    model_input_device,
    neighbor_patch_indices,
    online_vision_patch_window,
    prepare_resume_csv,
    spatial_merge_unit_of,
    validate_sequence_aligned_tensors,
)


DEFAULT_MODEL = "/mnt/vilab/scratch/arshia/models/Qwen2.5-VL-7B-Instruct"
DEFAULT_IMAGES = "/mnt/vilab/scratch/arshia/datasets/MMVP/MMVP Images"
DEFAULT_MANIFEST = str(
    Path(__file__).resolve().parent
    / "official_eagle/Qwen2.5-VL-7B-MMVP-VQA.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Activation-patching attribution on MMVP-300")
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


def prepare_mmvp_prompt_inputs(
    processor,
    image: Image.Image,
    question: str,
    device: torch.device,
) -> Dict[str, Any]:
    """Reproduce EAGLE's MMVP prompt: image plus the unmodified question."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    batch = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt",
    )
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def target_ids_from_manifest(content: Dict[str, Any]) -> List[int]:
    positions = [int(value) for value in content["selected_interpretation_token_id"]]
    target_ids = [int(value) for value in content["selected_interpretation_token_word_id"]]
    if not target_ids:
        raise ValueError("empty MMVP target token sequence")
    if positions != list(range(len(target_ids))):
        raise ValueError(
            "This adapter expects EAGLE's sentence-level protocol (all generated "
            f"tokens in order); got positions={positions[:10]}..."
        )
    return target_ids


def build_mmvp_teacher_batch(
    prompt_batch: Dict[str, Any],
    target_token_ids: Sequence[int],
    device: torch.device,
) -> Dict[str, Any]:
    return build_teacher_forced_prefix_batch(prompt_batch, target_token_ids[:-1], device)


@torch.inference_mode()
def target_token_probabilities(
    model,
    teacher_batch: Dict[str, Any],
    prompt_length: int,
    target_token_ids: Sequence[int],
) -> List[float]:
    """Teacher-forced p(y_t | image, question, y_<t) for every answer token."""
    validate_sequence_aligned_tensors(teacher_batch, "MMVP teacher-forced forward")
    outputs = model(**teacher_batch, use_cache=False, return_dict=True)
    logits = outputs.logits[0].float()
    count = len(target_token_ids)
    first_logit = int(prompt_length) - 1
    last_logit = first_logit + count
    if first_logit < 0 or last_logit > logits.shape[0]:
        raise RuntimeError(
            f"target logit range [{first_logit}, {last_logit}) exceeds "
            f"sequence length {logits.shape[0]}"
        )
    selected_logits = logits[first_logit:last_logit]
    probabilities = torch.softmax(selected_logits, dim=-1)
    ids = torch.tensor(target_token_ids, dtype=torch.long, device=probabilities.device)
    gathered = probabilities.gather(1, ids[:, None]).squeeze(1)
    return [float(value) for value in gathered.tolist()]


def mean_probability(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty probability sequence")
    return float(sum(values) / len(values))


@torch.inference_mode()
def score_sample(
    processor,
    model,
    content: Dict[str, Any],
    image_path: Path,
    start_layer: int,
    end_layer: int,
    neighbor_mode: str,
    include_center: bool,
    baseline: str,
    inject_mode: str,
    self_check: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    device = model_input_device(model)
    target_ids = target_ids_from_manifest(content)
    question = str(content["question"])
    image = Image.open(image_path).convert("RGB")

    source_prompt = prepare_mmvp_prompt_inputs(processor, image, question, device)
    grid_h, grid_w, resized_h, resized_w = infer_merged_vision_grid(source_prompt)
    baseline_image = make_baseline_canvas(resized_w, resized_h, baseline)
    baseline_prompt = prepare_mmvp_prompt_inputs(processor, baseline_image, question, device)
    source_prompt_length = int(source_prompt["input_ids"].shape[1])
    baseline_prompt_length = int(baseline_prompt["input_ids"].shape[1])
    if source_prompt_length != baseline_prompt_length:
        raise RuntimeError("source and baseline prompt token lengths differ")

    source_teacher = build_mmvp_teacher_batch(source_prompt, target_ids, device)
    baseline_teacher = build_mmvp_teacher_batch(baseline_prompt, target_ids, device)

    source_probs = target_token_probabilities(
        model, source_teacher, source_prompt_length, target_ids
    )
    baseline_probs = target_token_probabilities(
        model, baseline_teacher, baseline_prompt_length, target_ids
    )
    original_score = mean_probability(source_probs)
    baseline_score = mean_probability(baseline_probs)

    patch_count = grid_h * grid_w
    vision_module = get_vision_module(model)
    merge_unit = spatial_merge_unit_of(vision_module)
    raw_length = patch_count * merge_unit
    reverse_window_index = compute_reverse_window_index(
        vision_module, source_prompt["image_grid_thw"], patch_count
    )
    source_block_inputs = collect_source_vision_block_inputs(
        model, source_teacher, start_layer, end_layer
    )

    if self_check:
        positions = merged_patches_to_raw_positions(
            list(range(patch_count)), reverse_window_index, merge_unit, raw_length
        )
        with online_vision_patch_window(
            model,
            source_block_inputs,
            [(position, position) for position in positions],
            start_layer,
            end_layer,
            inject_mode=inject_mode,
        ):
            patched_all = mean_probability(
                target_token_probabilities(
                    model, baseline_teacher, baseline_prompt_length, target_ids
                )
            )
        print(
            f"[self-check] patch-ALL={patched_all:.6f} "
            f"original={original_score:.6f} baseline={baseline_score:.6f}"
        )

    target_text = processor.tokenizer.decode(
        target_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    rows: List[Dict[str, Any]] = []
    for patch_index in tqdm(range(patch_count), desc="MMVP visual patches", leave=False):
        neighborhood = neighbor_patch_indices(
            patch_index,
            grid_h,
            grid_w,
            mode=neighbor_mode,
            include_center=include_center,
        )
        raw_positions = merged_patches_to_raw_positions(
            neighborhood, reverse_window_index, merge_unit, raw_length
        )
        with online_vision_patch_window(
            model,
            source_block_inputs,
            [(position, position) for position in raw_positions],
            start_layer,
            end_layer,
            inject_mode=inject_mode,
        ):
            patched_probs = target_token_probabilities(
                model, baseline_teacher, baseline_prompt_length, target_ids
            )
        patched_score = mean_probability(patched_probs)
        gain = patched_score - baseline_score
        denominator = original_score - baseline_score
        recovery = gain / denominator if abs(denominator) > 1e-12 else float("nan")
        row, col = divmod(patch_index, grid_w)
        rows.append(
            {
                "image_path": content["image_filename"],
                "question_id": content["question_id"],
                "question": question,
                "options": json.dumps(content["options"], ensure_ascii=False),
                "ground_truth_answer": content["answer"],
                "generated_answer": content.get("generate_sentence", target_text),
                "target_token_ids": json.dumps(target_ids),
                "num_target_tokens": len(target_ids),
                "metric_target": "mean_teacher_forced_generated_answer_token_probability",
                "original_target_probability": original_score,
                "baseline_target_probability": baseline_score,
                "patched_target_probability": patched_score,
                "patched_yes_probability": patched_score,
                "insertion_score": patched_score,
                "attribution_score": patched_score,
                "probability_gain_vs_baseline": gain,
                "recovery_fraction": recovery,
                "patched_target_log_probability": math.log(max(patched_score, 1e-45)),
                "start_layer": start_layer,
                "end_layer": end_layer,
                "patch_site": "vision_encoder_blocks",
                "inject_mode": inject_mode,
                "canvas_mode": "full",
                "baseline": baseline,
                "neighbor_mode": neighbor_mode,
                "include_center": include_center,
                "source_grid_h": grid_h,
                "source_grid_w": grid_w,
                "resized_height": resized_h,
                "resized_width": resized_w,
                "center_patch_index": patch_index,
                "center_patch_row": row,
                "center_patch_col": col,
                "patched_patch_indices": json.dumps(neighborhood),
            }
        )

    sorted_rows = sorted(rows, key=lambda item: item["attribution_score"], reverse=True)
    ranks = {
        int(item["center_patch_index"]): rank
        for rank, item in enumerate(sorted_rows, start=1)
    }
    for item in rows:
        rank = ranks[int(item["center_patch_index"])]
        item["rank_by_target_probability"] = rank
        item["rank_by_yes_probability"] = rank
        item["rank_by_attribution_score"] = rank

    return rows, {
        "image_path": content["image_filename"],
        "question_id": content["question_id"],
        "original": original_score,
        "baseline": baseline_score,
        "num_target_tokens": len(target_ids),
    }


def main() -> None:
    args = parse_args()
    if args.overwrite and args.resume:
        raise ValueError("--overwrite and --resume cannot be combined")
    ensure_parent_dir(args.output_csv)
    if args.overwrite and os.path.exists(args.output_csv):
        os.remove(args.output_csv)
    complete = prepare_resume_csv(args.output_csv) if args.resume else set()

    with open(args.eval_list, encoding="utf-8") as stream:
        contents = json.load(stream)
    if len(contents) != 300:
        raise ValueError(f"expected official MMVP-300 manifest, found {len(contents)}")
    images_dir = Path(args.images_dir)
    missing = [item["image_filename"] for item in contents if not (images_dir / item["image_filename"]).is_file()]
    if missing:
        raise FileNotFoundError(f"missing {len(missing)} MMVP images, first: {missing[:5]}")

    end = len(contents) if args.end < 0 else min(args.end, len(contents))
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
    include_center = not args.exclude_center

    for index in tqdm(range(args.begin, end), desc="MMVP samples"):
        content = contents[index]
        if index in complete:
            print(f"[resume] sample={index} image={content['image_filename']}")
            continue
        try:
            rows, summary = score_sample(
                processor,
                model,
                content,
                images_dir / content["image_filename"],
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
        for row in rows:
            row["dataset_sample_index"] = index
        append_dict_rows_to_csv(args.output_csv, rows)
        print(
            f"[done] sample={index} image={summary['image_path']} "
            f"tokens={summary['num_target_tokens']} original={summary['original']:.6f} "
            f"baseline={summary['baseline']:.6f}",
            flush=True,
        )
        torch.cuda.empty_cache()

    print(f"Saved MMVP attribution CSV: {args.output_csv}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
