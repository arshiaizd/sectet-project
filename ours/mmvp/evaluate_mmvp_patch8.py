#!/usr/bin/env python3
"""Evaluate MMVP patch rankings with insertion/deletion, 8 patches per step."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from our_method_mmvp import (
    MAX_PIXELS,
    MIN_PIXELS,
    build_mmvp_teacher_batch,
    mean_probability,
    prepare_mmvp_prompt_inputs,
    target_token_probabilities,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--input-csv", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--begin", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1)
    parser.add_argument("--patches-per-step", type=int, default=8)
    return parser.parse_args()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def load_groups(paths: list[Path]) -> list[list[dict[str, str]]]:
    grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
    for path in paths:
        with path.open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                grouped[int(row["dataset_sample_index"])].append(row)
    return [grouped[index] for index in sorted(grouped)]


def validate_group(rows: list[dict[str, str]]) -> None:
    first = rows[0]
    expected = int(first["source_grid_h"]) * int(first["source_grid_w"])
    indices = {int(row["center_patch_index"]) for row in rows}
    ranks = {int(row["rank_by_target_probability"]) for row in rows}
    if len(rows) != expected or indices != set(range(expected)):
        raise ValueError(f"{first['image_path']}: incomplete patch grid")
    if ranks != set(range(1, expected + 1)):
        raise ValueError(f"{first['image_path']}: incomplete patch ranks")


def output_complete(path: Path, steps: int, target_count: int) -> bool:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return (
            len(value["region_area"]) == steps
            and len(value["insertion_score"]) == steps
            and len(value["deletion_score"]) == steps
            and all(len(row) == target_count for row in value["insertion_word_score"])
            and all(len(row) == target_count for row in value["deletion_word_score"])
            and abs(float(value["region_area"][-1]) - 1.0) < 1e-12
        )
    except (OSError, ValueError, TypeError, KeyError, IndexError, json.JSONDecodeError):
        return False


def alter_patches(
    insertion: np.ndarray,
    deletion: np.ndarray,
    source: np.ndarray,
    baseline: np.ndarray,
    patch_indices: list[int],
    grid_w: int,
    patch_h: int,
    patch_w: int,
) -> None:
    for index in patch_indices:
        row, col = divmod(index, grid_w)
        y0, y1 = row * patch_h, (row + 1) * patch_h
        x0, x1 = col * patch_w, (col + 1) * patch_w
        insertion[y0:y1, x0:x1] = source[y0:y1, x0:x1]
        deletion[y0:y1, x0:x1] = baseline[y0:y1, x0:x1]


@torch.inference_mode()
def image_scores(processor, model, image: np.ndarray, question: str, target_ids: list[int]):
    prompt = prepare_mmvp_prompt_inputs(
        processor, Image.fromarray(image), question, model.device
    )
    prompt_length = int(prompt["input_ids"].shape[1])
    teacher = build_mmvp_teacher_batch(prompt, target_ids, model.device)
    words = target_token_probabilities(model, teacher, prompt_length, target_ids)
    return mean_probability(words), words


@torch.inference_mode()
def evaluate(rows, processor, model, images_dir: Path, patches_per_step: int):
    validate_group(rows)
    first = rows[0]
    grid_h, grid_w = int(first["source_grid_h"]), int(first["source_grid_w"])
    resized_h, resized_w = int(first["resized_height"]), int(first["resized_width"])
    if resized_h % grid_h or resized_w % grid_w:
        raise ValueError(f"{first['image_path']}: invalid patch geometry")
    patch_h, patch_w = resized_h // grid_h, resized_w // grid_w
    source = np.asarray(
        Image.open(images_dir / first["image_path"]).convert("RGB").resize((resized_w, resized_h)),
        dtype=np.uint8,
    ).copy()
    baseline_value = 255 if first["baseline"] == "white" else 0
    baseline = np.full_like(source, baseline_value)
    insertion, deletion = baseline.copy(), source.copy()
    target_ids = [int(value) for value in json.loads(first["target_token_ids"])]
    ranked = sorted(rows, key=lambda row: int(row["rank_by_target_probability"]))
    indices = [int(row["center_patch_index"]) for row in ranked]
    result: dict[str, Any] = {
        "image_path": first["image_path"],
        "question_id": int(first["question_id"]),
        "dataset_sample_index": int(first["dataset_sample_index"]),
        "question": first["question"],
        "generated_answer": first["generated_answer"],
        "target_token_ids": target_ids,
        "metric_target": "mean_teacher_forced_generated_answer_token_probability",
        "ranking_field": "rank_by_target_probability",
        "perturbation_unit": "merged_vision_patch",
        "patches_per_step": patches_per_step,
        "grid_height": grid_h,
        "grid_width": grid_w,
        "baseline": first["baseline"],
        "insertion_score": [],
        "deletion_score": [],
        "insertion_word_score": [],
        "deletion_word_score": [],
        "region_area": [],
    }
    for begin in tqdm(range(0, len(indices), patches_per_step), leave=False):
        changed_indices = indices[begin : begin + patches_per_step]
        alter_patches(
            insertion, deletion, source, baseline, changed_indices,
            grid_w, patch_h, patch_w,
        )
        insertion_score, insertion_words = image_scores(
            processor, model, insertion, first["question"], target_ids
        )
        deletion_score, deletion_words = image_scores(
            processor, model, deletion, first["question"], target_ids
        )
        changed = min(begin + patches_per_step, len(indices))
        result["insertion_score"].append(insertion_score)
        result["deletion_score"].append(deletion_score)
        result["insertion_word_score"].append(insertion_words)
        result["deletion_word_score"].append(deletion_words)
        result["region_area"].append(changed / len(indices))
    return result


def main() -> None:
    args = parse_args()
    if args.patches_per_step <= 0:
        raise ValueError("--patches-per-step must be positive")
    json_dir = args.output_dir / "json"
    json_dir.mkdir(parents=True, exist_ok=True)
    groups = load_groups(args.input_csv)
    end = 300 if args.end < 0 else args.end
    groups = [
        rows for rows in groups
        if args.begin <= int(rows[0]["dataset_sample_index"]) < end
    ]
    for rows in groups:
        validate_group(rows)

    processor = AutoProcessor.from_pretrained(
        args.model_id, trust_remote_code=True, use_fast=False,
        min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16, trust_remote_code=True,
        low_cpu_mem_usage=True, device_map="auto",
    ).eval()

    for rows in tqdm(groups, desc=f"MMVP eval [{args.begin},{end})"):
        first = rows[0]
        steps = math.ceil(len(rows) / args.patches_per_step)
        target_count = len(json.loads(first["target_token_ids"]))
        output = json_dir / f"{int(first['dataset_sample_index']):03d}_{Path(first['image_path']).stem}.json"
        if output_complete(output, steps, target_count):
            print(f"[resume] {first['image_path']}")
            continue
        result = evaluate(rows, processor, model, args.images_dir, args.patches_per_step)
        atomic_json(output, result)
        print(f"[done] {first['image_path']} steps={steps}", flush=True)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
