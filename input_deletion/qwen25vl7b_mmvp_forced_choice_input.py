#!/usr/bin/env python3
"""Input-level insertion+necessity attribution on frozen MMVP answer tokens."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from our_method_insertion_deletion_input_online import (
    append_dict_rows_to_csv, ensure_parent_dir, infer_merged_vision_grid,
    make_baseline_canvas, mask_patches_on_input_image, prepare_resume_csv,
    reveal_patches_on_baseline, square_radius_neighbor_indices,
)
from ours.mmvp.our_method_mmvp import (
    MAX_PIXELS, MIN_PIXELS, build_mmvp_teacher_batch, mean_probability,
    prepare_mmvp_prompt_inputs, target_token_probabilities,
)
from shared.mmvp_target_utils import prompt_and_targets


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-id", required=True); p.add_argument("--images-dir", required=True)
    p.add_argument("--eval-list", required=True); p.add_argument("--output-csv", required=True)
    p.add_argument("--begin", type=int, default=0); p.add_argument("--end", type=int, default=-1)
    p.add_argument("--resume", action="store_true"); return p.parse_args()


@torch.inference_mode()
def probability(processor, model, image, prompt, ids):
    batch = prepare_mmvp_prompt_inputs(processor, image, prompt, model.device)
    teacher = build_mmvp_teacher_batch(batch, ids, model.device)
    return mean_probability(target_token_probabilities(model, teacher, batch["input_ids"].shape[1], ids))


def main():
    args = parse_args(); ensure_parent_dir(args.output_csv)
    complete = prepare_resume_csv(args.output_csv) if args.resume else set()
    records = json.loads(Path(args.eval_list).read_text(encoding="utf-8"))
    processor = AutoProcessor.from_pretrained(
        args.model_id, use_fast=False, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True
    ).eval()
    end = len(records) if args.end < 0 else min(args.end, len(records))
    for index in tqdm(range(args.begin, end), desc="MMVP input-level"):
        if index in complete: continue
        record = records[index]; prompt, _, ids = prompt_and_targets(record)
        source = Image.open(Path(args.images_dir) / record["image_path"]).convert("RGB")
        source_batch = prepare_mmvp_prompt_inputs(processor, source, prompt, model.device)
        gh, gw, rh, rw = infer_merged_vision_grid(source_batch); baseline = make_baseline_canvas(rw, rh, "white")
        original = probability(processor, model, source, prompt, ids)
        base = probability(processor, model, baseline, prompt, ids)
        rows = []
        for patch in tqdm(range(gh * gw), desc="MMVP input patches", leave=False):
            neighbors = square_radius_neighbor_indices(patch, gh, gw, 1, True)
            insertion = probability(processor, model, reveal_patches_on_baseline(source, rw, rh, neighbors, gw, "white"), prompt, ids)
            deletion = probability(processor, model, mask_patches_on_input_image(source, rw, rh, neighbors, gw, "white"), prompt, ids)
            score = insertion + (1.0 - deletion); row, col = divmod(patch, gw)
            rows.append({
                "image_path": record["image_path"], "question_id": record["source_question_id"],
                "question": prompt, "options": json.dumps(record["options"], ensure_ascii=False),
                "ground_truth_answer": record["ground_truth_option_text"],
                "generated_answer": record["predicted_option_text"],
                "target_token_ids": json.dumps(ids), "num_target_tokens": len(ids),
                "metric_target": "mean_teacher_forced_generated_answer_token_probability",
                "original_target_probability": original, "baseline_target_probability": base,
                "patched_target_probability": insertion, "patched_yes_probability": insertion,
                "insertion_score": insertion, "deletion_score": deletion,
                "necessity_score": 1.0 - deletion, "attribution_score": score,
                "baseline": "white", "source_grid_h": gh, "source_grid_w": gw,
                "resized_height": rh, "resized_width": rw, "center_patch_index": patch,
                "center_patch_row": row, "center_patch_col": col,
                "patched_patch_indices": json.dumps(neighbors),
                "probability_gain_vs_baseline": insertion - base,
                "recovery_fraction": (insertion - base) / (original - base) if abs(original-base)>1e-12 else float("nan"),
                "patched_target_log_probability": math.log(max(insertion, 1e-45)),
            })
        ranks = {r["center_patch_index"]: rank for rank, r in enumerate(sorted(rows, key=lambda x:x["attribution_score"], reverse=True), 1)}
        for row in rows:
            rank = ranks[row["center_patch_index"]]
            row.update({"rank_by_target_probability":rank, "rank_by_yes_probability":rank,
                        "rank_by_attribution_score":rank, "dataset_sample_index":index,
                        "predicted_option_index":record["predicted_option_index"],
                        "predicted_option_text":record["predicted_option_text"],
                        "ground_truth_option_index":record["ground_truth_option_index"],
                        "ground_truth_option_text":record["ground_truth_option_text"],
                        "prediction_correct":record["prediction_correct"],
                        "target_selection_protocol":record["target_protocol"]})
        append_dict_rows_to_csv(args.output_csv, rows); torch.cuda.empty_cache()


if __name__ == "__main__":
    torch.set_grad_enabled(False); main()
