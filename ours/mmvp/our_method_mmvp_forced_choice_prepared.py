#!/usr/bin/env python3
"""Our MMVP method using the shared frozen model-answer manifest."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from our_method_mmvp import (
    MAX_PIXELS, MIN_PIXELS, append_dict_rows_to_csv, ensure_parent_dir,
    prepare_resume_csv, score_sample,
)
from shared.mmvp_target_utils import prompt_and_targets


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-id", required=True); p.add_argument("--images-dir", required=True)
    p.add_argument("--eval-list", required=True); p.add_argument("--output-csv", required=True)
    p.add_argument("--begin", type=int, default=0); p.add_argument("--end", type=int, default=-1)
    p.add_argument("--resume", action="store_true"); p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main():
    args = parse_args(); ensure_parent_dir(args.output_csv)
    if args.overwrite and os.path.exists(args.output_csv): os.remove(args.output_csv)
    complete = prepare_resume_csv(args.output_csv) if args.resume else set()
    records = json.loads(Path(args.eval_list).read_text(encoding="utf-8"))
    processor = AutoProcessor.from_pretrained(
        args.model_id, use_fast=False, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True
    ).eval()
    end = len(records) if args.end < 0 else min(args.end, len(records))
    for index in tqdm(range(args.begin, end), desc="MMVP ours"):
        if index in complete: continue
        record = dict(records[index]); _, positions, answer_ids = prompt_and_targets(record)
        record.update({
            "question_id": record["source_question_id"], "question": record["prompt"],
            "answer": record["ground_truth_option_text"],
            "generate_sentence": record["predicted_option_text"],
            "selected_interpretation_token_id": positions,
            "selected_interpretation_token_word_id": answer_ids,
        })
        rows, _ = score_sample(
            processor, model, record, Path(args.images_dir) / record["image_path"],
            0, -1, "square", True, "white", "norm_preserve", False,
        )
        for row in rows:
            row.update({
                "dataset_sample_index": index,
                "predicted_option_index": record["predicted_option_index"],
                "predicted_option_text": record["predicted_option_text"],
                "ground_truth_option_index": record["ground_truth_option_index"],
                "ground_truth_option_text": record["ground_truth_option_text"],
                "prediction_correct": record["prediction_correct"],
                "target_selection_protocol": record["target_protocol"],
            })
        append_dict_rows_to_csv(args.output_csv, rows); torch.cuda.empty_cache()


if __name__ == "__main__":
    torch.set_grad_enabled(False); main()
