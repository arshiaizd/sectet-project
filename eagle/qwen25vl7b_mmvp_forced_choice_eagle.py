#!/usr/bin/env python3
"""EAGLE on MMVP-150, attributing every token of the predicted exact option."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from shared.mmvp_target_utils import prompt_and_targets
from submodular_vision import MLLMSubModularExplanationVision
from utils import SubRegionDivision, mkdir


def upstream_module():
    path = Path(__file__).with_name("Qwen25-VL-7B-coco-object.py")
    spec = importlib.util.spec_from_file_location("qwen_eagle_coco", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--Datasets", required=True)
    parser.add_argument("--eval-list", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--division-number", type=int, default=64)
    parser.add_argument("--lambda1", type=float, default=1.0)
    parser.add_argument("--lambda2", type=float, default=1.0)
    parser.add_argument("--attention-implementation", default="auto")
    return parser.parse_args()


def main(args):
    upstream = upstream_module()
    rank, _, world_size, device, dtype, attention = upstream.configure_worker(
        args.attention_implementation
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id, torch_dtype=dtype, device_map={"": device},
        attn_implementation=attention,
    ).eval()
    processor = AutoProcessor.from_pretrained(args.model_id, use_fast=False)
    adaptor = upstream.QwenVLAdaptor(model=model, processor=processor, device=device)
    explainer = MLLMSubModularExplanationVision(
        adaptor, lambda1=args.lambda1, lambda2=args.lambda2
    )
    records = json.loads(Path(args.eval_list).read_text(encoding="utf-8"))[rank::world_size]
    root = Path(args.save_dir) / f"slico-{args.lambda1}-{args.lambda2}-division-number-{args.division_number}"
    json_dir, npy_dir = root / "json", root / "npy"
    json_dir.mkdir(parents=True, exist_ok=True); npy_dir.mkdir(parents=True, exist_ok=True)

    for record in tqdm(records, desc=f"MMVP EAGLE rank {rank}"):
        stem = Path(record["image_path"]).stem
        json_path, npy_path = json_dir / f"{stem}.json", npy_dir / f"{stem}.npy"
        if upstream.outputs_are_complete(str(json_path), str(npy_path)):
            continue
        started = time.perf_counter()
        image_path = Path(args.Datasets) / record["image_path"]
        prompt, positions, target_ids = prompt_and_targets(record)
        messages = [{"role": "user", "content": [
            {"type": "image", "image": str(image_path)}, {"type": "text", "text": prompt}
        ]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, padding=True, return_tensors="pt").to(device)
        prompt_ids = inputs["input_ids"][0].detach().cpu().tolist()
        adaptor.image_only_text = text
        adaptor.generated_ids = torch.tensor([prompt_ids + target_ids], dtype=torch.long, device=device)
        adaptor.target_token_position = np.asarray(positions) + len(prompt_ids)
        adaptor.selected_interpretation_token_word_id = target_ids
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(image_path)
        region_size = int((image.shape[0] * image.shape[1] / args.division_number) ** 0.5)
        regions = SubRegionDivision(image, mode="slico", region_size=region_size)
        selected, saved = explainer(image, regions)
        saved.update({
            "image_path": record["image_path"], "prompt": prompt,
            "predicted_option_text": record["predicted_option_text"],
            "ground_truth_option_text": record["ground_truth_option_text"],
            "prediction_correct": record["prediction_correct"],
            "selected_interpretation_token_id": positions,
            "selected_interpretation_token_word_id": target_ids,
            "output_word_id": target_ids,
            "output_words": processor.tokenizer.convert_ids_to_tokens(target_ids),
            "target_protocol": record["target_protocol"],
            "runtime_seconds": time.perf_counter() - started,
        })
        upstream.atomic_save_npy(str(npy_path), np.asarray(selected))
        upstream.atomic_save_json(str(json_path), saved)


if __name__ == "__main__":
    main(parse_args())
