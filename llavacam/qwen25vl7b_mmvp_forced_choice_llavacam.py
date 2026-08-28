#!/usr/bin/env python3
"""LLaVA-CAM on MMVP-150, targeting all predicted option tokens."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from baselines.llavacam import LLaVACAM
from shared.mmvp_target_utils import prompt_and_targets


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--Datasets", required=True)
    parser.add_argument("--eval-list", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--save-dir", required=True)
    return parser.parse_args()


def complete(path: Path) -> bool:
    try:
        value = np.load(path, mmap_mode="r"); valid = value.ndim == 2 and value.size > 0
        del value; return valid
    except (OSError, ValueError, EOFError):
        return False


def main(args):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id, torch_dtype="auto", device_map="auto"
    ).eval()
    processor = AutoProcessor.from_pretrained(args.model_id, use_fast=False)
    explainer = LLaVACAM(model, processor, model.model.layers[26].post_attention_layernorm)
    records = json.loads(Path(args.eval_list).read_text(encoding="utf-8"))
    output = Path(args.save_dir); npy_dir = output / "npy"; vis_dir = output / "visualization"
    npy_dir.mkdir(parents=True, exist_ok=True); vis_dir.mkdir(parents=True, exist_ok=True)
    for record in tqdm(records, desc="MMVP LLaVA-CAM"):
        stem = Path(record["image_path"]).stem; npy_path = npy_dir / f"{stem}.npy"
        if complete(npy_path): continue
        image_path = Path(args.Datasets) / record["image_path"]
        prompt, positions, target_ids = prompt_and_targets(record)
        messages = [{"role": "user", "content": [
            {"type": "image", "image": str(image_path)}, {"type": "text", "text": prompt}
        ]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, padding=True, return_tensors="pt")
        prompt_ids = inputs["input_ids"][0].tolist()
        explainer.generated_ids = torch.tensor([prompt_ids + target_ids], dtype=torch.long, device=model.device)
        explainer.target_token_position = np.asarray(positions) + len(prompt_ids)
        explainer.selected_interpretation_token_word_id = target_ids
        heatmap = np.nan_to_num(np.asarray(explainer.generate_smooth_cam(str(image_path), prompt), dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        temporary = npy_path.with_suffix(".npy.tmp")
        with temporary.open("wb") as stream: np.save(stream, heatmap)
        temporary.replace(npy_path)
        original = cv2.imread(str(image_path))
        colored = cv2.applyColorMap(np.uint8(255 * cv2.resize(heatmap, (original.shape[1], original.shape[0]))), cv2.COLORMAP_JET)
        cv2.imwrite(str(vis_dir / record["image_path"]), np.clip(0.4 * colored + original, 0, 255).astype(np.uint8))


if __name__ == "__main__":
    main(parse_args())
