#!/usr/bin/env python3
"""Official LLaVA-CAM InternVL3.5 attribution on the audited COCO-250 set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from baselines.llavacam import LLaVACAM
from shared.internvl35_utils import (
    CAPTION_PROMPT,
    DEFAULT_MODEL_ID,
    atomic_save_npy,
    load_internvl,
    npy_is_complete,
    prepare_inputs,
)


def parse_args():
    parser = argparse.ArgumentParser(description="LLaVA-CAM for InternVL3.5-HF")
    parser.add_argument("--Datasets", required=True)
    parser.add_argument("--eval-list", required=True)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--save-dir", required=True)
    return parser.parse_args()


def main(args):
    model, processor, _, _ = load_internvl(args.model_id)
    layers = model.model.language_model.layers
    if len(layers) <= 32:
        raise RuntimeError(
            f"Official InternVL3.5-8B LLaVA-CAM expects layer 32; model has {len(layers)} layers"
        )
    explainer = LLaVACAM(
        model,
        processor,
        layers[32].post_attention_layernorm,
        mode="internvl",
    )

    contents = json.loads(Path(args.eval_list).read_text(encoding="utf-8"))
    output_dir = Path(args.save_dir)
    npy_dir = output_dir / "npy"
    vis_dir = output_dir / "visualization"
    npy_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    for content in tqdm(contents, desc="InternVL LLaVA-CAM"):
        stem = Path(content["image_path"]).stem
        output_path = npy_dir / f"{stem}.npy"
        if npy_is_complete(output_path):
            continue

        image_path = Path(args.Datasets) / Path(content["image_path"]).name
        inputs = prepare_inputs(
            processor, str(image_path), model.device, next(model.parameters()).dtype
        )
        caption_ids = [int(value) for value in content["output_word_id"]]
        target_index = int(content["target_generated_index"])
        target_id = int(content["target_generated_id"])
        if not 0 <= target_index < len(caption_ids):
            raise IndexError(f"{content['image_path']}: invalid target index {target_index}")
        if caption_ids[target_index] != target_id:
            raise ValueError(f"{content['image_path']}: target token ID mismatch")

        explainer.generated_ids = torch.tensor(
            [content["generated_ids"]], dtype=torch.long, device=model.device
        )
        explainer.target_token_position = np.asarray([target_index]) + inputs["input_ids"].shape[1]
        explainer.selected_interpretation_token_word_id = [target_id]
        heatmap = np.asarray(
            explainer.generate_smooth_cam(str(image_path), CAPTION_PROMPT),
            dtype=np.float32,
        )
        atomic_save_npy(output_path, heatmap)

        original = cv2.imread(str(image_path))
        if original is not None:
            resized = cv2.resize(heatmap, (original.shape[1], original.shape[0]))
            colored = cv2.applyColorMap(np.uint8(255 * resized), cv2.COLORMAP_JET)
            overlay = np.clip(0.4 * colored + original, 0, 255).astype(np.uint8)
            cv2.imwrite(str(vis_dir / Path(content["image_path"]).name), overlay)


if __name__ == "__main__":
    main(parse_args())
