#!/usr/bin/env python3
"""Dense insertion/deletion evaluation for InternVL3.5 attribution maps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
from torch import nn
import torchvision.transforms.functional as TF
from tqdm import tqdm

from shared.internvl35_utils import (
    CAPTION_PROMPT,
    DEFAULT_MODEL_ID,
    atomic_write_json,
    curve_json_is_complete,
    load_internvl,
    prepare_inputs,
)


def parse_args():
    parser = argparse.ArgumentParser(description="InternVL dense faithfulness")
    parser.add_argument("--Datasets", required=True)
    parser.add_argument("--eval-list", required=True)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--division-number", type=int, default=64)
    parser.add_argument("--eval-dir", required=True)
    parser.add_argument("--invert-map", action="store_true")
    return parser.parse_args()


class InternVLAdaptor(torch.nn.Module):
    def __init__(self, model, processor, dtype):
        super().__init__()
        self.model, self.processor, self.dtype = model, processor, dtype
        self.softmax = nn.Softmax(dim=-1)
        self.generated_ids = None
        self.target_token_position = None
        self.target_id = None

    def forward(self, image):
        if isinstance(image, torch.Tensor) and image.shape[-1] == 3:
            image = TF.to_pil_image(image[..., [2, 1, 0]].permute(2, 0, 1).clamp(0, 255).byte())
        inputs = prepare_inputs(self.processor, image, self.model.device, self.dtype)
        teacher_ids = self.generated_ids[:, : self.target_token_position]
        inputs["input_ids"] = teacher_ids
        inputs["attention_mask"] = torch.ones_like(teacher_ids)
        for stale_key in ("position_ids", "cache_position", "rope_deltas"):
            inputs.pop(stale_key, None)
        with torch.inference_mode():
            logits = self.model(**inputs, return_dict=True, use_cache=False).logits
            probabilities = self.softmax(logits[:, self.target_token_position - 1])
        return probabilities[:, self.target_id]


def perturbed(image, mask, rate, mode):
    flattened = mask.flatten()
    count = int(len(flattened) * rate)
    order = np.argsort(-flattened)
    changed = np.zeros_like(flattened) if mode == "insertion" else np.ones_like(flattened)
    changed[order[:count]] = 1 if mode == "insertion" else 0
    return (image * changed.reshape(mask.shape[0], mask.shape[1], 1)).astype(np.uint8)


def main(args):
    model, processor, _, dtype = load_internvl(args.model_id)
    adaptor = InternVLAdaptor(model, processor, dtype)
    records = json.loads(Path(args.eval_list).read_text(encoding="utf-8"))
    eval_dir = Path(args.eval_dir)
    output_dir = eval_dir / "json"
    output_dir.mkdir(parents=True, exist_ok=True)

    for content in tqdm(records, desc="InternVL faithfulness"):
        stem = Path(content["image_path"]).stem
        output_path = output_dir / f"{stem}.json"
        if curve_json_is_complete(output_path, args.division_number):
            continue
        image_path = Path(args.Datasets) / Path(content["image_path"]).name
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(image_path)
        saliency = np.asarray(np.load(eval_dir / "npy" / f"{stem}.npy"), dtype=np.float32)
        if args.invert_map:
            saliency = 1.0 - saliency
        if saliency.shape != image.shape[:2]:
            saliency = cv2.resize(saliency, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)

        caption_ids = [int(value) for value in content["output_word_id"]]
        target_index = int(content["target_generated_index"])
        target_id = int(content["target_generated_id"])
        if not 0 <= target_index < len(caption_ids) or caption_ids[target_index] != target_id:
            raise ValueError(f"{content['image_path']}: invalid audited target")
        prompt = prepare_inputs(processor, str(image_path), model.device, dtype)
        adaptor.generated_ids = torch.tensor([content["generated_ids"]], dtype=torch.long, device=model.device)
        adaptor.target_token_position = prompt["input_ids"].shape[1] + target_index
        adaptor.target_id = target_id

        result = {
            "insertion_score": [], "deletion_score": [],
            "insertion_word_score": [], "deletion_word_score": [], "region_area": [],
            "select_category": content["select_category"], "location": content["location"],
            "segmentation": content["segmentation"], "selected_coco_caption": content["selected_coco_caption"],
            "target_caption_phrase": content["target_caption_phrase"], "target_generated_index": target_index,
            "target_generated_id": target_id, "internvl_model_id": args.model_id,
            "prompt": CAPTION_PROMPT,
        }
        for step in range(1, args.division_number + 1):
            rate = step / args.division_number
            insertion = Image.fromarray(cv2.cvtColor(perturbed(image, saliency, rate, "insertion"), cv2.COLOR_BGR2RGB))
            deletion = Image.fromarray(cv2.cvtColor(perturbed(image, saliency, rate, "deletion"), cv2.COLOR_BGR2RGB))
            insertion_score = adaptor(insertion).item()
            deletion_score = adaptor(deletion).item()
            result["region_area"].append(rate)
            result["insertion_score"].append(insertion_score)
            result["deletion_score"].append(deletion_score)
            result["insertion_word_score"].append([insertion_score])
            result["deletion_word_score"].append([deletion_score])
        atomic_write_json(output_path, result)


if __name__ == "__main__":
    main(parse_args())
