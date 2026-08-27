#!/usr/bin/env python3
"""Official EAGLE InternVL3.5 object attribution adapted to the audited COCO-250 run."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
import torchvision.transforms.functional as TF
from tqdm import tqdm

from shared.internvl35_utils import (
    CAPTION_PROMPT,
    DEFAULT_MODEL_ID,
    atomic_save_npy,
    atomic_write_json,
    load_internvl,
    npy_is_complete,
    prepare_inputs,
)
from submodular_vision import MLLMSubModularExplanationVision
from utils import SubRegionDivision


def parse_args():
    parser = argparse.ArgumentParser(description="EAGLE attribution for InternVL3.5-HF")
    parser.add_argument("--Datasets", required=True)
    parser.add_argument("--eval-list", required=True)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--superpixel-algorithm", default="slico", choices=["slico", "seeds"])
    parser.add_argument("--lambda1", type=float, default=1.0)
    parser.add_argument("--lambda2", type=float, default=1.0)
    parser.add_argument("--division-number", type=int, default=64)
    parser.add_argument("--begin", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1)
    parser.add_argument("--save-dir", required=True)
    return parser.parse_args()


def outputs_are_complete(json_path: Path, npy_path: Path) -> bool:
    try:
        saved = json.loads(json_path.read_text(encoding="utf-8"))
        required = {
            "insertion_score",
            "deletion_score",
            "smdl_score",
            "region_area",
            "selected_interpretation_token_id",
            "selected_interpretation_token_word_id",
        }
        return required.issubset(saved) and npy_is_complete(npy_path, dimensions=4)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


class InternVLAdaptor(torch.nn.Module):
    """Model adapter expected by EAGLE's unchanged submodular implementation."""

    def __init__(self, model, processor, dtype):
        super().__init__()
        self.model = model
        self.processor = processor
        self.dtype = dtype
        self.softmax = nn.Softmax(dim=-1)
        self.text_prompt = CAPTION_PROMPT
        self.generated_ids = None
        self.target_token_position = None
        self.selected_interpretation_token_word_id = None

    def forward(self, image):
        if isinstance(image, torch.Tensor) and image.shape[-1] == 3:
            image_tensor = image[..., [2, 1, 0]].permute(2, 0, 1)
            image = TF.to_pil_image(image_tensor.clamp(0, 255).byte())

        inputs = prepare_inputs(
            self.processor, image, self.model.device, self.dtype, self.text_prompt
        )
        teacher_ids = self.generated_ids[:, : max(self.target_token_position)]
        inputs["input_ids"] = teacher_ids
        inputs["attention_mask"] = torch.ones_like(teacher_ids)
        for stale_key in ("position_ids", "cache_position", "rope_deltas"):
            inputs.pop(stale_key, None)

        with torch.inference_mode():
            logits = self.model(
                **inputs, return_dict=True, use_cache=False
            ).logits[:, self.target_token_position - 1]
        probabilities = self.softmax(logits)
        selected_ids = torch.as_tensor(
            self.selected_interpretation_token_word_id,
            dtype=torch.long,
            device=self.model.device,
        )
        indices = selected_ids.unsqueeze(0).unsqueeze(-1)
        return probabilities.gather(dim=2, index=indices).squeeze(-1)[0]


def main(args):
    model, processor, tokenizer, dtype = load_internvl(args.model_id)
    adaptor = InternVLAdaptor(model, processor, dtype)
    explainer = MLLMSubModularExplanationVision(
        adaptor, lambda1=args.lambda1, lambda2=args.lambda2
    )

    contents = json.loads(Path(args.eval_list).read_text(encoding="utf-8"))
    end = None if args.end == -1 else args.end
    contents = contents[args.begin:end]

    save_dir = Path(args.save_dir) / (
        f"{args.superpixel_algorithm}-{args.lambda1}-{args.lambda2}-"
        f"division-number-{args.division_number}"
    )
    json_dir = save_dir / "json"
    npy_dir = save_dir / "npy"
    json_dir.mkdir(parents=True, exist_ok=True)
    npy_dir.mkdir(parents=True, exist_ok=True)

    for content in tqdm(contents, desc="InternVL EAGLE"):
        stem = Path(content["image_path"]).stem
        json_path = json_dir / f"{stem}.json"
        npy_path = npy_dir / f"{stem}.npy"
        if outputs_are_complete(json_path, npy_path):
            tqdm.write(f"Skipping completed image: {content['image_path']}")
            continue

        image_path = Path(args.Datasets) / Path(content["image_path"]).name
        inputs = prepare_inputs(processor, str(image_path), model.device, dtype)
        target_index = int(content["target_generated_index"])
        target_id = int(content["target_generated_id"])
        caption_ids = [int(value) for value in content["output_word_id"]]
        if not 0 <= target_index < len(caption_ids):
            raise IndexError(
                f"{content['image_path']}: target index {target_index} outside "
                f"caption length {len(caption_ids)}"
            )
        if caption_ids[target_index] != target_id:
            raise ValueError(f"{content['image_path']}: target token ID mismatch")

        adaptor.generated_ids = torch.tensor(
            [content["generated_ids"]], dtype=torch.long, device=model.device
        )
        adaptor.target_token_position = np.asarray([target_index]) + inputs["input_ids"].shape[1]
        adaptor.selected_interpretation_token_word_id = [target_id]

        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(image_path)
        region_size = int(
            (image.shape[0] * image.shape[1] / args.division_number) ** 0.5
        )
        regions = SubRegionDivision(
            image, mode=args.superpixel_algorithm, region_size=region_size
        )
        selected_regions, saved = explainer(image, regions)
        saved.update(
            {
                "selected_interpretation_token_id": [target_index],
                "selected_interpretation_token_word_id": [target_id],
                "select_category": content["select_category"],
                "words": content["target_generated_token"],
                "location": content["location"],
                "segmentation": content["segmentation"],
                "output_word_id": caption_ids,
                "output_words": tokenizer.convert_ids_to_tokens(caption_ids),
                "selected_coco_caption": content["selected_coco_caption"],
                "target_caption_phrase": content["target_caption_phrase"],
                "internvl_model_id": args.model_id,
            }
        )
        atomic_save_npy(npy_path, np.asarray(selected_regions))
        atomic_write_json(json_path, saved)


if __name__ == "__main__":
    main(parse_args())
