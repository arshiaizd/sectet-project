#!/usr/bin/env python3
"""Official IGOS++ InternVL3.5 attribution on the audited COCO-250 set."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from baselines.IGOS_pp.IGOS_pp import gen_explanations_internvl
from shared.internvl35_utils import CAPTION_PROMPT, DEFAULT_MODEL_ID, atomic_save_npy, load_internvl, npy_is_complete


def parse_args():
    parser = argparse.ArgumentParser(description="IGOS++ for InternVL3.5-HF")
    parser.add_argument("--Datasets", required=True)
    parser.add_argument("--eval-list", required=True)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--save-dir", required=True)
    return parser.parse_args()


def main(args):
    model, processor, tokenizer, _ = load_internvl(args.model_id)
    contents = json.loads(Path(args.eval_list).read_text(encoding="utf-8"))
    output_dir = Path(args.save_dir)
    npy_dir, vis_dir = output_dir / "npy", output_dir / "visualization"
    npy_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    for content in tqdm(contents, desc="InternVL IGOS++"):
        stem = Path(content["image_path"]).stem
        output_path = npy_dir / f"{stem}.npy"
        if npy_is_complete(output_path):
            continue
        image_path = Path(args.Datasets) / Path(content["image_path"]).name
        caption_ids = [int(value) for value in content["output_word_id"]]
        target_index = int(content["target_generated_index"])
        target_id = int(content["target_generated_id"])
        if not 0 <= target_index < len(caption_ids):
            raise IndexError(f"{content['image_path']}: invalid target index {target_index}")
        if caption_ids[target_index] != target_id:
            raise ValueError(f"{content['image_path']}: target token ID mismatch")
        try:
            heatmap, visualization = gen_explanations_internvl(
                model, processor, Image.open(image_path).convert("RGB"), CAPTION_PROMPT,
                tokenizer, positions=[target_index], select_word_id=[target_id],
                generated_token_ids=caption_ids,
            )
            atomic_save_npy(output_path, np.asarray(heatmap, dtype=np.float32))
            cv2.imwrite(str(vis_dir / Path(content["image_path"]).name), visualization)
        except Exception:
            print(f"Error in IGOS++ for {image_path}", flush=True)
            traceback.print_exc()


if __name__ == "__main__":
    main(parse_args())
