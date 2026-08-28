#!/usr/bin/env python3
"""TAM on MMVP-150; average one TAM map per predicted answer token."""

from __future__ import annotations

import argparse
import importlib.util
import json
import traceback
from pathlib import Path

import numpy as np
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from shared.mmvp_target_utils import prompt_and_targets


def upstream_module():
    path = Path(__file__).with_name("qwen25vl7b_coco_object_tam.py")
    spec = importlib.util.spec_from_file_location("qwen_tam_coco", path)
    module = importlib.util.module_from_spec(spec); assert spec.loader is not None
    spec.loader.exec_module(module); return module


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--Datasets", required=True); p.add_argument("--eval-list", required=True)
    p.add_argument("--model-id", required=True); p.add_argument("--save-dir", required=True)
    return p.parse_args()


def main(args):
    upstream = upstream_module()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id, torch_dtype="auto", device_map="auto"
    ).eval()
    processor = AutoProcessor.from_pretrained(args.model_id, use_fast=False)
    records = json.loads(Path(args.eval_list).read_text(encoding="utf-8"))
    output = Path(args.save_dir); npy_dir = output / "npy"; vis_dir = output / "vis"
    npy_dir.mkdir(parents=True, exist_ok=True); vis_dir.mkdir(parents=True, exist_ok=True)
    for record in tqdm(records, desc="MMVP TAM"):
        stem = Path(record["image_path"]).stem; path = npy_dir / f"{stem}.npy"
        if upstream.output_is_complete(str(path)): continue
        image_path = Path(args.Datasets) / record["image_path"]
        prompt, positions, answer_ids = prompt_and_targets(record)
        try:
            maps = [
                np.asarray(upstream.tam_demo_for_qwen25_vl(
                    model, processor, str(image_path), prompt, token_id=position,
                    save_path=str(vis_dir / f"{stem}_token{position}.jpg"),
                    forced_caption_ids=answer_ids,
                ), dtype=np.float32)
                for position in positions
            ]
            upstream.atomic_save_npy(str(path), np.nan_to_num(np.mean(np.stack(maps), axis=0), nan=0.0, posinf=0.0, neginf=0.0))
        except Exception:
            print(f"Error in MMVP TAM for {image_path}", flush=True); traceback.print_exc()


if __name__ == "__main__":
    main(parse_args())
