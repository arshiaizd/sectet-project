#!/usr/bin/env python3
"""Official TAM InternVL3.5 implementation adapted to audited COCO-250 targets."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

from PIL import Image
import numpy as np
import torch
from tqdm import tqdm
from transformers import LogitsProcessor, LogitsProcessorList

from baselines.tam_for_internvl import TAM
from shared.internvl35_utils import (
    CAPTION_PROMPT,
    DEFAULT_MODEL_ID,
    atomic_save_npy,
    load_internvl,
    npy_is_complete,
    prepare_inputs,
)


class ForceCaptionTokens(LogitsProcessor):
    """Force the audited caption while retaining generate() hidden states for TAM."""

    def __init__(self, prompt_length, token_ids):
        self.prompt_length = int(prompt_length)
        self.token_ids = [int(token_id) for token_id in token_ids]

    def __call__(self, input_ids, scores):
        step = input_ids.shape[1] - self.prompt_length
        if 0 <= step < len(self.token_ids):
            forced = torch.full_like(scores, -float("inf"))
            forced[:, self.token_ids[step]] = 0
            return forced
        return scores


def parse_args():
    parser = argparse.ArgumentParser(description="TAM attribution for InternVL3.5-HF")
    parser.add_argument("--Datasets", required=True)
    parser.add_argument("--eval-list", required=True)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--save-dir", required=True)
    return parser.parse_args()


def tam_demo(
    model,
    processor,
    dtype,
    image_path,
    target_index,
    caption_ids,
    save_path,
):
    inputs = prepare_inputs(processor, str(image_path), model.device, dtype)
    if not caption_ids:
        raise ValueError("Audited caption tokenized to an empty sequence")
    outputs = model.generate(
        **inputs,
        do_sample=False,
        num_beams=1,
        min_new_tokens=len(caption_ids),
        max_new_tokens=len(caption_ids),
        output_hidden_states=True,
        return_dict_in_generate=True,
        logits_processor=LogitsProcessorList(
            [ForceCaptionTokens(inputs["input_ids"].shape[1], caption_ids)]
        ),
    )
    generated_ids = outputs.sequences
    logits = [model.lm_head(features[-1]) for features in outputs.hidden_states]
    answer_ids = generated_ids[:, inputs["input_ids"].shape[1] :]
    print(
        "TAM diagnostic: target_index={}, generated_steps={}, answer={!r}".format(
            target_index,
            len(logits),
            processor.batch_decode(answer_ids, skip_special_tokens=True)[0],
        ),
        flush=True,
    )
    if len(logits) != len(caption_ids):
        raise RuntimeError(
            f"TAM generated {len(logits)} steps for {len(caption_ids)} forced tokens"
        )

    image_token_id = int(getattr(model.config, "image_token_id", None) or 151671)
    special_ids = {
        "img_id": [image_token_id],
        "prompt_id": [151653, [151645, 198, 151644, 77091]],
        "answer_id": [[198, 151644, 77091, 198], -1],
    }
    return TAM(
        generated_ids[0].cpu().tolist(),
        (16, 16),
        logits,
        special_ids,
        Image.open(image_path).convert("RGB"),
        processor,
        str(save_path),
        int(target_index),
        [],
        True,
    )


def main(args):
    model, processor, _, dtype = load_internvl(args.model_id)
    contents = json.loads(Path(args.eval_list).read_text(encoding="utf-8"))
    output_dir = Path(args.save_dir)
    npy_dir = output_dir / "npy"
    vis_dir = output_dir / "vis"
    npy_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    for content in tqdm(contents, desc="InternVL TAM"):
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
            heatmap = tam_demo(
                model,
                processor,
                dtype,
                image_path,
                target_index,
                caption_ids,
                vis_dir / Path(content["image_path"]).name,
            )
            atomic_save_npy(output_path, np.asarray(heatmap))
        except Exception:
            print(f"Error in TAM for {image_path}", flush=True)
            traceback.print_exc()


if __name__ == "__main__":
    main(parse_args())
