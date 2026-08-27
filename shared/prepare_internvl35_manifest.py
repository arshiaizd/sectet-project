#!/usr/bin/env python3
"""Retokenize the audited COCO-250 captions for InternVL3.5-HF."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from transformers import AutoConfig, AutoProcessor, AutoTokenizer

from shared.internvl35_utils import (
    CAPTION_PROMPT,
    DEFAULT_MODEL_ID,
    align_audited_target,
    atomic_write_json,
    image_messages,
    validate_internvl8b_config,
    require_internvl_transformers,
)


def existing_is_current(path: Path, model_id: str, source_sha256: str, count: int) -> bool:
    try:
        records = json.loads(path.read_text(encoding="utf-8"))
        return (
            len(records) == count
            and all(record["internvl_model_id"] == model_id for record in records)
            and all(
                record["internvl_source_manifest_sha256"] == source_sha256
                for record in records
            )
        )
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--coco-root", type=Path, required=True)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    require_internvl_transformers()

    source_bytes = args.input.read_bytes()
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    records = json.loads(source_bytes)
    if len(records) != 250:
        raise ValueError(f"Expected 250 audited records, found {len(records)}")
    count = args.limit if args.limit is not None else len(records)
    if not 1 <= count <= len(records):
        raise ValueError("--limit must be between 1 and 250")
    if not args.force and existing_is_current(
        args.output, args.model_id, source_sha256, count
    ):
        print(f"InternVL manifest already current: {args.output}")
        return
    records = records[:count]

    config = AutoConfig.from_pretrained(args.model_id, trust_remote_code=True)
    validate_internvl8b_config(config)

    processor = AutoProcessor.from_pretrained(
        args.model_id, trust_remote_code=True, use_fast=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id, trust_remote_code=True, use_fast=True
    )
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError("InternVL target preparation requires a fast tokenizer.")

    converted = []
    for case_index, original in enumerate(records):
        image_path = args.coco_root / Path(original["image_path"]).name
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        alignment = align_audited_target(tokenizer, original)
        prompt_inputs = processor.apply_chat_template(
            image_messages(str(image_path), CAPTION_PROMPT),
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        prompt_ids = prompt_inputs["input_ids"][0].tolist()

        record = dict(original)
        record.update(
            {
                "caption_token_ids": alignment["caption_ids"],
                "target_caption_token_span": alignment["target_token_span"],
                "target_generated_index": alignment["target_index"],
                "target_generated_id": alignment["target_id"],
                "target_generated_token": alignment["target_token"],
                "target_generated_token_piece": alignment["target_token"],
                "output_word_id": alignment["caption_ids"],
                "generated_ids": prompt_ids + alignment["caption_ids"],
                "generate_sentence": original["selected_coco_caption"],
                "internvl_model_id": args.model_id,
                "internvl_source_manifest_sha256": source_sha256,
                "internvl_case_index": case_index,
                "target_index_convention": (
                    "zero-based; spans are half-open; target is the final InternVL "
                    "caption token overlapping the audited target character span"
                ),
            }
        )
        converted.append(record)

    if len({record["image_path"] for record in converted}) != count:
        raise ValueError("Converted manifest has duplicate image paths")
    atomic_write_json(args.output, converted)
    print(f"Wrote {len(converted)} InternVL records to {args.output}")
    print(f"Source manifest SHA-256: {source_sha256}")


if __name__ == "__main__":
    main()
