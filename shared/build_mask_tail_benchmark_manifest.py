#!/usr/bin/env python3
"""Build the shared EAGLE/ours manifest from the manually audited 250 cases."""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records = json.loads(args.input.read_text(encoding="utf-8"))
    if len(records) != 250:
        raise ValueError(f"expected 250 records, found {len(records)}")

    seen = set()
    benchmark = []
    for index, source in enumerate(records):
        record = dict(source)
        image_path = record["image_path"]
        if image_path in seen:
            raise ValueError(f"duplicate image at case {index + 1}: {image_path}")
        seen.add(image_path)

        required = (
            "selected_coco_caption",
            "target_caption_phrase",
            "target_generated_index",
            "target_generated_id",
            "caption_token_ids",
            "select_category",
            "segmentation",
            "location",
        )
        missing = [key for key in required if key not in record]
        if missing:
            raise ValueError(f"case {index + 1} missing {missing}")

        token_index = int(record["target_generated_index"])
        caption_ids = [int(value) for value in record["caption_token_ids"]]
        if not 0 <= token_index < len(caption_ids):
            raise ValueError(f"invalid target index at case {index + 1}")
        if caption_ids[token_index] != int(record["target_generated_id"]):
            raise ValueError(f"target id mismatch at case {index + 1}")

        # EAGLE uses the token id/index above. Our yes/no method needs a human
        # object label, so expose the complete audited phrase instead of a BPE
        # fragment (for example, the final "is" token in "skis").
        record["target_generated_token_piece"] = record.get(
            "target_generated_token", ""
        )
        record["target_generated_token"] = record["target_caption_phrase"]
        record["output_word_id"] = caption_ids
        record["generate_sentence"] = record["selected_coco_caption"]
        record["benchmark_case_index"] = index
        benchmark.append(record)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(
        json.dumps(benchmark, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(f"wrote {len(benchmark)} unique cases to {args.output}")


if __name__ == "__main__":
    main()
