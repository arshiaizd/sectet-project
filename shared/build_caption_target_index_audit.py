#!/usr/bin/env python3
"""Attach caption-relative Qwen token indices to a manually reviewed subset."""

import argparse
import csv
import json
import re
from pathlib import Path

from transformers import AutoTokenizer


REJECTED = {
    "000000110784.jpg": "caption says people, but the COCO mask covers one person",
    "000000144932.jpg": "caption says tug boats, but the COCO mask covers one boat",
    "000000262487.jpg": "'up to bat' is an idiom, not a direct noun reference to the bat",
    "000000471991.jpg": "caption says chairs, but the COCO mask covers one of several chairs",
    "000000483531.jpg": "caption says two beds, but the COCO mask is a single bed instance",
    "000000492878.jpg": "caption says toothbrushes, but the COCO mask is one toothbrush",
    "000000512564.jpg": "caption says a couple of buses, but the COCO mask is one bus",
}

PHRASE_OVERRIDES = {
    "000000376625.jpg": "trolley car",
}

CAVEATS = {
    50: "target bowl appears inside an artwork",
    78: "target is explicitly a miniature banana",
    93: "caption uses collective term 'luggage' for a suitcase instance",
    112: "caption says wheelchair for the COCO chair category",
    143: "very large vase target (mask fraction 0.8216)",
    167: "caption says boogie board for the COCO surfboard category",
    178: "target vase appears inside a painting",
    188: "target clock is visible as a reflection",
    206: "target television is visible in/through the mirror",
    247: "caption says tram for the COCO train category",
}


def find_phrase(caption: str, phrase: str) -> tuple[int, int]:
    parts = phrase.split()
    pattern = r"(?<!\w)" + r"\s+".join(map(re.escape, parts)) + r"(?!\w)"
    match = re.search(pattern, caption, flags=re.IGNORECASE)
    if match is None:
        raise ValueError(f"phrase {phrase!r} not found in caption {caption!r}")
    return match.span()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    records = json.loads(args.input.read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, use_fast=True, local_files_only=True
    )
    audited = []
    for case, source in enumerate(records, start=1):
        record = dict(source)
        caption = source["caption_target_match"]["caption"]
        image_path = source["image_path"]
        phrase = PHRASE_OVERRIDES.get(
            image_path, source["caption_target_match"]["alias"]
        )
        char_start, char_end = find_phrase(caption, phrase)

        words = list(re.finditer(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", caption))
        word_indices = [
            index
            for index, match in enumerate(words)
            if match.start() < char_end and match.end() > char_start
        ]
        encoded = tokenizer(
            caption,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        token_indices = [
            index
            for index, (start, end) in enumerate(encoded["offset_mapping"])
            if start < char_end and end > char_start
        ]
        if not word_indices or not token_indices:
            raise RuntimeError(f"empty target span for case {case}")

        target_token_index = token_indices[-1]
        target_token_id = int(encoded["input_ids"][target_token_index])
        status = "reject" if image_path in REJECTED else (
            "accept_with_caveat" if case in CAVEATS else "accept"
        )
        record.update(
            {
                "audit_case": case,
                "audit_status": status,
                "audit_note": REJECTED.get(
                    image_path, CAVEATS.get(case, "clean")
                ),
                "selected_coco_caption": caption,
                "target_caption_phrase": caption[char_start:char_end],
                "target_caption_char_span": [char_start, char_end],
                "target_caption_word_span": [word_indices[0], word_indices[-1] + 1],
                "target_caption_word_index": word_indices[-1],
                "caption_token_ids": [int(value) for value in encoded["input_ids"]],
                "target_caption_token_span": [token_indices[0], token_indices[-1] + 1],
                "target_generated_index": target_token_index,
                "target_generated_id": target_token_id,
                "target_generated_token": tokenizer.convert_ids_to_tokens(
                    target_token_id
                ),
                "target_index_convention": (
                    "zero-based; spans are half-open; EAGLE target is the final "
                    "Qwen caption token overlapping the target phrase"
                ),
            }
        )
        audited.append(record)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(audited, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    fields = [
        "audit_case", "image_path", "select_category", "audit_status",
        "audit_note", "selected_coco_caption", "target_caption_phrase",
        "target_caption_char_span", "target_caption_word_span",
        "target_caption_word_index", "target_caption_token_span",
        "target_generated_index", "target_generated_id",
        "target_generated_token", "mask_fraction",
    ]
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in audited:
            writer.writerow(
                {
                    field: json.dumps(record[field])
                    if isinstance(record[field], list)
                    else record[field]
                    for field in fields
                }
            )

    counts = {
        status: sum(record["audit_status"] == status for record in audited)
        for status in ("accept", "accept_with_caveat", "reject")
    }
    print(json.dumps({"cases": len(audited), **counts}, indent=2))


if __name__ == "__main__":
    main()
