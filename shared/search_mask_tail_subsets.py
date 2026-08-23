#!/usr/bin/env python3
"""Search random COCO subsets while penalizing very large target masks.

Each seed samples unique images, then randomly chooses one caption-valid,
unambiguous COCO instance target per sampled image. Only non-crowd polygon
annotations whose category occurs exactly once in that image and whose target
is named in at least one official COCO caption are eligible. EAGLE images can
be excluded with an evaluation-list JSON.

The ranking is deliberately independent of model results.  It is
lexicographic, in this order:

1. number of targets whose rasterized mask covers >= ``--large-threshold``;
2. CVaR90: mean mask fraction among the largest 10 percent of targets;
3. 95th percentile mask fraction;
4. 75th percentile mask fraction;
5. mean mask fraction;
6. maximum mask fraction;
7. category coverage (more categories wins);
8. seed (smaller wins, for deterministic final ties).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instances", type=Path, required=True)
    parser.add_argument("--captions", type=Path, required=True)
    parser.add_argument("--exclude-eval-list", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--subset-size", type=int, default=250)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--num-seeds", type=int, default=100_000)
    parser.add_argument("--large-threshold", type=float, default=0.40)
    return parser.parse_args()


# Ordinary caption terms that map clearly to COCO categories. Generic words
# such as "board", "bag", and "vehicle" are intentionally excluded.
CAPTION_SYNONYMS: dict[str, tuple[str, ...]] = {
    "person": ("people", "man", "men", "woman", "women", "boy", "boys", "girl", "girls", "child", "children", "kid", "kids", "baby", "babies"),
    "bicycle": ("bike", "bikes", "cyclist", "cyclists"),
    "car": ("automobile", "automobiles"),
    "motorcycle": ("motorbike", "motorbikes", "motor cycle", "motor cycles"),
    "airplane": ("aeroplane", "aeroplanes", "plane", "planes", "aircraft", "jet", "jets"),
    "train": ("locomotive", "locomotives", "tram", "trams"),
    "truck": ("pickup truck", "pickup trucks"),
    "boat": ("sailboat", "sailboats", "ship", "ships"),
    "traffic light": ("stoplight", "stoplights", "traffic signal", "traffic signals"),
    "fire hydrant": ("hydrant", "hydrants"),
    "cat": ("kitten", "kittens"),
    "dog": ("puppy", "puppies"),
    "horse": ("pony", "ponies", "stallion", "stallions", "mare", "mares", "donkey", "donkeys"),
    "sheep": ("lamb", "lambs"),
    "cow": ("cattle", "calf", "calves", "bull", "bulls", "ox", "oxen"),
    "backpack": ("rucksack", "rucksacks", "bookbag", "bookbags"),
    "umbrella": ("parasol", "parasols"),
    "handbag": ("purse", "purses"),
    "tie": ("necktie", "neckties"),
    "suitcase": ("luggage",),
    "frisbee": ("flying disc", "flying discs"),
    "skis": ("ski",),
    "snowboard": ("snow board", "snow boards"),
    "sports ball": ("ball", "balls"),
    "baseball bat": ("bat", "bats"),
    "baseball glove": ("mitt", "mitts"),
    "skateboard": ("skate board", "skate boards"),
    "surfboard": ("surf board", "surf boards", "bodyboard", "bodyboards", "boogie board", "boogie boards"),
    "tennis racket": ("tennis racquet", "tennis racquets", "racket", "rackets", "racquet", "racquets"),
    "wine glass": ("wineglass", "wineglasses", "goblet", "goblets"),
    "cup": ("mug", "mugs"),
    "donut": ("doughnut", "doughnuts"),
    "chair": ("stool", "stools"),
    "couch": ("sofa", "sofas"),
    "potted plant": ("houseplant", "houseplants"),
    "dining table": ("table", "tables"),
    "tv": ("television", "televisions"),
    "mouse": ("mice",),
    "remote": ("remote control", "remote controls", "game controller", "game controllers"),
    "cell phone": ("cellphone", "cellphones", "mobile phone", "mobile phones", "smartphone", "smartphones", "phone", "phones"),
    "refrigerator": ("fridge", "fridges"),
    "clock": ("clocktower", "clocktowers", "clock tower", "clock towers"),
    "teddy bear": ("stuffed bear", "stuffed bears", "teddy", "teddies"),
    "hair drier": ("hair dryer", "hair dryers", "blow dryer", "blow dryers"),
    "toothbrush": ("tooth brush", "tooth brushes"),
    "knife": ("knives",),
}


def pluralize_phrase(phrase: str) -> str:
    words = phrase.split()
    last = words[-1]
    irregular = {"mouse": "mice", "knife": "knives", "sheep": "sheep"}
    if last in irregular:
        words[-1] = irregular[last]
    elif last.endswith(("s", "x", "z", "ch", "sh")):
        words[-1] = last + "es"
    elif len(last) > 1 and last.endswith("y") and last[-2] not in "aeiou":
        words[-1] = last[:-1] + "ies"
    else:
        words[-1] = last + "s"
    return " ".join(words)


def normalize_caption(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.casefold()))


def caption_target_match(category: str, captions: list[str]) -> tuple[str, str] | None:
    aliases = (category, pluralize_phrase(category), *CAPTION_SYNONYMS.get(category, ()))
    normalized_aliases = tuple(normalize_caption(alias) for alias in aliases)
    for caption in captions:
        padded_caption = f" {normalize_caption(caption)} "
        for alias, normalized_alias in zip(aliases, normalized_aliases):
            if f" {normalized_alias} " not in padded_caption:
                if category == "tv" and normalized_alias.startswith("television"):
                    if any(token.startswith("television") for token in padded_caption.split()):
                        return caption, alias
                continue
            if (
                category == "bicycle"
                and normalized_alias in {"bike", "bikes"}
                and any(
                    term in padded_caption
                    for term in (" motorcycle ", " motorcycles ", " motorcyclist ", " motorcyclists ")
                )
            ):
                continue
            return caption, alias
    return None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def image_id_from_record(record: dict) -> int:
    if "image_id" in record:
        return int(record["image_id"])
    return int(Path(record["image_path"]).stem)


def rasterized_mask_pixels(segmentation: list, width: int, height: int) -> int:
    mask = np.zeros((height, width), dtype=np.uint8)
    for polygon in segmentation:
        if not isinstance(polygon, list) or len(polygon) < 6:
            continue
        points = np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
        points = np.rint(points).astype(np.int32)
        points[:, 0] = np.clip(points[:, 0], 0, width - 1)
        points[:, 1] = np.clip(points[:, 1], 0, height - 1)
        cv2.fillPoly(mask, [points], 1)
    return int(np.count_nonzero(mask))


def quantile(sorted_values: list[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("cannot calculate a quantile of an empty sequence")
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def subset_metrics(records: list[dict], large_threshold: float) -> dict:
    fractions = sorted(float(record["mask_fraction"]) for record in records)
    tail_count = max(1, math.ceil(0.10 * len(fractions)))
    categories = {int(record["category_id"]) for record in records}
    return {
        "very_large_count": sum(value >= large_threshold for value in fractions),
        "cvar90": sum(fractions[-tail_count:]) / tail_count,
        "q95": quantile(fractions, 0.95),
        "q75": quantile(fractions, 0.75),
        "mean": sum(fractions) / len(fractions),
        "maximum": fractions[-1],
        "category_count": len(categories),
    }


def objective(metrics: dict, seed: int) -> tuple:
    return (
        metrics["very_large_count"],
        metrics["cvar90"],
        metrics["q95"],
        metrics["q75"],
        metrics["mean"],
        metrics["maximum"],
        -metrics["category_count"],
        seed,
    )


def load_candidates(args: argparse.Namespace) -> tuple[dict[int, list[dict]], dict]:
    coco = json.loads(args.instances.read_text(encoding="utf-8"))
    images = {int(image["id"]): image for image in coco["images"]}
    categories = {
        int(category["id"]): category["name"] for category in coco["categories"]
    }

    excluded: set[int] = set()
    if args.exclude_eval_list:
        records = json.loads(args.exclude_eval_list.read_text(encoding="utf-8"))
        excluded = {image_id_from_record(record) for record in records}

    annotations_by_image: dict[int, list[dict]] = defaultdict(list)
    for annotation in coco["annotations"]:
        image_id = int(annotation["image_id"])
        if image_id in excluded:
            continue
        annotations_by_image[image_id].append(annotation)

    captions_by_image: dict[int, list[str]] = defaultdict(list)
    captions = json.loads(args.captions.read_text(encoding="utf-8"))
    for annotation in captions["annotations"]:
        captions_by_image[int(annotation["image_id"])].append(
            annotation["caption"]
        )

    structurally_eligible_targets = 0
    caption_rejected_targets = 0
    candidates: dict[int, list[dict]] = {}
    for image_id, annotations in annotations_by_image.items():
        image = images[image_id]
        width = int(image["width"])
        height = int(image["height"])
        category_counts = Counter(int(a["category_id"]) for a in annotations)
        image_candidates = []
        for annotation in annotations:
            if int(annotation.get("iscrowd", 0)) != 0:
                continue
            segmentation = annotation.get("segmentation")
            if not isinstance(segmentation, list) or not segmentation:
                continue
            category_id = int(annotation["category_id"])
            if category_counts[category_id] != 1:
                continue
            mask_pixels = rasterized_mask_pixels(
                annotation["segmentation"], width, height
            )
            if mask_pixels == 0:
                continue
            structurally_eligible_targets += 1
            category_name = categories[category_id]
            match = caption_target_match(
                category_name, captions_by_image.get(image_id, [])
            )
            if match is None:
                caption_rejected_targets += 1
                continue
            matched_caption, matched_alias = match
            x, y, box_width, box_height = map(float, annotation["bbox"])
            image_candidates.append(
                {
                    "image_id": image_id,
                    "image_path": image["file_name"],
                    "width": width,
                    "height": height,
                    "annotation_id": int(annotation["id"]),
                    "category_id": category_id,
                    "select_category": category_name,
                    "bbox_xywh": annotation["bbox"],
                    "location": [x, y, x + box_width, y + box_height],
                    "segmentation": annotation["segmentation"],
                    "mask_pixels": mask_pixels,
                    "image_pixels": width * height,
                    "mask_fraction": mask_pixels / (width * height),
                    "coco_area": float(annotation["area"]),
                    "coco_captions": captions_by_image.get(image_id, []),
                    "caption_target_match": {
                        "caption": matched_caption,
                        "alias": matched_alias,
                    },
                }
            )
        if image_candidates:
            candidates[image_id] = sorted(
                image_candidates, key=lambda record: record["annotation_id"]
            )

    pool_info = {
        "coco_images": len(images),
        "excluded_images": len(excluded),
        "structurally_eligible_targets": structurally_eligible_targets,
        "caption_rejected_targets": caption_rejected_targets,
        "eligible_images": len(candidates),
        "eligible_targets": sum(len(values) for values in candidates.values()),
    }
    return candidates, pool_info


def sample_seed(
    candidates: dict[int, list[dict]], subset_size: int, seed: int
) -> list[dict]:
    rng = random.Random(seed)
    image_ids = rng.sample(list(candidates), subset_size)
    return [rng.choice(candidates[image_id]) for image_id in image_ids]


def main() -> None:
    args = parse_args()
    if args.subset_size <= 0:
        raise ValueError("--subset-size must be positive")
    if args.num_seeds <= 0:
        raise ValueError("--num-seeds must be positive")
    if not 0.0 < args.large_threshold <= 1.0:
        raise ValueError("--large-threshold must be in (0, 1]")

    candidates, pool_info = load_candidates(args)
    if args.subset_size > len(candidates):
        raise ValueError(
            f"requested {args.subset_size} images but only {len(candidates)} are eligible"
        )

    rows = []
    best_key = None
    best_seed = None
    best_records = None
    for offset in range(args.num_seeds):
        seed = args.seed_start + offset
        records = sample_seed(candidates, args.subset_size, seed)
        metrics = subset_metrics(records, args.large_threshold)
        key = objective(metrics, seed)
        rows.append({"seed": seed, **metrics})
        if best_key is None or key < best_key:
            best_key = key
            best_seed = seed
            best_records = records

    rows.sort(key=lambda row: objective(row, int(row["seed"])))
    assert best_seed is not None and best_records is not None
    best_records = sorted(best_records, key=lambda record: record["image_path"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    subset_path = args.output_dir / f"best_subset_{args.subset_size}.json"
    leaderboard_path = args.output_dir / "seed_leaderboard.csv"
    metadata_path = args.output_dir / "search_metadata.json"

    subset_path.write_text(
        json.dumps(best_records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    fieldnames = [
        "rank",
        "seed",
        "very_large_count",
        "cvar90",
        "q95",
        "q75",
        "mean",
        "maximum",
        "category_count",
    ]
    with leaderboard_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for rank, row in enumerate(rows, start=1):
            writer.writerow({"rank": rank, **row})

    eligibility_prefix = "non-EAGLE image; " if args.exclude_eval_list else ""
    metadata = {
        "instances": str(args.instances.resolve()),
        "instances_sha256": sha256(args.instances),
        "captions": str(args.captions.resolve()),
        "captions_sha256": sha256(args.captions),
        "exclude_eval_list": (
            str(args.exclude_eval_list.resolve()) if args.exclude_eval_list else None
        ),
        "subset_size": args.subset_size,
        "seed_start": args.seed_start,
        "num_seeds": args.num_seeds,
        "large_threshold": args.large_threshold,
        "objective_order": [
            "very_large_count",
            "cvar90",
            "q95",
            "q75",
            "mean",
            "maximum",
            "negative_category_count",
            "seed",
        ],
        "mask_fraction_definition": (
            "foreground pixels after OpenCV polygon rasterization divided by "
            "image width times image height"
        ),
        "eligibility": (
            f"{eligibility_prefix}non-crowd polygon; category occurs exactly once "
            "in the image; nonempty rasterized mask; at least one official COCO "
            "caption contains the category name or an explicit mapped synonym"
        ),
        "pool": pool_info,
        "best_seed": best_seed,
        "best_metrics": subset_metrics(best_records, args.large_threshold),
        "outputs": {
            "subset": str(subset_path.resolve()),
            "leaderboard": str(leaderboard_path.resolve()),
        },
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
