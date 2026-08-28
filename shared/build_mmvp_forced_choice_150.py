#!/usr/bin/env python3
"""Build a balanced, one-image-per-pair forced-choice MMVP-150 manifest."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_SOURCE = Path(
    "/mnt/vilab/scratch/arshia/projects/izadi/ours/mmvp/official_eagle/"
    "Qwen2.5-VL-7B-MMVP-VQA.json"
)
DEFAULT_OUTPUT = Path(
    "/mnt/vilab/scratch/arshia/projects/izadi/shared/"
    "mmvp_forced_choice_150_seed_20260828.json"
)
DEFAULT_IMAGES = Path("/mnt/vilab/scratch/arshia/datasets/MMVP/MMVP Images")
PROMPT_VERSION = "mmvp_exact_option_text_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--seed", type=int, default=20260828)
    return parser.parse_args()


def forced_choice_prompt(question: str, options: list[str]) -> str:
    if len(options) != 2:
        raise ValueError(f"expected exactly two options, found {len(options)}")
    return (
        "Answer the visual multiple-choice question by selecting exactly one option.\n\n"
        f"Question: {question}\n\n"
        "Options:\n"
        f"A. {options[0]}\n"
        f"B. {options[1]}\n\n"
        "Respond with exactly the complete text of one option and nothing else. "
        "Do not include an option letter, explanation, punctuation, or any other text."
    )


def normalize_answer(value: Any) -> str:
    answer = str(value).strip().lower().replace("(", "").replace(")", "")
    if answer not in {"a", "b"}:
        raise ValueError(f"unsupported answer label: {value!r}")
    return answer


def validate_pair(first: dict[str, Any], second: dict[str, Any], pair_index: int) -> None:
    if first["question"] != second["question"]:
        raise ValueError(f"pair {pair_index} has different questions")
    if first["options"] != second["options"]:
        raise ValueError(f"pair {pair_index} has different options")
    if len(first["options"]) != 2 or first["options"][0] == first["options"][1]:
        raise ValueError(f"pair {pair_index} has invalid options")
    answers = {normalize_answer(first["answer"]), normalize_answer(second["answer"])}
    if answers != {"a", "b"}:
        raise ValueError(f"pair {pair_index} does not contain complementary answers")


def main() -> None:
    args = parse_args()
    source = json.loads(args.source.read_text(encoding="utf-8"))
    if len(source) != 300:
        raise ValueError(f"expected MMVP-300 source, found {len(source)} entries")

    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for pair_index, start in enumerate(range(0, len(source), 2)):
        first, second = source[start], source[start + 1]
        validate_pair(first, second, pair_index)
        pairs.append((first, second))
    if len(pairs) != 150:
        raise ValueError(f"expected 150 pairs, found {len(pairs)}")

    rng = random.Random(args.seed)
    choose_a_pairs = set(rng.sample(range(len(pairs)), len(pairs) // 2))
    output: list[dict[str, Any]] = []
    for pair_index, pair in enumerate(pairs):
        desired_answer = "a" if pair_index in choose_a_pairs else "b"
        selected = next(
            item for item in pair if normalize_answer(item["answer"]) == desired_answer
        )
        partner = next(item for item in pair if item is not selected)
        options = [str(value).strip() for value in selected["options"]]
        answer_index = 0 if desired_answer == "a" else 1
        image_path = args.images_dir / selected["image_filename"]
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        output.append(
            {
                "dataset_sample_index": len(output),
                "mmvp_pair_index": pair_index,
                "source_question_id": str(selected["question_id"]),
                "source_partner_question_id": str(partner["question_id"]),
                "image_filename": str(selected["image_filename"]),
                "partner_image_filename": str(partner["image_filename"]),
                "question": str(selected["question"]),
                "options": options,
                "allowed_outputs": options,
                "ground_truth_label": desired_answer.upper(),
                "ground_truth_option_index": answer_index,
                "ground_truth_option_text": options[answer_index],
                "prompt": forced_choice_prompt(str(selected["question"]), options),
                "prompt_version": PROMPT_VERSION,
                "selection_seed": args.seed,
                "selection_protocol": (
                    "one image per consecutive MMVP complementary pair; "
                    "75 answer-A pairs and 75 answer-B pairs selected by seeded sampling"
                ),
                "attribution_target_protocol": (
                    "attribute the model-selected exact option text; never substitute "
                    "the ground-truth option when the model prediction is incorrect"
                ),
            }
        )

    answer_counts = Counter(item["ground_truth_label"] for item in output)
    pair_indices = {int(item["mmvp_pair_index"]) for item in output}
    image_names = {str(item["image_filename"]) for item in output}
    if answer_counts != {"A": 75, "B": 75}:
        raise RuntimeError(f"unexpected answer balance: {answer_counts}")
    if pair_indices != set(range(150)) or len(image_names) != 150:
        raise RuntimeError("subset does not contain exactly one unique image per pair")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"saved={args.output}")
    print(f"samples={len(output)} pairs={len(pair_indices)} answers={dict(answer_counts)}")
    print(f"seed={args.seed} prompt_version={PROMPT_VERSION}")


if __name__ == "__main__":
    main()
