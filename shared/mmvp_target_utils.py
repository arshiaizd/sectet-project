"""Small helpers shared by MMVP baseline adapters."""

from __future__ import annotations

from typing import Any


def prompt_and_targets(record: dict[str, Any]) -> tuple[str, list[int], list[int]]:
    prompt = str(record["prompt"])
    answer_ids = [int(value) for value in record["predicted_answer_token_ids"]]
    positions = [int(value) for value in record["target_generated_indices"]]
    if not answer_ids or positions != list(range(len(answer_ids))):
        raise ValueError(f"{record.get('image_path')}: invalid answer target sequence")
    if [int(value) for value in record["target_generated_ids"]] != answer_ids:
        raise ValueError(f"{record.get('image_path')}: target ID mismatch")
    return prompt, positions, answer_ids
