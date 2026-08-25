# -*- coding: utf-8 -*-
"""
Yes/no activation-patching attribution on the EAGLE Qwen2.5-VL-7B COCO object
dataset, with the intervention applied inside the VISION ENCODER.

This is a variant of ``our_method_insertion_online.py``. EVERYTHING is identical
to that script -- the yes/no prompting, the 3x3 patch neighborhood, the
baseline canvas, the norm-preserving source->baseline injection, the yes-token
scoring, and all CSV plumbing -- EXCEPT the location of the patch: instead of the
language DECODER layers, the source activations are injected into the VISION
ENCODER blocks (``model.visual.blocks``).

For every EAGLE sample and every merged visual patch (with its 3x3 neighborhood):

1. Build the yes/no question from the cleaned EAGLE target token.
2. Run the ORIGINAL image and cache the source vision-encoder block INPUTS.
3. Run a black/white baseline image with the same yes/no prompt.
4. Inject the source vision-encoder activations into the corresponding raw-patch
   positions of the baseline forward, over vision blocks [start_layer, end_layer],
   norm-preserving to the destination activation (identical rule to the decoder
   version, applied one level earlier in the network).
5. Read the summed softmax probability of the yes token variants.
6. Optionally compute a deletion score for the center patch's clipped 3x3
   neighborhood:
     * ``deletion_mode=none``: skip deletion and preserve the original method.
     * ``deletion_mode=input``: paint the 3x3 pixels with the baseline color on
       the full input image, then measure p(yes).
     * ``deletion_mode=embedding``: in a full-image forward, replace the 3x3
       source activation rows with the corresponding baseline activation rows
       over an independently configurable vision-block interval (default 0..7),
       then measure p(yes).
7. With deletion enabled, rank by
   ``insertion_score + (1 - deletion_score)``; otherwise rank by insertion.

Encoder position mapping (the only genuinely new logic)
-------------------------------------------------------
The vision encoder processes RAW patches: sequence length S = 4 * grid_h * grid_w
(merge_size**2 raw patches per merged patch). Qwen2.5-VL also permutes the raw
patches into WINDOW order before the blocks (self.get_window_index). A merged
patch p (raster order, matching the language-side visual token order) is one
"merged unit"; in window order that unit sits at slot reverse[p] =
argsort(window_index)[p], and its raw tokens occupy positions
    reverse[p] * spatial_merge_unit + {0 .. spatial_merge_unit-1}.
If the model exposes no window index (e.g. Qwen2-VL), natural order is used
(reverse = identity). Vision-block hidden states are 2-D ([S, C]); the hooks
handle both 2-D and 3-D just in case.

IMPORTANT: this encoder mapping is version-sensitive and was NOT run on a real
model in this environment. Validate on your model with the built-in check:
patching the FULL 3x3 union over ALL patches should drive p(yes) strongly toward
the original image's p(yes); patching nothing leaves it at the baseline.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import string
from contextlib import contextmanager
from typing import Any, Dict, List, Sequence, Tuple

from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


# -----------------------------------------------------------------------------
# Defaults matching EAGLE's Qwen2.5-VL-7B object experiment
# -----------------------------------------------------------------------------
DEFAULT_MODEL_ID = "/home/mmd/Qwen2.5-VL-7B-Instruct"
DEFAULT_COCO_ROOT = "/home/mmd/asal/EAGLE/datasets/coco/val2017"
DEFAULT_EVAL_LIST = "/home/mmd/asal/EAGLE/datasets/coco_single_target_once_qwen25vl-7B-subset100.json"
DEFAULT_OUTPUT_CSV = "./our_method_insertion_online_encoder.csv"

SYSTEM_PROMPT = "You are a precise vision assistant. Follow instructions exactly."

YES_NO_PROMPT_TEMPLATE = (
    "Is there a {object_label} in the image or not? "
    "Answer with exactly one word: yes or no."
)

# Candidate surface forms are used in addition to a vocabulary scan. Only
# candidates that tokenize to a single token are included, because this script
# measures the next-token probability at logits[:, -1, :].
YES_SURFACE_FORMS = (
    "yes", "Yes", "YES",
    " yes", " Yes", " YES",
    "\nyes", "\nYes", "\nYES",
    "yes.", "Yes.", "YES.",
    " yes.", " Yes.", " YES.",
    "yes!", "Yes!", "YES!",
    " yes!", " Yes!", " YES!",
)

PATCH_SIZE = 14
MERGE_SIZE = 2
EFFECTIVE_PATCH = PATCH_SIZE * MERGE_SIZE

# These are the standard Qwen2.5-VL limits used by the original implementation.
MIN_PIXELS = 56 * 56
MAX_PIXELS = 28 * 28 * 1280


# -----------------------------------------------------------------------------
# Arguments
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Score yes-token probability with the user's visual activation-patching method."
        )
    )
    parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--coco-root", type=str, default=DEFAULT_COCO_ROOT)
    parser.add_argument("--eval-list", type=str, default=DEFAULT_EVAL_LIST)
    parser.add_argument("--output-csv", type=str, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument(
        "--question-template",
        type=str,
        default=YES_NO_PROMPT_TEMPLATE,
        help=(
            "Yes/no question template. It must contain {object_label}, which is "
            "filled from each EAGLE sample's cleaned target_generated_token "
            "when available, falling back to select_category."
        ),
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=SYSTEM_PROMPT,
        help="System prompt used before the image yes/no question.",
    )

    parser.add_argument("--begin", type=int, default=0)
    parser.add_argument(
        "--end",
        type=int,
        default=30,
        help="Exclusive dataset end index. -1 means all remaining samples.",
    )

    # Vision-ENCODER block range (configurable, exactly like the decoder version's
    # layer range). Qwen2.5-VL-7B has 32 vision blocks (indices 0..31). -1 end
    # means the last block. e.g. --start-layer 0 --end-layer 31 patches all blocks.
    parser.add_argument("--start-layer", type=int, default=0)
    parser.add_argument("--end-layer", type=int, default=-1)
    parser.add_argument(
        "--deletion-mode",
        "--deletion_mode",
        dest="deletion_mode",
        type=str,
        default="none",
        choices=["none", "input", "embedding"],
        help=(
            "Deletion mechanism. 'none' preserves the original insertion-only "
            "encoder method. 'input' removes the center patch and its clipped "
            "3x3 neighborhood from the input image. 'embedding' replaces those "
            "vision-token activations with baseline-image activations over "
            "--deletion-start-layer..--deletion-end-layer."
        ),
    )
    parser.add_argument(
        "--deletion-start-layer",
        "--deletion_start_layer",
        dest="deletion_start_layer",
        type=int,
        default=0,
        help="First vision-encoder block used for embedding deletion (default 0).",
    )
    parser.add_argument(
        "--deletion-end-layer",
        "--deletion_end_layer",
        dest="deletion_end_layer",
        type=int,
        default=7,
        help=(
            "Last vision-encoder block used for embedding deletion, inclusive "
            "(default 7; -1 means the final vision block)."
        ),
    )
    parser.add_argument(
        "--canvas-mode",
        type=str,
        default="full",
        choices=["full", "patch"],
        help=(
            "Destination the source activations are patched onto. 'full' "
            "(default, original behavior): a black/white canvas at the ORIGINAL "
            "image size, with the 3x3 neighborhood injected in place. 'patch': a "
            "small canvas sized EXACTLY to the selected patch + its padding "
            "(the neighborhood bounding box, e.g. 3x3 patches = 84x84 px), with "
            "the neighborhood's full-frame source activations injected into it."
        ),
    )
    parser.add_argument(
        "--inject-mode",
        type=str,
        default="norm_preserve",
        choices=["norm_preserve", "raw"],
        help=(
            "How the source vector is written into the baseline activation. "
            "'norm_preserve' (default, same rule as the decoder script): scale "
            "the source direction to the destination (baseline) norm. 'raw': "
            "write the exact source vector with NO rescaling -- use this to test "
            "whether norm-preservation is distorting single-patch magnitudes in "
            "the vision encoder."
        ),
    )
    parser.add_argument(
        "--self-check",
        action="store_true",
        help=(
            "For the FIRST processed sample, also patch the FULL union of all "
            "patches and print p(yes); it should land close to the original "
            "image's p(yes) if the encoder position mapping is correct."
        ),
    )

    parser.add_argument(
        "--neighbor-mode",
        type=str,
        default="square",
        choices=["center", "cross", "square"],
        help=(
            "center: patch only the selected visual token; cross: center + 4-neighbors; "
            "square: center + 8-neighbors."
        ),
    )
    parser.add_argument(
        "--exclude-center",
        action="store_true",
        help="Exclude the center patch in cross/square modes.",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default="white",
        choices=["black", "white"],
        help="Baseline image used as the activation-patching target.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete an existing output CSV before starting.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Preserve and skip complete per-image CSV groups. Any incomplete "
            "group left by an interrupted write is discarded and recomputed."
        ),
    )
    return parser.parse_args()


# -----------------------------------------------------------------------------
# Basic helpers
# -----------------------------------------------------------------------------
def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)


def append_dict_rows_to_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    """Append dictionaries to CSV using Python's standard library only."""
    if not rows:
        return

    file_has_content = os.path.exists(path) and os.path.getsize(path) > 0

    if file_has_content:
        with open(path, "r", encoding="utf-8", newline="") as csv_file:
            reader = csv.reader(csv_file)
            try:
                fieldnames = next(reader)
            except StopIteration as error:
                raise RuntimeError(f"Existing CSV has no header: {path}") from error
    else:
        fieldnames = list(rows[0].keys())

    expected_fields = set(fieldnames)
    for row_index, row in enumerate(rows):
        actual_fields = set(row.keys())
        if actual_fields != expected_fields:
            missing = sorted(expected_fields - actual_fields)
            extra = sorted(actual_fields - expected_fields)
            raise ValueError(
                f"CSV schema mismatch at row {row_index}: "
                f"missing={missing}, extra={extra}"
            )

    with open(path, "a", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if not file_has_content:
            writer.writeheader()
        writer.writerows(rows)


def prepare_resume_csv(path: str) -> set[int]:
    """Keep complete image groups, drop partial groups, and return their indices."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return set()

    with open(path, "r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise RuntimeError(f"Existing CSV has no header: {path}")
        fieldnames = list(reader.fieldnames)
        all_rows = list(reader)

    groups: Dict[int, List[Dict[str, str]]] = {}
    for row in all_rows:
        try:
            sample_index = int(row["dataset_sample_index"])
        except (KeyError, TypeError, ValueError):
            continue
        groups.setdefault(sample_index, []).append(row)

    complete_indices = set()
    kept_rows: List[Dict[str, str]] = []
    for sample_index, rows in groups.items():
        try:
            grid_h = int(rows[0]["source_grid_h"])
            grid_w = int(rows[0]["source_grid_w"])
            expected = grid_h * grid_w
            patch_indices = {int(row["center_patch_index"]) for row in rows}
            one_image = len({row["image_path"] for row in rows}) == 1
            complete = (
                one_image
                and len(rows) == expected
                and patch_indices == set(range(expected))
            )
        except (KeyError, TypeError, ValueError):
            complete = False
        if complete:
            complete_indices.add(sample_index)
            kept_rows.extend(rows)

    if len(kept_rows) != len(all_rows):
        temporary_path = path + ".resume.tmp"
        with open(temporary_path, "w", encoding="utf-8", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(kept_rows)
        os.replace(temporary_path, path)
        print(
            f"[resume] retained {len(complete_indices)} complete images and "
            f"dropped {len(all_rows) - len(kept_rows)} incomplete rows"
        )
    else:
        print(f"[resume] found {len(complete_indices)} complete images")

    return complete_indices


def idx_to_rc(idx: int, grid_w: int) -> Tuple[int, int]:
    return idx // grid_w, idx % grid_w


def rc_to_idx(row: int, col: int, grid_w: int) -> int:
    return row * grid_w + col


def neighbor_patch_indices(
    idx: int,
    grid_h: int,
    grid_w: int,
    mode: str,
    include_center: bool,
) -> List[int]:
    row, col = idx_to_rc(idx, grid_w)

    if mode == "center":
        return [idx]

    locations: List[Tuple[int, int]] = []
    if include_center:
        locations.append((row, col))

    if mode == "cross":
        offsets = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    elif mode == "square":
        offsets = [
            (dr, dc)
            for dr in (-1, 0, 1)
            for dc in (-1, 0, 1)
            if not (dr == 0 and dc == 0)
        ]
    else:
        raise ValueError(f"Unknown neighbor mode: {mode}")

    for dr, dc in offsets:
        rr, cc = row + dr, col + dc
        if 0 <= rr < grid_h and 0 <= cc < grid_w:
            locations.append((rr, cc))

    output: List[int] = []
    seen = set()
    for rr, cc in locations:
        patch_idx = rc_to_idx(rr, cc, grid_w)
        if patch_idx not in seen:
            seen.add(patch_idx)
            output.append(patch_idx)
    return output


def make_baseline_canvas(width: int, height: int, baseline: str) -> Image.Image:
    color = (0, 0, 0) if baseline == "black" else (255, 255, 255)
    return Image.new("RGB", (width, height), color=color)


def deletion_neighborhood_indices(
    idx: int,
    grid_h: int,
    grid_w: int,
) -> List[int]:
    """Return the center patch plus its clipped 8-neighborhood (a 3x3 square)."""
    return neighbor_patch_indices(
        idx,
        grid_h,
        grid_w,
        mode="square",
        include_center=True,
    )


def mask_patches_on_input_image(
    source_image: Image.Image,
    resized_w: int,
    resized_h: int,
    patch_indices: Sequence[int],
    grid_w: int,
    baseline: str,
    patch_px: int = EFFECTIVE_PATCH,
) -> Image.Image:
    """Paint selected merged-patch cells with the baseline color on the input."""
    import numpy as np

    resized = source_image.convert("RGB").resize(
        (int(resized_w), int(resized_h)), Image.BICUBIC
    )
    array = np.array(resized)
    fill = 0 if baseline == "black" else 255
    for patch_idx in patch_indices:
        row, col = idx_to_rc(int(patch_idx), grid_w)
        y0, y1 = row * patch_px, (row + 1) * patch_px
        x0, x1 = col * patch_px, (col + 1) * patch_px
        array[y0:y1, x0:x1, :] = fill
    return Image.fromarray(array)


def clean_eagle_token_for_prompt(token_text: Any) -> str:
    """
    Convert tokenizer/display tokens like ``Ġperson`` into prompt text like ``person``.

    The EAGLE JSON stores Qwen tokenizer-style strings in target_generated_token.
    For byte-level/SentencePiece tokenizers, leading markers such as ``Ġ`` or ``▁``
    mean a word boundary/space and should not be shown to the model in the prompt.
    """
    if token_text is None:
        return ""

    text = str(token_text)

    # Common tokenizer artifacts:
    #   Ġ = byte-level BPE leading-space marker
    #   ▁ = SentencePiece word-boundary marker
    #   Ċ = byte-level BPE newline marker
    text = (
        text.replace("Ġ", " ")
        .replace("▁", " ")
        .replace("Ċ", " ")
    )

    # Remove common special-token wrappers if a malformed entry ever includes them.
    for marker in ("<|", "|>", "<", ">"):
        text = text.replace(marker, " ")

    # For object labels, punctuation attached to the token is not useful in the
    # yes/no prompt. Example: ``Ġperson,`` -> ``person``.
    extra_quotes = "“”‘’«»‹›„‚"
    text = text.strip().strip(string.punctuation + extra_quotes).strip()

    # Collapse any whitespace introduced by artifact replacement.
    return " ".join(text.split())


def object_label_from_eagle_sample(content: Dict[str, Any], tokenizer) -> Tuple[str, str, str]:
    """
    Return (clean_object_label, raw_source_text, source_field_name).

    Prefer target_generated_token because it corresponds to the target word being
    analyzed by EAGLE. Fall back to decoding target_generated_id, then to
    select_category if needed.
    """
    raw_target_token = content.get("target_generated_token", "")
    cleaned = clean_eagle_token_for_prompt(raw_target_token)
    if cleaned:
        return cleaned, str(raw_target_token), "target_generated_token"

    if "target_generated_id" in content:
        try:
            token_id = int(content["target_generated_id"])
            decoded = tokenizer.decode(
                [token_id],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            cleaned = clean_eagle_token_for_prompt(decoded)
            if cleaned:
                return cleaned, decoded, "decoded_target_generated_id"
        except Exception:
            pass

    raw_category = content.get("select_category", "")
    cleaned = clean_eagle_token_for_prompt(raw_category)
    if cleaned:
        return cleaned, str(raw_category), "select_category"

    raise KeyError(
        "Could not build object label: target_generated_token, target_generated_id, "
        "and select_category are all missing or empty after cleaning."
    )


def model_input_device(model: nn.Module) -> torch.device:
    # For device_map="auto", model.device is normally the input/embedding device.
    if hasattr(model, "device"):
        return torch.device(model.device)
    return next(model.parameters()).device


# -----------------------------------------------------------------------------
# Qwen input and teacher-forcing helpers
# -----------------------------------------------------------------------------
def prepare_yes_no_prompt_inputs(
    processor,
    pil_image: Image.Image,
    prompt_text: str,
    system_prompt: str,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Prepare a Qwen chat prompt containing one image and a yes/no question."""
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_text},
            ],
        },
    ]
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    batch = processor(
        text=[text],
        images=[pil_image],
        padding=True,
        return_tensors="pt",
    )
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.to(device)
    return dict(batch)


def copy_tensor_batch(batch: Dict[str, Any]) -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    for key, value in batch.items():
        output[key] = value.clone() if isinstance(value, torch.Tensor) else value
    return output


def validate_sequence_aligned_tensors(batch: Dict[str, Any], where: str) -> None:
    """Validate tensors that must have the same sequence length as input_ids."""
    if "input_ids" not in batch:
        raise KeyError(f"{where}: batch does not contain input_ids.")

    sequence_length = int(batch["input_ids"].shape[1])
    for key in ("attention_mask", "mm_token_type_ids", "token_type_ids"):
        value = batch.get(key)
        if isinstance(value, torch.Tensor) and value.ndim == 2:
            if int(value.shape[1]) != sequence_length:
                raise RuntimeError(
                    f"{where}: {key} has sequence length {value.shape[1]}, "
                    f"but input_ids has sequence length {sequence_length}."
                )


def build_teacher_forced_prefix_batch(
    prompt_batch: Dict[str, Any],
    previous_output_token_ids: Sequence[int],
    device: torch.device,
) -> Dict[str, Any]:
    """
    Append only output tokens before the target token.

    All sequence-aligned tensors are extended consistently. In particular,
    Qwen2.5-VL may return ``mm_token_type_ids`` from the processor; generated
    assistant tokens are text tokens and therefore receive modality type 0.

    The final input position predicts the target token, so logits[:, -1, :]
    represents p(target | prompt, image, previous outputs).
    """
    batch = copy_tensor_batch(prompt_batch)
    validate_sequence_aligned_tensors(batch, "before prefix append")

    prefix_ids = torch.tensor(
        [list(map(int, previous_output_token_ids))],
        dtype=batch["input_ids"].dtype,
        device=device,
    )

    if prefix_ids.numel() > 0:
        old_length = int(batch["input_ids"].shape[1])
        batch["input_ids"] = torch.cat([batch["input_ids"], prefix_ids], dim=1)

        # Rebuild rather than incrementally concatenate, so attention_mask is
        # guaranteed to match the newly extended input_ids exactly.
        attention_dtype = batch.get("attention_mask", batch["input_ids"]).dtype
        batch["attention_mask"] = torch.ones(
            batch["input_ids"].shape,
            dtype=attention_dtype,
            device=batch["input_ids"].device,
        )

        # New assistant-prefix tokens are ordinary text tokens. Qwen2.5-VL's
        # modality convention is text=0, image=1, video=2.
        for key in ("mm_token_type_ids", "token_type_ids"):
            value = batch.get(key)
            if isinstance(value, torch.Tensor) and value.ndim == 2:
                if int(value.shape[1]) != old_length:
                    raise RuntimeError(
                        f"Cannot extend {key}: its length is {value.shape[1]}, "
                        f"but the original input_ids length is {old_length}."
                    )
                text_types = torch.zeros(
                    prefix_ids.shape,
                    dtype=value.dtype,
                    device=value.device,
                )
                batch[key] = torch.cat([value, text_types], dim=1)

    # These tensors are sequence-length dependent. Let Qwen recompute them for
    # the extended multimodal sequence.
    batch.pop("position_ids", None)
    batch.pop("cache_position", None)
    batch.pop("rope_deltas", None)

    validate_sequence_aligned_tensors(batch, "after prefix append")
    return batch


def find_vision_token_positions(
    processor,
    batch: Dict[str, torch.Tensor],
) -> List[int]:
    tokens = processor.tokenizer.convert_ids_to_tokens(
        batch["input_ids"][0].tolist()
    )

    positions: List[int] = []
    inside_image = False
    for pos, token in enumerate(tokens):
        if token == "<|vision_start|>":
            inside_image = True
            continue
        if token == "<|vision_end|>":
            inside_image = False
            continue
        if inside_image:
            positions.append(pos)
    return positions


def infer_merged_vision_grid(
    prompt_batch: Dict[str, torch.Tensor],
) -> Tuple[int, int, int, int]:
    """Return merged-token grid and corresponding processor-resized dimensions."""
    if "image_grid_thw" not in prompt_batch:
        raise RuntimeError("Qwen processor output does not contain image_grid_thw.")

    grid_thw = prompt_batch["image_grid_thw"][0]
    raw_grid_h = int(grid_thw[1].item())
    raw_grid_w = int(grid_thw[2].item())

    if raw_grid_h % MERGE_SIZE != 0 or raw_grid_w % MERGE_SIZE != 0:
        raise RuntimeError(
            f"Unexpected Qwen vision grid {raw_grid_h}x{raw_grid_w}; "
            f"not divisible by merge size {MERGE_SIZE}."
        )

    grid_h = raw_grid_h // MERGE_SIZE
    grid_w = raw_grid_w // MERGE_SIZE
    resized_h = raw_grid_h * PATCH_SIZE
    resized_w = raw_grid_w * PATCH_SIZE
    return grid_h, grid_w, resized_h, resized_w


# -----------------------------------------------------------------------------
# Vision-ENCODER activation patching
# -----------------------------------------------------------------------------
def get_vision_blocks(model: nn.Module) -> nn.ModuleList:
    """Return the vision-encoder transformer blocks (model.visual.blocks)."""
    explicit_paths = [
        ("visual", "blocks"),
        ("model", "visual", "blocks"),
        ("model", "model", "visual", "blocks"),
        ("vision_tower", "blocks"),
        ("model", "vision_tower", "blocks"),
    ]
    for path in explicit_paths:
        obj: Any = model
        valid = True
        for name in path:
            if not hasattr(obj, name):
                valid = False
                break
            obj = getattr(obj, name)
        if valid and isinstance(obj, nn.ModuleList):
            return obj

    # Fallback: a ModuleList whose block looks like a VISION block -- it has an
    # ``attn`` and ``norm1`` but NOT ``self_attn`` (which marks a decoder block).
    for _, module in model.named_modules():
        if (
            isinstance(module, nn.ModuleList)
            and len(module) > 0
            and hasattr(module[0], "attn")
            and hasattr(module[0], "norm1")
            and not hasattr(module[0], "self_attn")
        ):
            return module

    raise RuntimeError("Could not find the Qwen vision encoder blocks.")


def get_vision_module(model: nn.Module) -> Any:
    """Return the vision transformer module (owner of get_window_index)."""
    for path in [("visual",), ("model", "visual"), ("model", "model", "visual"), ("vision_tower",), ("model", "vision_tower")]:
        obj: Any = model
        valid = True
        for name in path:
            if not hasattr(obj, name):
                valid = False
                break
            obj = getattr(obj, name)
        if valid and hasattr(obj, "blocks"):
            return obj
    return None


def spatial_merge_unit_of(vision_module: Any) -> int:
    """merge_size**2 raw patches per merged patch (4 for Qwen2.5-VL)."""
    if vision_module is not None and hasattr(vision_module, "spatial_merge_unit"):
        try:
            return int(vision_module.spatial_merge_unit)
        except Exception:
            pass
    return int(MERGE_SIZE * MERGE_SIZE)


def compute_reverse_window_index(
    vision_module: Any,
    grid_thw: torch.Tensor,
    num_units: int,
) -> "torch.Tensor | None":
    """
    reverse[p] = the WINDOW-order slot of original merged unit p, so that the raw
    tokens of merged patch p live at reverse[p]*spatial_merge_unit + {0..u-1} in
    the sequence the blocks see. Returns None when the model applies no window
    reordering (then natural order is used).
    """
    if vision_module is None or not hasattr(vision_module, "get_window_index"):
        return None
    try:
        window_index, _ = vision_module.get_window_index(grid_thw)
    except Exception:
        return None
    window_index = window_index.detach().to("cpu").long().view(-1)
    if int(window_index.numel()) != int(num_units):
        # Unexpected shape -> fall back to natural order rather than mis-map.
        return None
    reverse = torch.argsort(window_index)
    return reverse


def merged_patches_to_raw_positions(
    merged_indices: Sequence[int],
    reverse_window_index: "torch.Tensor | None",
    spatial_merge_unit: int,
    seq_len: int,
) -> List[int]:
    """Map merged patch indices to their raw-patch positions in block order."""
    positions: List[int] = []
    for p in merged_indices:
        p = int(p)
        slot = int(reverse_window_index[p].item()) if reverse_window_index is not None else p
        base = slot * spatial_merge_unit
        for k in range(spatial_merge_unit):
            pos = base + k
            if 0 <= pos < seq_len:
                positions.append(pos)
    return positions


def build_patch_canvas_and_map(
    processor,
    model: nn.Module,
    neighborhood: Sequence[int],
    grid_w: int,
    full_reverse_window_index: "torch.Tensor | None",
    spatial_merge_unit: int,
    full_raw_seq_len: int,
    baseline: str,
    question_prompt: str,
    system_prompt: str,
    device,
) -> Tuple[Dict[str, torch.Tensor], List[Tuple[int, int]]]:
    """
    Build the small 3x3-style baseline canvas for the "patch" canvas mode and the
    (dest_pos, src_pos) map that injects the neighborhood's FULL-frame source
    activations into that small canvas.

    The small canvas is exactly the bounding box of the neighborhood in merged
    patches (e.g. 3x3 for a center patch, clipped at borders), sized
    (small_gw*28) x (small_gh*28) px so each merged patch is one 28px cell.
    Neighborhood merged patch at full-grid (r, c) maps to small-canvas raster
    slot (r-rmin, c-cmin); each maps 4 raw sub-patches (source frame -> small
    frame) via the respective window indices.
    """
    rc = [idx_to_rc(int(p), grid_w) for p in neighborhood]
    rows = [r for r, _ in rc]
    cols = [c for _, c in rc]
    rmin, rmax = min(rows), max(rows)
    cmin, cmax = min(cols), max(cols)
    small_gh = rmax - rmin + 1
    small_gw = cmax - cmin + 1

    canvas = make_baseline_canvas(
        small_gw * EFFECTIVE_PATCH, small_gh * EFFECTIVE_PATCH, baseline
    )
    small_prompt = prepare_yes_no_prompt_inputs(
        processor, canvas, question_prompt, system_prompt, device
    )
    s_gh, s_gw, _, _ = infer_merged_vision_grid(small_prompt)
    if (s_gh, s_gw) != (small_gh, small_gw):
        raise RuntimeError(
            f"3x3 canvas regridded to {s_gh}x{s_gw}, expected {small_gh}x{small_gw} "
            "(the processor may have resized a too-small canvas; increase the "
            "neighborhood or use --canvas-mode full)."
        )

    vision_module = get_vision_module(model)
    small_units = small_gh * small_gw
    small_raw_seq = small_units * spatial_merge_unit
    small_reverse = compute_reverse_window_index(
        vision_module, small_prompt["image_grid_thw"], small_units
    )

    position_map: List[Tuple[int, int]] = []
    for p, (r, c) in zip(neighborhood, rc):
        small_merged_j = (r - rmin) * small_gw + (c - cmin)
        src_raw = merged_patches_to_raw_positions(
            [int(p)], full_reverse_window_index, spatial_merge_unit, full_raw_seq_len
        )
        dst_raw = merged_patches_to_raw_positions(
            [small_merged_j], small_reverse, spatial_merge_unit, small_raw_seq
        )
        for k in range(min(len(src_raw), len(dst_raw))):
            position_map.append((dst_raw[k], src_raw[k]))

    return small_prompt, position_map


def _split_vision_hidden(hidden: torch.Tensor) -> Tuple[torch.Tensor, bool]:
    """Vision block hidden states are [S, C]; tolerate [1, S, C] too."""
    is_3d = hidden.ndim == 3
    seq = hidden[0] if is_3d else hidden  # [S, C]
    return seq, is_3d


@torch.inference_mode()
def collect_source_vision_block_inputs(
    model: nn.Module,
    source_prompt_batch: Dict[str, torch.Tensor],
    start_layer: int,
    end_layer: int,
) -> Dict[int, torch.Tensor]:
    """
    Cache SOURCE vision-encoder block INPUTS ([S, C] per block) over the block
    interval, using forward PRE-hooks (input == the hidden state entering block i).
    """
    blocks = get_vision_blocks(model)
    first = max(0, int(start_layer))
    last = int(end_layer)
    if last < 0:
        last = len(blocks) - 1
    last = min(last, len(blocks) - 1)
    if first > last:
        raise ValueError(
            f"Invalid vision-block interval [{start_layer}, {end_layer}] for "
            f"{len(blocks)} vision blocks."
        )

    captured: Dict[int, torch.Tensor] = {}
    handles = []

    def make_capture(layer_idx: int):
        def hook(_module, inputs):
            hidden = inputs[0]
            if isinstance(hidden, torch.Tensor):
                seq, _ = _split_vision_hidden(hidden)
                captured[layer_idx] = seq.detach().clone()
            return None
        return hook

    for layer_idx in range(first, last + 1):
        handles.append(blocks[layer_idx].register_forward_pre_hook(make_capture(layer_idx)))
    try:
        model(**source_prompt_batch, use_cache=False, return_dict=True)
    finally:
        for handle in handles:
            handle.remove()

    return captured


@contextmanager
def online_vision_patch_window(
    model: nn.Module,
    source_block_inputs: Dict[int, torch.Tensor],
    position_map: Sequence[Tuple[int, int]],
    start_layer: int,
    end_layer: int,
    inject_mode: str = "norm_preserve",
):
    """
    Inject cached SOURCE vision-block activations into a destination forward, over
    vision blocks [start_layer, end_layer].

    position_map is a list of (dest_pos, src_pos) pairs: the destination-sequence
    row ``dest_pos`` receives the source row ``src_pos`` (from the cached full
    source encode). For the full-canvas mode dest==src; for the 3x3-canvas mode
    the source rows come from the full frame while the destinations index the
    small canvas.

    inject_mode:
      "norm_preserve" -> source direction scaled to the destination norm.
      "raw"           -> exact source vector, no rescaling.
    """
    blocks = get_vision_blocks(model)
    first = max(0, int(start_layer))
    last = int(end_layer)
    if last < 0:
        last = len(blocks) - 1
    last = min(last, len(blocks) - 1)
    handles = []
    pairs = [(int(d), int(s)) for (d, s) in position_map]

    def make_hook(layer_idx: int):
        source_seq = source_block_inputs.get(layer_idx, None)

        def hook(_module, inputs):
            hidden = inputs[0]
            if not isinstance(hidden, torch.Tensor) or source_seq is None:
                return inputs
            seq, is_3d = _split_vision_hidden(hidden)
            if seq.ndim != 2:
                return inputs

            patched = seq.clone()
            src = source_seq.to(device=patched.device, dtype=patched.dtype)
            d_len = patched.size(0)
            s_len = src.size(0)
            for dest_pos, src_pos in pairs:
                if dest_pos >= d_len or src_pos >= s_len:
                    continue
                source_vector = src[src_pos]
                if inject_mode == "raw":
                    patched[dest_pos] = source_vector
                else:  # norm_preserve
                    target_norm = patched[dest_pos].norm().clamp_min(1e-6)
                    source_norm = source_vector.norm().clamp_min(1e-6)
                    patched[dest_pos] = source_vector * (target_norm / source_norm)

            new_hidden = patched.unsqueeze(0) if is_3d else patched
            return (new_hidden,) + tuple(inputs[1:])

        return hook

    for layer_idx in range(first, last + 1):
        handles.append(blocks[layer_idx].register_forward_pre_hook(make_hook(layer_idx)))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


# -----------------------------------------------------------------------------
# Yes-token probability
# -----------------------------------------------------------------------------
def format_yes_no_prompt(question_template: str, object_label: str) -> str:
    object_label = str(object_label).strip()
    if not object_label:
        raise ValueError("Cannot build a yes/no prompt because object_label is empty.")
    try:
        return question_template.format(object_label=object_label)
    except KeyError as error:
        raise KeyError(
            "--question-template must contain the placeholder {object_label}."
        ) from error


def normalize_yes_candidate(text: str) -> str:
    # Decode may return strings like " Yes", "yes.", "YES!", etc.
    # Stripping whitespace and punctuation lets us collect yes variants while
    # avoiding unrelated tokens such as "yesterday".
    extra_quotes = "“”‘’«»‹›„‚"
    return text.strip().strip(string.punctuation + extra_quotes).casefold()


def collect_yes_token_ids(tokenizer) -> Tuple[List[int], List[str]]:
    """Collect single-token ids whose decoded text is a yes variant."""
    yes_ids = set()

    # 1) Directly try common surface forms.
    for surface in YES_SURFACE_FORMS:
        token_ids = tokenizer.encode(surface, add_special_tokens=False)
        if len(token_ids) == 1:
            yes_ids.add(int(token_ids[0]))

    # 2) Scan the tokenizer vocabulary for decoded single-token yes variants.
    # This catches tokenizer-specific forms such as leading-space tokens.
    vocab = tokenizer.get_vocab()
    for token_id in sorted(set(int(x) for x in vocab.values())):
        try:
            decoded = tokenizer.decode(
                [token_id],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        except Exception:
            continue
        if normalize_yes_candidate(decoded) == "yes":
            yes_ids.add(token_id)

    yes_ids_sorted = sorted(yes_ids)
    yes_decoded = [
        tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        for token_id in yes_ids_sorted
    ]
    return yes_ids_sorted, yes_decoded


@torch.inference_mode()
def yes_token_probability(
    model: nn.Module,
    prompt_batch: Dict[str, torch.Tensor],
    yes_token_ids: Sequence[int],
) -> float:
    """Return summed next-token probability mass over all yes token ids."""
    validate_sequence_aligned_tensors(prompt_batch, "yes-token forward")
    outputs = model(
        **prompt_batch,
        use_cache=False,
        return_dict=True,
    )
    logits = outputs.logits[0, -1, :].float()

    valid_yes_ids = [
        int(token_id)
        for token_id in yes_token_ids
        if 0 <= int(token_id) < logits.numel()
    ]
    if not valid_yes_ids:
        raise ValueError("No yes token ids are inside the model vocabulary.")

    probabilities = torch.softmax(logits, dim=-1)
    yes_id_tensor = torch.tensor(
        valid_yes_ids,
        dtype=torch.long,
        device=probabilities.device,
    )
    return float(probabilities.index_select(0, yes_id_tensor).sum().item())


# -----------------------------------------------------------------------------
# One EAGLE sample
# -----------------------------------------------------------------------------
@torch.inference_mode()
def score_eagle_sample(
    processor,
    model: nn.Module,
    content: Dict[str, Any],
    image_path: str,
    start_layer: int,
    end_layer: int,
    neighbor_mode: str,
    include_center: bool,
    baseline: str,
    question_template: str,
    system_prompt: str,
    yes_token_ids: Sequence[int],
    yes_token_decoded: Sequence[str],
    self_check: bool = False,
    inject_mode: str = "norm_preserve",
    canvas_mode: str = "full",
    deletion_mode: str = "none",
    deletion_start_layer: int = 0,
    deletion_end_layer: int = 7,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if deletion_mode not in {"none", "input", "embedding"}:
        raise ValueError(
            "deletion_mode must be one of: none, input, embedding; "
            f"got {deletion_mode!r}."
        )

    device = model_input_device(model)
    tokenizer = processor.tokenizer

    object_label, raw_object_label, object_label_source = object_label_from_eagle_sample(
        content,
        tokenizer,
    )
    question_prompt = format_yes_no_prompt(question_template, object_label)

    source_image = Image.open(image_path).convert("RGB")
    source_prompt = prepare_yes_no_prompt_inputs(
        processor,
        source_image,
        question_prompt,
        system_prompt,
        device,
    )

    grid_h, grid_w, resized_h, resized_w = infer_merged_vision_grid(source_prompt)
    baseline_image = make_baseline_canvas(resized_w, resized_h, baseline)
    baseline_prompt = prepare_yes_no_prompt_inputs(
        processor,
        baseline_image,
        question_prompt,
        system_prompt,
        device,
    )

    expected_patch_count = grid_h * grid_w

    # Vision-encoder geometry: raw sequence length and merged->raw mapping.
    vision_module = get_vision_module(model)
    spatial_merge_unit = spatial_merge_unit_of(vision_module)
    grid_thw = source_prompt["image_grid_thw"]
    raw_seq_len = expected_patch_count * spatial_merge_unit
    reverse_window_index = compute_reverse_window_index(
        vision_module, grid_thw, expected_patch_count
    )

    original_yes_probability = yes_token_probability(
        model,
        source_prompt,
        yes_token_ids,
    )
    baseline_yes_probability = yes_token_probability(
        model,
        baseline_prompt,
        yes_token_ids,
    )

    # Cache SOURCE vision-encoder block inputs (all raw positions) once.
    source_block_inputs = collect_source_vision_block_inputs(
        model,
        source_prompt,
        start_layer,
        end_layer,
    )

    # Embedding deletion is the dual of insertion: insertion writes cached
    # SOURCE rows into a baseline forward, whereas deletion writes cached
    # BASELINE rows into a full-source forward. Cache the baseline once for the
    # independently configurable deletion interval.
    deletion_baseline_block_inputs: Dict[int, torch.Tensor] = {}
    if deletion_mode == "embedding":
        deletion_baseline_block_inputs = collect_source_vision_block_inputs(
            model,
            baseline_prompt,
            deletion_start_layer,
            deletion_end_layer,
        )

    # Optional structural self-check: patch the union of ALL patches; p(yes)
    # should move strongly toward the ORIGINAL image's p(yes) if the mapping is
    # correct. (Norm-preserving injection makes it approximate, not exact.)
    if self_check:
        all_positions = merged_patches_to_raw_positions(
            list(range(expected_patch_count)),
            reverse_window_index,
            spatial_merge_unit,
            raw_seq_len,
        )
        all_map = [(p, p) for p in all_positions]
        with online_vision_patch_window(
            model, source_block_inputs, all_map, start_layer, end_layer,
            inject_mode=inject_mode,
        ):
            full_patch_yes = yes_token_probability(model, baseline_prompt, yes_token_ids)
        print(
            "[self-check] patch-ALL p(yes)="
            f"{full_patch_yes:.6f} | original={original_yes_probability:.6f} | "
            f"baseline={baseline_yes_probability:.6f} "
            f"(window_index={'used' if reverse_window_index is not None else 'natural-order'}, "
            f"raw_seq_len={raw_seq_len}, spatial_merge_unit={spatial_merge_unit})"
        )

    output_token_ids = [int(x) for x in content.get("output_word_id", [])]
    full_output_text = (
        tokenizer.decode(
            output_token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if output_token_ids
        else ""
    )

    target_output_index = content.get("target_generated_index", "")
    target_token_id = content.get("target_generated_id", "")

    rows: List[Dict[str, Any]] = []
    for patch_index in tqdm(
        range(expected_patch_count),
        desc="Visual patches",
        leave=False,
    ):
        neighborhood = neighbor_patch_indices(
            patch_index,
            grid_h,
            grid_w,
            mode=neighbor_mode,
            include_center=include_center,
        )

        if canvas_mode == "patch":
            # 3x3-canvas mode: build a small canvas the size of the neighborhood
            # and inject the neighborhood's full-frame source activations into it.
            dest_prompt, position_map = build_patch_canvas_and_map(
                processor,
                model,
                neighborhood,
                grid_w,
                reverse_window_index,
                spatial_merge_unit,
                raw_seq_len,
                baseline,
                question_prompt,
                system_prompt,
                device,
            )
        else:
            # full-canvas mode (default): inject in place on the full baseline.
            raw_patch_positions = merged_patches_to_raw_positions(
                neighborhood,
                reverse_window_index,
                spatial_merge_unit,
                raw_seq_len,
            )
            position_map = [(p, p) for p in raw_patch_positions]
            dest_prompt = baseline_prompt

        with online_vision_patch_window(
            model,
            source_block_inputs,
            position_map,
            start_layer,
            end_layer,
            inject_mode=inject_mode,
        ):
            patched_yes_probability = yes_token_probability(
                model,
                dest_prompt,
                yes_token_ids,
            )

        # Keep the original encoder method's patched probability as insertion.
        insertion_score = float(patched_yes_probability)

        # Deletion always uses center + clipped 8-neighborhood (3x3), even when
        # insertion was configured with a different --neighbor-mode.
        deletion_neighborhood = deletion_neighborhood_indices(
            patch_index,
            grid_h,
            grid_w,
        )

        if deletion_mode == "none":
            deletion_score = float("nan")
            necessity_score = float("nan")
            attribution_score = insertion_score
            deletion_mechanism = "skipped"
        elif deletion_mode == "input":
            deletion_image = mask_patches_on_input_image(
                source_image,
                resized_w,
                resized_h,
                deletion_neighborhood,
                grid_w,
                baseline,
            )
            deletion_prompt = prepare_yes_no_prompt_inputs(
                processor,
                deletion_image,
                question_prompt,
                system_prompt,
                device,
            )
            deletion_grid_h, deletion_grid_w, _, _ = infer_merged_vision_grid(
                deletion_prompt
            )
            if (deletion_grid_h, deletion_grid_w) != (grid_h, grid_w):
                raise RuntimeError(
                    "Input-deletion image regridded to "
                    f"{deletion_grid_h}x{deletion_grid_w}, expected "
                    f"{grid_h}x{grid_w}."
                )
            deletion_score = float(
                yes_token_probability(model, deletion_prompt, yes_token_ids)
            )
            necessity_score = 1.0 - deletion_score
            attribution_score = insertion_score + necessity_score
            deletion_mechanism = "input_pixel_masking_3x3"
        else:  # deletion_mode == "embedding"
            deletion_raw_positions = merged_patches_to_raw_positions(
                deletion_neighborhood,
                reverse_window_index,
                spatial_merge_unit,
                raw_seq_len,
            )
            deletion_position_map = [
                (position, position) for position in deletion_raw_positions
            ]
            with online_vision_patch_window(
                model,
                deletion_baseline_block_inputs,
                deletion_position_map,
                deletion_start_layer,
                deletion_end_layer,
                inject_mode=inject_mode,
            ):
                deletion_score = float(
                    yes_token_probability(model, source_prompt, yes_token_ids)
                )
            necessity_score = 1.0 - deletion_score
            attribution_score = insertion_score + necessity_score
            deletion_mechanism = "baseline_activation_replacement_3x3"

        probability_gain = insertion_score - baseline_yes_probability
        denominator = original_yes_probability - baseline_yes_probability
        recovery_fraction = (
            probability_gain / denominator
            if abs(denominator) > 1e-12
            else float("nan")
        )

        patch_row, patch_col = idx_to_rc(patch_index, grid_w)
        rows.append(
            {
                "prompt": question_prompt,
                "image_path": content["image_path"],

                "original_yes_probability": float(original_yes_probability),
                "original_target_probability": float(original_yes_probability),

                "baseline_yes_probability": float(baseline_yes_probability),
                "baseline_target_probability": float(baseline_yes_probability),

                "patched_yes_probability": float(patched_yes_probability),
                "patched_target_probability": float(patched_yes_probability),

                # patched_yes_probability keeps its historical meaning and is
                # exactly the insertion component. Rank downstream by
                # attribution_score when deletion is enabled.
                "insertion_score": float(insertion_score),
                "deletion_score": float(deletion_score),
                "necessity_score": float(necessity_score),
                "attribution_score": float(attribution_score),
                "deletion_mode": deletion_mode,
                "deletion_mechanism": deletion_mechanism,
                "deletion_start_layer": (
                    int(deletion_start_layer)
                    if deletion_mode == "embedding"
                    else ""
                ),
                "deletion_end_layer": (
                    int(deletion_end_layer)
                    if deletion_mode == "embedding"
                    else ""
                ),
                "deletion_patch_indices": json.dumps(deletion_neighborhood),
                "num_deletion_patches": int(len(deletion_neighborhood)),

                "select_category": content.get("select_category", ""),
                "object_label": object_label,
                "object_label_raw": raw_object_label,
                "object_label_source": object_label_source,

                "yes_token_ids": json.dumps([int(x) for x in yes_token_ids]),
                "yes_token_decoded": json.dumps(
                    list(yes_token_decoded),
                    ensure_ascii=False,
                ),

                "target_generated_index": target_output_index,
                "target_token_id": int(target_token_id),
                "target_token_dataset": content.get("target_generated_token", ""),

                "full_output_text": full_output_text,

                "start_layer": int(start_layer),
                "end_layer": int(end_layer),
                "patch_site": "vision_encoder_blocks",
                "inject_mode": inject_mode,
                "canvas_mode": canvas_mode,
                "baseline": baseline,
                "neighbor_mode": neighbor_mode,
                "include_center": bool(include_center),

                "source_grid_h": int(grid_h),
                "source_grid_w": int(grid_w),
                "resized_height": int(resized_h),
                "resized_width": int(resized_w),

                "center_patch_index": int(patch_index),
                "center_patch_row": int(patch_row),
                "center_patch_col": int(patch_col),

                "patched_patch_indices": json.dumps(neighborhood),

                "yes_probability_gain_vs_baseline": float(probability_gain),
                "probability_gain_vs_baseline": float(probability_gain),

                "recovery_fraction": float(recovery_fraction),

                "patched_yes_log_probability": float(
                    math.log(max(patched_yes_probability, 1e-45))
                ),
                "patched_target_log_probability": float(
                    math.log(max(patched_yes_probability, 1e-45))
                ),
            }
        )

    # With deletion enabled, the combined insight+necessity objective drives the
    # effective rank. In none mode attribution_score == insertion_score, exactly
    # preserving the original method.
    rows_sorted = sorted(
        rows,
        key=lambda row: row["attribution_score"],
        reverse=True,
    )

    rank_by_patch = {
        int(row["center_patch_index"]): rank
        for rank, row in enumerate(rows_sorted, start=1)
    }

    insertion_rows_sorted = sorted(
        rows,
        key=lambda row: row["insertion_score"],
        reverse=True,
    )
    insertion_rank_by_patch = {
        int(row["center_patch_index"]): rank
        for rank, row in enumerate(insertion_rows_sorted, start=1)
    }

    for row in rows:
        patch_idx = int(row["center_patch_index"])
        rank = rank_by_patch[patch_idx]

        # Compatibility ranks follow the effective attribution objective.
        row["rank_by_yes_probability"] = rank
        row["rank_by_target_probability"] = rank
        row["rank_by_attribution_score"] = rank
        row["rank_by_insertion_only"] = insertion_rank_by_patch[patch_idx]

    sample_summary = {
        "image_path": content["image_path"],
        "select_category": content.get("select_category", ""),
        "object_label": object_label,
        "object_label_raw": raw_object_label,
        "object_label_source": object_label_source,

        # Prompt fields.
        "question_prompt": question_prompt,
        "user_prompt": question_prompt,
        "system_prompt": system_prompt,

        # Yes/no probabilities.
        "original_yes_probability": float(original_yes_probability),
        "baseline_yes_probability": float(baseline_yes_probability),
        "visual_dependence": float(
            original_yes_probability - baseline_yes_probability
        ),

        # Compatibility aliases. These are yes probabilities, but named this way
        # so downstream EAGLE-style evaluators can read the CSV.
        "original_target_probability": float(original_yes_probability),
        "baseline_target_probability": float(baseline_yes_probability),

        "number_of_visual_patches": int(expected_patch_count),
        "deletion_mode": deletion_mode,
        "deletion_start_layer": (
            int(deletion_start_layer) if deletion_mode == "embedding" else None
        ),
        "deletion_end_layer": (
            int(deletion_end_layer) if deletion_mode == "embedding" else None
        ),
    }

    return rows, sample_summary


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    ensure_parent_dir(args.output_csv)

    if args.overwrite and args.resume:
        raise ValueError("--overwrite and --resume cannot be used together")
    if args.overwrite and os.path.exists(args.output_csv):
        os.remove(args.output_csv)

    completed_indices = (
        prepare_resume_csv(args.output_csv) if args.resume else set()
    )

    if not os.path.exists(args.eval_list):
        raise FileNotFoundError(f"EAGLE eval list not found: {args.eval_list}")
    if not os.path.isdir(args.coco_root):
        raise NotADirectoryError(f"COCO image directory not found: {args.coco_root}")

    with open(args.eval_list, "r", encoding="utf-8") as file:
        contents = json.load(file)
    if not isinstance(contents, list):
        raise TypeError("EAGLE eval-list JSON must contain a list of samples.")

    end = None if args.end < 0 else args.end
    selected_contents = contents[args.begin:end]

    processor = AutoProcessor.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        use_fast=False,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
    )

    if torch.cuda.is_available():
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_id,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            device_map="auto",
        ).eval()
    else:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_id,
            torch_dtype=torch.float32,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        ).to("cpu").eval()

    yes_token_ids, yes_token_decoded = collect_yes_token_ids(processor.tokenizer)
    if not yes_token_ids:
        raise RuntimeError("Could not find any single-token yes variants.")
    print(
        "Using yes token ids: "
        f"{yes_token_ids} decoded as {yes_token_decoded}"
    )

    include_center = not args.exclude_center
    print(
        "Patch site: VISION ENCODER blocks "
        f"[{args.start_layer}, {args.end_layer if args.end_layer >= 0 else 'last'}], "
        f"neighbor_mode={args.neighbor_mode}, include_center={include_center}, "
        f"inject_mode={args.inject_mode}, canvas_mode={args.canvas_mode}, "
        f"deletion_mode={args.deletion_mode}"
    )
    if args.deletion_mode == "embedding":
        print(
            "Embedding deletion: replace the clipped 3x3 source window with "
            "baseline activations over vision blocks "
            f"[{args.deletion_start_layer}, "
            f"{args.deletion_end_layer if args.deletion_end_layer >= 0 else 'last'}]"
        )
    elif args.deletion_mode == "input":
        print(
            "Input deletion: paint the clipped 3x3 source window with the "
            f"{args.baseline} baseline color."
        )

    for local_index, content in enumerate(
        tqdm(selected_contents, desc="EAGLE object samples"),
        start=args.begin,
    ):
        if local_index in completed_indices:
            print(
                f"[resume] skipping complete sample={local_index}, "
                f"image={content.get('image_path')}"
            )
            continue

        image_path = os.path.join(args.coco_root, content["image_path"])
        if not os.path.exists(image_path):
            print(f"[skip] Missing image: {image_path}")
            continue

        try:
            rows, summary = score_eagle_sample(
                processor=processor,
                model=model,
                content=content,
                image_path=image_path,
                start_layer=args.start_layer,
                end_layer=args.end_layer,
                neighbor_mode=args.neighbor_mode,
                include_center=include_center,
                baseline=args.baseline,
                question_template=args.question_template,
                system_prompt=args.system_prompt,
                yes_token_ids=yes_token_ids,
                yes_token_decoded=yes_token_decoded,
                self_check=(args.self_check and local_index == args.begin),
                inject_mode=args.inject_mode,
                canvas_mode=args.canvas_mode,
                deletion_mode=args.deletion_mode,
                deletion_start_layer=args.deletion_start_layer,
                deletion_end_layer=args.deletion_end_layer,
            )
        except Exception as error:
            print(
                f"[error] sample_index={local_index}, image={content.get('image_path')}: "
                f"{type(error).__name__}: {error}"
            )
            continue

        for row in rows:
            row["dataset_sample_index"] = int(local_index)

        append_dict_rows_to_csv(args.output_csv, rows)

        print(
            "[done] "
            f"sample={local_index}, image={summary['image_path']}, "
            f"object_label={summary['object_label']!r} "
            f"from {summary['object_label_source']}={summary['object_label_raw']!r}, "
            f"p_yes_original={summary['original_yes_probability']:.6f}, "
            f"p_yes_baseline={summary['baseline_yes_probability']:.6f}, "
            f"visual_dependence={summary['visual_dependence']:.6f}"
        )

        del rows
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(
        f"Saved patch-level attribution to: {args.output_csv}\n"
        "Rank downstream by 'attribution_score'. "
        "'patched_yes_probability' remains insertion_score for compatibility."
    )


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
