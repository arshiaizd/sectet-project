# -*- coding: utf-8 -*-
"""
EAGLE-style insight+necessity attribution at the PATCH level, computed entirely
by INPUT (pixel) manipulation, non-accumulative, with a yes/no prompt.

This is the EAGLE objective (insertion + (1 - deletion)), but:
  * scored on 28px Qwen visual PATCHES instead of superpixels,
  * every patch scored INDEPENDENTLY (no accumulative greedy search),
  * both halves done as INPUT edits + a fresh forward (no activation patching),
  * the scoring target is p(yes) to a yes/no object-presence prompt
    ("Is there a <object> in the image ...? yes/no") rather than EAGLE's
    teacher-forced caption token.

For every EAGLE sample and every visual patch, using a 3x3 patch window
(--neighbor-radius, default 1) around the patch:

  insertion_score  (INPUT-level "insight")
    Start from the empty baseline canvas (black/white) and paint back ONLY the
    real image pixels inside the 3x3 window; everything else stays baseline.
    Run a forward, read p(yes). This is EAGLE's MLLM(alpha * source): reveal only
    the region on an empty canvas. High == the region alone is sufficient
    evidence of the object.

  deletion_score   (INPUT-level "necessity" probe)
    Start from the real image and paint the 3x3 window with the baseline color;
    everything else stays real. Run a forward, read p(yes). This is EAGLE's
    MLLM((1 - alpha) * source): reveal everything EXCEPT the region. Low ==
    removing the region destroys the object evidence (the region is necessary).

Combined attribution score (EAGLE's insight + necessity):

    attribution_score = insertion_score + (1 - deletion_score)

insertion enters with a + sign and deletion with a - sign. High for a patch that
is BOTH sufficient on its own (high insertion) AND necessary (low deletion ->
high 1-deletion). Patches are ranked by ``attribution_score`` (descending); each
patch's score is an independent scalar, so a plain sort is the ranking (no greedy
ordering). ``patched_yes_probability`` keeps its old meaning (== insertion_score)
for backward compatibility.

There is NO activation patching anywhere in this script: both halves are pure
input-image edits (reveal_patches_on_baseline / mask_patches_on_input_image)
followed by an ordinary vision-tower + LLM forward.

Ranking mode (--score-mode)
---------------------------
  both (default)
    Run BOTH halves. attribution_score comes from --combine-mode (insertion
    combined with the necessity term 1-deletion) and drives the ranking.
  insertion_only
    Rank by the input-level insertion (insight) score ALONE. attribution_score
    is set == insertion_score, the deletion forward is SKIPPED entirely (~half
    the forwards saved), and deletion_score / necessity_score are written as NaN
    with deletion_mechanism='skipped'. --combine-mode is ignored in this mode.

Downstream: rank by ``--score-column attribution_score`` (in both modes this is
the correct ranking column; in insertion_only it equals insertion_score, in both
it equals the combined objective). ``patched_yes_probability`` always keeps its
old meaning (== insertion_score) for backward compatibility.
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
DEFAULT_OUTPUT_CSV = "./our_method_insertion_deletion_input_online.csv"

# Chebyshev radius of the patch window used for BOTH insertion and deletion:
# radius r -> a (2r+1)x(2r+1) square. r=1 == 3x3.
DEFAULT_NEIGHBOR_RADIUS = 1

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

    parser.add_argument(
        "--neighbor-radius",
        type=int,
        default=DEFAULT_NEIGHBOR_RADIUS,
        help=(
            "Chebyshev radius of the square patch window used for BOTH insertion "
            "and deletion (center always included). radius r -> (2r+1)x(2r+1) "
            f"patches. Default {DEFAULT_NEIGHBOR_RADIUS} (3x3)."
        ),
    )
    parser.add_argument(
        "--score-mode",
        type=str,
        default="both",
        choices=["both", "insertion_only"],
        help=(
            "Which halves drive the ranking / attribution_score. "
            "'both' (default): run BOTH insertion and deletion and set "
            "attribution_score from --combine-mode (insertion combined with the "
            "necessity term 1-deletion). 'insertion_only': rank by the "
            "input-level insertion (insight) score ALONE -- attribution_score is "
            "set equal to insertion_score, the deletion forward is SKIPPED (about "
            "half the forwards saved), and deletion_score / necessity_score are "
            "written as NaN with deletion_mechanism='skipped'. In insertion_only "
            "mode --combine-mode has no effect."
        ),
    )
    parser.add_argument(
        "--combine-mode",
        type=str,
        default="insertion_plus_necessity",
        choices=["insertion_plus_necessity", "insertion_minus_necessity"],
        help=(
            "How to combine the two halves into attribution_score (only used when "
            "--score-mode both). "
            "'insertion_plus_necessity' (default): "
            "insertion_score + (1 - deletion_score) -- EAGLE's insight+necessity; "
            "insertion has a + sign and deletion a - sign (opposite signs). "
            "'insertion_minus_necessity': insertion_score - (1 - deletion_score). "
            "Both are computed from the same forwards, so switching needs no "
            "re-run -- every component is in the CSV."
        ),
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default="white",
        choices=["black", "white"],
        help=(
            "Baseline color used both as the empty insertion canvas and as the "
            "deletion fill color."
        ),
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


def square_radius_neighbor_indices(
    idx: int,
    grid_h: int,
    grid_w: int,
    radius: int,
    include_center: bool = True,
) -> List[int]:
    """
    All patch indices within Chebyshev distance `radius` of `idx` (a square
    window of side 2*radius+1), clipped to the grid. radius=0 -> just the center;
    radius=1 -> the same 3x3 window as neighbor_patch_indices(mode="square").

    Used for the (larger) DELETION neighborhood. The center is included by
    default because deletion is about removing the patch itself.
    """
    row, col = idx_to_rc(idx, grid_w)
    r = max(0, int(radius))

    output: List[int] = []
    seen = set()
    for dr in range(-r, r + 1):
        for dc in range(-r, r + 1):
            if dr == 0 and dc == 0 and not include_center:
                continue
            rr, cc = row + dr, col + dc
            if 0 <= rr < grid_h and 0 <= cc < grid_w:
                patch_idx = rc_to_idx(rr, cc, grid_w)
                if patch_idx not in seen:
                    seen.add(patch_idx)
                    output.append(patch_idx)
    return output


def make_baseline_canvas(width: int, height: int, baseline: str) -> Image.Image:
    color = (0, 0, 0) if baseline == "black" else (255, 255, 255)
    return Image.new("RGB", (width, height), color=color)


def mask_patches_on_input_image(
    source_image: Image.Image,
    resized_w: int,
    resized_h: int,
    patch_indices: Sequence[int],
    grid_w: int,
    baseline: str,
    patch_px: int = EFFECTIVE_PATCH,
) -> Image.Image:
    """
    INPUT-level deletion: return the processor-resized source image with the
    given patch cells painted over with the baseline color.

    This is a genuine pixel manipulation of the model input (as opposed to the
    activation-level insertion). The image is first resized to the exact
    processor dimensions (resized_w x resized_h, both multiples of patch_px), so
    each patch index maps to an exact patch_px x patch_px cell that lines up with
    the merged visual-token grid. Re-processing this image reproduces the same
    grid (Qwen's smart_resize is idempotent on already-valid sizes).
    """
    import numpy as np

    resized = source_image.convert("RGB").resize(
        (int(resized_w), int(resized_h)), Image.BICUBIC
    )
    arr = np.array(resized)
    fill = 0 if baseline == "black" else 255
    for idx in patch_indices:
        r, c = idx_to_rc(int(idx), grid_w)
        y0, y1 = r * patch_px, (r + 1) * patch_px
        x0, x1 = c * patch_px, (c + 1) * patch_px
        arr[y0:y1, x0:x1, :] = fill
    return Image.fromarray(arr)


def reveal_patches_on_baseline(
    source_image: Image.Image,
    resized_w: int,
    resized_h: int,
    patch_indices: Sequence[int],
    grid_w: int,
    baseline: str,
    patch_px: int = EFFECTIVE_PATCH,
) -> Image.Image:
    """
    INPUT-level insertion: return an image that is the baseline canvas everywhere
    EXCEPT the given patch cells, which show the real (resized) source pixels.

    This is EAGLE's MLLM(alpha * source): reveal only the selected region on an
    otherwise empty canvas. It is the pixel-level dual of
    ``mask_patches_on_input_image``.
    """
    import numpy as np

    resized = source_image.convert("RGB").resize(
        (int(resized_w), int(resized_h)), Image.BICUBIC
    )
    src_arr = np.array(resized)
    fill = 0 if baseline == "black" else 255
    out = np.full_like(src_arr, fill)
    for idx in patch_indices:
        r, c = idx_to_rc(int(idx), grid_w)
        y0, y1 = r * patch_px, (r + 1) * patch_px
        x0, x1 = c * patch_px, (c + 1) * patch_px
        out[y0:y1, x0:x1, :] = src_arr[y0:y1, x0:x1, :]
    return Image.fromarray(out)


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
    neighbor_radius: int,
    baseline: str,
    question_template: str,
    system_prompt: str,
    yes_token_ids: Sequence[int],
    yes_token_decoded: Sequence[str],
    combine_mode: str,
    score_mode: str = "both",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
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

    # Both halves are INPUT-level pixel edits + a fresh forward; there is no
    # activation caching or patching in this script.

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
        # Shared square window (default 3x3) used for BOTH insertion and deletion.
        neighborhood = square_radius_neighbor_indices(
            patch_index,
            grid_h,
            grid_w,
            radius=neighbor_radius,
            include_center=True,
        )

        # --- Insertion half: INPUT-LEVEL "insight" -------------------------
        # Reveal ONLY this window's real pixels on the empty baseline canvas,
        # then run a fresh forward. == EAGLE's MLLM(alpha * source).
        insertion_image = reveal_patches_on_baseline(
            source_image,
            resized_w,
            resized_h,
            neighborhood,
            grid_w,
            baseline,
        )
        insertion_prompt = prepare_yes_no_prompt_inputs(
            processor,
            insertion_image,
            question_prompt,
            system_prompt,
            device,
        )
        ins_grid_h, ins_grid_w, _, _ = infer_merged_vision_grid(insertion_prompt)
        if (ins_grid_h, ins_grid_w) != (grid_h, grid_w):
            raise RuntimeError(
                f"Insertion image regridded to {ins_grid_h}x{ins_grid_w}, "
                f"expected {grid_h}x{grid_w}."
            )
        insertion_score = float(
            yes_token_probability(model, insertion_prompt, yes_token_ids)
        )

        # --- Deletion half: INPUT-LEVEL "necessity" probe ------------------
        # Paint this window with the baseline color on the real image, then run
        # a fresh forward. == EAGLE's MLLM((1 - alpha) * source).
        # In --score-mode insertion_only this half is SKIPPED entirely (it never
        # feeds the ranking), saving one forward per patch. The deletion columns
        # are then written as NaN so no one mistakes a skipped probe for a real 0.
        if score_mode == "insertion_only":
            deletion_score = float("nan")
            necessity_score = float("nan")
            attribution_score = insertion_score
            deletion_mechanism = "skipped"
        else:
            deletion_image = mask_patches_on_input_image(
                source_image,
                resized_w,
                resized_h,
                neighborhood,
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
            del_grid_h, del_grid_w, _, _ = infer_merged_vision_grid(deletion_prompt)
            if (del_grid_h, del_grid_w) != (grid_h, grid_w):
                raise RuntimeError(
                    f"Deletion image regridded to {del_grid_h}x{del_grid_w}, "
                    f"expected {grid_h}x{grid_w}."
                )
            deletion_score = float(
                yes_token_probability(model, deletion_prompt, yes_token_ids)
            )

            # necessity = 1 - deletion_score (EAGLE's necessity term).
            necessity_score = 1.0 - deletion_score
            if combine_mode == "insertion_minus_necessity":
                attribution_score = insertion_score - necessity_score
            else:  # "insertion_plus_necessity" (default): insight + necessity
                attribution_score = insertion_score + necessity_score
            deletion_mechanism = "input_pixel_masking"

        # patched_yes_probability keeps its old meaning == insertion_score.
        patched_yes_probability = insertion_score

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

                # Insertion / deletion / combined attribution (both input-level).
                "insertion_score": float(insertion_score),
                "deletion_score": float(deletion_score),
                "necessity_score": float(necessity_score),
                "attribution_score": float(attribution_score),
                "score_mode": score_mode,
                "combine_mode": (
                    combine_mode if score_mode == "both" else "n/a_insertion_only"
                ),
                "neighbor_radius": int(neighbor_radius),
                "insertion_mechanism": "input_pixel_reveal",
                "deletion_mechanism": deletion_mechanism,
                "window_patch_indices": json.dumps(neighborhood),
                "num_window_patches": int(len(neighborhood)),

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

                "baseline": baseline,

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

    # Rank by the combined attribution score (descending). Each patch's score is
    # an independent scalar, so a plain sort is the correct ranking (no greedy
    # ordering is involved here).
    rows_sorted = sorted(
        rows,
        key=lambda row: row["attribution_score"],
        reverse=True,
    )
    rank_by_patch = {
        int(row["center_patch_index"]): rank
        for rank, row in enumerate(rows_sorted, start=1)
    }

    # Reference: the base method's insertion-only ranking.
    insertion_sorted = sorted(
        rows,
        key=lambda row: row["insertion_score"],
        reverse=True,
    )
    insertion_rank_by_patch = {
        int(row["center_patch_index"]): rank
        for rank, row in enumerate(insertion_sorted, start=1)
    }

    for row in rows:
        patch_idx = int(row["center_patch_index"])
        rank = rank_by_patch[patch_idx]

        # Effective attribution rank == rank by combined attribution_score.
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

        # Combined-scoring configuration.
        "neighbor_radius": int(neighbor_radius),
        "score_mode": score_mode,
        "combine_mode": (
            combine_mode if score_mode == "both" else "n/a_insertion_only"
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

    if args.score_mode == "insertion_only":
        print(
            "EAGLE-style patch attribution (all INPUT-level, non-accumulative): "
            f"INSERTION-ONLY ranking, window radius={args.neighbor_radius} "
            f"({2*args.neighbor_radius+1}x{2*args.neighbor_radius+1} patches), "
            f"baseline={args.baseline} (deletion half skipped)"
        )
    else:
        print(
            "EAGLE-style patch attribution (all INPUT-level, non-accumulative): "
            f"insertion+deletion window radius={args.neighbor_radius} "
            f"({2*args.neighbor_radius+1}x{2*args.neighbor_radius+1} patches), "
            f"combine_mode={args.combine_mode}, baseline={args.baseline}"
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
                neighbor_radius=args.neighbor_radius,
                baseline=args.baseline,
                question_template=args.question_template,
                system_prompt=args.system_prompt,
                yes_token_ids=yes_token_ids,
                yes_token_decoded=yes_token_decoded,
                combine_mode=args.combine_mode,
                score_mode=args.score_mode,
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
        "Saved patch-level insertion+deletion attribution to: "
        f"{args.output_csv}\n"
        "Rank downstream by the 'attribution_score' column "
        "(patched_yes_probability == insertion_score only)."
    )


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
