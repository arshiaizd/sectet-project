# -*- coding: utf-8 -*-
"""
Point Game evaluation for COCO — direct port of EAGLE's official
evals/eval_point_game.py, extended to score our own attribution methods.

The evaluation BODY (segmentation_to_mask, point_game, point_game_mask, and the
EAGLE add_value attribution-map construction) is reproduced VERBATIM from the
official script the user supplied, so numbers match EAGLE exactly. Only the
saliency-map *source* is generalized, via --map-source:

  eagle       Score EAGLE's own outputs, exactly like the official script:
              iterate over the json/ files under --explanation-dir, load the
              matching npy/ S_set, rebuild the attribution map with add_value(),
              and read location/segmentation FROM EAGLE'S OWN json (not the
              dataset). This is byte-for-byte the official main() loop.

  patch       Score our patch-level CSV (our_method_first30.csv /
              our_method_insertion_online*.csv). Group rows by
              dataset_sample_index; rebuild a (source_grid_h x source_grid_w)
              score grid from center_patch_index + the score column; resize to
              the original image size (EAGLE-style bilinear); score with the
              SAME point_game / point_game_mask primitives. location and
              segmentation come from the dataset entry at that
              dataset_sample_index.

  superpixel  Score our superpixel-level CSV
              (our_method_superpixel_first30.csv, independent OR accumulative).
              Regenerate the SLICO superpixels at the image's ORIGINAL
              resolution from the division_number stored in the CSV (same
              region_size = sqrt(H*W/division_number) formula the attribution
              script used), fill each superpixel's mask with its score, and
              score with the SAME primitives. location/segmentation again come
              from the dataset entry.

For 'patch' and 'superpixel', the score COLUMN used to rank/colour is
auto-detected: 'attribution_score' if present (the accumulative superpixel
method's marginal-increment column — its patched_yes_probability is the
CUMULATIVE F(S_r) and would invert the map), otherwise
'patched_yes_probability'. Override with --score-column.

FAIRNESS FIX — unified Point Game criterion (--pg-criterion)
------------------------------------------------------------
The original argmax-pixel test is UNFAIR across methods. EAGLE's map paints a
whole superpixel with one constant value, so the argmax "point" is really an
entire blob and the hit succeeds if ANY pixel of the top superpixel touches the
box/mask. Our patch map is bilinearly interpolated to a single sharp peak, so
its argmax pixel must fall EXACTLY inside the region. Same nominal test, very
different leniency.

--pg-criterion centroid (default, the fair one) removes this asymmetry: for
every method it reduces the top-scoring REGION to a SINGLE representative point
and tests whether THAT point lies in the box/mask.
  * eagle / superpixel : the top-scoring superpixel -> centroid = mean (x, y)
                         over all its pixels.
  * patch              : the top-scoring patch -> its center pixel (which is the
                         mean of the patch's pixel coordinates).
This is identical in spirit for both families: "centre of mass of the winning
region", so neither method benefits from region size or interpolation shape.

--pg-criterion argmax reproduces the ORIGINAL EAGLE test verbatim (argmax pixel
inside region), for backward comparison.

--patch-mode {linear, centroid} (only for --map-source patch) picks how the
patch saliency is turned into its representative point:
  * linear   : the CURRENT behaviour — build the grid, resize to the original
               image with cv2.INTER_LINEAR, then take the argmax pixel (under
               --pg-criterion argmax) OR the centroid of the interpolated map's
               top region. Kept for backward comparison.
  * centroid : skip interpolation; take the single top-scoring patch and use its
               geometric center pixel as the point. This is the patch analogue
               of the superpixel centroid and the fair default.
(When --pg-criterion centroid and --patch-mode centroid are combined — the
recommended fair setting — the patch point is exactly the top patch's center.)

An EXCEPTED list at the top of this file names images to ignore entirely (their
GT box/mask does not match the target object); they are skipped in all sources.

Outputs: per-image CSV (--out) and a summary JSON (--out-summary) with
Point Game (Box) and Point Game (Mask) means, matching the two numbers the
official script prints.

Usage (see also HANDOFF_SUMMARY.md section 10):

  # EAGLE (identical to official eval_point_game.py)
  python eval_point_game_coco.py --map-source eagle \\
      --explanation-dir interpretation_results/Qwen2.5-VL-7B-coco-object/slico-1.0-1.0-division-number-64 \\
      --coco-root datasets/coco/val2017 \\
      --out ./pg_eagle.csv --out-summary ./pg_eagle.json

  # our patch-level method
  python eval_point_game_coco.py --map-source patch \\
      --csv ./auc_results_insertion \\
      --eval-list ./coco_single_target_once_qwen25vl-7B.json \\
      --coco-root datasets/coco/val2017 \\
      --out ./pg_patch_input.csv --out-summary ./pg_patch_input.json

  # our superpixel-level method (independent or accumulative)
  python eval_point_game_coco.py --map-source superpixel \\
      --csv ./our_method_superpixel_first30.csv \\
      --eval-list ./coco_single_target_once_qwen25vl-7B.json \\
      --coco-root datasets/coco/val2017 \\
      --out ./pg_superpixel.csv --out-summary ./pg_superpixel.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd


# =============================================================================
# Images to IGNORE during Point Game evaluation.
# =============================================================================
# List image file names (as they appear in the dataset / EAGLE json, e.g.
# "000000012345.jpg") whose ground-truth box/mask does NOT match the target
# object, so they would only add noise to the Point Game score. These are
# skipped in EVERY map-source (eagle / patch / superpixel) and are reported in
# the summary under "num_excepted_skipped". Fill this in manually.
EXCEPTED: List[str] = [
   "000000147740.jpg",
   "000000102411.jpg",
   "000000184324.jpg",
]


def _normalize_image_name(name: str) -> str:
    """Compare by basename so entries like 'val2017/xxx.jpg' still match."""
    return os.path.basename(str(name)).strip()


_EXCEPTED_SET = None


def is_excepted(image_name: str) -> bool:
    """True if image_name is in the EXCEPTED ignore-list (basename match)."""
    global _EXCEPTED_SET
    if _EXCEPTED_SET is None:
        _EXCEPTED_SET = {_normalize_image_name(n) for n in EXCEPTED}
    return _normalize_image_name(image_name) in _EXCEPTED_SET


# =============================================================================
# VERBATIM from EAGLE's evals/eval_point_game.py (the file the user supplied)
# =============================================================================
def segmentation_to_mask(segmentation, w, h):
    """
    Convert COCO-style segmentation(s) to a binary mask of shape (h, w).
    Supports single polygon [x1,y1,...], multiple polygons [[...],[...]], and RLE.
    Verbatim from EAGLE's eval_point_game.py.
    """
    mask = np.zeros((h, w), dtype=np.uint8)

    if segmentation is None:
        return mask

    if isinstance(segmentation, dict) and "counts" in segmentation:
        try:
            import pycocotools.mask as maskUtils
            rle = segmentation
            if not isinstance(rle["counts"], bytes):
                rle = maskUtils.frPyObjects([segmentation], h, w)[0]
            m = maskUtils.decode(rle)
            return (m.astype(np.uint8)).clip(0, 1)
        except Exception as e:
            raise ValueError(f"RLE segmentation provided but pycocotools decode failed: {e}")

    if isinstance(segmentation, (list, tuple, np.ndarray)) and len(segmentation) > 0:
        if isinstance(segmentation[0], (list, tuple, np.ndarray)):
            polygons = segmentation
        else:
            polygons = [segmentation]
    else:
        return mask

    for poly in polygons:
        if poly is None or len(poly) < 6:
            continue
        pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
        pts = np.round(pts).astype(np.int32)
        pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
        cv2.fillPoly(mask, [pts], 1)

    return mask


def point_game_mask(segmentation, saliency_map):
    """Verbatim from EAGLE. 1 iff the saliency max lies inside the segmentation."""
    h, w = saliency_map.shape
    empty = segmentation_to_mask(segmentation, w, h)

    mask_bbox = saliency_map * empty

    if mask_bbox.max() == saliency_map.max():
        return 1
    else:
        return 0


def point_game(bbox, saliency_map):
    """
    Verbatim from EAGLE (including the `w, h = saliency_map.shape` order, which
    EAGLE's own comment marks as "Not bug"). bbox is unpacked x1, y1, x2, y2 and
    passed straight from the json/dataset "location" field with no transform.
    """
    x1, y1, x2, y2 = bbox
    w, h = saliency_map.shape   # Not bug

    empty = np.zeros((w, h))
    empty[y1:y2, x1:x2] = 1
    mask_bbox = saliency_map * empty

    if mask_bbox.max() == saliency_map.max():
        return 1
    else:
        return 0


# =============================================================================
# FAIR criterion: reduce the top-scoring REGION to a single point, then test
# whether that point lies inside the box / segmentation. Used by
# --pg-criterion centroid for every method, so EAGLE superpixels and our
# patches/superpixels are judged identically ("centre of mass of the winner").
# =============================================================================
def point_game_box_point(bbox, point_xy) -> int:
    """1 iff point (x, y) lies inside bbox [x1, y1, x2, y2].

    Uses the SAME half-open convention as EAGLE's point_game (empty[y1:y2,
    x1:x2] = 1), i.e. x1 <= x < x2 and y1 <= y < y2, so the two criteria agree
    when the representative point coincides with the argmax pixel.
    """
    x1, y1, x2, y2 = bbox
    px, py = point_xy
    px_i = int(round(px))
    py_i = int(round(py))
    return int(x1 <= px_i < x2 and y1 <= py_i < y2)


def point_game_mask_point(segmentation, point_xy, w, h) -> int:
    """1 iff point (x, y) lies inside the segmentation mask of size (h, w)."""
    mask = segmentation_to_mask(segmentation, w, h)
    px_i = int(round(point_xy[0]))
    py_i = int(round(point_xy[1]))
    if 0 <= py_i < h and 0 <= px_i < w:
        return int(mask[py_i, px_i] > 0)
    return 0


def argmax_point_xy(saliency_map: np.ndarray) -> Tuple[float, float]:
    """(x, y) of the global-max pixel of a saliency map (row=y, col=x)."""
    idx = int(np.argmax(saliency_map))
    r, c = divmod(idx, saliency_map.shape[1])
    return float(c), float(r)


def top_region_centroid_from_labels(
    labels: np.ndarray,
    region_scores: Dict[int, float],
) -> Tuple[float, float]:
    """
    Given a per-pixel integer label map and a {label: score} dict, pick the
    highest-scoring label and return the centroid (mean x, mean y) over ALL of
    that region's pixels. This is EAGLE's / our superpixels' fair point.
    """
    if not region_scores:
        raise ValueError("empty region_scores")
    top_label = max(region_scores.items(), key=lambda kv: kv[1])[0]
    ys, xs = np.where(labels == int(top_label))
    if xs.size == 0:
        raise ValueError(f"top region label {top_label} has no pixels")
    return float(xs.mean()), float(ys.mean())


def top_patch_center_xy(
    scores_by_patch: Dict[int, float],
    grid_h: int,
    grid_w: int,
    orig_h: int,
    orig_w: int,
) -> Tuple[float, float]:
    """
    Centroid point for --map-source patch, --patch-mode centroid: take the
    top-scoring patch (grid cell) and return the center pixel of the image
    rectangle that cell maps to. The cell (r, c) covers original-image columns
    [c*cw, (c+1)*cw) and rows [r*rh, (r+1)*rh); its center is the mean of those
    pixel coordinates, i.e. the patch centroid.
    """
    if not scores_by_patch:
        raise ValueError("empty scores_by_patch")
    top_idx = max(scores_by_patch.items(), key=lambda kv: kv[1])[0]
    r, c = divmod(int(top_idx), grid_w)
    cell_w = orig_w / float(grid_w)
    cell_h = orig_h / float(grid_h)
    cx = (c + 0.5) * cell_w
    cy = (r + 0.5) * cell_h
    # clamp into the image just in case of rounding at the border
    cx = min(max(cx, 0.0), orig_w - 1.0)
    cy = min(max(cy, 0.0), orig_h - 1.0)
    return float(cx), float(cy)


# =============================================================================
# VERBATIM add_value from EAGLE's visualization/visualization.py
# (used only by --map-source eagle, exactly as the official script does)
# =============================================================================
def add_value(S_set, json_file):
    single_mask = np.zeros_like(S_set[0]).astype(np.float16)

    value_list_1 = np.array(json_file["smdl_score"])
    value_list_2 = np.array(
        [np.mean(1 - np.array(json_file["org_score"]) + np.array(json_file["baseline_score"]))]
        + json_file["smdl_score"][:-1]
    )
    value_list = value_list_1 - value_list_2

    values = []
    value = 0
    for smdl_single_mask, smdl_value in zip(S_set, value_list):
        value = value - abs(smdl_value)
        single_mask[smdl_single_mask == 1] = value
        values.append(value)

    attribution_map = single_mask - single_mask.min()
    attribution_map = attribution_map / attribution_map.max()
    return attribution_map, np.array(values)


# =============================================================================
# Saliency-map construction for OUR methods (patch / superpixel)
# =============================================================================
PATCH_COLUMN = "center_patch_index"
SUPERPIXEL_COLUMN = "superpixel_id"
ACCUMULATIVE_SCORE_COLUMN = "attribution_score"
GROUP_COLUMN = "dataset_sample_index"


def resolve_score_column(df: pd.DataFrame, override: str) -> str:
    """
    attribution_score if present (accumulative superpixel method's marginal
    increment dF -- its patched_yes_probability is the cumulative F(S_r) and
    would invert the map), else patched_yes_probability. --score-column wins.
    """
    if override:
        if override not in df.columns:
            raise KeyError(f"--score-column {override!r} not in CSV. "
                           f"Available: {sorted(df.columns)}")
        return override
    if ACCUMULATIVE_SCORE_COLUMN in df.columns:
        return ACCUMULATIVE_SCORE_COLUMN
    return "patched_yes_probability"


def saliency_from_patch_rows(df_sample: pd.DataFrame, orig_h: int, orig_w: int,
                             score_column: str):
    """
    Rebuild a (grid_h x grid_w) score grid from center_patch_index, resize to
    the ORIGINAL image size EAGLE-style (bilinear), and min-max normalize.

    Returns (saliency, grid_h, grid_w, scores_by_patch) so the caller can either
    take the interpolated map (linear mode) or the top patch's center (centroid
    mode) without recomputing.
    """
    grid_h = int(df_sample["source_grid_h"].iloc[0])
    grid_w = int(df_sample["source_grid_w"].iloc[0])
    scores = (df_sample.groupby(PATCH_COLUMN)[score_column].mean().to_dict())
    scores_by_patch = {int(k): float(v) for k, v in scores.items()}

    grid = np.zeros((grid_h, grid_w), dtype=np.float32)
    for patch_index, s in scores_by_patch.items():
        r, c = divmod(int(patch_index), grid_w)
        if 0 <= r < grid_h and 0 <= c < grid_w:
            grid[r, c] = float(s)

    sal = cv2.resize(grid, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    sal = sal - sal.min()
    sal = sal / (sal.max() + 1e-8)
    return sal, grid_h, grid_w, scores_by_patch


def saliency_from_superpixel_rows(df_sample: pd.DataFrame, image_bgr: np.ndarray,
                                  score_column: str):
    """
    Regenerate SLICO superpixels at ORIGINAL resolution from the CSV's
    division_number (same region_size = sqrt(H*W/division_number) formula the
    attribution script used), fill each superpixel's mask with its score, then
    min-max normalize. Point Game scores against original-resolution bbox /
    segmentation, so building the map directly at original resolution avoids a
    resize round-trip.

    Returns (saliency, labels, scores_by_superpixel) so the caller can take the
    filled map (argmax) or the top superpixel's centroid (centroid mode).
    """
    if not hasattr(cv2, "ximgproc"):
        raise RuntimeError("cv2.ximgproc is required (install opencv-contrib-python).")
    if "division_number" not in df_sample.columns:
        raise KeyError("superpixel CSV is missing 'division_number'.")

    division_number = int(df_sample["division_number"].iloc[0])
    h, w = image_bgr.shape[:2]
    region_size = max(int((h * w / division_number) ** 0.5), 1)
    slic = cv2.ximgproc.createSuperpixelSLIC(image_bgr, region_size=region_size, ruler=20.0)
    slic.iterate(20)
    labels = slic.getLabels()

    scores_by_superpixel: Dict[int, float] = {}
    sal = np.zeros((h, w), dtype=np.float32)
    for _, row in df_sample.iterrows():
        sp_id = int(row[SUPERPIXEL_COLUMN])
        scores_by_superpixel[sp_id] = float(row[score_column])
        m = labels == sp_id
        if m.any():
            sal[m] = float(row[score_column])

    sal = sal - sal.min()
    sal = sal / (sal.max() + 1e-8)
    return sal, labels, scores_by_superpixel


# =============================================================================
# Unified per-image scorer: given a representative POINT (centroid criterion) or
# a saliency map (argmax criterion), return (pg_box, pg_mask).
# =============================================================================
def score_point_game(
    pg_criterion: str,
    location,
    segmentation,
    orig_h: int,
    orig_w: int,
    point_xy: Optional[Tuple[float, float]] = None,
    saliency_map: Optional[np.ndarray] = None,
) -> Tuple[int, int]:
    """
    pg_criterion == 'centroid': use point_xy against box/mask (fair, unified).
    pg_criterion == 'argmax'  : use the ORIGINAL EAGLE argmax-pixel test on the
                                saliency_map (verbatim primitives).
    """
    if pg_criterion == "centroid":
        if point_xy is None:
            raise ValueError("centroid criterion requires point_xy")
        pg_box = point_game_box_point(location, point_xy)
        pg_mask = point_game_mask_point(segmentation, point_xy, orig_w, orig_h)
        return int(pg_box), int(pg_mask)
    # argmax (original EAGLE test)
    if saliency_map is None:
        raise ValueError("argmax criterion requires saliency_map")
    pg_box = point_game(location, saliency_map)
    pg_mask = point_game_mask(segmentation, saliency_map)
    return int(pg_box), int(pg_mask)


# =============================================================================
# Main — three sources, one scoring body
# =============================================================================
def run(args) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    pg_box_value: List[int] = []
    pg_seg_value: List[int] = []
    num_excepted = 0

    if args.map_source == "eagle":
        # ------ EXACTLY the official main() loop ------
        json_root = os.path.join(args.explanation_dir, "json")
        npy_root = os.path.join(args.explanation_dir, "npy")
        if not os.path.isdir(json_root):
            raise FileNotFoundError(f"No json/ under {args.explanation_dir}")
        json_names = sorted(f for f in os.listdir(json_root) if f.endswith(".json"))

        for json_name in json_names:
            try:
                image_name = json_name.replace(".json", ".jpg")
                if is_excepted(image_name):
                    num_excepted += 1
                    rows.append({"image_path": image_name, "pg_box": np.nan,
                                 "pg_mask": np.nan, "note": "excepted (ignored)"})
                    continue
                with open(os.path.join(json_root, json_name), encoding="utf-8") as f:
                    saved = json.load(f)
                image = cv2.imread(os.path.join(args.coco_root, image_name))
                if image is None:
                    rows.append({"image_path": image_name, "pg_box": np.nan,
                                 "pg_mask": np.nan, "note": "image not found"})
                    continue
                orig_h, orig_w = image.shape[:2]

                npy_path = os.path.join(npy_root, json_name.replace(".json", ".npy"))
                S_set = np.load(npy_path)
                attribution_map, values = add_value(S_set, saved)
                # Saliency at original resolution (used by the argmax criterion
                # and always emitted so both criteria see the same source map).
                attribution_map_full = cv2.resize(
                    attribution_map.astype(float), (orig_w, orig_h))

                point_xy = None
                if args.pg_criterion == "centroid":
                    # Fair point: centroid of the TOP-scoring superpixel. values[i]
                    # is the attribution assigned to S_set[i]; the top superpixel is
                    # argmax(values). Resize its mask to original resolution, then
                    # take the mean (x, y) over its pixels.
                    top_i = int(np.argmax(values))
                    top_mask = (S_set[top_i] > 0).astype(np.float32)
                    top_mask_full = cv2.resize(
                        top_mask, (orig_w, orig_h),
                        interpolation=cv2.INTER_NEAREST)
                    ys, xs = np.where(top_mask_full > 0)
                    if xs.size == 0:
                        # degenerate: fall back to argmax pixel of the map
                        point_xy = argmax_point_xy(attribution_map_full)
                    else:
                        point_xy = (float(xs.mean()), float(ys.mean()))

                pg_box, pg_seg = score_point_game(
                    args.pg_criterion, saved["location"], saved["segmentation"],
                    orig_h, orig_w, point_xy=point_xy,
                    saliency_map=attribution_map_full)
                pg_box_value.append(pg_box)
                pg_seg_value.append(pg_seg)
                rows.append({"image_path": image_name,
                             "select_category": saved.get("select_category", ""),
                             "pg_box": pg_box, "pg_mask": pg_seg, "note": ""})
            except Exception as e:
                print(f"[skip] {json_name}: {type(e).__name__}: {e}")
                continue

    elif args.map_source == "dense":
        # ------ dense TAM / LLaVA-CAM heatmaps ------
        with open(args.eval_list, encoding="utf-8") as f:
            dataset = json.load(f)
        npy_root = os.path.join(args.explanation_dir, "npy")
        selected = (dataset[args.begin:args.end] if args.end > 0
                    else dataset[args.begin:])

        for offset, content in enumerate(selected, start=args.begin):
            image_name = content["image_path"]
            try:
                if is_excepted(image_name):
                    num_excepted += 1
                    rows.append({"image_path": image_name,
                                 "dataset_sample_index": offset,
                                 "pg_box": np.nan, "pg_mask": np.nan,
                                 "note": "excepted (ignored)"})
                    continue
                image = cv2.imread(os.path.join(args.coco_root, image_name))
                if image is None:
                    raise FileNotFoundError(f"image not found: {image_name}")
                orig_h, orig_w = image.shape[:2]
                saliency = np.load(
                    os.path.join(npy_root, os.path.splitext(image_name)[0] + ".npy")
                )
                if saliency.ndim != 2 or not np.isfinite(saliency).all():
                    raise ValueError(f"invalid dense saliency map: {saliency.shape}")
                if saliency.shape != (orig_h, orig_w):
                    saliency = cv2.resize(
                        saliency.astype(float), (orig_w, orig_h),
                        interpolation=cv2.INTER_LINEAR,
                    )
                point_xy = (argmax_point_xy(saliency)
                            if args.pg_criterion == "centroid" else None)
                pg_box, pg_seg = score_point_game(
                    args.pg_criterion, content["location"], content["segmentation"],
                    orig_h, orig_w, point_xy=point_xy, saliency_map=saliency
                )
                pg_box_value.append(pg_box)
                pg_seg_value.append(pg_seg)
                rows.append({"image_path": image_name,
                             "dataset_sample_index": offset,
                             "select_category": content.get("select_category", ""),
                             "pg_box": pg_box, "pg_mask": pg_seg, "note": ""})
            except Exception as error:
                print(f"[skip] {image_name}: {type(error).__name__}: {error}")
                rows.append({"image_path": image_name,
                             "dataset_sample_index": offset,
                             "pg_box": np.nan, "pg_mask": np.nan,
                             "note": str(error)})

    else:
        # ------ our patch / superpixel CSV ------
        with open(args.eval_list, encoding="utf-8") as f:
            dataset = json.load(f)

        df = pd.concat(
            [pd.read_csv(csv_path) for csv_path in args.csv],
            ignore_index=True,
        )
        if GROUP_COLUMN not in df.columns:
            raise KeyError(f"CSV missing {GROUP_COLUMN!r}, needed to match rows "
                           f"to their dataset entry (location/segmentation).")
        score_column = resolve_score_column(df, args.score_column.strip())
        print(f"map_source={args.map_source}  score_column={score_column!r}")

        sample_indices = sorted(int(x) for x in df[GROUP_COLUMN].unique())
        sel = (sample_indices[args.begin:args.end] if args.end > 0
               else sample_indices[args.begin:])

        for sample_index in sel:
            try:
                if not (0 <= sample_index < len(dataset)):
                    rows.append({"image_path": "", "pg_box": np.nan, "pg_mask": np.nan,
                                 "note": f"index {sample_index} out of dataset range"})
                    continue
                content = dataset[sample_index]
                image_name = content["image_path"]
                if is_excepted(image_name):
                    num_excepted += 1
                    rows.append({"image_path": image_name,
                                 "dataset_sample_index": sample_index,
                                 "pg_box": np.nan, "pg_mask": np.nan,
                                 "note": "excepted (ignored)"})
                    continue
                image = cv2.imread(os.path.join(args.coco_root, image_name))
                if image is None:
                    rows.append({"image_path": image_name, "pg_box": np.nan,
                                 "pg_mask": np.nan, "note": "image not found"})
                    continue
                orig_h, orig_w = image.shape[:2]

                df_sample = df[df[GROUP_COLUMN] == sample_index]
                if len(df_sample) == 0:
                    rows.append({"image_path": image_name, "pg_box": np.nan,
                                 "pg_mask": np.nan, "note": "no rows for sample"})
                    continue

                point_xy = None
                if args.map_source == "patch":
                    saliency, grid_h, grid_w, scores_by_patch = \
                        saliency_from_patch_rows(df_sample, orig_h, orig_w, score_column)
                    if args.pg_criterion == "centroid":
                        if args.patch_mode == "centroid":
                            # Fair patch point: center of the top-scoring patch.
                            point_xy = top_patch_center_xy(
                                scores_by_patch, grid_h, grid_w, orig_h, orig_w)
                        else:  # patch_mode == "linear"
                            # Centroid of the interpolated map's top region is not
                            # well-defined (single peak), so use the interpolated
                            # argmax pixel as the representative point.
                            point_xy = argmax_point_xy(saliency)
                else:  # superpixel
                    saliency, labels, scores_by_sp = \
                        saliency_from_superpixel_rows(df_sample, image, score_column)
                    if args.pg_criterion == "centroid":
                        point_xy = top_region_centroid_from_labels(labels, scores_by_sp)

                pg_box, pg_seg = score_point_game(
                    args.pg_criterion, content["location"], content["segmentation"],
                    orig_h, orig_w, point_xy=point_xy, saliency_map=saliency)
                pg_box_value.append(pg_box)
                pg_seg_value.append(pg_seg)
                rows.append({"image_path": image_name,
                             "dataset_sample_index": sample_index,
                             "select_category": content.get("select_category", ""),
                             "pg_box": pg_box, "pg_mask": pg_seg, "note": ""})
            except Exception as e:
                print(f"[skip] sample {sample_index}: {type(e).__name__}: {e}")
                continue

    box_mean = float(np.array(pg_box_value).mean()) if pg_box_value else float("nan")
    mask_mean = float(np.array(pg_seg_value).mean()) if pg_seg_value else float("nan")

    summary = {
        "map_source": args.map_source,
        "pg_criterion": args.pg_criterion,
        "patch_mode": (args.patch_mode if args.map_source == "patch" else None),
        "num_scored": len(pg_box_value),
        "num_excepted_skipped": num_excepted,
        "num_excepted_in_list": len(EXCEPTED),
        "point_game_box_mean": box_mean,
        "point_game_mask_mean": mask_mean,
    }

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        pd.DataFrame(rows).to_csv(args.out, index=False)
    if args.out_summary:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_summary)) or ".", exist_ok=True)
        with open(args.out_summary, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    # Print the exact two numbers the official script prints.
    print(f"[pg_criterion={args.pg_criterion}"
          + (f", patch_mode={args.patch_mode}" if args.map_source == "patch" else "")
          + (f", excepted_skipped={num_excepted}" if num_excepted else "")
          + "]")
    print("Point Game (Box): {}".format(box_mean))
    print("Point Game (Mask): {}".format(mask_mean))
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EAGLE-faithful Point Game on COCO "
                                            "(EAGLE / our-patch / our-superpixel).")
    p.add_argument("--map-source", choices=["eagle", "dense", "patch", "superpixel"], required=True)

    p.add_argument("--pg-criterion", choices=["centroid", "argmax"], default="centroid",
                   help="Point Game criterion. 'centroid' (default, FAIR): reduce "
                        "the top-scoring region to a single representative point "
                        "(superpixel centroid for eagle/superpixel; top-patch "
                        "center for patch) and test that point against box/mask, "
                        "so all methods are judged identically. 'argmax': the "
                        "ORIGINAL EAGLE test (global-max pixel inside region).")
    p.add_argument("--patch-mode", choices=["linear", "centroid"], default="centroid",
                   help="Only for --map-source patch. 'linear': current behaviour "
                        "(grid -> cv2.INTER_LINEAR resize; representative point is "
                        "the interpolated argmax pixel under --pg-criterion "
                        "centroid). 'centroid' (default): use the top-scoring "
                        "patch's geometric center as the point (no interpolation), "
                        "the patch analogue of the superpixel centroid.")

    # our-method inputs
    p.add_argument("--csv", type=str, nargs="+", default=None,
                   help="One or more our_method CSVs (required for "
                        "--map-source patch/superpixel).")
    p.add_argument("--eval-list", type=str, default=None,
                   help="dataset json (required for patch/superpixel) — supplies "
                        "location/segmentation by dataset_sample_index.")
    p.add_argument("--score-column", type=str, default="",
                   help="Override the ranking column. Default: attribution_score "
                        "if present (accumulative superpixel), else "
                        "patched_yes_probability.")

    # eagle inputs
    p.add_argument("--explanation-dir", type=str, default=None,
                   help="EAGLE slico-... dir with json/ and npy/ "
                        "(required for --map-source eagle).")

    p.add_argument("--coco-root", type=str, required=True)
    p.add_argument("--begin", type=int, default=0)
    p.add_argument("--end", type=int, default=-1)
    p.add_argument("--out", type=str, default="./point_game_per_image.csv")
    p.add_argument("--out-summary", type=str, default="./point_game_summary.json")
    args = p.parse_args()

    if args.map_source in ("eagle", "dense") and not args.explanation_dir:
        p.error(f"--map-source {args.map_source} requires --explanation-dir")
    if args.map_source == "dense" and not args.eval_list:
        p.error("--map-source dense requires --eval-list")
    if args.map_source in ("patch", "superpixel"):
        if not args.csv:
            p.error(f"--map-source {args.map_source} requires --csv")
        if not args.eval_list:
            p.error(f"--map-source {args.map_source} requires --eval-list")
    return args


if __name__ == "__main__":
    run(parse_args())
