# Qwen2.5-VL visual attribution benchmark

This repository evaluates visual-attribution methods for Qwen2.5-VL-7B on a
shared COCO 2017 validation benchmark. It contains the current activation-
patching method, an input insertion/deletion baseline, EAGLE, and additional
EAGLE baselines.

Model weights, COCO images, Conda environments, and generated experiment
outputs are deliberately not tracked.

## Canonical data

- `shared/coco_mask_tail_30_benchmark.json`: 30-case pilot.
- `shared/coco_mask_tail_250_benchmark.json`: canonical 250-case benchmark.
- `shared/eval_point_game_coco.py`: shared centroid-based Point Game evaluator.

The earlier EAGLE-baseline experiment manifest and its four worker shards are
also retained under `shared/` because TAM, LLaVA-CAM, and IGOS++ launchers still
reference them.

Each record contains the image ID, audited COCO caption and target phrase, COCO
bounding box, and segmentation mask. Every method must use these shared files;
method-specific copies should not be generated.

## Repository layout

- `ours/`: current vision-encoder activation-patching method. The discarded
  legacy implementation is not retained.
- `input_deletion/`: independent input-space insertion + deletion baseline.
- `eagle/`: EAGLE runner and evaluation utilities.
- `tam/`, `llavacam/`, `igos_pp/`: additional baseline integrations.
- `shared/`: canonical manifests and shared evaluation/data utilities.
- `reports/`: report-generation source code; generated reports are ignored.

## Environment

The EPFL RunAI jobs use:

```text
/mnt/vilab/scratch/arshia/conda-envs/izadi
```

The verified environment uses Python 3.11, PyTorch 2.5.1+cu124,
Transformers 4.49.0, and OpenCV contrib 4.11. Install a CUDA-compatible
PyTorch build first, then:

```bash
python -m pip install -r requirements.txt
```

Expected local assets on the cluster:

```text
/mnt/vilab/scratch/arshia/models/Qwen2.5-VL-7B-Instruct
/mnt/vilab/scratch/arshia/datasets/coco/val2017
```

## Portable 250-image launchers

For clone-and-run commands covering EAGLE, our method, the input-level
baseline, TAM, LLaVA-CAM, and IGOS++ on one to four GPUs, see
[`PORTABLE_RUNS.md`](PORTABLE_RUNS.md). Every launcher takes the COCO
location as its required argument and uses the same audited 250-case manifest.

## Current methods

### Activation patching (`ours/`)

The current method injects source vision-encoder activations into a white
baseline and scores summed yes-token probability with:

```text
Is there a <object> in the image or not? Answer with exactly one word: yes or no.
```

### Input insertion/deletion (`input_deletion/`)

The new baseline reveals a patch neighborhood on an empty image and removes the
same neighborhood from the source image. Its default attribution is:

```text
insertion_score + (1 - deletion_score)
```

Its uploaded prompt and scoring defaults are preserved as-is. See
`input_deletion/README.md` for the exact configuration.

## Evaluation

Patch methods use eight merged vision patches per faithfulness step. Evaluation
launchers produce:

- per-image insertion/deletion curves;
- aggregate AUC metrics;
- Point Game box and segmentation-mask metrics using the shared centroid
  evaluator.

Generated outputs are stored below each method's `results/` directory and are
ignored by Git.

## Upstream provenance

The EAGLE integrations originate from:

```text
https://github.com/RuoyuChen10/EAGLE
commit 3cf4f656e09a0ee41f71dcf18aa6bcac47217f64
```

Untouched upstream snapshots are retained under baseline `official_source/`
directories where available. See `BASELINES_SETUP.md` for infrastructure-only
adaptations. Upstream code remains subject to its original licensing terms.

Before making this repository public, select a license for original project
code and confirm that redistributed upstream snapshots comply with their
licenses.
