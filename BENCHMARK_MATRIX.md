# Two-model, two-dataset experiment matrix

This is the canonical run/status guide for the five retained methods:
`ours`, `input_level`, `eagle`, `tam`, and `llavacam`. IGOS++ is not part of
this matrix. The checkpoints are `Qwen/Qwen2.5-VL-7B-Instruct` and
`OpenGVLab/InternVL3_5-8B-HF`.

## Implementation status

| Model | Dataset | Ours | Input-level | EAGLE | TAM | LLaVA-CAM |
|---|---|---:|---:|---:|---:|---:|
| Qwen2.5-VL-7B | COCO mask-tail-150 | completed | completed | runnable | completed | completed |
| Qwen2.5-VL-7B | MMVP forced-choice-150 | runnable | runnable | runnable | runnable | runnable |
| InternVL3.5-8B-HF | COCO-250 | empty for later | not implemented | runnable | runnable | runnable |
| InternVL3.5-8B-HF | MMVP forced-choice-150 | empty for later | not implemented | runnable | runnable | runnable |

“Runnable” means attribution and faithfulness evaluation launchers exist.
“Completed” means final numbers for the current cohort are tracked in `benchmark/results.json`, so the
dispatcher reuses them by default. “Not implemented” is deliberate: the
dispatcher exits instead of silently applying Qwen-specific code to InternVL or
the old free-form MMVP protocol to the new forced-choice subset.

Our method on InternVL is intentionally empty, as requested. The audit also
found that the input-level method has no InternVL implementation; that second
gap is recorded rather than hidden. InternVL EAGLE/TAM/LLaVA-CAM support both the COCO caption-target pipeline and the shared forced-choice MMVP-150 answer-token protocol.

## Canonical data protocols

- Current reported COCO results use
  `shared/coco_mask_tail_150_drop100_smallest_masks.json`. This is the 150-case
  cohort left after excluding the 100 lowest target-mask pixel counts from the
  audited 250-case benchmark (ties broken by image ID). It requires COCO 2017
  `val2017` images and includes audited captions, targets, boxes, and masks.
- MMVP uses `shared/mmvp_forced_choice_150_seed_20260828.json`: one seeded image
  from each complementary pair and exactly 75 A/75 B answers. The model emits
  exactly one complete option text. Attribution targets the model-selected
  option, even when the prediction is wrong.
- MMVP has no localization ground truth here, so Pointing Game is not reported.
- Older free-form MMVP-300 numbers are not interchangeable with this 150-image
  forced-choice protocol. They remain in the registry only as provenance.

## Setup elsewhere

```bash
git clone https://github.com/arshiaizd/sectet-project.git
cd sectet-project

# Install a CUDA-compatible torch/torchvision build first.
python -m pip install -r requirements.txt
python -m pip install -r requirements-internvl.txt  # InternVL only
```

External assets, which are not committed:

- COCO `val2017` image directory;
- MMVP `MMVP Images` directory;
- Qwen2.5-VL-7B-Instruct weights or Hugging Face access;
- InternVL3.5-8B-HF weights or Hugging Face access.

Every implemented launcher detects visible GPUs, uses at most four, and falls
back to one. `--num-gpus` is a maximum.

## Unified command

```bash
./scripts/run_experiment.sh \
  --method METHOD --dataset DATASET --model MODEL \
  --data-dir DATA_DIR --model-path MODEL_ID_OR_LOCAL_PATH \
  --num-gpus 4 --stage all
```

Values:

- method: `ours`, `input_level`, `eagle`, `tam`, `llavacam`;
- dataset: `coco`, `mmvp`;
- model: `qwen`, `internvl`;
- stage: `attribution`, `evaluation`, `all`.

Completed Qwen/COCO cells print tracked results and do not load a model. Add
`--force-rerun` only to regenerate one. Outputs default to
`runs/MODEL/DATASET/METHOD`; `--output-dir` puts them elsewhere. Resubmitting
the same command resumes at existing per-image/CSV boundaries.

### Four completed Qwen/COCO mask-tail-150 experiments

These commands return stored final numbers. Add `--force-rerun`, `--data-dir`,
and `--model-path` to rerun.

```bash
for method in ours input_level tam llavacam; do
  ./scripts/run_experiment.sh --method "$method" --dataset coco --model qwen
done
```

EAGLE remains runnable, but its previous COCO-250 numbers are not mixed with
this 150-case result cohort.

### Completed Qwen/COCO mask-tail-250 cohort

All five methods also have aggregate results on the full mask-tail 250 cohort.
They are kept separately in
`reports/previews/mask_tail_250_metrics_summary.csv`. The EAGLE aggregate was
supplied after the run; its provenance record is
`reports/previews/eagle_mask_tail_250_supplied_metrics.json`. EAGLE per-image
results, non-Pointing-Game standard errors, sensitive sample counts, and runtime
were not available and are therefore left empty rather than inferred.

Example forced rerun:

```bash
./scripts/run_experiment.sh \
  --method eagle --dataset coco --model qwen --force-rerun \
  --data-dir /datasets/coco \
  --model-path /models/Qwen2.5-VL-7B-Instruct \
  --output-dir /experiments/qwen/coco/eagle \
  --num-gpus 4 --stage all
```

### Qwen/MMVP forced-choice-150

All five Qwen methods use the same frozen exact-option prediction and answer-token targets:

```bash
for method in ours input_level eagle tam llavacam; do
  ./scripts/run_experiment.sh \
    --method "$method" --dataset mmvp --model qwen \
    --data-dir "/datasets/MMVP/MMVP Images" \
    --model-path /models/Qwen2.5-VL-7B-Instruct \
    --output-dir "/experiments/qwen/mmvp/$method" \
    --num-gpus 4 --stage all
done
```

### InternVL/COCO-250 baselines

```bash
for method in eagle tam llavacam; do
  ./scripts/run_experiment.sh \
    --method "$method" --dataset coco --model internvl \
    --data-dir /datasets/coco \
    --model-path /models/InternVL3_5-8B-HF \
    --output-dir "/experiments/internvl/coco/$method" \
    --num-gpus 4 --stage all
done
```

## Inspect without GPUs

```bash
python benchmark/show_results.py
python benchmark/show_results.py --model qwen --dataset coco
python benchmark/show_results.py --model internvl --json
```

`benchmark/results.json` is the source of truth for final numbers, protocol
notes, missing implementations, and legacy results that must not be mixed into
the current matrix.
