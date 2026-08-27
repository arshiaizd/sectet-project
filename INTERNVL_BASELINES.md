# InternVL3.5-8B baseline runs

This port covers the official EAGLE baselines only: EAGLE, TAM, LLaVA-CAM,
and IGOS++. The custom methods in `ours/` and `input_deletion/` are intentionally
unchanged.

The implementation originates from the official EAGLE repository at commit
`3cf4f656e09a0ee41f71dcf18aa6bcac47217f64`:

```text
https://github.com/RuoyuChen10/EAGLE
```

InternVL3.5 requires Transformers 4.52.1 or newer. The project environment is
now pinned to 4.55.0. Install the dedicated file after installing a suitable
CUDA PyTorch build:

```bash
python -m pip install -r requirements-internvl.txt
```

The attribution algorithms, target layers, and optimization hyperparameters
are adapted from the upstream InternVL3.5-4B implementation. The verified 8B-HF architecture retains the same layer-32 hook and 256-token (16x16) visual grid. The experiment
infrastructure changes previously used for Qwen are reapplied here:

- all methods use `shared/coco_mask_tail_250_benchmark.json`;
- the audited caption is retokenized for the selected InternVL checkpoint;
- the final model token overlapping the audited target character span is used;
- the same audited caption and target are teacher-forced in every baseline;
- per-image files are atomic and completed images can be resumed safely;
- runs use up to four visible GPUs and automatically fall back to one;
- dense maps use the same 64-step black-baseline faithfulness evaluation and
  centroid Pointing Game metric as the Qwen experiment;
- IGOS++ maps are inverted during evaluation, matching upstream EAGLE;
- BF16 is used when supported, otherwise FP16 is used.

## Attribution

The first argument selects the method and the second is either the COCO root
or its `val2017` directory:

```bash
./scripts/run_internvl_baseline_250.sh eagle /datasets/coco
./scripts/run_internvl_baseline_250.sh tam /datasets/coco
./scripts/run_internvl_baseline_250.sh llavacam /datasets/coco
./scripts/run_internvl_baseline_250.sh igos_pp /datasets/coco
```

The optional third argument is a local checkpoint or Hugging Face model ID.
It defaults to `OpenGVLab/InternVL3_5-8B-HF`. The optional fourth argument is
the output directory:

```bash
NUM_GPUS=1 ./scripts/run_internvl_baseline_250.sh \
  tam /datasets/coco /models/InternVL3_5-8B-HF /experiments/internvl-tam
```

## Evaluation

Use the same arguments and output path as attribution:

```bash
./scripts/evaluate_internvl_baseline_250.sh eagle /datasets/coco
./scripts/evaluate_internvl_baseline_250.sh tam /datasets/coco
./scripts/evaluate_internvl_baseline_250.sh llavacam /datasets/coco
./scripts/evaluate_internvl_baseline_250.sh igos_pp /datasets/coco
```

Evaluation writes aggregate AUC text, per-image Pointing Game CSV, and a
Pointing Game summary JSON inside the method output directory. The scripts
validate exact coverage before reporting success.

The checkpoint-specific retokenized manifest is cached below
`shared/generated/`, which is excluded from Git.
