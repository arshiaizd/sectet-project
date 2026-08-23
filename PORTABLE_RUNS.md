# Running the 250-image benchmark on another machine

All attribution launchers use the same audited manifest:

```text
shared/coco_mask_tail_250_benchmark.json
```

They accept either the COCO root containing `val2017/` or the `val2017/`
directory itself. Before loading a model, each launcher verifies that all 250
required images exist and that the manifest contains 250 unique paths.

## 1. Clone and install

```bash
git clone https://github.com/arshiaizd/sectet-project.git
cd sectet-project

# Create/activate a Python environment first, and install a CUDA-compatible
# PyTorch + torchvision build for the target machine.
python -m pip install -r requirements.txt
```

The launchers use the active environment's `python`. Override it with
`PYTHON_BIN=/path/to/python` when necessary.

## 2. Run a method

The only required argument is the COCO dataset location:

```bash
./scripts/run_eagle_250.sh /datasets/coco
./scripts/run_ours_250.sh /datasets/coco
./scripts/run_input_deletion_250.sh /datasets/coco
./scripts/run_tam_250.sh /datasets/coco
./scripts/run_llavacam_250.sh /datasets/coco
./scripts/run_igos_pp_250.sh /datasets/coco
```

By default, the scripts load `Qwen/Qwen2.5-VL-7B-Instruct` from Hugging Face.
A local checkpoint or a different cache-compatible model location can be the
second argument:

```bash
./scripts/run_eagle_250.sh \
  /datasets/coco/val2017 \
  /models/Qwen2.5-VL-7B-Instruct
```

An optional third argument sets the output directory:

```bash
./scripts/run_eagle_250.sh \
  /datasets/coco \
  /models/Qwen2.5-VL-7B-Instruct \
  /experiments/eagle-250
```

The same positional interface is used by every launcher:

```text
COCO_DIR [MODEL_ID_OR_PATH] [OUTPUT_DIR]
```

## GPU selection and one-GPU fallback

The scripts inspect `torch.cuda.device_count()`, use at most four visible GPUs,
and automatically fall back to one worker when only one GPU is visible. They
also respect an existing `CUDA_VISIBLE_DEVICES` mapping.

To deliberately use fewer GPUs, set `NUM_GPUS`:

```bash
NUM_GPUS=1 ./scripts/run_eagle_250.sh /datasets/coco
NUM_GPUS=2 ./scripts/run_ours_250.sh /datasets/coco
```

One complete Qwen2.5-VL-7B model is loaded per worker. A GPU therefore needs
enough memory for one model plus the selected attribution method's activations.
The scripts require CUDA and do not silently run these experiments on CPU.

## Resume and validation

All runners use method-specific output directories. EAGLE and the heatmap
baselines atomically save per-image files and skip readable completed outputs.
The two patch-CSV methods use their existing `--resume` mechanism. Worker-count
specific CSV directories prevent collisions if a run is restarted with a
different number of GPUs.

After workers exit, each launcher validates exact coverage of all 250 manifest
images and returns a nonzero status if an output is missing or invalid.

Generated outputs live under each method's ignored `results/` directory unless
an explicit output path is supplied.

## Offline execution

Pass a local model path and enable the standard Transformers offline switches:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  ./scripts/run_eagle_250.sh \
  /datasets/coco \
  /models/Qwen2.5-VL-7B-Instruct
```

## Unified all-input runner

For a single entry point, use `scripts/run_benchmark_250.sh`. It performs
runtime dependency checks, validates all 250 COCO images, optionally installs
Python requirements, optionally downloads a pinned Qwen snapshot, selects one
or more methods, applies the one-to-four GPU fallback, runs attribution, resumes
completed work, and validates all 250 outputs.

```bash
./scripts/run_benchmark_250.sh \
  --method eagle \
  --method ours \
  --coco-dir /datasets/coco \
  --model /models/Qwen2.5-VL-7B-Instruct \
  --num-gpus 1 \
  --output-root /experiments/coco250
```

To download and pin the model as part of the run:

```bash
./scripts/run_benchmark_250.sh \
  --method all \
  --coco-dir /datasets/coco \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --model-revision HUGGING_FACE_COMMIT_HASH \
  --download-model-to /models/qwen25vl7b \
  --output-root /experiments/coco250
```

Run `./scripts/run_benchmark_250.sh --help` for every accepted input. The
`--install-deps` option installs `requirements.txt`, but a CUDA-compatible
PyTorch and torchvision build must already be installed because their correct
wheel index depends on the target machine.
