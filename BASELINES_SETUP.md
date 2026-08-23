# Visual attribution baseline setup

## Canonical evaluation set

- File: `shared/coco_eval_250_seed_20260815.json`
- Records: 250 unique COCO validation images
- Selection seed: `20260815`
- SHA-256: `8efa57d84af25aab013bc98a2deeed5ee6ffd094fec7bb401379f07ecb678804`
- Four shared strided shards: `63 / 63 / 62 / 62` records

Every method reads the same shared files. No baseline owns or regenerates the
selection.

## Source provenance

Official repository: `https://github.com/RuoyuChen10/EAGLE`

Pinned commit: `3cf4f656e09a0ee41f71dcf18aa6bcac47217f64`

Untouched upstream runner and core snapshots are retained under each method's
`official_source/` directory. The operative core attribution mathematics is
unchanged.

The official repository contains a Qwen2.5-VL-7B COCO-object runner for
LLaVA-CAM. Its Qwen COCO-object IGOS++ and TAM runners are published under the
3B directory, so those two operative runners change only the model path to the
same local Qwen2.5-VL-7B checkpoint used by EAGLE.

Infrastructure-only adaptations in operative files:

1. Load `/mnt/vilab/scratch/arshia/models/Qwen2.5-VL-7B-Instruct` offline.
2. Read the shared 250-example worker shards.
3. Save outputs atomically and skip only readable, complete outputs.
4. Process the full worker shard in the faithfulness stage; the upstream
   inference script contains a task-specific `contents[240:]` slice.
5. Remove stale `position_ids`, `cache_position`, and `rope_deltas` after
   teacher-forced `input_ids` replacement for Transformers 4.49 compatibility.
6. Run one independent process per visible GPU. Attribution formulas,
   hyperparameters, target layers, and heatmap calculations are unchanged.

## Collision-free layout

- `eagle/results/eagle_250_seed_20260815/`
- `llavacam/results/qwen25vl7b_coco_object_250_seed_20260815/LLaVACAM/`
- `igos_pp/results/qwen25vl7b_coco_object_250_seed_20260815/IGOS_PP/`
- `tam/results/qwen25vl7b_coco_object_250_seed_20260815/TAM/`

Each baseline has two stages:

1. `run_attribution_250.sh` creates `npy/` heatmaps.
2. `run_faithfulness_250.sh` evaluates the saved maps and creates `json/`
   insertion/deletion curves.

Both stages validate all 250 expected IDs before returning success. They are
safe to rerun after interruption.
