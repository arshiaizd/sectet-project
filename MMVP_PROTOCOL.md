# Shared forced-choice MMVP-150 protocol

All implemented methods consume the same model-specific target manifest. The
manifest is prepared once per checkpoint by
`shared/prepare_mmvp_forced_choice_targets.py` and then reused by every method.

For each sample, the model receives the image and the exact prompt stored in
`shared/mmvp_forced_choice_150_seed_20260828.json`. Decoding is greedy and
constrained to exactly one of the two complete option texts. The resulting
predicted option and all of its tokenizer IDs are frozen before attribution.
Ground truth is used only to record prediction correctness; it never replaces
an incorrect model prediction as the attribution target.

Attribution targets only the generated option tokens. The question,
instructions, and option-list tokens condition the model but are not themselves
attribution targets. For a multi-token answer `y_1 ... y_T`, scoring uses
teacher-forced probabilities `p(y_t | image, prompt, y_<t)` for every answer
token.

Method aggregation:

- Ours: mean answer-token probability for every activation-patched region.
- Input-level: mean answer-token probability for insertion and deletion, then
  the unchanged `insertion + (1 - deletion)` attribution objective.
- EAGLE: its unchanged submodular objective averages the vector of answer-token
  probabilities.
- TAM: one answer-token TAM is generated for every token and the maps are
  averaged.
- LLaVA-CAM: gradients target the joint score of all answer tokens. Its final
  normalized map is unchanged by using sum rather than mean because the two
  differ only by a positive constant.

Faithfulness evaluation uses the same prompt, frozen predicted answer, token
IDs, and teacher-forced per-token probabilities. Patch methods change eight
merged patches per step. Dense methods use 64 equal pixel-fraction steps. All
MMVP aggregate evaluations use sensitivity threshold `0.2`. Pointing Game is
not reported because this MMVP setup has no localization masks or boxes.

Runnable combinations:

- Qwen2.5-VL-7B-Instruct: ours, input-level, EAGLE, TAM, LLaVA-CAM.
- InternVL3.5-8B-HF: EAGLE, TAM, LLaVA-CAM.
- Our method and input-level remain unimplemented for InternVL.

The unified interface is:

```bash
./scripts/run_experiment.sh \
  --method METHOD --dataset mmvp --model MODEL \
  --data-dir "/datasets/MMVP/MMVP Images" \
  --model-path /models/CHECKPOINT \
  --output-dir /experiments/MODEL/mmvp/METHOD \
  --num-gpus 4 --stage all
```

The first run for a checkpoint creates a resumable cached target manifest under
`shared/generated/`. Concurrent jobs use a file lock, ensuring that every method
reads a complete identical manifest.
