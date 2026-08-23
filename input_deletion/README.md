# Input insertion/deletion baseline

This baseline performs no activation patching. For each merged Qwen visual
patch and its clipped 3×3 neighborhood, it runs two independent pixel-space
interventions:

- insertion: reveal the neighborhood on an otherwise white image;
- deletion: replace the neighborhood with white pixels in the original image.

The default ranking is:

```text
attribution_score = insertion_score + (1 - deletion_score)
```

The run scripts explicitly use the uploaded defaults: radius 1, `score-mode
both`, `insertion_plus_necessity`, and a white baseline. The source prompt is
also preserved unchanged from the uploaded implementation.

## Entry points

- `run_mask_tail_30_4gpu.sh`: first 30 canonical cases;
- `run_evaluation_30_4gpu.sh`: patch-8 faithfulness, AUC, and Point Game;
- `run_mask_tail_250_4gpu.sh`: all 250 canonical cases;
- `run_evaluation_250_4gpu.sh`: full evaluation.

Attribution launchers are restart-safe. Completed image groups are retained;
an incomplete trailing group is removed and recomputed.
