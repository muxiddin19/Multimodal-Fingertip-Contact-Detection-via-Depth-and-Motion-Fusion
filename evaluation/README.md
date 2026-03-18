# Evaluation

Scripts for evaluating depth estimation accuracy and contact detection performance.

## Depth Evaluation

```bash
python evaluate_depth.py \
    --checkpoint ../checkpoints/depth_anything_v2_vits_d405_finetuned.pth \
    --split-file ../depth_finetuning/splits/custom_d405/test.txt \
    --max-depth 0.5
```

## Metrics

| Metric | Description |
|--------|-------------|
| MAE (mm) | Mean absolute error in millimeters |
| RMSE (mm) | Root mean square error |
| abs_rel | Absolute relative error |
| delta1 (%) | % of pixels with max(pred/gt, gt/pred) < 1.25 |
| SiLog | Scale-invariant logarithmic error |
