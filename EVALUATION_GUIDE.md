# Evaluation Guide

This project evaluates models using the same protocol for fair comparison.

## Primary Metric

Primary metric:

```text
Macro F1-score
```

Reason:

- tomato disease classes may be imbalanced,
- accuracy can hide poor performance on minority classes.

## Required Classification Metrics

For every model, report:

- accuracy,
- macro precision,
- macro recall,
- macro F1-score,
- per-class precision,
- per-class recall,
- per-class F1-score,
- confusion matrix.

## Required Efficiency Metrics

For every model, report:

- number of parameters,
- FLOPs/MACs,
- checkpoint size in MB,
- average CPU inference time per image,
- average GPU inference time per image if GPU is available,
- throughput images/second.

Inference time should be averaged over at least 100 test images.

## Required Robustness Metrics

For every model and corruption type, report:

- clean accuracy,
- corrupted accuracy,
- clean macro F1,
- corrupted macro F1,
- absolute drop,
- relative drop percentage.

## Required Output Files

```text
results/experiments.csv
results/model_comparison.csv
results/robustness_comparison.csv
results/confusion_matrices/<model_name>_cm.png
results/classification_reports/<model_name>_report.csv
```

## Experiments CSV Columns

```csv
run_id,model,date,epochs_run,best_val_f1,test_accuracy,test_precision_macro,test_recall_macro,test_f1_macro,num_params,flops_macs,model_size_mb,inference_time_ms_cpu,inference_time_ms_gpu,throughput_img_per_sec,notes
```

## Robustness CSV Columns

```csv
model,corruption_type,clean_accuracy,corrupted_accuracy,clean_f1_macro,corrupted_f1_macro,absolute_drop,relative_drop_percent
```

## Final Model Selection Logic

Do not select the final recommended model by accuracy alone.

Use this logic:

1. Remove models with weak macro F1.
2. Compare model size and inference latency.
3. Compare robustness drop.
4. Check Grad-CAM quality on clean and corrupted samples.
5. Choose the best accuracy-efficiency-robustness-explainability trade-off.

## Discussion Questions

Use results to answer:

- Which model is most accurate on clean data?
- Which model is fastest?
- Which model is smallest?
- Which model is most robust under corrupted images?
- Which corruption hurts performance the most?
- Does the best accuracy model also provide better Grad-CAM explanations?
- Is MobileViT-XS worth the extra complexity compared with CNN-only lightweight models?
