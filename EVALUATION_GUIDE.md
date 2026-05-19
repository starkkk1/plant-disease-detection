# Evaluation Guide

This project must evaluate each model using the same protocol.

## 1. Primary Metric

Primary metric:

```txt
Macro F1-score
```

Reason:

- The tomato dataset may be imbalanced.
- Accuracy can hide poor performance on minority disease classes.

## 2. Required Classification Metrics

For every model, report:

- Accuracy
- Macro precision
- Macro recall
- Macro F1-score
- Per-class precision
- Per-class recall
- Per-class F1-score
- Confusion matrix

## 3. Required Efficiency Metrics

For every model, report:

- Number of parameters
- Model checkpoint size in MB
- Average CPU inference time per image
- Average GPU inference time per image if GPU is available

Inference time should be averaged over at least 100 test images.

## 4. Required Output Files

```txt
results/experiments.csv
results/model_comparison.csv
results/confusion_matrices/<model_name>_cm.png
results/classification_reports/<model_name>_report.csv
```

## 5. Experiments CSV Columns

```csv
run_id,model,date,epochs_run,best_val_f1,test_accuracy,test_precision_macro,test_recall_macro,test_f1_macro,num_params,model_size_mb,inference_time_ms_cpu,inference_time_ms_gpu,notes
```

## 6. Final Model Selection

The final recommended model should not be selected by accuracy alone.

Use this decision logic:

1. Remove models with weak macro F1.
2. Compare model size and inference speed.
3. Check Grad-CAM quality on correct and wrong predictions.
4. Choose the best accuracy-efficiency-interpretability trade-off.

## 7. Paper Discussion Questions

Use evaluation results to answer:

- Which model is most accurate?
- Which model is fastest?
- Which model is smallest?
- Which model has the best macro F1?
- Does a smaller model lose too much accuracy?
- Does the best accuracy model also provide better Grad-CAM explanations?
