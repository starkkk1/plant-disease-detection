"""Reproduce Sprint 01 audit + CPU baseline for all five local checkpoints.

Run from repository root. Saves individual predictions and an aggregate report.
Never trains, downloads weights, or substitutes randomly initialized models.
"""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from torchvision.datasets import ImageFolder
from scripts.benchmark import benchmark_model
from src.inference.predictor import load_class_names
from src.inference.reporting import (
    collect_samples, create_predictor, dataset_evidence, evaluate_samples,
    metadata, positive_int, save_report,
)

MODELS = [
    ("convnext_tiny_best", "convnext"),
    ("efficientnet_b0_best", "efficientnet_b0"),
    ("efficientnet_b0_distilled_best", "distillation_effnetb0"),
    ("mobilenetv3_small_100_best", "mobilenet_v3_small"),
    ("mobilenetv3_small_100_distilled_best", "distillation_mobilenetv3"),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/new-data-removal"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/sprint-01"))
    parser.add_argument("--threads", type=positive_int, default=4)
    parser.add_argument("--iterations", type=positive_int, default=100)
    parser.add_argument("--limit-per-class", type=positive_int,
                        help="Explicit smoke subset; omit for complete test/eval baseline")
    args = parser.parse_args()
    class_map = args.data_root / "class_to_idx.json"
    names = load_class_names(class_map)
    training_map = ImageFolder(args.data_root / "train").class_to_idx
    if training_map != {name: index for index, name in enumerate(names)}:
        raise ValueError("Training ImageFolder indices disagree with class_to_idx.json")
    aggregate = {"training_class_map_verified": True, "class_names": names,
                 "limit_per_class": args.limit_per_class, "models": []}
    for stem, config_name in MODELS:
        print(f"Auditing {stem}...", flush=True)
        predictor, config = create_predictor(SimpleNamespace(
            config=f"configs/{config_name}.yaml", checkpoint=args.checkpoint_dir / f"{stem}.pth",
            class_map=class_map, model_name=None, model_version=None,
            device="cpu", threads=args.threads,
        ))
        report = metadata(predictor, config)
        test_samples = collect_samples(args.data_root / "test", names, args.limit_per_class)
        # Round-robin the classes so latency measures varied real images.
        by_class = [[path for path, label in test_samples if label == i] for i in range(len(names))]
        images = [group[i] for i in range(10) for group in by_class if len(group) > i]
        report["latency"] = benchmark_model(predictor, images, iterations=args.iterations)
        report["evaluations"] = {}
        for split in ("test", "eval"):
            samples = test_samples if split == "test" else collect_samples(
                args.data_root / split, names, args.limit_per_class)
            metrics = evaluate_samples(predictor, samples)
            evidence = dataset_evidence(samples, args.data_root / split, names)
            save_report(args.output_dir / f"{stem}_{split}.json",
                        {**metadata(predictor, config), "dataset": evidence,
                         "limit_per_class": args.limit_per_class, "metrics": metrics})
            report["evaluations"][split] = {
                "dataset": evidence, "metrics": {k: v for k, v in metrics.items() if k != "predictions"}}
            print(f"  {split}: n={len(samples)}, acc={metrics['accuracy']:.6f}, "
                  f"macro-F1={metrics['macro_f1']:.6f}", flush=True)
        report["real_image_prediction"] = predictor.predict(test_samples[0][0])
        aggregate["models"].append(report)
        save_report(args.output_dir / "baseline.json", aggregate)
        print(f"  CPU p50={report['latency']['p50_ms']:.2f} ms, "
              f"p95={report['latency']['p95_ms']:.2f} ms", flush=True)
    print(f"Saved complete baseline to {args.output_dir / 'baseline.json'}", flush=True)


if __name__ == "__main__":
    main()
