"""Benchmark trained artifacts on real images, including all predict overhead."""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from src.inference.reporting import (
    add_model_arguments, collect_samples, create_predictor, dataset_evidence,
    evaluate_samples, metadata, positive_int, save_report,
)


def benchmark_model(predictor, images, warmup=10, iterations=100):
    if not images or warmup < 0 or iterations <= 0:
        raise ValueError("Benchmark needs real images, warmup >= 0 and iterations > 0")

    def synchronize():
        if predictor.device.type == "cuda":
            torch.cuda.synchronize(predictor.device)

    for index in range(warmup):
        predictor.predict(images[index % len(images)], top_k=1)
    synchronize()
    timings = []
    for index in range(iterations):
        synchronize()
        start = time.perf_counter_ns()
        predictor.predict(images[index % len(images)], top_k=1)
        synchronize()
        timings.append((time.perf_counter_ns() - start) / 1e6)
    return {"batch_size": 1, "warmup": warmup, "iterations": iterations,
            "p50_ms": float(np.percentile(timings, 50)),
            "p95_ms": float(np.percentile(timings, 95)), "mean_ms": float(np.mean(timings)),
            "samples_ms": timings,
            "scope": "file decode + RGB + resize + normalize + model + softmax/top1 serialization",
            "excludes": "model loading, Grad-CAM, network and API overhead",
            "image_paths": [str(path) for path in images]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_arguments(parser)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--image", type=Path, help="Real image for latency only")
    inputs.add_argument("--dataset", type=Path, help="Labeled ImageFolder; labels use training map")
    parser.add_argument("--iterations", type=positive_int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--limit-per-class", type=positive_int,
                        help="Explicit subset of labeled data; default uses all images")
    parser.add_argument("--evaluate", action="store_true", help="Also measure accuracy/macro-F1")
    args = parser.parse_args()
    if args.warmup < 0 or (args.evaluate and not args.dataset):
        parser.error("warmup must be >= 0; --evaluate requires --dataset")
    predictor, config = create_predictor(args)
    report = metadata(predictor, config)
    if args.dataset:
        samples = collect_samples(args.dataset, predictor.class_names, args.limit_per_class)
        images = [path for path, _ in samples]
        report["dataset"] = dataset_evidence(samples, args.dataset, predictor.class_names)
        report["limit_per_class"] = args.limit_per_class
    else:
        images = [args.image]
        from src.inference.predictor import sha256_file
        report["image_sha256"] = sha256_file(args.image)
    report["latency"] = benchmark_model(predictor, images, args.warmup, args.iterations)
    if args.evaluate:
        report["metrics"] = evaluate_samples(predictor, samples)
    save_report(args.output, report)
    print(f"{predictor.model_version}: p50={report['latency']['p50_ms']:.2f} ms, "
          f"p95={report['latency']['p95_ms']:.2f} ms; saved {args.output}")


if __name__ == "__main__":
    main()
