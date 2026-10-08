"""Shared artifact selection and reproducible dataset evaluation."""
import argparse
import hashlib
import json
import platform
from importlib.metadata import version
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from torchvision.datasets.folder import IMG_EXTENSIONS

from src.inference.predictor import PlantDiseasePredictor, sha256_file
from src.inference.preprocess import IMAGENET_MEAN, IMAGENET_STD
from src.utils.config import load_config


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def add_model_arguments(parser):
    parser.add_argument("--config", default="configs/mobilenet_v3_small.yaml")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--class-map", type=Path, help="Defaults to config data.class_map")
    parser.add_argument("--model-name", help="Defaults to config model.name")
    parser.add_argument("--model-version")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--threads", type=positive_int, default=4)
    parser.add_argument("--output", type=Path, required=True)


def create_predictor(args):
    torch.set_num_threads(args.threads)
    config = load_config(args.config)
    data = config.get("data", {})
    normal = config.get("normalization", {})
    return PlantDiseasePredictor(
        args.checkpoint, args.class_map or data["class_map"],
        model_name=args.model_name or config["model"]["name"],
        model_version=args.model_version, image_size=data.get("image_size", 224),
        device=args.device, mean=normal.get("mean", IMAGENET_MEAN),
        std=normal.get("std", IMAGENET_STD),
    ), config


def collect_samples(dataset_path, class_names, limit_per_class=None):
    """Remap by name: a subset split's ImageFolder indices may differ from train."""
    root = Path(dataset_path)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {root}")
    directories = sorted(path for path in root.iterdir() if path.is_dir())
    unknown = {path.name for path in directories} - set(class_names)
    if unknown:
        raise ValueError(f"Dataset contains labels absent from training class map: {sorted(unknown)}")
    class_indices = {name: index for index, name in enumerate(class_names)}
    counts = Counter()
    samples = []
    for directory in directories:
        label = class_indices[directory.name]
        for path in sorted(directory.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in IMG_EXTENSIONS:
                continue
            if limit_per_class is None or counts[label] < limit_per_class:
                samples.append((str(path), label))
                counts[label] += 1
    if not samples:
        raise ValueError(f"No labeled images found: {dataset_path}")
    return samples


def dataset_evidence(samples, root, class_names):
    manifest = [
        {"path": Path(path).relative_to(root).as_posix(), "class_id": label,
         "sha256": sha256_file(path)} for path, label in samples
    ]
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    counts = Counter(label for _, label in samples)
    return {"root": str(Path(root).resolve()), "sample_count": len(samples),
            "class_counts": {name: counts[index] for index, name in enumerate(class_names)},
            "manifest_sha256": digest}


def metadata(predictor, config):
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": predictor.model_name, "model_version": predictor.model_version,
        "checkpoint": str(predictor.checkpoint_path),
        "checkpoint_sha256": predictor.checkpoint_sha256,
        "checkpoint_size_mib": predictor.checkpoint_path.stat().st_size / 1024**2,
        "parameter_count": sum(p.numel() for p in predictor.model.parameters()),
        "class_names": predictor.class_names, "class_map_sha256": predictor.class_map_sha256,
        "input_shape": [1, 3, predictor.image_size, predictor.image_size],
        "output_shape": [1, len(predictor.class_names)],
        "preprocessing": {"resize": "OpenCV INTER_LINEAR, direct square resize",
                          "color": "RGB", "dtype": "float32",
                          "mean": config.get("normalization", {}).get("mean", IMAGENET_MEAN),
                          "std": config.get("normalization", {}).get("std", IMAGENET_STD)},
        "environment": {"platform": platform.platform(), "processor": platform.processor(),
                        "python": platform.python_version(), "torch": torch.__version__,
                        "device": str(predictor.device), "threads": torch.get_num_threads(),
                        "packages": {name: version(name) for name in
                                     ("torchvision", "timm", "numpy", "Pillow", "scikit-learn", "PyYAML")}},
    }


def evaluate_samples(predictor, samples):
    targets, predictions, rows = [], [], []
    for path, label in samples:
        result = predictor.predict(path, top_k=1)["predictions"][0]
        targets.append(label)
        predictions.append(result["class_id"])
        rows.append({"path": path, "target": label, **result})
    labels = list(range(len(predictor.class_names)))
    return {
        "sample_count": len(targets), "accuracy": float(accuracy_score(targets, predictions)),
        "macro_f1": float(f1_score(targets, predictions, labels=labels, average="macro", zero_division=0)),
        "macro_precision": float(precision_score(targets, predictions, labels=labels, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(targets, predictions, labels=labels, average="macro", zero_division=0)),
        "macro_policy": "all training classes; zero_division=0, including absent classes",
        "confusion_matrix": confusion_matrix(targets, predictions, labels=labels).tolist(),
        "predictions": rows,
    }


def save_report(path, report):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
