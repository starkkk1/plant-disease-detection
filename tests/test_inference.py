"""Contract tests use tiny fabricated weights, never report them as a baseline."""
import json

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from src.datasets.transforms import get_transforms
from src.inference.predictor import PlantDiseasePredictor
from src.inference.preprocess import build_transform
from src.inference.reporting import collect_samples, evaluate_samples


@pytest.mark.parametrize("mode", ["RGB", "L", "RGBA"])
def test_tensor_shape_dtype_and_rgb(mode):
    image = Image.new(mode, (61, 37))
    tensor = build_transform(32)(image)
    assert tensor.shape == (3, 32, 32)
    assert tensor.dtype == torch.float32
    assert tensor.is_contiguous()
    assert torch.isfinite(tensor).all()


@pytest.mark.parametrize("size", [0, -1, True, 1.5])
def test_invalid_image_size(size):
    with pytest.raises(ValueError):
        build_transform(size)


def test_preprocessing_matches_training_validation():
    config = {"data": {"image_size": 32},
              "normalization": {"mean": [0.1, 0.2, 0.3], "std": [0.4, 0.5, 0.6]}}
    rgb = np.random.default_rng(42).integers(0, 256, (47, 83, 3), dtype=np.uint8)
    expected = get_transforms(config, split="val")(image=rgb)["image"]
    actual = build_transform(32, **config["normalization"])(Image.fromarray(rgb))
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def test_normalized_pixel_values():
    tensor = build_transform(8, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))(
        Image.new("RGB", (19, 7), (255, 0, 255)))
    torch.testing.assert_close(tensor[:, 0, 0], torch.tensor([1., -1., 1.]))


class TinyClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.bias = nn.Parameter(torch.arange(num_classes, dtype=torch.float32))

    def forward(self, tensor):
        assert torch.is_inference_mode_enabled()
        assert not self.training
        assert tensor.device.type == "cpu"
        assert tensor.shape == (1, 3, 32, 32)
        return self.bias.expand(tensor.shape[0], -1)


@pytest.fixture
def artifacts(tmp_path, monkeypatch):
    def factory(name, pretrained, num_classes):
        assert pretrained is False
        return TinyClassifier(num_classes)
    monkeypatch.setattr("src.inference.predictor.timm.create_model", factory)
    classes = tmp_path / "classes.json"
    classes.write_text(json.dumps({"healthy": 1, "diseased": 0}), encoding="utf-8")
    checkpoint = tmp_path / "weights.pth"
    torch.save({"bias": torch.tensor([4., -4.])}, checkpoint)
    return checkpoint, classes


@pytest.mark.parametrize("wrapper", [None, "model", "state_dict"])
def test_checkpoint_loaded_and_prediction_uses_training_index(artifacts, tmp_path, wrapper):
    checkpoint, classes = artifacts
    if wrapper:
        weights = torch.load(checkpoint, weights_only=True)
        torch.save({wrapper: weights}, checkpoint)
    predictor = PlantDiseasePredictor(checkpoint, classes, image_size=32)
    image_path = tmp_path / "real-file.png"
    Image.new("L", (31, 67), 128).save(image_path)
    result = predictor.predict(image_path)
    assert result["model_version"].startswith("mobilenetv3_small_100:")
    assert result["predictions"][0]["class_id"] == 0
    assert result["predictions"][0]["label"] == "diseased"
    assert result["predictions"][0]["confidence"] > 0.99
    assert len(result["predictions"]) == 2
    assert sum(item["confidence"] for item in result["predictions"]) == pytest.approx(1.)
    json.dumps(result)


def test_missing_or_incompatible_weights_fail(artifacts):
    checkpoint, classes = artifacts
    torch.save({"bias": torch.zeros(3)}, checkpoint)
    with pytest.raises(RuntimeError):
        PlantDiseasePredictor(checkpoint, classes)
    with pytest.raises(FileNotFoundError):
        PlantDiseasePredictor(checkpoint.with_name("missing.pth"), classes)


def test_invalid_image_and_top_k(artifacts, tmp_path):
    predictor = PlantDiseasePredictor(*artifacts, image_size=32)
    bad_image = tmp_path / "broken.png"
    bad_image.write_text("not an image")
    with pytest.raises(ValueError, match="Invalid or unreadable"):
        predictor.predict(bad_image)
    for top_k in (0, 3, True, 1.5):
        with pytest.raises(ValueError):
            predictor.predict(Image.new("RGB", (8, 8)), top_k=top_k)


@pytest.mark.parametrize("output", [torch.zeros(1, 3), torch.tensor([[float("nan"), 0.]])])
def test_invalid_model_outputs_fail(artifacts, monkeypatch, output):
    predictor = PlantDiseasePredictor(*artifacts, image_size=32)
    monkeypatch.setattr(predictor.model, "forward", lambda tensor: output)
    with pytest.raises(RuntimeError):
        predictor.predict(Image.new("RGB", (8, 8)))


def test_subset_dataset_remaps_labels_and_reports_absent_classes(artifacts, tmp_path):
    predictor = PlantDiseasePredictor(*artifacts, image_size=32)
    root = tmp_path / "split"
    (root / "healthy").mkdir(parents=True)
    Image.new("RGB", (8, 8)).save(root / "healthy" / "leaf.png")
    samples = collect_samples(root, predictor.class_names)
    assert samples[0][1] == 1  # local ImageFolder index is 0, training index is 1
    metrics = evaluate_samples(predictor, samples)
    assert metrics["accuracy"] == 0.
    assert metrics["confusion_matrix"] == [[0, 0], [1, 0]]
    (root / "untrained-class").mkdir()
    Image.new("RGB", (8, 8)).save(root / "untrained-class" / "leaf.png")
    with pytest.raises(ValueError, match="absent from training"):
        collect_samples(root, predictor.class_names)
