"""Real Grad-CAM++ on deterministic tiny weights, not baseline measurements."""
import base64
import io
import json

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from src.inference.explain import ExplanationUnavailable, explain_image
from src.inference.predictor import PlantDiseasePredictor


class TinyCamModel(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.blocks = nn.Sequential(nn.Conv2d(3, 2, 1, bias=False), nn.ReLU())
        self.classifier = nn.Linear(2, num_classes, bias=False)
        with torch.no_grad():
            self.blocks[0].weight.zero_()
            self.blocks[0].weight[0, 0, 0, 0] = 1
            self.blocks[0].weight[1, 1, 0, 0] = 1
            self.classifier.weight.copy_(torch.eye(2))

    def forward(self, tensor):
        return self.classifier(self.blocks(tensor).mean(dim=(2, 3)))


@pytest.fixture
def cam_predictor(tmp_path, monkeypatch):
    monkeypatch.setattr("src.inference.predictor.timm.create_model", lambda *args, **kwargs: TinyCamModel())
    checkpoint = tmp_path / "cam.pth"
    torch.save(TinyCamModel().state_dict(), checkpoint)
    mapping = tmp_path / "classes.json"
    mapping.write_text(json.dumps({"red": 0, "green": 1}))
    return PlantDiseasePredictor(checkpoint, mapping, image_size=32)


def decode_png(value):
    with Image.open(io.BytesIO(base64.b64decode(value, validate=True))) as image:
        assert image.format == "PNG"
        return np.asarray(image).copy()


def test_targets_pngs_and_no_model_or_hook_mutation(cam_predictor):
    pixels = np.zeros((32, 48, 3), dtype=np.uint8)
    pixels[:, :24, 0] = 255
    pixels[:, 24:, 1] = 255
    image = Image.fromarray(pixels)
    before = cam_predictor.predict(image, top_k=2)
    weights = {key: value.clone() for key, value in cam_predictor.model.state_dict().items()}
    results = [explain_image(cam_predictor, image, top_k=2, class_id=class_id) for class_id in (0, 1)]
    maps = []
    for class_id, result in enumerate(results):
        explanation = result["explanation"]
        assert explanation["target"]["class_id"] == class_id
        assert explanation["method"] == "gradcam++" and explanation["has_signal"]
        assert explanation["target_layer"] == "blocks.1"
        assert result["predictions"] == before["predictions"]
        assert decode_png(explanation["overlay_png_base64"]).shape == (32, 32, 3)
        heatmap = decode_png(explanation["heatmap_png_base64"])
        assert heatmap.shape == (32, 32) and heatmap.max() > heatmap.min()
        maps.append(heatmap)
    assert maps[0][:, :16].mean() > maps[0][:, 16:].mean()
    assert maps[1][:, 16:].mean() > maps[1][:, :16].mean()
    assert cam_predictor.predict(image, top_k=2) == before
    assert not cam_predictor.model.training
    for key, value in cam_predictor.model.state_dict().items():
        torch.testing.assert_close(value, weights[key])
    assert all(parameter.grad is None for parameter in cam_predictor.model.parameters())
    assert all(not module._forward_hooks and not module._backward_hooks for module in cam_predictor.model.modules())


@pytest.mark.parametrize("class_id", [-1, 2, True])
def test_invalid_target(cam_predictor, class_id):
    with pytest.raises(ValueError):
        explain_image(cam_predictor, Image.new("RGB", (32, 32)), class_id=class_id)


def test_unsupported_architecture(cam_predictor):
    cam_predictor.model_name = "unconfigured"
    with pytest.raises(ExplanationUnavailable):
        explain_image(cam_predictor, Image.new("RGB", (32, 32)))
