"""Inference preprocessing matching src/datasets/transforms.py validation transform."""

import cv2
import numpy as np
import torch
from PIL import Image

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(image_size: int = 224, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """Match validation's OpenCV INTER_LINEAR resize and A.Normalize exactly.

    Accept PIL images in any mode, return float32 RGB CHW tensors. No cropping,
    EXIF rotation or stochastic augmentation is applied during inference.
    """
    if not isinstance(image_size, int) or isinstance(image_size, bool) or image_size <= 0:
        raise ValueError("image_size must be positive")
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    if (mean.shape != (3,) or std.shape != (3,) or not np.isfinite(mean).all()
            or not np.isfinite(std).all() or (std <= 0).any()):
        raise ValueError("mean/std must contain three finite values; std must be positive")
    mean_pixels = mean * 255.0
    inverse_std = 1.0 / (std * 255.0)

    def transform(image):
        if not isinstance(image, Image.Image):
            raise TypeError("transform expects a PIL Image")
        rgb = np.asarray(image.convert("RGB"))
        resized = cv2.resize(rgb, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
        normalized = (resized.astype(np.float32) - mean_pixels) * inverse_std
        return torch.from_numpy(np.ascontiguousarray(normalized.transpose(2, 0, 1)))

    return transform
