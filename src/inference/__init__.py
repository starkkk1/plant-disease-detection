"""Lightweight classification inference for deployment (no Grad-CAM/Qdrant dependencies)."""

from .predictor import PlantDiseasePredictor

__all__ = ["PlantDiseasePredictor"]
