from typing import Literal

from pydantic import BaseModel, Field


class Prediction(BaseModel):
    class_id: int = Field(ge=0)
    label: str
    confidence: float = Field(ge=0, le=1)


class PredictionResponse(BaseModel):
    request_id: str
    model_version: str
    is_mock: Literal[False] = False
    processing_time_ms: float = Field(ge=0)
    predictions: list[Prediction]


class HealthResponse(BaseModel):
    request_id: str
    api_version: Literal["1.0"] = "1.0"
    status: Literal["ok", "unavailable"]
    model_version: str | None
    is_mock: Literal[False] = False
    max_upload_bytes: int
    max_image_pixels: int
    supported_formats: list[str] = ["image/jpeg", "image/png"]


class Explanation(BaseModel):
    method: Literal["gradcam++"]
    target: Prediction
    target_layer: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    has_signal: bool
    overlay_png_base64: str
    heatmap_png_base64: str


class ExplanationResponse(PredictionResponse):
    explanation: Explanation


class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    request_id: str
    error: ErrorDetail
