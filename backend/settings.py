import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    checkpoint: Path = field(default_factory=lambda: Path(os.getenv(
        "MODEL_CHECKPOINT", str(ROOT / "checkpoints/mobilenetv3_small_100_best.pth"))))
    class_map: Path = field(default_factory=lambda: Path(os.getenv(
        "MODEL_CLASS_MAP", str(ROOT / "configs/class_to_idx.sprint01.json"))))
    model_name: str = field(default_factory=lambda: os.getenv("MODEL_NAME", "mobilenetv3_small_100"))
    image_size: int = field(default_factory=lambda: int(os.getenv("MODEL_IMAGE_SIZE", "224")))
    threads: int = field(default_factory=lambda: int(os.getenv("MODEL_THREADS", "4")))
    max_upload_bytes: int = field(default_factory=lambda: int(os.getenv("MAX_UPLOAD_BYTES", str(5 * 1024**2))))
    max_image_pixels: int = field(default_factory=lambda: int(os.getenv("MAX_IMAGE_PIXELS", "20000000")))
    inference_timeout: float = field(default_factory=lambda: float(os.getenv("PREDICT_TIMEOUT_SECONDS", "20")))
    queue_timeout: float = field(default_factory=lambda: float(os.getenv("QUEUE_TIMEOUT_SECONDS", "1")))
    cors_origins: tuple[str, ...] = field(default_factory=lambda: tuple(
        origin.strip() for origin in os.getenv("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",")
        if origin.strip()))

    def __post_init__(self):
        if any(value <= 0 for value in (self.image_size, self.threads, self.max_upload_bytes,
                                       self.max_image_pixels, self.inference_timeout, self.queue_timeout)):
            raise ValueError("Sizes, threads and timeouts must be positive")
        if "*" in self.cors_origins:
            raise ValueError("Configure explicit CORS origins instead of a wildcard")
