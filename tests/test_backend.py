import io
import time
import uuid
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.main import create_app
from backend.settings import Settings


class FakePredictor:
    class_names = [f"class_{index}" for index in range(11)]
    model_version = "test-fixture:weights"

    def predict(self, image, top_k=3):
        assert image.mode == "RGB"
        return {"model_version": self.model_version, "predictions": [
            {"class_id": i, "label": self.class_names[i], "confidence": 1 / top_k}
            for i in range(top_k)]}


def image_bytes(format="PNG", size=(17, 23), mode="RGB"):
    buffer = io.BytesIO()
    Image.new(mode, size).save(buffer, format=format)
    return buffer.getvalue()


@pytest.fixture
def settings(tmp_path):
    return Settings(checkpoint=tmp_path / "absent.pth", class_map=tmp_path / "absent.json")


@pytest.fixture
def client(settings):
    with TestClient(create_app(settings, lambda _: FakePredictor())) as client:
        yield client


def assert_error(response, status, code):
    assert response.status_code == status
    assert response.json()["error"]["code"] == code
    assert response.headers["x-request-id"] == response.json()["request_id"]
    uuid.UUID(response.json()["request_id"])


def test_health_and_openapi(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model_version"] == "test-fixture:weights"
    assert response.json()["is_mock"] is False  # fixture injected only by tests
    assert client.get("/docs").status_code == 200
    schema = client.get("/openapi.json").json()
    assert all(path in schema["paths"] for path in ("/health", "/predict", "/explain"))
    assert "multipart/form-data" in schema["paths"]["/predict"]["post"]["requestBody"]["content"]
    assert "multipart/form-data" in schema["paths"]["/explain"]["post"]["requestBody"]["content"]


def explanation_fixture(predictor, image, top_k=3, class_id=None):
    assert image.mode == "RGB"
    result = predictor.predict(image, top_k=top_k)
    target = result["predictions"][0] if class_id is None else {
        "class_id": class_id, "label": predictor.class_names[class_id], "confidence": 0.1}
    result["explanation"] = {"method": "gradcam++", "target": target, "target_layer": "test-layer",
        "width": 17, "height": 23, "has_signal": True,
        "overlay_png_base64": "fixture", "heatmap_png_base64": "fixture"}
    return result


@pytest.mark.parametrize("class_id", [None, "9"])
def test_explain_contract(client, monkeypatch, class_id):
    monkeypatch.setattr("src.inference.explain.explain_image", explanation_fixture)
    data = {"top_k": "2"}
    if class_id is not None:
        data["class_id"] = class_id
    response = client.post("/explain", files={"file": ("leaf.png", image_bytes(), "image/png")}, data=data)
    assert response.status_code == 200
    body = response.json()
    assert body["request_id"] == response.headers["x-request-id"]
    assert body["is_mock"] is False and len(body["predictions"]) == 2
    assert body["explanation"]["target"]["class_id"] == (0 if class_id is None else 9)


@pytest.mark.parametrize("class_id", ["-1", "11", "nope"])
def test_explain_invalid_target(client, class_id):
    assert_error(client.post("/explain", files={"file": ("leaf.png", image_bytes(), "image/png")},
                            data={"class_id": class_id}), 422, "invalid_request")


@pytest.mark.parametrize("mime,data,status,code", [
    ("text/plain", b"bad", 415, "unsupported_image"),
    ("image/png", b"bad", 400, "invalid_image"),
])
def test_explain_uses_image_validation(client, mime, data, status, code):
    assert_error(client.post("/explain", files={"file": ("leaf", data, mime)}), status, code)


def test_explain_unavailable_and_failure_are_safe(client, monkeypatch):
    from src.inference.explain import ExplanationUnavailable
    def unavailable(*args, **kwargs):
        raise ExplanationUnavailable("private/path")
    monkeypatch.setattr("src.inference.explain.explain_image", unavailable)
    files = {"file": ("leaf.png", image_bytes(), "image/png")}
    response = client.post("/explain", files=files)
    assert_error(response, 503, "explanation_unavailable")
    assert "private" not in response.text
    def broken(*args, **kwargs):
        raise RuntimeError("private/path")
    monkeypatch.setattr("src.inference.explain.explain_image", broken)
    response = client.post("/explain", files=files)
    assert_error(response, 500, "explanation_failed")
    assert "private" not in response.text
    assert client.post("/predict", files=files).status_code == 200


def test_explain_timeout_shares_prediction_slot(settings, monkeypatch):
    def slow(*args, **kwargs):
        time.sleep(0.2)
        return explanation_fixture(*args, **kwargs)
    monkeypatch.setattr("src.inference.explain.explain_image", slow)
    with TestClient(create_app(replace(settings, inference_timeout=0.03, queue_timeout=0.01),
                               lambda _: FakePredictor())) as client:
        files = {"file": ("leaf.png", image_bytes(), "image/png")}
        assert_error(client.post("/explain", files=files), 504, "prediction_timeout")
        assert_error(client.post("/predict", files=files), 503, "service_busy")
        assert client.get("/health").status_code == 200
        time.sleep(0.22)
        assert client.post("/predict", files=files).status_code == 200


@pytest.mark.parametrize("format,mime,mode", [("PNG", "image/png", "RGBA"),
    ("PNG", "image/png", "L"), ("JPEG", "image/jpeg", "RGB")])
def test_valid_image_contract(client, format, mime, mode):
    request_id = str(uuid.uuid4())
    response = client.post("/predict", files={"file": ("leaf", image_bytes(format, mode=mode), mime)},
                           data={"top_k": "2"}, headers={"X-Request-ID": request_id})
    assert response.status_code == 200
    body = response.json()
    assert body["request_id"] == request_id == response.headers["x-request-id"]
    assert body["processing_time_ms"] >= 0
    assert body["model_version"] == "test-fixture:weights"
    assert body["predictions"][0] == {"class_id": 0, "label": "class_0", "confidence": 0.5}
    assert len(body["predictions"]) == 2


@pytest.mark.parametrize("data,mime,status,code", [
    (b"", "image/png", 400, "invalid_image"),
    (b"garbage", "image/jpeg", 400, "invalid_image"),
    (b"garbage", "text/plain", 415, "unsupported_image"),
    (image_bytes("GIF"), "image/png", 415, "unsupported_image"),
    (image_bytes("PNG"), "image/jpeg", 415, "unsupported_image"),
])
def test_reject_invalid_images(client, data, mime, status, code):
    assert_error(client.post("/predict", files={"file": ("leaf.png", data, mime)}), status, code)


@pytest.mark.parametrize("top_k", ["0", "12", "nope"])
def test_top_k_validation(client, top_k):
    assert_error(client.post("/predict", files={"file": ("leaf.png", image_bytes(), "image/png")},
                            data={"top_k": top_k}), 422, "invalid_request")


def test_missing_file(client):
    assert_error(client.post("/predict", data={"top_k": 3}), 422, "invalid_request")


def test_upload_and_pixel_limits(settings):
    with TestClient(create_app(replace(settings, max_upload_bytes=128, max_image_pixels=100),
                               lambda _: FakePredictor())) as client:
        assert_error(client.post("/predict", files={"file": ("leaf.png", b"a" * 129, "image/png")}),
                     413, "file_too_large")
        assert_error(client.post("/predict", files={"file": ("leaf.png", image_bytes(), "image/png")}),
                     413, "image_too_large")
        assert_error(client.post("/predict", content=b"a" * 66000,
                                 headers={"Content-Type": "multipart/form-data; boundary=x"}),
                     413, "request_too_large")


def test_chunked_request_body_is_bounded(settings):
    with TestClient(create_app(replace(settings, max_upload_bytes=128), lambda _: FakePredictor())) as client:
        chunks = iter([b"--x\r\nContent-Disposition: form-data; name=\"file\"; filename=\"leaf.png\"\r\n"
                       b"Content-Type: image/png\r\n\r\n", b"a" * 66000, b"\r\n--x--\r\n"])
        response = client.post("/predict", content=chunks,
                               headers={"Content-Type": "multipart/form-data; boundary=x"})
        assert response.status_code == 413
        assert response.headers["x-request-id"] == response.json()["request_id"]


def test_unavailable_model_is_explicit(settings):
    def missing(_):
        raise FileNotFoundError("private/checkpoint/path")
    with TestClient(create_app(settings, missing)) as client:
        response = client.get("/health")
        assert response.status_code == 503
        assert response.json()["model_version"] is None
        assert "private" not in response.text
        assert_error(client.post("/predict", files={"file": ("leaf.png", image_bytes(), "image/png")}),
                     503, "model_unavailable")


def test_internal_errors_do_not_disclose_paths(settings):
    class Broken(FakePredictor):
        def predict(self, *args, **kwargs):
            raise RuntimeError("private/path/internal_details")
    with TestClient(create_app(settings, lambda _: Broken())) as client:
        response = client.post("/predict", files={"file": ("leaf.png", image_bytes(), "image/png")})
        assert_error(response, 500, "prediction_failed")
        assert "private" not in response.text


def test_timeout_keeps_slot_until_worker_finishes(settings):
    class SlowOnce(FakePredictor):
        first = True
        def predict(self, *args, **kwargs):
            if self.first:
                self.first = False
                time.sleep(0.2)
            return super().predict(*args, **kwargs)
    with TestClient(create_app(replace(settings, inference_timeout=0.03, queue_timeout=0.01),
                               lambda _: SlowOnce())) as client:
        file = {"file": ("leaf.png", image_bytes(), "image/png")}
        assert_error(client.post("/predict", files=file), 504, "prediction_timeout")
        assert_error(client.post("/predict", files=file), 503, "service_busy")
        assert client.get("/health").status_code == 200
        time.sleep(0.22)
        assert client.post("/predict", files=file).status_code == 200


def test_cors_only_allows_configured_origins(client):
    headers = {"Origin": "http://localhost:3000", "Access-Control-Request-Method": "POST",
               "Access-Control-Request-Headers": "x-request-id"}
    response = client.options("/predict", headers=headers)
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"
    headers["Origin"] = "https://unapproved.example"
    assert client.options("/predict", headers=headers).status_code == 400
    response = client.post("/predict", files={"file": ("leaf.png", b"garbage", "image/png")},
                           headers={"Origin": "http://localhost:3000"})
    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"
