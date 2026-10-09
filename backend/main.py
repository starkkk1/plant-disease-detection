"""Run: python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000."""
import asyncio
import io
import json
import logging
import time
import uuid
import warnings
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError
from starlette.exceptions import HTTPException as StarletteHTTPException

from backend.schemas import ErrorResponse, HealthResponse, PredictionResponse, ExplanationResponse
from backend.settings import Settings

logger = logging.getLogger("plant_api")
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    logger.addHandler(logging.StreamHandler())


def error(status, code, message):
    return HTTPException(status_code=status, detail={"code": code, "message": message})


class RequestMiddleware:
    """Bound the entire multipart body and attach IDs even to rejected requests."""
    def __init__(self, app, max_bytes):
        self.app, self.max_bytes = app, max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        headers = dict(scope.get("headers", []))
        try:
            request_id = str(uuid.UUID(headers.get(b"x-request-id", b"").decode("ascii")))
        except (ValueError, UnicodeError):
            request_id = str(uuid.uuid4())
        scope.setdefault("state", {})["request_id"] = request_id
        start, received, status = time.perf_counter(), 0, 500

        async def respond(message):
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                message.setdefault("headers", []).append((b"x-request-id", request_id.encode()))
            await send(message)

        async def limited_receive():
            nonlocal received
            message = await receive()
            received += len(message.get("body", b""))
            if received > self.max_bytes:
                raise error(413, "request_too_large", "Request body exceeds the upload limit.")
            return message

        try:
            length = headers.get(b"content-length")
            if length and int(length) > self.max_bytes:
                response = JSONResponse(status_code=413, content={"request_id": request_id,
                    "error": {"code": "request_too_large", "message": "Request body exceeds the upload limit."}})
                await response(scope, receive, respond)
            else:
                await self.app(scope, limited_receive, respond)
        finally:
            logger.info(json.dumps({"request_id": request_id, "method": scope["method"],
                                   "path": scope["path"], "status": status,
                                   "duration_ms": round((time.perf_counter() - start) * 1000, 2)}))


def decode_image(data, content_type, settings):
    if not data:
        raise error(400, "invalid_image", "The image is empty or cannot be decoded.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as image:
                expected = {"image/jpeg": "JPEG", "image/png": "PNG"}[content_type]
                if image.format != expected:
                    raise error(415, "unsupported_image", "Image content must match JPEG or PNG media type.")
                if image.width * image.height > settings.max_image_pixels:
                    raise error(413, "image_too_large", "Image exceeds the pixel limit.")
                image.verify()
            with Image.open(io.BytesIO(data)) as image:
                return image.convert("RGB")
    except (Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise error(413, "image_too_large", "Image exceeds the pixel limit.") from None
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError):
        raise error(400, "invalid_image", "The image is empty or cannot be decoded.") from None


def load_predictor(settings):
    import torch
    from src.inference import PlantDiseasePredictor
    torch.set_num_threads(settings.threads)
    return PlantDiseasePredictor(settings.checkpoint, settings.class_map,
                                 model_name=settings.model_name, image_size=settings.image_size,
                                 device="cpu")


def create_app(settings=None, predictor_factory=None):
    settings = settings or Settings()
    factory = predictor_factory or load_predictor

    @asynccontextmanager
    async def lifespan(application):
        application.state.predictor = None
        application.state.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="prediction")
        application.state.slot = asyncio.Semaphore(1)
        try:
            application.state.predictor = await asyncio.to_thread(factory, settings)
        except Exception:
            logger.exception("Model initialization failed; prediction service unavailable")
        yield
        application.state.executor.shutdown(wait=True, cancel_futures=True)

    application = FastAPI(title="Tomato Leaf Classification API", version="1.0", lifespan=lifespan)
    application.add_middleware(RequestMiddleware, max_bytes=settings.max_upload_bytes + 64 * 1024)
    application.add_middleware(CORSMiddleware, allow_origins=list(settings.cors_origins),
                               allow_credentials=False, allow_methods=["GET", "POST"],
                               allow_headers=["Content-Type", "X-Request-ID"], expose_headers=["X-Request-ID"])

    @application.exception_handler(StarletteHTTPException)
    async def http_error(request, exc):
        detail = exc.detail if isinstance(exc.detail, dict) else {
            "code": "http_error", "message": str(exc.detail)}
        return JSONResponse(status_code=exc.status_code, content={
            "request_id": request.state.request_id, "error": detail}, headers=exc.headers)

    @application.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        return JSONResponse(status_code=422, content={"request_id": request.state.request_id,
            "error": {"code": "invalid_request", "message": "Provide an image, top_k between 1 and 11, and an optional valid class_id for /explain."}})

    @application.get("/health", response_model=HealthResponse,
                     responses={503: {"model": HealthResponse}})
    async def health(request: Request):
        predictor = application.state.predictor
        body = HealthResponse(request_id=request.state.request_id,
            status="ok" if predictor else "unavailable", model_version=predictor.model_version if predictor else None,
            max_upload_bytes=settings.max_upload_bytes, max_image_pixels=settings.max_image_pixels)
        return JSONResponse(status_code=200 if predictor else 503, content=body.model_dump())

    @application.post("/predict", response_model=PredictionResponse,
        responses={code: {"model": ErrorResponse} for code in (400, 413, 415, 422, 500, 503, 504)})
    async def predict(request: Request, file: UploadFile = File(...), top_k: int = Form(3, ge=1, le=11)):
        return await run_image_request(request, file, top_k)

    @application.post("/explain", response_model=ExplanationResponse,
        responses={code: {"model": ErrorResponse} for code in (400, 413, 415, 422, 500, 503, 504)})
    async def explain(request: Request, file: UploadFile = File(...),
                      top_k: int = Form(3, ge=1, le=11), class_id: int | None = Form(None, ge=0, le=10)):
        return await run_image_request(request, file, top_k, explain=True, class_id=class_id)

    async def run_image_request(request, file, top_k, explain=False, class_id=None):
        started = time.perf_counter()
        try:
            if file.content_type not in ("image/jpeg", "image/png"):
                raise error(415, "unsupported_image", "Only image/jpeg and image/png are supported.")
            data = await file.read(settings.max_upload_bytes + 1)
            if len(data) > settings.max_upload_bytes:
                raise error(413, "file_too_large", "Image exceeds the file size limit.")
        finally:
            await file.close()
        predictor = application.state.predictor
        if predictor is None:
            raise error(503, "model_unavailable", "The model is unavailable. Try again later.")
        if top_k > len(predictor.class_names):
            raise error(422, "invalid_request", "top_k exceeds the number of trained classes.")
        if class_id is not None and class_id >= len(predictor.class_names):
            raise error(422, "invalid_request", "class_id exceeds the number of trained classes.")
        try:
            await asyncio.wait_for(application.state.slot.acquire(), settings.queue_timeout)
        except TimeoutError:
            raise error(503, "service_busy", "The model is busy. Try again shortly.") from None

        def work():
            image = decode_image(data, file.content_type, settings)
            try:
                if explain:
                    from src.inference.explain import explain_image, ExplanationUnavailable
                    try:
                        return explain_image(predictor, image, top_k=top_k, class_id=class_id)
                    except ExplanationUnavailable:
                        raise error(503, "explanation_unavailable", "Grad-CAM++ is unavailable for this model.") from None
                return predictor.predict(image, top_k=top_k)
            finally:
                image.close()

        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(application.state.executor, work)
        # A timed-out native CPU call cannot be killed. Retain its slot until completion.
        def completed(done):
            application.state.slot.release()
            if not done.cancelled():
                done.exception()  # consume late errors after timeout/client cancellation
        future.add_done_callback(completed)
        try:
            result = await asyncio.wait_for(asyncio.shield(future), settings.inference_timeout)
            response_type = ExplanationResponse if explain else PredictionResponse
            return response_type(request_id=request.state.request_id, **result,
                                      processing_time_ms=(time.perf_counter() - started) * 1000)
        except TimeoutError:
            raise error(504, "prediction_timeout", "Prediction timed out. Try again shortly.") from None
        except HTTPException:
            raise
        except Exception:
            logger.exception("Prediction failed for request %s", request.state.request_id)
            raise error(500, "explanation_failed" if explain else "prediction_failed",
                        "Explanation failed. Try again later." if explain else
                        "Prediction failed. Try another image or retry later.") from None

    return application


app = create_app()
