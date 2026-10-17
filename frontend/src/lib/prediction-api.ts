export const MAX_UPLOAD_BYTES = 5 * 1024 * 1024;
export const PREDICT_TIMEOUT_MS = 25_000;

export interface Prediction {
  class_id: number;
  label: string;
  confidence: number;
}

export interface PredictionResult {
  request_id: string;
  model_version: string;
  is_mock: boolean;
  processing_time_ms: number;
  predictions: Prediction[];
}

export interface ExplanationResult extends PredictionResult {
  explanation: {
    method: "gradcam++";
    target: Prediction;
    target_layer: string;
    width: number;
    height: number;
    has_signal: boolean;
    overlay_png_base64: string;
    heatmap_png_base64: string;
  };
}

export interface PredictionAdapter {
  predict(file: File, signal?: AbortSignal): Promise<PredictionResult>;
}

const messages: Record<string, string> = {
  unsupported_image: "Only JPEG or PNG images are supported. Convert your image and try again.",
  invalid_image: "This image could not be read. Choose another photo.",
  file_too_large: "This image exceeds 5 MB. Choose a smaller photo.",
  request_too_large: "This image exceeds the upload limit. Choose a smaller photo.",
  image_too_large: "The image resolution is too large. Resize it and try again.",
  invalid_request: "The request is invalid. Select your image again and retry.",
  model_unavailable: "The analysis service is not ready. Please try again later.",
  service_busy: "The service is busy. Try again in a few seconds.",
  prediction_timeout: "Analysis took too long. Please try again.",
  prediction_failed: "This image could not be analyzed. Try again or choose another photo.",
  explanation_unavailable: "Grad-CAM++ is unavailable for this model.",
  explanation_failed: "The explanation could not be created. Please try again.",
};

export class PredictionError extends Error {
  constructor(public code: string, message: string, public requestId?: string) {
    super(message);
    this.name = "PredictionError";
  }
}

export function validateFile(file: File): void {
  if (!["image/jpeg", "image/png"].includes(file.type)) {
    throw new PredictionError("unsupported_image", messages.unsupported_image);
  }
  if (!file.size) throw new PredictionError("invalid_image", messages.invalid_image);
  if (file.size > MAX_UPLOAD_BYTES) throw new PredictionError("file_too_large", messages.file_too_large);
}

export function resolveApiBaseUrl(pageOrigin: string, configured?: string): string {
  if (configured) return configured.replace(/\/+$/, "");
  const url = new URL(pageOrigin);
  // HTTPS deployments proxy /api to FastAPI on the same origin.
  if (url.protocol === "https:") return `${url.origin}/api`;
  // HTTP development and the existing IP demo retain the direct API port.
  url.port = "8000";
  return url.origin;
}

export function apiBaseUrl(): string {
  return resolveApiBaseUrl(
    typeof window !== "undefined" ? window.location.origin : "http://localhost:3000",
    process.env.NEXT_PUBLIC_API_URL,
  );
}

function parseResult(data: unknown): PredictionResult {
  if (!data || typeof data !== "object") throw new Error("invalid response");
  const result = data as PredictionResult;
  if (typeof result.request_id !== "string" || !result.request_id ||
      typeof result.model_version !== "string" || !result.model_version || result.is_mock !== false ||
      !Number.isFinite(result.processing_time_ms) || result.processing_time_ms < 0 ||
      !Array.isArray(result.predictions) || !result.predictions.length ||
      result.predictions.some((prediction) => !prediction || !Number.isInteger(prediction.class_id) ||
        prediction.class_id < 0 || typeof prediction.label !== "string" || !prediction.label.trim() ||
        !Number.isFinite(prediction.confidence) || prediction.confidence < 0 || prediction.confidence > 1)) {
    throw new Error("invalid response");
  }
  return result;
}

async function requestImage<T>(file: File, endpoint: string, parse: (data: unknown) => T,
                               signal?: AbortSignal, classId?: number): Promise<T> {
    validateFile(file);
    const controller = new AbortController();
    let timedOut = false;
    const abort = () => controller.abort();
    signal?.addEventListener("abort", abort, { once: true });
    if (signal?.aborted) controller.abort();
    const timer = setTimeout(() => { timedOut = true; controller.abort(); }, PREDICT_TIMEOUT_MS);
    try {
      const form = new FormData();
      form.append("file", file);
      form.append("top_k", "3");
      if (classId !== undefined) form.append("class_id", String(classId));
      const response = await fetch(`${apiBaseUrl()}/${endpoint}`, { method: "POST", body: form, signal: controller.signal });
      const data = await response.json().catch(() => null);
      if (!response.ok) {
        const code = data?.error?.code || "http_error";
        throw new PredictionError(code, messages[code] || "The service encountered an error. Please try again later.",
                                  data?.request_id || response.headers.get("X-Request-ID") || undefined);
      }
      if (endpoint === "predict" && data?.model_status && Array.isArray(data.predictions) &&
          data.predictions.some((prediction: Record<string, unknown> | null) => prediction && "class_name" in prediction)) {
        throw new PredictionError("api_version_mismatch",
          "The service is using an outdated API version. The analysis service needs an update.");
      }
      try { return parse(data); }
      catch { throw new PredictionError("invalid_response", "The service returned an invalid result. Please try again."); }
    } catch (error) {
      if (error instanceof PredictionError) throw error;
      if (timedOut) throw new PredictionError("prediction_timeout", messages.prediction_timeout);
      if (signal?.aborted) throw new PredictionError("cancelled", "Analysis cancelled.");
      throw new PredictionError("api_unavailable", "Cannot connect to the analysis service. Check your connection and try again.");
    } finally {
      clearTimeout(timer);
      signal?.removeEventListener("abort", abort);
    }
}

export const realAdapter: PredictionAdapter = {
  predict: (file, signal) => requestImage(file, "predict", parseResult, signal),
};

function parseExplanation(data: unknown): ExplanationResult {
  const result = parseResult(data) as ExplanationResult;
  const explanation = result.explanation;
  const png = (value: unknown) => typeof value === "string" && value.startsWith("iVBORw0KGgo") &&
    value.length < 4_000_000 && /^[A-Za-z0-9+/]+={0,2}$/.test(value);
  if (!explanation || explanation.method !== "gradcam++" ||
      !Number.isInteger(explanation.width) || explanation.width <= 0 ||
      !Number.isInteger(explanation.height) || explanation.height <= 0 ||
      typeof explanation.has_signal !== "boolean" || typeof explanation.target_layer !== "string" ||
      !explanation.target_layer || !png(explanation.overlay_png_base64) || !png(explanation.heatmap_png_base64)) {
    throw new Error("invalid explanation");
  }
  parseResult({ ...result, predictions: [explanation.target] });
  return result;
}

export function explainImage(file: File, signal?: AbortSignal, classId?: number): Promise<ExplanationResult> {
  return requestImage(file, "explain", parseExplanation, signal, classId);
}
