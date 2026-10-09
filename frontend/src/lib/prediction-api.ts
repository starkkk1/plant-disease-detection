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
  unsupported_image: "Chỉ hỗ trợ ảnh JPEG hoặc PNG. Hãy đổi định dạng ảnh rồi thử lại.",
  invalid_image: "Không đọc được ảnh này. Hãy chọn ảnh khác.",
  file_too_large: "Ảnh vượt quá 5 MB. Hãy chọn ảnh nhỏ hơn.",
  request_too_large: "Ảnh vượt quá giới hạn tải lên. Hãy chọn ảnh nhỏ hơn.",
  image_too_large: "Độ phân giải ảnh quá lớn. Hãy giảm kích thước ảnh rồi thử lại.",
  invalid_request: "Yêu cầu chưa hợp lệ. Hãy chọn lại ảnh và thử lại.",
  model_unavailable: "Dịch vụ nhận diện chưa sẵn sàng. Vui lòng thử lại sau.",
  service_busy: "Dịch vụ đang bận. Vui lòng thử lại sau vài giây.",
  prediction_timeout: "Nhận diện mất quá nhiều thời gian. Vui lòng thử lại.",
  prediction_failed: "Chưa thể nhận diện ảnh này. Hãy thử lại hoặc chọn ảnh khác.",
  explanation_unavailable: "Grad-CAM++ chưa sẵn sàng cho mô hình này.",
  explanation_failed: "Chưa thể tạo bản đồ giải thích. Vui lòng thử lại.",
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

export function apiBaseUrl(): string {
  // Local default follows the page hostname, so a phone does not call its own localhost.
  const configured = process.env.NEXT_PUBLIC_API_URL;
  if (configured) return configured.replace(/\/+$/, "");
  if (typeof window !== "undefined") {
    const url = new URL(window.location.origin);
    url.port = "8000";
    return url.origin;
  }
  return "http://localhost:8000";
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
        throw new PredictionError(code, messages[code] || "Dịch vụ gặp lỗi. Vui lòng thử lại sau.",
                                  data?.request_id || response.headers.get("X-Request-ID") || undefined);
      }
      if (endpoint === "predict" && data?.model_status && Array.isArray(data.predictions) &&
          data.predictions.some((prediction: Record<string, unknown> | null) => prediction && "class_name" in prediction)) {
        throw new PredictionError("api_version_mismatch",
          "Dịch vụ đang dùng phiên bản API cũ. Cần cập nhật dịch vụ nhận diện.");
      }
      try { return parse(data); }
      catch { throw new PredictionError("invalid_response", "Dịch vụ trả về kết quả chưa hợp lệ. Vui lòng thử lại."); }
    } catch (error) {
      if (error instanceof PredictionError) throw error;
      if (timedOut) throw new PredictionError("prediction_timeout", messages.prediction_timeout);
      if (signal?.aborted) throw new PredictionError("cancelled", "Đã huỷ nhận diện.");
      throw new PredictionError("api_unavailable", "Không kết nối được dịch vụ nhận diện. Hãy kiểm tra kết nối và thử lại.");
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
