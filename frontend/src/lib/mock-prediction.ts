import { PredictionAdapter, PredictionError, validateFile } from "./prediction-api";

// Explicit demo mode only. Never substituted automatically after an API failure.
export const mockAdapter: PredictionAdapter = {
  async predict(file, signal) {
    validateFile(file);
    await new Promise<void>((resolve, reject) => {
      const abort = () => {
        clearTimeout(timer);
        reject(new PredictionError("cancelled", "Đã huỷ nhận diện."));
      };
      const timer = setTimeout(() => { signal?.removeEventListener("abort", abort); resolve(); }, 600);
      signal?.addEventListener("abort", abort, { once: true });
      if (signal?.aborted) abort();
    });
    return {
      request_id: "demo-local", model_version: "demo-mock-v1", is_mock: true, processing_time_ms: 600,
      predictions: [
        { class_id: 0, label: "Tomato___Bacterial_spot", confidence: 0.82 },
        { class_id: 1, label: "Tomato___Early_blight", confidence: 0.12 },
        { class_id: 2, label: "Tomato___Late_blight", confidence: 0.06 },
      ],
    };
  },
};
