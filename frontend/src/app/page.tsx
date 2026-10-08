"use client";

import { useEffect, useRef, useState } from "react";
import { Camera, Check, ChevronRight, FlaskConical, ImagePlus, Leaf, Loader2, RotateCcw, X } from "lucide-react";
import { PredictionError, PredictionResult, realAdapter, validateFile } from "@/lib/prediction-api";
import { mockAdapter } from "@/lib/mock-prediction";

const diseaseNames: Record<string, string> = {
  Tomato___Bacterial_spot: "Đốm vi khuẩn",
  Tomato___Early_blight: "Bệnh cháy lá sớm",
  Tomato___Late_blight: "Bệnh mốc sương",
  Tomato___Leaf_Mold: "Bệnh mốc lá",
  Tomato___Septoria_leaf_spot: "Đốm lá Septoria",
  "Tomato___Spider_mites Two-spotted_spider_mite": "Nhện đỏ hai chấm",
  Tomato___Target_Spot: "Bệnh đốm vòng",
  Tomato___Tomato_Yellow_Leaf_Curl_Virus: "Virus xoăn vàng lá",
  Tomato___Tomato_mosaic_virus: "Virus khảm cà chua",
  Tomato___healthy: "Lá khỏe",
  unknown: "Chưa xác định",
};

export default function Home() {
  const [image, setImage] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<PredictionError | null>(null);
  const [demo, setDemo] = useState(false);
  const gallery = useRef<HTMLInputElement>(null);
  const camera = useRef<HTMLInputElement>(null);
  const active = useRef<AbortController | null>(null);
  const generation = useRef(0);

  useEffect(() => {
    if (!image) { setPreview(null); return; }
    const url = URL.createObjectURL(image);
    setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [image]);
  useEffect(() => () => { active.current?.abort(); }, []);

  function reset() {
    active.current?.abort();
    generation.current += 1;
    setImage(null); setResult(null); setError(null); setBusy(false);
  }

  function choose(file?: File) {
    if (!file) return;
    reset();
    try { validateFile(file); setImage(file); }
    catch (error) { setError(error as PredictionError); }
  }

  async function analyze() {
    if (!image || busy) return;
    const operation = ++generation.current;
    const controller = new AbortController();
    active.current = controller;
    setBusy(true); setError(null); setResult(null);
    try {
      const response = await (demo ? mockAdapter : realAdapter).predict(image, controller.signal);
      if (generation.current === operation) setResult(response);
    } catch (error) {
      if (generation.current === operation && error instanceof PredictionError && error.code !== "cancelled") {
        setError(error);
      }
    } finally {
      if (generation.current === operation) { setBusy(false); active.current = null; }
    }
  }

  return (
    <main className="leaf-app">
      <div className="leaf-shell">
        <header className="leaf-header">
          <a href="/" className="leaf-brand" aria-label="Lá, trang chủ"><Leaf size={22} /><span>Lá<span className="brand-dot">.</span></span></a>
          <span className="leaf-header-note">Dành cho lá cà chua</span>
        </header>

        <section className="leaf-intro">
          <p className="leaf-eyebrow">TỪ MỘT CHIẾC LÁ</p>
          <h1>Hiểu cây của bạn<br /><span>qua một bức ảnh.</span></h1>
          <p>Chụp hoặc chọn ảnh lá cà chua rõ nét để nhận diện dấu hiệu bệnh.</p>
        </section>

        <div className="leaf-workspace">
          <section className="leaf-card" aria-labelledby="upload-title">
            <div className="leaf-section-heading"><span className="leaf-step">01</span><h2 id="upload-title">Ảnh chiếc lá</h2></div>
            <div className={`leaf-preview ${preview ? "has-image" : ""}`}
                 onDragOver={(event) => event.preventDefault()}
                 onDrop={(event) => { event.preventDefault(); if (!busy) choose(event.dataTransfer.files[0]); }}>
              {preview ? (
                <>
                  {/* Blob URLs are local previews, not optimized remote images. */}
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img src={preview} alt="Ảnh lá đã chọn" />
                  <button className="leaf-remove" onClick={reset} aria-label={busy ? "Huỷ và bỏ ảnh" : "Bỏ ảnh"}><X size={18} /></button>
                  <span className="leaf-image-label"><Check size={13} />Ảnh đã sẵn sàng</span>
                </>
              ) : (
                <div className="leaf-empty-preview"><div className="leaf-icon-ring"><Leaf size={42} strokeWidth={1.3} /></div>
                  <p>Thêm một chiếc lá</p><span>Chụp gần, đủ sáng và thấy rõ bề mặt lá.</span></div>
              )}
            </div>

            <input ref={gallery} aria-label="Chọn ảnh lá" type="file" accept="image/jpeg,image/png" className="sr-only" disabled={busy}
              onChange={(event) => { choose(event.target.files?.[0]); event.target.value = ""; }} />
            <input ref={camera} aria-label="Chụp ảnh lá" type="file" accept="image/jpeg,image/png" capture="environment" className="sr-only" disabled={busy}
              onChange={(event) => { choose(event.target.files?.[0]); event.target.value = ""; }} />
            <div className="leaf-upload-actions">
              <button onClick={() => camera.current?.click()} disabled={busy}><Camera size={19} />Chụp ảnh</button>
              <button onClick={() => gallery.current?.click()} disabled={busy}><ImagePlus size={19} />Chọn ảnh</button>
            </div>
            <p className="leaf-file-note">JPEG hoặc PNG · Tối đa 5 MB</p>
            <button className="leaf-analyze" onClick={analyze} disabled={!image || busy}>
              {busy ? <><Loader2 className="leaf-spinner" size={19} />Đang nhận diện…</> : <>{error && image ? "Thử lại" : "Nhận diện chiếc lá"}<ChevronRight size={19} /></>}
            </button>
            {busy && <button className="leaf-text-button" onClick={() => { active.current?.abort(); generation.current += 1; setBusy(false); }}>Huỷ nhận diện</button>}
          </section>

          <section className="leaf-card leaf-results" aria-labelledby="results-title" aria-busy={busy}>
            <div className="leaf-section-heading"><span className="leaf-step">02</span><h2 id="results-title">Kết quả nhận diện</h2></div>
            <div aria-live="polite" aria-atomic="true" className="leaf-result-content">
              {busy ? (
                <div className="leaf-result-placeholder"><div className="leaf-icon-ring"><Loader2 className="leaf-spinner" size={32} /></div><h3>Đang xem ảnh chiếc lá</h3><p>Kết quả sẽ hiển thị ngay khi xử lý xong.</p></div>
              ) : error ? (
                <div className="leaf-error" role="alert"><h3>Chưa thể nhận diện</h3><p>{error.message}</p>
                  {error.requestId && <details><summary>Mã yêu cầu để hỗ trợ</summary><code>{error.requestId}</code></details>}
                </div>
              ) : result ? (
                <div className="leaf-result-data">
                  {result.is_mock && <div className="leaf-demo-banner"><FlaskConical size={16} /><span>Kết quả mẫu · Không phải phân tích ảnh này</span></div>}
                  <p className="leaf-result-caption">Dấu hiệu phù hợp nhất</p>
                  <h3 className="leaf-primary-label">{diseaseNames[result.predictions[0].label] || result.predictions[0].label.replace("Tomato___", "").replace(/_/g, " ")}</h3>
                  <div className="leaf-confidence"><span>{(result.predictions[0].confidence * 100).toFixed(1)}<small>%</small></span><p>Điểm tin cậy<br />của mô hình</p></div>
                  <div className="leaf-prediction-list">{result.predictions.map((prediction, index) => (
                    <div key={prediction.class_id} className="leaf-prediction-row"><div><span>{diseaseNames[prediction.label] || prediction.label.replace("Tomato___", "").replace(/_/g, " ")}</span><strong>{(prediction.confidence * 100).toFixed(1)}%</strong></div>
                      <div className="leaf-bar"><span style={{ width: `${prediction.confidence * 100}%` }} className={index === 0 ? "primary" : ""} /></div>
                    </div>
                  ))}</div>
                  <p className="leaf-model-version">Phiên bản mô hình <code>{result.model_version}</code></p>
                  <button className="leaf-text-button" onClick={reset}><RotateCcw size={15} />Nhận diện ảnh khác</button>
                </div>
              ) : (
                <div className="leaf-result-placeholder"><div className="leaf-icon-ring"><Leaf size={32} strokeWidth={1.4} /></div><h3>{image ? "Ảnh đã sẵn sàng" : "Chưa có ảnh chiếc lá"}</h3><p>{image ? "Bấm “Nhận diện chiếc lá” để xem kết quả." : "Chọn hoặc chụp ảnh để bắt đầu."}</p></div>
              )}
            </div>
            <p className="leaf-result-note">Kết quả tham khảo từ hình ảnh. Đối chiếu triệu chứng thực tế trước khi đưa ra quyết định chăm sóc cây.</p>
          </section>
        </div>

        <footer className="leaf-footer">
          <label className="leaf-demo-toggle"><input type="checkbox" checked={demo} disabled={busy} onChange={(event) => { reset(); setDemo(event.target.checked); }} /><FlaskConical size={15} />Dùng kết quả mẫu để xem giao diện</label>
          {demo && <p className="leaf-demo-note">Chế độ minh hoạ · Không gọi dịch vụ nhận diện</p>}
          <a href="/research">Công cụ nghiên cứu và tìm kiếm <ChevronRight size={13} /></a>
        </footer>
      </div>
    </main>
  );
}
