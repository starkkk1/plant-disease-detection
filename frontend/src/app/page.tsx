"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import { Camera, ChevronRight, FlaskConical, ImagePlus, Leaf, Loader2, RotateCcw, ScanLine, X } from "lucide-react";
import { ExplanationResult, explainImage, PredictionError, PredictionResult, realAdapter, validateFile } from "@/lib/prediction-api";
import { mockAdapter } from "@/lib/mock-prediction";
import { ScrollReveal } from "@/components/scroll-reveal";

const diseaseNames: Record<string, string> = {
  Tomato___Bacterial_spot: "Bacterial spot",
  Tomato___Early_blight: "Early blight",
  Tomato___Late_blight: "Late blight",
  Tomato___Leaf_Mold: "Leaf mold",
  Tomato___Septoria_leaf_spot: "Septoria leaf spot",
  "Tomato___Spider_mites Two-spotted_spider_mite": "Two-spotted spider mite",
  Tomato___Target_Spot: "Target spot",
  Tomato___Tomato_Yellow_Leaf_Curl_Virus: "Tomato yellow leaf curl virus",
  Tomato___Tomato_mosaic_virus: "Tomato mosaic virus",
  Tomato___healthy: "Healthy leaf",
  unknown: "Unknown",
};

export default function Home() {
  const [image, setImage] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<PredictionError | null>(null);
  const [demo, setDemo] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [explanation, setExplanation] = useState<ExplanationResult | null>(null);
  const [explaining, setExplaining] = useState(false);
  const [explainError, setExplainError] = useState<PredictionError | null>(null);
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
    setExplanation(null); setExplaining(false); setExplainError(null);
    setDragging(false);
  }

  function choose(file?: File) {
    if (!file) return;
    reset();
    try { validateFile(file); setImage(file); }
    catch (error) { setError(error as PredictionError); }
  }

  async function analyze() {
    if (!image || busy || explaining) return;
    const operation = ++generation.current;
    const controller = new AbortController();
    active.current = controller;
    setBusy(true); setError(null); setResult(null);
    setExplanation(null); setExplainError(null);
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

  async function explain() {
    if (!image || !result || result.is_mock || busy || explaining) return;
    const operation = ++generation.current;
    const controller = new AbortController();
    active.current = controller;
    setExplaining(true); setExplainError(null);
    try {
      const response = await explainImage(image, controller.signal, result.predictions[0].class_id);
      if (response.model_version !== result.model_version) {
        throw new PredictionError("model_changed", "The model has changed. Analyze the image again before viewing its explanation.");
      }
      if (generation.current === operation) setExplanation(response);
    } catch (error) {
      if (generation.current === operation && error instanceof PredictionError && error.code !== "cancelled") {
        setExplainError(error);
      }
    } finally {
      if (generation.current === operation) { setExplaining(false); active.current = null; }
    }
  }

  return (
    <main className="leaf-app">
      <div className="leaf-shell">
        <header className="leaf-header">
          <Link href="/" className="leaf-brand" aria-label="Leaf home"><Leaf size={23} strokeWidth={1.7} /><span>Leaf</span></Link>
          <span className="leaf-header-note">Tomato leaf analysis</span>
        </header>

        <ScrollReveal className="leaf-intro">
          <h1>A closer look at your leaf.</h1>
          <p>Upload a tomato leaf photo. Explore the signs, one image at a time.</p>
        </ScrollReveal>

        <div className="leaf-workspace">
          <ScrollReveal className="leaf-card" aria-labelledby="upload-title">
            <div className="leaf-section-heading"><h2 id="upload-title">Leaf image</h2><span>01</span></div>
            <div className={`leaf-preview ${preview ? "has-image" : ""} ${dragging ? "is-dragging" : ""}`}
                 onDragOver={(event) => { event.preventDefault(); if (!busy && !explaining) setDragging(true); }}
                 onDragLeave={(event) => { if (!event.currentTarget.contains(event.relatedTarget as Node | null)) setDragging(false); }}
                 onDrop={(event) => { event.preventDefault(); setDragging(false); if (!busy && !explaining) choose(event.dataTransfer.files[0]); }}>
              {preview ? (
                <>
                  {/* Blob URLs are local previews, not optimized remote images. */}
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img src={preview} alt="Selected leaf" />
                  <button className="leaf-remove" onClick={reset} aria-label={busy || explaining ? "Cancel and remove image" : "Remove image"}><X size={18} /></button>
                </>
              ) : (
                <div className="leaf-empty-preview"><div className="leaf-icon-ring"><ImagePlus size={29} strokeWidth={1.4} /></div>
                  <p>Drop your leaf photo here</p><span>Use a clear, well-lit photo of a single leaf.</span></div>
              )}
            </div>

            <input ref={gallery} aria-label="Choose leaf image" type="file" accept="image/jpeg,image/png" className="sr-only" disabled={busy || explaining}
              onChange={(event) => { choose(event.target.files?.[0]); event.target.value = ""; }} />
            <input ref={camera} aria-label="Take leaf photo" type="file" accept="image/jpeg,image/png" capture="environment" className="sr-only" disabled={busy || explaining}
              onChange={(event) => { choose(event.target.files?.[0]); event.target.value = ""; }} />
            <div className="leaf-upload-actions">
              <button onClick={() => camera.current?.click()} disabled={busy || explaining}><Camera size={19} />Take photo</button>
              <button onClick={() => gallery.current?.click()} disabled={busy || explaining}><ImagePlus size={19} />Choose image</button>
            </div>
            <p className="leaf-file-note">JPG or PNG · Up to 5 MB</p>
            <button className="leaf-analyze" onClick={analyze} disabled={!image || busy || explaining}>
              {busy ? <><Loader2 className="leaf-spinner" size={19} />Analyzing…</> : <>{error && image ? "Try again" : "Analyze leaf"}<ChevronRight size={19} /></>}
            </button>
            {busy && <button className="leaf-text-button" onClick={() => { active.current?.abort(); generation.current += 1; setBusy(false); }}>Cancel analysis</button>}
          </ScrollReveal>

          <ScrollReveal className="leaf-card leaf-results" aria-labelledby="results-title" aria-busy={busy}>
            <div className="leaf-section-heading"><h2 id="results-title">Analysis</h2><span>02</span></div>
            <div aria-live="polite" aria-atomic="true" className="leaf-result-content">
              {busy ? (
                <div className="leaf-result-placeholder"><div className="leaf-icon-ring"><Loader2 className="leaf-spinner" size={32} /></div><h3>Looking at your leaf</h3><p>Your results will appear in a moment.</p></div>
              ) : error ? (
                <div className="leaf-error" role="alert"><h3>Analysis unavailable</h3><p>{error.message}</p>
                  {error.requestId && <details><summary>Support reference</summary><code>{error.requestId}</code></details>}
                </div>
              ) : result ? (
                <div className="leaf-result-data">
                  {result.is_mock && <div className="leaf-demo-banner"><FlaskConical size={16} /><span>Sample results · Not an analysis of this image</span></div>}
                  <p className="leaf-result-caption">Top prediction</p>
                  <h3 className="leaf-primary-label">{diseaseNames[result.predictions[0].label] || result.predictions[0].label.replace("Tomato___", "").replace(/_/g, " ")}</h3>
                  <div className="leaf-confidence"><span>{(result.predictions[0].confidence * 100).toFixed(1)}<small>%</small></span><p>Model confidence</p></div>
                  <div className="leaf-prediction-list">{result.predictions.map((prediction, index) => (
                    <div key={prediction.class_id} className="leaf-prediction-row"><div><span>{diseaseNames[prediction.label] || prediction.label.replace("Tomato___", "").replace(/_/g, " ")}</span><strong>{(prediction.confidence * 100).toFixed(1)}%</strong></div>
                      <div className="leaf-bar"><span style={{ width: `${prediction.confidence * 100}%` }} className={index === 0 ? "primary" : ""} /></div>
                    </div>
                  ))}</div>
                  <details className="leaf-model-version"><summary>Analysis details</summary><p>Model <code>{result.model_version}</code></p></details>
                  {!result.is_mock && <div className="leaf-explanation" aria-busy={explaining}>
                    <button className="leaf-text-button" onClick={explain} disabled={explaining}>
                      {explaining ? <><Loader2 className="leaf-spinner" size={16} />Creating explanation…</> :
                        explainError ? "Retry explanation" : "Explain prediction"}
                    </button>
                    {explaining && <button className="leaf-text-button" onClick={() => {
                      active.current?.abort(); generation.current += 1; setExplaining(false);
                    }}>Cancel explanation</button>}
                    {explainError && <p role="alert" className="leaf-explain-error">{explainError.message}</p>}
                  </div>}
                  <button className="leaf-text-button" onClick={reset}><RotateCcw size={15} />Analyze another leaf</button>
                </div>
              ) : (
                <div className="leaf-result-placeholder"><div className="leaf-icon-ring"><ScanLine size={29} strokeWidth={1.4} /></div><h3>{image ? "Ready to analyze" : "Your results, here"}</h3><p>{image ? "Select Analyze leaf to get started." : "Add a leaf photo to see the model’s predictions."}</p></div>
              )}
            </div>
            <p className="leaf-result-note">Model predictions are a guide, not a diagnosis. Check the plant’s symptoms before taking action.</p>
          </ScrollReveal>
        </div>

        {explanation && preview && <ScrollReveal className="leaf-card leaf-visual-explanation" aria-labelledby="explanation-title" aria-live="polite">
          <div className="leaf-section-heading"><h2 id="explanation-title">A look behind the prediction</h2><span>Grad-CAM++</span></div>
          <div className="leaf-comparison">
            <figure>
              {/* Blob URLs are local previews, not optimized remote images. */}
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src={preview} alt="Original leaf for comparison" />
              <figcaption>Original image</figcaption>
            </figure>
            <figure>
              {/* The API supplies a validated PNG data URL for the explanation. */}
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src={`data:image/png;base64,${explanation.explanation.overlay_png_base64}`}
                   width={explanation.explanation.width} height={explanation.explanation.height}
                   alt="Grad-CAM++ overlay on the leaf" />
              <figcaption>Model attention</figcaption>
            </figure>
          </div>
          <p className="leaf-visual-note">Warm colors highlight regions that contributed to this prediction, not confirmed diseased areas.</p>
          {!explanation.explanation.has_signal && <p className="leaf-visual-note">No clear positive contribution was found for this class.</p>}
        </ScrollReveal>}

        <footer className="leaf-footer">
          <details className="leaf-preview-options">
            <summary>Preview options{demo && <span>Sample mode on</span>}</summary>
            <label className="leaf-demo-toggle"><input type="checkbox" checked={demo} disabled={busy || explaining} onChange={(event) => { reset(); setDemo(event.target.checked); }} />Use sample results</label>
            {demo && <p className="leaf-demo-note">Sample mode does not analyze your image.</p>}
          </details>
        </footer>
      </div>
    </main>
  );
}
