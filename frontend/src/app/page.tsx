"use client";
import { useState, useRef } from "react";
import { UploadCloud, Leaf, Loader2, Info } from "lucide-react";

export default function Home() {
  const [image, setImage] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [results, setResults] = useState<any[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleImageChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setImage(file);
      setPreviewUrl(URL.createObjectURL(file));
      setResults(null);
      setError(null);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const file = e.dataTransfer.files?.[0];
    if (file && file.type.startsWith("image/")) {
      setImage(file);
      setPreviewUrl(URL.createObjectURL(file));
      setResults(null);
      setError(null);
    }
  };

  const handleAnalyze = async () => {
    if (!image) return;
    setIsAnalyzing(true);
    setError(null);
    setResults(null);

    const formData = new FormData();
    formData.append("file", image);
    formData.append("top_k", "3");

    try {
      // Gọi tới API Python chạy ở cổng 8000
      const res = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.detail || "Đã có lỗi xảy ra");
      }

      const data = await res.json();
      setResults(data.predictions);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <main className="min-h-screen bg-gradient-to-br from-emerald-950 via-slate-900 to-teal-950 text-slate-200 p-8 font-sans flex flex-col items-center">
      {/* Header */}
      <div className="flex flex-col items-center mb-12 mt-8 animate-fade-in-down">
        <div className="p-4 bg-emerald-500/20 rounded-full mb-4 ring-1 ring-emerald-500/30 shadow-[0_0_30px_rgba(16,185,129,0.2)]">
          <Leaf className="w-12 h-12 text-emerald-400" />
        </div>
        <h1 className="text-5xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-emerald-400 to-teal-200 mb-4 tracking-tight text-center">
          Plant AI Diagnoser
        </h1>
        <p className="text-slate-400 text-lg max-w-xl text-center">
          Upload a photo of a tomato leaf and let our advanced AI model instantly detect any diseases with precision.
        </p>
      </div>

      <div className="w-full max-w-5xl grid md:grid-cols-2 gap-8 items-start">
        {/* Upload Section */}
        <div className="bg-white/5 backdrop-blur-xl border border-white/10 p-8 rounded-3xl shadow-2xl relative overflow-hidden group">
          <div className="absolute inset-0 bg-gradient-to-br from-emerald-500/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
          
          <h2 className="text-2xl font-bold mb-6 text-white flex items-center gap-2">
            <UploadCloud className="text-emerald-400" /> Upload Leaf
          </h2>

          <div
            className="border-2 border-dashed border-slate-600 hover:border-emerald-500 hover:bg-emerald-500/5 transition-all duration-300 rounded-2xl p-8 flex flex-col items-center justify-center text-center cursor-pointer min-h-[300px] relative overflow-hidden group/drop"
            onDragOver={(e) => e.preventDefault()}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
          >
            <input 
              type="file" 
              className="hidden" 
              ref={fileInputRef} 
              accept="image/*" 
              onChange={handleImageChange}
            />

            {previewUrl ? (
              <div className="absolute inset-0 w-full h-full">
                <img src={previewUrl} alt="Preview" className="w-full h-full object-cover" />
                <div className="absolute inset-0 bg-slate-900/40 flex items-center justify-center opacity-0 group-hover/drop:opacity-100 transition-opacity">
                  <span className="bg-black/60 px-4 py-2 rounded-lg text-white font-medium backdrop-blur-sm">Click to change image</span>
                </div>
              </div>
            ) : (
              <>
                <UploadCloud className="w-16 h-16 text-slate-500 mb-4 group-hover/drop:text-emerald-400 transition-colors" />
                <p className="text-slate-300 font-medium mb-1">Drag & drop your image here</p>
                <p className="text-slate-500 text-sm">or click to browse</p>
              </>
            )}
          </div>

          <button
            onClick={handleAnalyze}
            disabled={!image || isAnalyzing}
            className="w-full mt-6 bg-gradient-to-r from-emerald-500 to-teal-500 hover:from-emerald-400 hover:to-teal-400 disabled:from-slate-700 disabled:to-slate-700 disabled:text-slate-400 text-white font-bold py-4 rounded-xl shadow-lg hover:shadow-emerald-500/25 transition-all duration-300 flex items-center justify-center gap-2 text-lg"
          >
            {isAnalyzing ? (
              <>
                <Loader2 className="animate-spin w-6 h-6" /> Analyzing...
              </>
            ) : (
              "Diagnose Image"
            )}
          </button>
        </div>

        {/* Results Section */}
        <div className="bg-white/5 backdrop-blur-xl border border-white/10 p-8 rounded-3xl shadow-2xl h-full flex flex-col min-h-[450px]">
          <h2 className="text-2xl font-bold mb-6 text-white flex items-center gap-2">
            <Info className="text-teal-400" /> Results
          </h2>

          <div className="flex-1 flex flex-col justify-center">
            {error ? (
              <div className="bg-red-500/10 border border-red-500/20 rounded-xl p-4 text-red-400 text-center flex flex-col items-center">
                <p>{error}</p>
                <p className="text-sm mt-2 opacity-80">(Make sure the Python API server is running on port 8000)</p>
              </div>
            ) : results ? (
              <div className="space-y-8 animate-fade-in-up">
                {results.map((res, idx) => {
                  const percent = Math.round(res.probability * 100);
                  const isTop = idx === 0;
                  return (
                    <div key={idx} className="relative">
                      <div className="flex justify-between items-end mb-3">
                        <span className={`font-semibold ${isTop ? 'text-emerald-400 text-xl' : 'text-slate-300 text-lg'}`}>
                          {res.class_name.replace("Tomato___", "").replace(/_/g, " ")}
                        </span>
                        <span className={`font-mono font-bold ${isTop ? 'text-emerald-400 text-2xl' : 'text-slate-400 text-xl'}`}>
                          {percent}%
                        </span>
                      </div>
                      <div className="w-full bg-slate-800 rounded-full h-4 overflow-hidden">
                        <div 
                          className={`h-full rounded-full transition-all duration-1000 ease-out ${isTop ? 'bg-gradient-to-r from-emerald-500 to-teal-400 shadow-[0_0_10px_rgba(16,185,129,0.5)]' : 'bg-slate-600'}`} 
                          style={{ width: `${percent}%` }}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
            ) : (
              <div className="text-center text-slate-500">
                <Leaf className="w-20 h-20 opacity-20 mx-auto mb-6" />
                <p className="text-lg">Upload an image and hit <span className="text-emerald-400/80">Analyze</span> to see the AI's diagnosis.</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </main>
  );
}
