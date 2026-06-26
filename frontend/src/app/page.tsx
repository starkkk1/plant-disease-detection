"use client";
import { useState, useRef, useEffect } from "react";
import { UploadCloud, Leaf, Loader2, Info, Search } from "lucide-react";

export default function Home() {
  const [image, setImage] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  
  // Predict states
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [results, setResults] = useState<any[] | null>(null);
  
  // Search states
  const [isSearching, setIsSearching] = useState(false);
  const [searchResults, setSearchResults] = useState<any[] | null>(null);
  
  const [error, setError] = useState<string | null>(null);
  
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleImageChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setImage(file);
      setPreviewUrl(URL.createObjectURL(file));
      setResults(null);
      setSearchResults(null);
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
      setSearchResults(null);
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

  const handleSearch = async () => {
    if (!image) return;
    setIsSearching(true);
    setError(null);
    setSearchResults(null);

    const formData = new FormData();
    formData.append("file", image);
    formData.append("limit", "4"); // Lấy top 4 ảnh giống nhất

    try {
      const res = await fetch("http://127.0.0.1:8000/search/image", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.detail || "Đã có lỗi xảy ra");
      }

      const data = await res.json();
      setSearchResults(data.results);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setIsSearching(false);
    }
  };



  const hasContent = !!(results || searchResults || error);

  return (
    <main className="min-h-screen bg-slate-950 text-slate-200 p-8 font-sans flex flex-col items-center overflow-x-hidden relative">
      


      <div className="z-10 w-full flex flex-col items-center">
        {/* Header */}
        <div className="flex flex-col items-center mb-12 mt-8 animate-fade-in-down w-full">
        <div className="p-4 bg-emerald-500/20 rounded-full mb-4 ring-1 ring-emerald-500/30 shadow-[0_0_30px_rgba(16,185,129,0.2)]">
          <Leaf className="w-12 h-12 text-emerald-400" />
        </div>
        <h1 className="text-5xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-emerald-400 to-teal-200 mb-4 tracking-tight text-center">
          Tomato Leaf Doctor
        </h1>
        <p className="text-slate-400 text-lg max-w-xl text-center">
          Upload a photo of a tomato leaf and let our advanced AI model instantly detect any diseases with precision.
        </p>
      </div>

      <div className="w-full max-w-6xl grid lg:grid-cols-2 gap-8 items-stretch pb-20">
        {/* Upload Section */}
        <div className="bg-white/5 backdrop-blur-xl border border-white/10 p-8 rounded-3xl shadow-2xl relative overflow-hidden group flex flex-col">
          <div className="pointer-events-none absolute inset-0 bg-gradient-to-br from-emerald-500/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
          
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

          <div className="flex flex-col gap-3 mt-6">
            <button
              onClick={handleAnalyze}
              disabled={!image || isAnalyzing}
              className="w-full bg-gradient-to-r from-emerald-500 to-teal-500 hover:from-emerald-400 hover:to-teal-400 disabled:from-slate-700 disabled:to-slate-700 disabled:text-slate-400 text-white font-bold py-5 rounded-2xl shadow-lg hover:shadow-emerald-500/25 transition-all duration-300 flex items-center justify-center gap-2 text-2xl"
            >
              {isAnalyzing ? (
                <>
                  <Loader2 className="animate-spin w-7 h-7" /> Analyzing...
                </>
              ) : (
                "Diagnose"
              )}
            </button>

            <button
              onClick={handleSearch}
              disabled={!image || isSearching}
              className="w-full bg-transparent text-slate-400 hover:text-teal-400 disabled:text-slate-600 font-medium py-2 rounded-xl transition-all duration-300 flex items-center justify-center gap-2 text-sm"
            >
              {isSearching ? (
                <>
                  <Loader2 className="animate-spin w-4 h-4" /> Searching...
                </>
              ) : (
                <>
                  <Search className="w-4 h-4" /> Find Similar Cases in Database
                </>
              )}
            </button>
          </div>
        </div>

        {/* Results Section */}
        <div className="bg-white/5 backdrop-blur-xl border border-white/10 p-8 rounded-3xl shadow-2xl min-h-[450px] relative overflow-hidden flex flex-col">
          
          {/* Header Title Animating */}
          <div className={`absolute transition-all duration-700 ease-in-out z-10 flex flex-col items-center justify-center
            ${hasContent 
              ? 'top-8 left-8 translate-x-0 translate-y-0' 
              : 'top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2'}`}
          >
            <div className={`transition-all duration-500 flex flex-col items-center ${hasContent ? 'opacity-0 h-0 overflow-hidden' : 'opacity-100'}`}>
               <Leaf className="w-20 h-20 opacity-20 mb-6 text-slate-500" />
            </div>

            <h2 className={`font-bold text-white flex items-center gap-2 transition-all duration-700 ${hasContent ? 'text-2xl' : 'text-4xl'}`}>
              <Info className={`transition-all duration-700 ${hasContent ? 'w-6 h-6 text-teal-400' : 'w-10 h-10 text-slate-500'}`} /> 
              Results
            </h2>

            <div className={`transition-all duration-500 mt-6 ${hasContent ? 'opacity-0 h-0 overflow-hidden m-0' : 'opacity-100'}`}>
               <p className="text-slate-500 text-lg text-center whitespace-nowrap">Upload an image and hit <span className="text-emerald-400/80">Diagnose</span>.</p>
            </div>
          </div>

          <div className={`flex-1 flex flex-col transition-all duration-1000 delay-100 mt-12 w-full ${hasContent ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-8 pointer-events-none'}`}>
            {error ? (
              <div className="bg-red-500/10 border border-red-500/20 rounded-xl p-4 text-red-400 text-center flex flex-col items-center m-auto w-full">
                <p>{error}</p>
                <p className="text-sm mt-2 opacity-80">(Make sure the Python API server is running on port 8000)</p>
              </div>
            ) : results ? (
              <div className="space-y-8 w-full">
                <h3 className="text-emerald-400 font-semibold mb-4 border-b border-white/10 pb-2">AI Diagnosis (Model Prediction)</h3>
                {results.map((res, idx) => {
                  const percent = Math.round(res.probability * 100);
                  const isTop = idx === 0;
                  return (
                    <div key={idx} className="relative animate-fade-in-up" style={{ animationDelay: `${idx * 150}ms` }}>
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
                      
                      {isTop && res.gradcam_path && (
                        <div className="mt-6 rounded-2xl overflow-hidden border border-emerald-500/30 relative w-full aspect-video bg-black/60 group shadow-lg">
                           <div className="absolute inset-0 flex items-center justify-center text-slate-600 z-0">
                              <Loader2 className="animate-spin w-8 h-8" />
                           </div>
                           <img 
                              src={`http://127.0.0.1:8000/image?path=${encodeURIComponent(res.gradcam_path)}`} 
                              alt="Grad-CAM++ Heatmap" 
                              className="w-full h-full object-cover relative z-10 opacity-90 group-hover:opacity-100 group-hover:scale-105 transition-all duration-700"
                              onError={(e) => {
                                (e.target as HTMLImageElement).style.display = 'none';
                              }}
                           />
                           <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/90 via-black/50 to-transparent p-4 z-20 pointer-events-none">
                             <p className="text-emerald-300 text-sm font-semibold flex items-center gap-2">
                               <Info className="w-4 h-4"/> Grad-CAM++ Attention Map
                             </p>
                             <p className="text-slate-400 text-xs mt-1">
                               Vùng màu đỏ/cam là khu vực AI tập trung nhìn vào để chẩn đoán.
                             </p>
                           </div>
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            ) : searchResults ? (
               <div className="w-full">
                 <h3 className="text-teal-400 font-semibold mb-6 border-b border-white/10 pb-2">Similar Cases from Database (Qdrant)</h3>
                 <div className="grid grid-cols-2 gap-4">
                    {searchResults.map((item, idx) => (
                      <div key={idx} className="bg-black/40 rounded-xl overflow-hidden border border-white/5 relative group animate-fade-in-up" style={{ animationDelay: `${idx * 150}ms` }}>
                        <div className="aspect-square bg-slate-800 relative">
                           {/* Fetch image directly from FastAPI backend */}
                           <img 
                              src={`http://127.0.0.1:8000/image?path=${encodeURIComponent(item.payload.image_path)}`} 
                              alt="Similar case" 
                              className="w-full h-full object-cover group-hover:scale-110 transition-transform duration-500"
                              onError={(e) => {
                                (e.target as HTMLImageElement).src = "https://via.placeholder.com/200?text=Image+Not+Found";
                              }}
                           />
                        </div>
                        <div className="p-3 absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/90 to-transparent pt-8">
                           <p className="text-white text-sm font-semibold truncate" title={item.payload.disease_name}>
                              {item.payload.disease_name}
                           </p>
                           <p className="text-teal-400 text-xs mt-1">Similarity: {(item.score * 100).toFixed(1)}%</p>
                        </div>
                      </div>
                    ))}
                 </div>
               </div>
            ) : null}
          </div>
        </div>
      </div>
      </div>
    </main>
  );
}
