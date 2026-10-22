import React, { useRef, useState } from 'react';
import { UploadCloud, Film, CheckCircle2, AlertCircle, RefreshCw, Sparkles } from 'lucide-react';
import { uploadVideo } from '../services/api';

export default function UploadZone({ selectedFile, onFileSelected, isProcessing }) {
  const fileInputRef = useRef(null);
  const [isDragging, setIsDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState(null);

  const handleFileChange = async (e) => {
    const file = e.target.files?.[0];
    if (file) {
      await processSelectedFile(file);
    }
  };

  const handleDrop = async (e) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files?.[0];
    if (file) {
      await processSelectedFile(file);
    }
  };

  const processSelectedFile = async (file) => {
    setUploadError(null);
    setUploading(true);
    try {
      const response = await uploadVideo(file);
      onFileSelected({
        name: file.name,
        size_mb: response.data.size_mb,
        storage_path: response.data.storage_path,
        video_url: `/storage/${response.data.storage_path}`,
        isSample: false
      });
    } catch (err) {
      setUploadError(err.message || 'Error al subir el video');
    } finally {
      setUploading(false);
    }
  };

  const handleUseSample = () => {
    onFileSelected({
      name: 'sample_vertical_raw_take.mp4',
      size_mb: 42.5,
      storage_path: null,
      video_url: 'https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerBlazes.mp4',
      isSample: true
    });
    setUploadError(null);
  };

  return (
    <div className="bg-[#111827] border border-gray-800 rounded-2xl p-5 shadow-sm">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-semibold text-gray-200 uppercase tracking-wider flex items-center gap-2">
          <Film className="w-4 h-4 text-[#188ff0]" /> 1. Origen de Video Crudo
        </h3>
        <button
          type="button"
          onClick={handleUseSample}
          disabled={isProcessing || uploading}
          className="text-xs px-2.5 py-1 rounded-full bg-purple-500/10 hover:bg-purple-500/20 text-purple-300 border border-purple-500/30 flex items-center gap-1.5 transition-all disabled:opacity-50"
        >
          <Sparkles className="w-3.5 h-3.5 text-purple-400" />
          Cargar Video Demo
        </button>
      </div>

      <div
        onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={handleDrop}
        onClick={() => !uploading && !isProcessing && fileInputRef.current?.click()}
        className={`border-2 border-dashed rounded-xl p-6 text-center cursor-pointer transition-all ${
          isDragging
            ? 'border-[#188ff0] bg-blue-500/5'
            : 'border-gray-700 hover:border-gray-600 bg-gray-900/50'
        } ${uploading ? 'pointer-events-none opacity-70' : ''}`}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept="video/mp4,video/quicktime,video/x-matroska"
          onChange={handleFileChange}
          className="hidden"
        />

        {uploading ? (
          <div className="flex flex-col items-center justify-center gap-2 py-4">
            <RefreshCw className="w-8 h-8 text-[#188ff0] animate-spin" />
            <p className="text-sm font-medium text-gray-300">Subiendo video al storage local de FastAPI...</p>
            <p className="text-xs text-gray-500 font-mono">Stream directo a backend/storage/uploads</p>
          </div>
        ) : selectedFile ? (
          <div className="flex flex-col items-center justify-center gap-2 py-2">
            <div className="w-12 h-12 rounded-full bg-emerald-500/10 border border-emerald-500/30 flex items-center justify-center text-emerald-400">
              <CheckCircle2 className="w-6 h-6" />
            </div>
            <p className="text-sm font-semibold text-white truncate max-w-xs">{selectedFile.name}</p>
            <span className="text-xs text-gray-400 font-mono">
              {selectedFile.isSample ? 'Clip Sintético Demo' : `${selectedFile.size_mb} MB en storage`}
            </span>
            <span className="text-[11px] text-[#188ff0] hover:underline mt-1">
              Click o arrastra para reemplazar
            </span>
          </div>
        ) : (
          <div className="flex flex-col items-center justify-center gap-2 py-4">
            <div className="w-12 h-12 rounded-full bg-gray-800 flex items-center justify-center text-gray-400">
              <UploadCloud className="w-6 h-6" />
            </div>
            <p className="text-sm font-medium text-gray-300">
              Arrastra un archivo MP4 o haz clic para explorar
            </p>
            <p className="text-xs text-gray-500">
              Formatos soportados: MP4, MOV, MKV (Hasta 500 MB)
            </p>
          </div>
        )}
      </div>

      {uploadError && (
        <div className="mt-3 p-2.5 rounded-lg bg-rose-500/10 border border-rose-500/30 flex items-center gap-2 text-rose-400 text-xs">
          <AlertCircle className="w-4 h-4 flex-shrink-0" />
          <span>{uploadError}</span>
        </div>
      )}
    </div>
  );
}
