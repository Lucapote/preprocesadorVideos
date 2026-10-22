import React, { useState } from 'react';
import { Code2, Copy, Check, FileJson } from 'lucide-react';

export default function JsonViewer({ data }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    if (!data) return;
    navigator.clipboard.writeText(JSON.stringify(data, null, 2));
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="bg-[#111827] border border-gray-800 rounded-2xl p-5 shadow-sm space-y-3 flex flex-col h-full">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-gray-200 uppercase tracking-wider flex items-center gap-2">
          <FileJson className="w-4 h-4 text-emerald-400" /> 4. Manifiesto JSON Generado
        </h3>
        <button
          type="button"
          onClick={handleCopy}
          disabled={!data}
          className="text-xs px-2.5 py-1 rounded-lg bg-gray-800 hover:bg-gray-700 text-gray-300 border border-gray-700 flex items-center gap-1.5 transition-all disabled:opacity-40"
        >
          {copied ? (
            <>
              <Check className="w-3.5 h-3.5 text-emerald-400" /> Copiado
            </>
          ) : (
            <>
              <Copy className="w-3.5 h-3.5 text-gray-400" /> Copiar JSON
            </>
          )}
        </button>
      </div>

      <div className="bg-[#05070a] border border-gray-800 rounded-xl p-4 font-mono text-xs overflow-auto flex-1 max-h-96">
        {data ? (
          <pre className="text-emerald-400/90 whitespace-pre font-mono leading-relaxed">
            {JSON.stringify(data, null, 2)}
          </pre>
        ) : (
          <div className="text-gray-600 italic py-12 text-center">
            El payload JSON del resultado de la Fase 0 se desplegará aquí una vez finalizado el proceso.
          </div>
        )}
      </div>
    </div>
  );
}
