import React, { useEffect, useRef, useState } from 'react';
import { Terminal, Trash2, ArrowDown } from 'lucide-react';

export default function LogTerminal({ logs = [], isProcessing }) {
  const terminalEndRef = useRef(null);
  const [autoScroll, setAutoScroll] = useState(true);

  useEffect(() => {
    if (autoScroll) {
      terminalEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }
  }, [logs, autoScroll]);

  const getLogLevelClass = (level) => {
    switch (level?.toUpperCase()) {
      case 'SUCCESS':
        return 'text-emerald-400 font-semibold';
      case 'WARN':
        return 'text-amber-400 font-semibold';
      case 'ERROR':
        return 'text-rose-400 font-semibold';
      case 'INFO':
      default:
        return 'text-[#188ff0]';
    }
  };

  return (
    <div className="bg-[#05070a] border border-gray-800 rounded-2xl overflow-hidden flex flex-col h-80 shadow-2xl">
      {/* Terminal Titlebar */}
      <div className="bg-[#0d1117] px-4 py-2.5 border-b border-gray-800 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className="flex gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full bg-red-500/80 inline-block" />
            <span className="w-2.5 h-2.5 rounded-full bg-yellow-500/80 inline-block" />
            <span className="w-2.5 h-2.5 rounded-full bg-green-500/80 inline-block" />
          </div>
          <span className="text-xs font-mono text-gray-400 flex items-center gap-1.5 ml-2">
            <Terminal className="w-3.5 h-3.5 text-gray-500" />
            fastapi_pipeline.log
          </span>
        </div>

        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={() => setAutoScroll(!autoScroll)}
            className={`text-[11px] px-2 py-0.5 rounded border transition-colors flex items-center gap-1 ${
              autoScroll
                ? 'bg-blue-500/10 text-blue-400 border-blue-500/30'
                : 'bg-gray-800 text-gray-500 border-gray-700'
            }`}
          >
            <ArrowDown className="w-3 h-3" />
            Auto-scroll
          </button>
          {isProcessing && (
            <span className="inline-flex items-center gap-1 text-[11px] font-mono text-blue-400">
              <span className="w-1.5 h-1.5 rounded-full bg-blue-400 animate-ping" />
              LIVE
            </span>
          )}
        </div>
      </div>

      {/* Terminal Body */}
      <div className="p-4 font-mono text-xs overflow-y-auto flex-1 space-y-1.5 leading-relaxed selection:bg-blue-900/50">
        {logs.length === 0 ? (
          <div className="text-gray-600 italic py-6 text-center font-mono">
            Esperando orden de ejecución. Los logs del worker de segundo plano aparecerán aquí en tiempo real...
          </div>
        ) : (
          logs.map((log, index) => (
            <div key={index} className="flex items-start gap-2 text-gray-300 font-mono">
              <span className="text-gray-500 select-none text-[11px]">{log.timestamp}</span>
              <span className={`text-[11px] select-none ${getLogLevelClass(log.level)}`}>
                [{log.level}]
              </span>
              <span className="flex-1 break-all text-gray-200">{log.message}</span>
            </div>
          ))
        )}
        <div ref={terminalEndRef} />
      </div>
    </div>
  );
}
