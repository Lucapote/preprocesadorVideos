import React from 'react';
import { Activity, Cpu, Layers, Terminal } from 'lucide-react';

export default function Header({ backendConnected }) {
  return (
    <header className="border-b border-gray-800 bg-[#0d131f]/80 backdrop-blur-md sticky top-0 z-50 px-6 py-4">
      <div className="max-w-7xl mx-auto flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-[#188ff0] to-[#ef18d6] flex items-center justify-center shadow-lg shadow-blue-500/20">
            <Layers className="w-5 h-5 text-white" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="font-bold text-lg text-white tracking-tight">
                Preprocesador Lab
              </h1>
              <span className="px-2 py-0.5 text-[10px] font-mono uppercase tracking-wider font-semibold bg-blue-500/10 text-[#188ff0] border border-blue-500/20 rounded-full">
                Fase 0 Cockpit
              </span>
            </div>
            <p className="text-xs text-gray-400">
              Workbench de experimentación: Ingesta, aplausos, Whisper y manifest para CapCut
            </p>
          </div>
        </div>

        <div className="flex items-center gap-3 w-full md:w-auto justify-between md:justify-end">
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-gray-900 border border-gray-800 text-xs">
            <span
              className={`w-2 h-2 rounded-full ${
                backendConnected ? 'bg-emerald-500 shadow-sm shadow-emerald-500' : 'bg-rose-500 animate-pulse'
              }`}
            />
            <span className="text-gray-300 font-mono text-[11px]">
              {backendConnected ? 'FastAPI: Online (8000)' : 'FastAPI: Desconectado'}
            </span>
          </div>

          <div className="hidden sm:flex items-center gap-1.5 text-xs text-gray-400 font-mono">
            <span className="px-2 py-1 rounded bg-gray-900 border border-gray-800 flex items-center gap-1">
              <Cpu className="w-3.5 h-3.5 text-blue-400" /> FFmpeg
            </span>
            <span className="px-2 py-1 rounded bg-gray-900 border border-gray-800 flex items-center gap-1">
              <Terminal className="w-3.5 h-3.5 text-pink-400" /> Whisper
            </span>
          </div>
        </div>
      </div>
    </header>
  );
}
