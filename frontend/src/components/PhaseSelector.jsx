import React, { useState } from 'react';
import { Play, Loader2, CheckCircle2, AlertTriangle, FastForward, Mic, Volume2, Sparkles, Layers, Scissors } from 'lucide-react';

export default function PhaseSelector({
  onRunPhase0,
  onRunPhase1,
  onRunPhase2,
  onRunFullPipeline,
  isProcessing,
  currentTask,
  activePhase,
  hasFile
}) {
  const [selectedTab, setSelectedTab] = useState('phase2'); // 'phase2' | 'full' | 'phase0' | 'phase1'

  const getStatusBadge = () => {
    if (!currentTask && !isProcessing) {
      return (
        <span className="px-3 py-1 rounded-full text-xs font-mono bg-gray-800 text-gray-400 border border-gray-700">
          Standby
        </span>
      );
    }
    if (isProcessing) {
      return (
        <span className="px-3 py-1 rounded-full text-xs font-mono bg-blue-500/10 text-blue-400 border border-blue-500/30 flex items-center gap-1.5 animate-pulse">
          <Loader2 className="w-3 h-3 animate-spin" /> Procesando {activePhase ? `(${activePhase})` : ''}
        </span>
      );
    }
    if (currentTask?.status === 'completed') {
      return (
        <span className="px-3 py-1 rounded-full text-xs font-mono bg-emerald-500/10 text-emerald-400 border border-emerald-500/30 flex items-center gap-1.5">
          <CheckCircle2 className="w-3 h-3" /> Completado (100%)
        </span>
      );
    }
    if (currentTask?.status === 'failed') {
      return (
        <span className="px-3 py-1 rounded-full text-xs font-mono bg-rose-500/10 text-rose-400 border border-rose-500/30 flex items-center gap-1.5">
          <AlertTriangle className="w-3 h-3" /> Error
        </span>
      );
    }
    return (
      <span className="px-3 py-1 rounded-full text-xs font-mono bg-amber-500/10 text-amber-400 border border-amber-500/30">
        En cola
      </span>
    );
  };

  const handleExecute = () => {
    if (selectedTab === 'phase2') onRunPhase2();
    else if (selectedTab === 'phase0') onRunPhase0();
    else if (selectedTab === 'phase1') onRunPhase1();
    else onRunFullPipeline();
  };

  return (
    <div className="bg-[#111827] border border-gray-800 rounded-2xl p-5 shadow-sm space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-gray-200 uppercase tracking-wider flex items-center gap-2">
          <FastForward className="w-4 h-4 text-[#ef18d6]" /> 2. Pipeline de Laboratorio
        </h3>
        {getStatusBadge()}
      </div>

      {/* Segmented Controller (Pill Shape) */}
      <div className="bg-gray-900/90 p-1 rounded-full border border-gray-800 flex items-center gap-1">
        <button
          type="button"
          onClick={() => setSelectedTab('phase2')}
          className={`flex-1 py-1.5 px-3 rounded-full text-xs font-medium transition-all flex items-center justify-center gap-1.5 ${
            selectedTab === 'phase2'
              ? 'bg-gradient-to-r from-[#188ff0] to-[#ef18d6] text-white shadow-md font-semibold'
              : 'text-gray-400 hover:text-gray-200'
          }`}
        >
          <Scissors className="w-3.5 h-3.5" />
          <span>Fase 2 (Tomas In/Out)</span>
        </button>

        <button
          type="button"
          onClick={() => setSelectedTab('full')}
          className={`flex-1 py-1.5 px-3 rounded-full text-xs font-medium transition-all flex items-center justify-center gap-1.5 ${
            selectedTab === 'full'
              ? 'bg-gray-800 text-white shadow-md border border-gray-700 font-semibold'
              : 'text-gray-400 hover:text-gray-200'
          }`}
        >
          <Layers className="w-3.5 h-3.5" />
          <span>Fase 0 + 1 (Full)</span>
        </button>

        <button
          type="button"
          onClick={() => setSelectedTab('phase1')}
          className={`flex-1 py-1.5 px-3 rounded-full text-xs font-medium transition-all flex items-center justify-center gap-1.5 ${
            selectedTab === 'phase1'
              ? 'bg-gray-800 text-white shadow-md border border-gray-700 font-semibold'
              : 'text-gray-400 hover:text-gray-200'
          }`}
        >
          <Volume2 className="w-3.5 h-3.5 text-purple-400" />
          <span>Aplausos</span>
        </button>

        <button
          type="button"
          onClick={() => setSelectedTab('phase0')}
          className={`flex-1 py-1.5 px-3 rounded-full text-xs font-medium transition-all flex items-center justify-center gap-1.5 ${
            selectedTab === 'phase0'
              ? 'bg-gray-800 text-white shadow-md border border-gray-700 font-semibold'
              : 'text-gray-400 hover:text-gray-200'
          }`}
        >
          <Mic className="w-3.5 h-3.5 text-blue-400" />
          <span>Whisper</span>
        </button>
      </div>

      {/* Tab Description Card */}
      <div className="p-4 rounded-xl bg-gray-900/80 border border-gray-800 text-xs text-gray-300">
        {selectedTab === 'phase2' && (
          <div>
            <h4 className="font-semibold text-white flex items-center gap-1.5 mb-1">
              <Scissors className="w-3.5 h-3.5 text-[#188ff0]" /> Paso 2: Delimitación de Tomas (In / Out Points)
            </h4>
            <p className="text-gray-400 leading-relaxed">
              Calcula los puntos In y Out exactos eliminando el pre-roll muerto entre claqueta/aplauso y la primera palabra,
              preserva el discurso completo y marca automáticamente los descartes por repetición (retakes).
            </p>
          </div>
        )}

        {selectedTab === 'full' && (
          <div>
            <h4 className="font-semibold text-white flex items-center gap-1.5 mb-1">
              <Sparkles className="w-3.5 h-3.5 text-[#ef18d6]" /> Pipeline Completo (Fase 0 + 1)
            </h4>
            <p className="text-gray-400 leading-relaxed">
              Extrae el audio WAV, transcribe con Faster-Whisper, localiza marcas "bloque X, toma Y",
              y busca los aplausos acústicos (anclas físicas) en ventanas acotadas con SciPy.
            </p>
          </div>
        )}

        {selectedTab === 'phase0' && (
          <div>
            <h4 className="font-semibold text-white flex items-center gap-1.5 mb-1">
              <Mic className="w-3.5 h-3.5 text-[#188ff0]" /> Fase 0: Transcripción y Marcas de Voz
            </h4>
            <p className="text-gray-400 leading-relaxed">
              Convierte el audio a mono 16 kHz PCM, ejecuta Faster-Whisper localmente con timestamps
              a nivel de palabra y detecta la frase "bloque X, toma Y" con regex tolerante.
            </p>
          </div>
        )}

        {selectedTab === 'phase1' && (
          <div>
            <h4 className="font-semibold text-white flex items-center gap-1.5 mb-1">
              <Volume2 className="w-3.5 h-3.5 text-purple-400" /> Fase 1: Anclaje Acústico de Aplausos
            </h4>
            <p className="text-gray-400 leading-relaxed">
              Examina la ventana temporal [-1.5s, +2.5s] alrededor de cada frase para hallar
              el impacto del aplauso mediante la transformada de Hilbert y prominencia RMS.
            </p>
          </div>
        )}

        {currentTask && (
          <div className="mt-3 pt-3 border-t border-gray-800">
            <div className="flex justify-between text-xs text-gray-400 font-mono mb-1">
              <span>Tarea: {currentTask.task_id}</span>
              <span>{Math.round(currentTask.progress || 0)}%</span>
            </div>
            <div className="w-full bg-gray-800 h-1.5 rounded-full overflow-hidden">
              <div
                className="bg-gradient-to-r from-[#188ff0] to-[#ef18d6] h-full transition-all duration-300 ease-out"
                style={{ width: `${currentTask.progress || 0}%` }}
              />
            </div>
          </div>
        )}
      </div>

      {/* Main Action CTA Button */}
      <button
        type="button"
        onClick={handleExecute}
        disabled={isProcessing || !hasFile}
        className={`w-full h-12 rounded-full flex items-center justify-center gap-2 text-xs font-bold uppercase tracking-wider transition-all duration-200 ${
          !hasFile
            ? 'bg-gray-800 text-gray-500 cursor-not-allowed border border-gray-700'
            : isProcessing
            ? 'bg-blue-600/50 text-white cursor-wait'
            : 'bg-[#188ff0] hover:bg-blue-600 text-white shadow-lg shadow-blue-500/20 active:scale-[0.98]'
        }`}
      >
        {isProcessing ? (
          <>
            <Loader2 className="w-4 h-4 animate-spin" />
            Procesando Audio en Local...
          </>
        ) : (
          <>
            <Play className="w-4 h-4 fill-current" />
            {selectedTab === 'full' && 'Ejecutar Pipeline Completo (Fase 0 + 1)'}
            {selectedTab === 'phase0' && 'Ejecutar Fase 0: Transcripción y Marcas'}
            {selectedTab === 'phase1' && 'Ejecutar Fase 1: Detección de Aplausos'}
          </>
        )}
      </button>
    </div>
  );
}
