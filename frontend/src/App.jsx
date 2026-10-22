import React, { useState, useEffect, useRef } from 'react';
import Header from './components/Header';
import UploadZone from './components/UploadZone';
import PhaseSelector from './components/PhaseSelector';
import LogTerminal from './components/LogTerminal';
import VideoPlayer from './components/VideoPlayer';
import JsonViewer from './components/JsonViewer';
import { checkHealth, triggerPhase0, getTaskStatus, detectMarkers, runPhase1, runPhase2 } from './services/api';

export default function App() {
  const [backendConnected, setBackendConnected] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const [currentTask, setCurrentTask] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [activePhase, setActivePhase] = useState(null);
  const [logs, setLogs] = useState([]);
  const [resultData, setResultData] = useState(null);

  const pollingTimerRef = useRef(null);

  // Comprobar salud del backend periódicamente
  useEffect(() => {
    const pingBackend = async () => {
      try {
        await checkHealth();
        setBackendConnected(true);
      } catch (err) {
        setBackendConnected(false);
      }
    };
    pingBackend();
    const interval = setInterval(pingBackend, 5000);
    return () => clearInterval(interval);
  }, []);

  // Limpieza de timer de polling
  useEffect(() => {
    return () => {
      if (pollingTimerRef.current) clearInterval(pollingTimerRef.current);
    };
  }, []);

  const addLog = (message, level = 'INFO') => {
    setLogs((prev) => [
      ...prev,
      {
        timestamp: new Date().toLocaleTimeString(),
        level,
        message
      }
    ]);
  };

  const handleFileSelected = (fileData) => {
    setSelectedFile(fileData);
    setCurrentTask(null);
    setLogs([]);
    setResultData(null);
    addLog(`Archivo seleccionado: ${fileData.name} (${fileData.isSample ? 'Demo' : `${fileData.size_mb} MB`})`, 'SUCCESS');
  };

  // Ejecución de la Fase 0 (Detección de marcas y Whisper)
  const handleRunPhase0 = async () => {
    if (!selectedFile || isProcessing) return;

    setIsProcessing(true);
    setActivePhase('Fase 0: Whisper');
    setLogs([]);
    setResultData(null);

    addLog('[Fase 0] Inicializando extracción de audio mono 16 kHz con FFmpeg...');
    addLog('[Fase 0] Enviando stream a Faster-Whisper (CUDA/CPU int8)...');

    try {
      const response = await detectMarkers({
        filePath: selectedFile.storage_path,
        sampleMode: selectedFile.isSample
      });

      addLog(`[Fase 0] Transcripción completada (${response.total_duration_s}s).`, 'SUCCESS');
      addLog(`[Fase 0] Se detectaron ${response.markers_detected?.length || 0} marcas de voz ("bloque X, toma Y").`, 'SUCCESS');

      setResultData(response);
    } catch (err) {
      addLog(`[ERROR Fase 0] ${err.message}`, 'ERROR');
    } finally {
      setIsProcessing(false);
      setActivePhase(null);
    }
  };

  // Ejecución de la Fase 1 (Detección acústica de aplausos)
  const handleRunPhase1 = async () => {
    if (!selectedFile || isProcessing) return;

    setIsProcessing(true);
    setActivePhase('Fase 1: Aplausos');
    setLogs([]);

    addLog('[Fase 1] Inicializando análisis DSP de transitorios (NumPy + SciPy)...');
    addLog('[Fase 1] Buscando picos en ventanas acotadas [-1.5s, +2.5s] respecto a las frases...');

    try {
      const response = await runPhase1({
        filePath: selectedFile.storage_path,
        markers: resultData?.markers_detected || null,
        sampleMode: selectedFile.isSample
      });

      const confirmed = response.claps_detected?.filter((c) => c.clap_detected).length || 0;
      addLog(`[Fase 1] Análisis completado. ${confirmed} aplausos confirmados como anclas físicas.`, 'SUCCESS');

      // Enriquecer resultado combinando con los datos existentes
      setResultData((prev) => ({
        ...prev,
        ...response,
        markers_detected: prev?.markers_detected || response.markers_detected || []
      }));
    } catch (err) {
      addLog(`[ERROR Fase 1] ${err.message}`, 'ERROR');
    } finally {
      setIsProcessing(false);
      setActivePhase(null);
    }
  };

  // Ejecución de la Fase 2 (Delimitación de tomas In/Out y descartes)
  const handleRunPhase2 = async () => {
    if (!selectedFile || isProcessing) return;

    setIsProcessing(true);
    setActivePhase('Fase 2: Tomas In/Out');
    setLogs([]);
    setResultData(null);

    addLog('[Fase 2] Extrayendo audio y transcribiendo con Faster-Whisper a nivel palabra...');
    addLog('[Fase 2] Localizando claquetas y calculando anclas acústicas con SciPy...');
    addLog('[Fase 2] Delimitando puntos In/Out recortando pre-roll y silencios post-toma...');

    try {
      const response = await runPhase2({
        filePath: selectedFile.storage_path,
        sampleMode: selectedFile.isSample
      });

      addLog(`[Fase 2] Segmentación lista: ${response.total_takes} tomas procesadas.`, 'SUCCESS');
      addLog(`[Fase 2] ${response.kept_takes_count} tomas útiles (${response.usable_duration_s}s) &bull; ${response.discarded_takes_count} descartes.`, 'SUCCESS');

      setResultData(response);
    } catch (err) {
      addLog(`[ERROR Fase 2] ${err.message}`, 'ERROR');
    } finally {
      setIsProcessing(false);
      setActivePhase(null);
    }
  };

  // Pipeline Completo (Fase 0 + Fase 1 secuencial)
  const handleRunFullPipeline = async () => {
    if (!selectedFile || isProcessing) return;

    setIsProcessing(true);
    setActivePhase('Pipeline Completo');
    setLogs([]);
    setResultData(null);

    addLog('[Pipeline] Paso 1: Ejecutando Fase 0 (Extracción FFmpeg y Faster-Whisper)...');

    try {
      const phase0 = await detectMarkers({
        filePath: selectedFile.storage_path,
        sampleMode: selectedFile.isSample
      });

      addLog(`[Fase 0] Transcripción lista. ${phase0.markers_detected?.length || 0} frases detectadas.`, 'SUCCESS');
      setResultData(phase0);

      addLog('[Pipeline] Paso 2: Ejecutando Fase 1 (Análisis de envolvente de Hilbert y aplausos)...');

      const phase1 = await runPhase1({
        filePath: selectedFile.storage_path,
        markers: phase0.markers_detected,
        sampleMode: selectedFile.isSample
      });

      const confirmed = phase1.claps_detected?.filter((c) => c.clap_detected).length || 0;
      addLog(`[Fase 1] ${confirmed} aplausos anclados exitosamente. Pipeline listo para segmentación.`, 'SUCCESS');

      setResultData({
        ...phase0,
        ...phase1,
        markers_detected: phase0.markers_detected
      });
    } catch (err) {
      addLog(`[ERROR Pipeline] ${err.message}`, 'ERROR');
    } finally {
      setIsProcessing(false);
      setActivePhase(null);
    }
  };

  return (
    <div className="min-h-screen bg-[#0b0f17] text-gray-100 flex flex-col font-inter">
      <Header backendConnected={backendConnected} />

      <main className="flex-1 max-w-7xl w-full mx-auto p-4 md:p-6 lg:p-8 space-y-6">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Panel Izquierdo: Ingesta, Selector de Fases y Terminal */}
          <div className="lg:col-span-5 space-y-6">
            <UploadZone
              selectedFile={selectedFile}
              onFileSelected={handleFileSelected}
              isProcessing={isProcessing}
            />

            <PhaseSelector
              onRunPhase0={handleRunPhase0}
              onRunPhase1={handleRunPhase1}
              onRunPhase2={handleRunPhase2}
              onRunFullPipeline={handleRunFullPipeline}
              isProcessing={isProcessing}
              currentTask={currentTask}
              activePhase={activePhase}
              hasFile={Boolean(selectedFile)}
            />

            <LogTerminal
              logs={logs}
              isProcessing={isProcessing}
            />
          </div>

          {/* Panel Derecho: Reproductor de Video con Marcas y Visor JSON */}
          <div className="lg:col-span-7 space-y-6">
            <VideoPlayer
              videoUrl={resultData?.video_metadata?.video_url || selectedFile?.video_url}
              markers={resultData?.markers_detected || []}
              claps={resultData?.claps_detected || []}
              rawTranscript={resultData?.raw_transcript || ''}
              takes={resultData?.takes || []}
            />

            <JsonViewer data={resultData} />
          </div>
        </div>
      </main>

      <footer className="border-t border-gray-800/80 py-4 text-center text-xs text-gray-500 font-mono">
        Video Processing Pipeline Lab &bull; Fase 0 (Whisper) &amp; Fase 1 (Aplausos DSP) &bull; FastAPI + React
      </footer>
    </div>
  );
}
