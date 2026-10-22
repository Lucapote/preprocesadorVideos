import React, { useRef, useState } from 'react';
import { Play, Pause, Bookmark, Check, X, Tag, Volume2, Mic, FileText, Scissors, CheckCircle, AlertCircle } from 'lucide-react';

export default function VideoPlayer({
  videoUrl,
  markers = [],
  claps = [],
  rawTranscript = '',
  takes = []
}) {
  const videoRef = useRef(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);

  const togglePlay = () => {
    if (!videoRef.current) return;
    if (videoRef.current.paused) {
      videoRef.current.play();
      setIsPlaying(true);
    } else {
      videoRef.current.pause();
      setIsPlaying(false);
    }
  };

  const jumpToTime = (seconds) => {
    if (!videoRef.current) return;
    videoRef.current.currentTime = Math.max(0, seconds);
    videoRef.current.play();
    setIsPlaying(true);
  };

  const handleTimeUpdate = () => {
    if (videoRef.current) {
      setCurrentTime(videoRef.current.currentTime);
    }
  };

  const handleLoadedMetadata = () => {
    if (videoRef.current) {
      setDuration(videoRef.current.duration);
    }
  };

  const formatSeconds = (sec) => {
    if (typeof sec !== 'number' || isNaN(sec)) return '00:00.0';
    const mins = Math.floor(sec / 60);
    const secs = Math.floor(sec % 60);
    const ms = Math.floor((sec % 1) * 10);
    return `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}.${ms}`;
  };

  return (
    <div className="bg-[#111827] border border-gray-800 rounded-2xl p-5 shadow-sm space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-gray-200 uppercase tracking-wider flex items-center gap-2">
          <Bookmark className="w-4 h-4 text-[#188ff0]" /> 3. Inspección Visual y Verificación de Marcas
        </h3>
        <span className="font-mono text-xs text-gray-400 bg-gray-900 px-2.5 py-1 rounded-full border border-gray-800">
          {formatSeconds(currentTime)} / {formatSeconds(duration || 0)}
        </span>
      </div>

      {/* Video Container */}
      <div className="relative aspect-video bg-black rounded-xl overflow-hidden border border-gray-800 flex items-center justify-center group">
        {videoUrl ? (
          <video
            ref={videoRef}
            src={videoUrl}
            onTimeUpdate={handleTimeUpdate}
            onLoadedMetadata={handleLoadedMetadata}
            onPlay={() => setIsPlaying(true)}
            onPause={() => setIsPlaying(false)}
            className="w-full h-full object-contain"
            playsInline
          />
        ) : (
          <div className="text-gray-500 text-xs font-mono">
            Ningún video cargado en el reproductor
          </div>
        )}

        {/* Video Overlay Control Bar */}
        {videoUrl && (
          <div className="absolute bottom-0 inset-x-0 bg-gradient-to-t from-black/90 via-black/40 to-transparent p-3 opacity-90 transition-opacity flex items-center justify-between gap-3">
            <button
              type="button"
              onClick={togglePlay}
              className="w-8 h-8 rounded-full bg-white/20 hover:bg-white/30 text-white flex items-center justify-center backdrop-blur-sm transition-all"
            >
              {isPlaying ? <Pause className="w-4 h-4 fill-current" /> : <Play className="w-4 h-4 fill-current ml-0.5" />}
            </button>

            <input
              type="range"
              min="0"
              max={duration || 100}
              step="0.05"
              value={currentTime}
              onChange={(e) => jumpToTime(parseFloat(e.target.value))}
              className="flex-1 accent-[#188ff0] cursor-pointer h-1.5 bg-gray-700 rounded-lg"
            />

            <span className="text-[11px] font-mono text-gray-300">
              {formatSeconds(currentTime)}
            </span>
          </div>
        )}
      </div>

      {/* Acoustic Claps (Phase 1) Section */}
      {claps.length > 0 && (
        <div className="p-3.5 rounded-xl bg-purple-950/20 border border-purple-800/40 space-y-2">
          <div className="flex items-center justify-between">
            <h4 className="text-xs font-semibold text-purple-300 uppercase tracking-wider flex items-center gap-1.5">
              <Volume2 className="w-3.5 h-3.5 text-purple-400" /> Aplausos Detectados - Anclas Físicas (Fase 1):
            </h4>
            <span className="text-[11px] font-mono text-purple-400">
              {claps.filter((c) => c.clap_detected).length} confirmados
            </span>
          </div>

          <div className="flex flex-wrap gap-2">
            {claps.map((c, i) => (
              <button
                key={`clap-${i}`}
                type="button"
                onClick={() => jumpToTime(c.clap_timestamp)}
                className={`text-xs px-3 py-1.5 rounded-full border font-mono flex items-center gap-2 transition-all active:scale-95 ${
                  c.clap_detected
                    ? 'bg-purple-500/15 hover:bg-purple-500/25 text-purple-200 border-purple-500/40'
                    : 'bg-gray-800 text-gray-400 border-gray-700'
                }`}
              >
                <span className="font-bold">👏 B{c.bloque} T{c.toma}</span>
                <span className="font-mono text-purple-300">[{formatSeconds(c.clap_timestamp)}]</span>
                {c.clap_detected ? (
                  <span className="text-[10px] px-1.5 py-0.5 rounded-full bg-purple-900/60 text-purple-300">
                    Ratio {c.clap_confidence}x &bull; {c.delta_phrase_clap > 0 ? `+${c.delta_phrase_clap}s` : `${c.delta_phrase_clap}s`}
                  </span>
                ) : (
                  <span className="text-[10px] text-amber-400">(Fallback)</span>
                )}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Voice Markers (Phase 0) Section */}
      {markers.length > 0 && (
        <div className="p-3.5 rounded-xl bg-blue-950/20 border border-blue-800/40 space-y-2">
          <div className="flex items-center justify-between">
            <h4 className="text-xs font-semibold text-blue-300 uppercase tracking-wider flex items-center gap-1.5">
              <Mic className="w-3.5 h-3.5 text-blue-400" /> Frases Clave Detectadas (Fase 0):
            </h4>
            <span className="text-[11px] font-mono text-blue-400">
              {markers.length} marcas
            </span>
          </div>

          <div className="flex flex-wrap gap-2">
            {markers.map((m, i) => (
              <button
                key={`marker-${i}`}
                type="button"
                onClick={() => jumpToTime(m.start_time)}
                className="text-xs px-3 py-1.5 rounded-full border font-mono flex items-center gap-2 transition-all active:scale-95 bg-blue-500/15 hover:bg-blue-500/25 text-blue-200 border-blue-500/40"
              >
                <span className="font-bold">🎙️ B{m.bloque} T{m.toma}</span>
                <span>[{formatSeconds(m.start_time)} ➔ {formatSeconds(m.end_time)}]</span>
                <span className="text-[10px] opacity-75">
                  ({Math.round((m.confidence || 1.0) * 100)}%)
                </span>
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Segmented Takes (Phase 2) Section */}
      {takes && takes.length > 0 && (
        <div className="p-4 rounded-xl bg-gradient-to-b from-blue-950/20 to-gray-900/60 border border-blue-500/30 space-y-3">
          <div className="flex items-center justify-between">
            <h4 className="text-xs font-semibold text-blue-300 uppercase tracking-wider flex items-center gap-1.5">
              <Scissors className="w-4 h-4 text-[#188ff0]" /> 4. Tomas Segmentadas &amp; Puntos In/Out (Fase 2):
            </h4>
            <span className="text-[11px] font-mono text-emerald-400 bg-emerald-950/40 px-2.5 py-0.5 rounded-full border border-emerald-500/30">
              {takes.filter((t) => t.status === 'keep').length} a conservar &bull; {takes.filter((t) => t.status === 'discard').length} descartadas
            </span>
          </div>

          <div className="space-y-2.5">
            {takes.map((take, idx) => (
              <div
                key={take.id_toma || idx}
                className={`p-3 rounded-xl border transition-all ${
                  take.status === 'keep'
                    ? 'bg-emerald-950/20 border-emerald-500/40'
                    : 'bg-gray-900/80 border-gray-800 opacity-75 hover:opacity-100'
                }`}
              >
                <div className="flex items-center justify-between gap-2 mb-2">
                  <div className="flex items-center gap-2">
                    <span className="px-2 py-0.5 rounded-md bg-gray-800 text-white font-mono font-bold text-xs border border-gray-700">
                      B{take.bloque} T{take.toma}
                    </span>
                    {take.status === 'keep' ? (
                      <span className="px-2 py-0.5 rounded-full bg-emerald-500/20 text-emerald-300 font-mono text-[10px] border border-emerald-500/40 flex items-center gap-1">
                        <CheckCircle className="w-3 h-3" /> CONSERVAR {take.is_best_take ? '(Definitiva)' : ''}
                      </span>
                    ) : (
                      <span className="px-2 py-0.5 rounded-full bg-amber-500/20 text-amber-300 font-mono text-[10px] border border-amber-500/40 flex items-center gap-1">
                        <AlertCircle className="w-3 h-3" /> {take.discard_reason === 'retake' ? 'DESCARTE (Repetición)' : 'VACÍA'}
                      </span>
                    )}
                    <span className="text-[10px] font-mono text-gray-400">
                      Ancla: {take.anchor_type === 'clap' ? '👏 Aplauso' : '🎙️ Voz'}
                    </span>
                  </div>

                  <span className="text-xs font-mono font-semibold text-gray-300">
                    {take.duration}s útiles
                  </span>
                </div>

                {/* In / Out Jump Buttons */}
                <div className="flex items-center gap-2 mb-2">
                  <button
                    type="button"
                    onClick={() => jumpToTime(take.in_point)}
                    className="text-xs px-2.5 py-1 rounded-lg bg-blue-500/20 hover:bg-blue-500/30 text-blue-300 border border-blue-500/30 font-mono flex items-center gap-1 transition-all active:scale-95"
                    title="Saltar al punto In (primera palabra de discurso)"
                  >
                    <span>▶ In:</span>
                    <span className="font-bold">{formatSeconds(take.in_point)}</span>
                  </button>

                  <button
                    type="button"
                    onClick={() => jumpToTime(take.out_point)}
                    className="text-xs px-2.5 py-1 rounded-lg bg-purple-500/20 hover:bg-purple-500/30 text-purple-300 border border-purple-500/30 font-mono flex items-center gap-1 transition-all active:scale-95"
                    title="Saltar al punto Out (última palabra de discurso)"
                  >
                    <span>⏹ Out:</span>
                    <span className="font-bold">{formatSeconds(take.out_point)}</span>
                  </button>

                  <span className="text-[10px] text-gray-500 font-mono ml-auto">
                    ID: {take.take_id || take.id_toma}
                  </span>
                </div>

                {/* Internal Silences Detection (> 1.0s) */}
                {take.internal_silences && take.internal_silences.length > 0 && (
                  <div className="flex flex-wrap items-center gap-1.5 mb-2">
                    <span className="text-[10px] text-amber-400 font-mono flex items-center gap-1">
                      ⚠️ Silencios internos ({take.internal_silences.length}):
                    </span>
                    {take.internal_silences.map((silence, sIdx) => (
                      <span
                        key={sIdx}
                        className="text-[10px] px-2 py-0.5 rounded-full bg-amber-500/15 text-amber-300 font-mono border border-amber-500/30"
                      >
                        {silence.duration}s ({formatSeconds(silence.start_time)} ➔ {formatSeconds(silence.end_time)})
                      </span>
                    ))}
                  </div>
                )}

                {/* Acoustic Energy Phrases (CapCut Directors Metadata) */}
                {take.acoustic_phrases && take.acoustic_phrases.length > 0 && (
                  <div className="space-y-1.5 mb-2.5">
                    <span className="text-[10px] text-gray-400 font-mono block">
                      🎙️ Dinámicas de ritmo y energía acústica (dB):
                    </span>
                    <div className="space-y-1">
                      {take.acoustic_phrases.map((phrase, pIdx) => (
                        <div
                          key={pIdx}
                          className="flex items-center justify-between gap-2 p-1.5 rounded-lg bg-black/40 border border-gray-800/80 text-xs font-mono"
                        >
                          <div className="flex items-center gap-2 flex-1 min-w-0">
                            <button
                              type="button"
                              onClick={() => jumpToTime(phrase.start)}
                              className="text-[10px] text-blue-400 hover:text-blue-300 font-bold shrink-0 underline"
                              title="Saltar a esta frase"
                            >
                              [{formatSeconds(phrase.start)}]
                            </button>
                            <span className="truncate text-gray-300 italic">"{phrase.text}"</span>
                          </div>
                          <div className="flex items-center gap-1.5 shrink-0">
                            {phrase.pre_pause > 0 && (
                              <span className="text-[9px] px-1.5 py-0.5 rounded bg-gray-800 text-gray-400 border border-gray-700">
                                ⏸ {phrase.pre_pause}s
                              </span>
                            )}
                            <span
                              className={`text-[9px] font-bold px-2 py-0.5 rounded-full border ${
                                phrase.energy.includes('PICO_MAXIMO')
                                  ? 'bg-fuchsia-500/20 text-fuchsia-300 border-fuchsia-500/40'
                                  : phrase.energy.includes('ALTA')
                                  ? 'bg-blue-500/20 text-blue-300 border-blue-500/40'
                                  : phrase.energy.includes('BAJA')
                                  ? 'bg-amber-500/20 text-amber-300 border-amber-500/40'
                                  : 'bg-gray-800 text-gray-400 border-gray-700'
                              }`}
                            >
                              {phrase.energy}
                            </span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Speech text fallback if no phrases */}
                {(!take.acoustic_phrases || take.acoustic_phrases.length === 0) && (
                  (take.transcript_snippet || take.speech_text) ? (
                    <p className="text-xs text-gray-300 font-mono italic bg-black/30 p-2 rounded-lg border border-gray-800/80 mb-2">
                      "{take.transcript_snippet || take.speech_text}"
                    </p>
                  ) : (
                    <p className="text-[11px] text-gray-500 italic mb-2">
                      Sin discurso detectable en esta toma (pre-roll / toma abortada).
                    </p>
                  )
                )}

                {/* Physical File & URL status */}
                {take.rendered && take.video_url && (
                  <div className="flex items-center justify-between pt-2 border-t border-gray-800/60 text-[10px] font-mono">
                    <span className="text-emerald-400 flex items-center gap-1">
                      <Check className="w-3 h-3" /> Clip exportado con FFmpeg (48kHz &bull; 30fps CFR)
                    </span>
                    <a
                      href={take.video_url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-blue-400 hover:text-blue-300 underline font-semibold"
                    >
                      Descargar / Abrir MP4 ↗
                    </a>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Transcript Preview */}
      {rawTranscript && (
        <div className="p-3.5 rounded-xl bg-gray-900/70 border border-gray-800 space-y-1.5">
          <h4 className="text-xs font-semibold text-gray-400 uppercase tracking-wider flex items-center gap-1.5">
            <FileText className="w-3.5 h-3.5 text-emerald-400" /> Transcripción Fonética Whisper:
          </h4>
          <p className="text-xs text-gray-300 italic font-mono leading-relaxed line-clamp-3">
            "{rawTranscript}"
          </p>
        </div>
      )}

      {/* Empty State */}
      {markers.length === 0 && claps.length === 0 && (!takes || takes.length === 0) && (
        <div className="p-4 rounded-xl bg-gray-900/40 border border-gray-800/60 text-center">
          <p className="text-xs text-gray-500 italic">
            Ejecuta la Fase 2 o Fase 0/1 para ver los timestamps interactivos de aplausos, locución y tomas recortadas.
          </p>
        </div>
      )}
    </div>
  );
}
