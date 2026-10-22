import io
import wave
import numpy as np
import pytest
from app.services.silence_detector import (
    Silence,
    _db_volume_per_window,
    _smooth,
    _mark_silence_with_hysteresis,
    _silences_from_db,
    extraer_metadatos_acusticos
)


def create_synthetic_wav_bytes(dur_sec: float = 3.0, sr: int = 16000) -> bytes:
    """Crea un WAV mono 16-bit sintético: 1s tono (voz), 1s silencio, 1s tono."""
    t1 = np.linspace(0, 1.0, int(1.0 * sr), endpoint=False)
    tone1 = (0.5 * np.sin(2 * np.pi * 440 * t1) * 32767).astype(np.int16)
    
    silence = np.zeros(int(1.0 * sr), dtype=np.int16)
    
    t2 = np.linspace(0, 1.0, int(1.0 * sr), endpoint=False)
    tone2 = (0.4 * np.sin(2 * np.pi * 440 * t2) * 32767).astype(np.int16)
    
    data = np.concatenate([tone1, silence, tone2])
    
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(data.tobytes())
    return buf.getvalue()


def test_db_volume_and_hysteresis():
    sr = 16000
    # 0.5s tono alto (-6 dB aprox) + 1.0s silencio (-60 dB) + 0.5s tono alto
    s1 = np.ones(int(0.5 * sr), dtype=np.float32) * 16384.0
    s_sil = np.ones(int(1.0 * sr), dtype=np.float32) * 10.0
    s2 = np.ones(int(0.5 * sr), dtype=np.float32) * 16384.0
    samples = np.concatenate([s1, s_sil, s2])
    
    db_windows = _db_volume_per_window(samples, sr, window_ms=25.0)
    assert len(db_windows) > 0
    assert np.max(db_windows) > -10.0
    assert np.min(db_windows) < -40.0
    
    silences = _silences_from_db(db_windows, min_silence_duration_ms=500.0, window_ms=25.0)
    assert len(silences) >= 1
    # El silencio debe estar centrado alrededor de 0.5s - 1.5s
    s = silences[0]
    assert 0.4 <= s.start <= 0.6
    assert 1.4 <= s.end <= 1.6
    assert pytest.approx(s.duration, 0.2) == 1.0


def test_extraer_metadatos_acusticos_phrasing():
    from types import SimpleNamespace
    
    words = [
        SimpleNamespace(word="Hola", start=0.0, end=0.4),
        SimpleNamespace(word="mundo.", start=0.5, end=0.9), # Punto genera fin de frase
        SimpleNamespace(word="Segunda", start=2.0, end=2.4), # Pausa de 1.1s
        SimpleNamespace(word="oración", start=2.5, end=2.9),
    ]
    
    # Probando sin audio físico (fallback de frases)
    res = extraer_metadatos_acusticos("non_existent.wav", words)
    assert len(res) == 2
    assert res[0]["text"] == "Hola mundo."
    assert res[1]["text"] == "Segunda oración"
    assert pytest.approx(res[1]["pre_pause"], 0.05) == 1.10
