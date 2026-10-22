import numpy as np
import pytest
import soundfile as sf
from pathlib import Path
from app.models.markers import MarkerResult, ClapResult
from app.services.clap_detector import (
    compute_amplitude_envelope,
    verify_transient_characteristics,
    detect_clap_in_signal,
    detect_claps_for_markers
)


def generate_synthetic_clap(
    duration_s: float = 5.0,
    sr: int = 16000,
    clap_time_s: float = 3.5,
    clap_amplitude: float = 0.85,
    noise_level: float = 0.02
) -> np.ndarray:
    """
    Genera una señal de audio sintética con ruido de fondo y un impulso
    transitorio de aplauso (subida ultra rápida < 3ms + decaimiento exponencial < 50ms).
    """
    total_samples = int(duration_s * sr)
    # Ruido blanco base de fondo
    rng = np.random.default_rng(seed=42)
    signal = rng.normal(0, noise_level, total_samples).astype(np.float32)

    clap_idx = int(clap_time_s * sr)
    # Impulso de aplauso: ráfaga de alta frecuencia con caída exponencial
    decay_samples = int(0.060 * sr)  # 60 ms
    t = np.arange(decay_samples) / sr
    
    # Envolvente: ataque en 2ms (32 muestras) y decaimiento exponencial tau = 15ms
    attack_samples = int(0.002 * sr)
    attack_curve = np.linspace(0.0, 1.0, attack_samples)
    decay_curve = np.exp(-t[attack_samples:] / 0.015)
    envelope = np.concatenate([attack_curve, decay_curve])[:decay_samples]

    # Modulación con componente ruidosa de banda ancha (típica de choque de palmas)
    burst = rng.uniform(-1.0, 1.0, decay_samples) * envelope * clap_amplitude

    end_idx = min(total_samples, clap_idx + decay_samples)
    actual_len = end_idx - clap_idx
    signal[clap_idx:end_idx] += burst[:actual_len].astype(np.float32)

    return signal


def test_clean_clap_detected():
    """
    Verifica que un aplauso limpio y nítido se detecta con exactitud milimétrica.
    """
    sr = 16000
    expected_clap_time = 3.20
    signal = generate_synthetic_clap(
        duration_s=5.0,
        sr=sr,
        clap_time_s=expected_clap_time,
        clap_amplitude=0.90,
        noise_level=0.015
    )

    phrase_start_rel = 1.0
    phrase_end_rel = 2.8

    detection = detect_clap_in_signal(
        signal=signal,
        sr=sr,
        phrase_start_rel=phrase_start_rel,
        phrase_end_rel=phrase_end_rel
    )

    assert detection is not None, "El detector debió identificar el aplauso sintético"
    assert detection["is_transient"] is True
    assert detection["confidence"] >= 4.0
    detected_time = detection["timestamp_rel"]
    assert abs(detected_time - expected_clap_time) < 0.02, (
        f"Timestamp detectado {detected_time}s difiere de {expected_clap_time}s"
    )


def test_continuous_noise_no_false_positive():
    """
    Verifica que ruido continuo o murmullo sin picos de ataque abrupto
    NO dispara una detección falsa de aplauso.
    """
    sr = 16000
    total_samples = int(4.0 * sr)
    rng = np.random.default_rng(seed=123)
    
    # Ruido continuo uniforme
    signal = rng.normal(0, 0.05, total_samples).astype(np.float32)

    # Añadir onda sinusoidal continua que emula vocal sostenida (sin ataque abrupto)
    t = np.arange(total_samples) / sr
    vowel = 0.3 * np.sin(2 * np.pi * 300 * t).astype(np.float32)
    signal += vowel

    detection = detect_clap_in_signal(
        signal=signal,
        sr=sr,
        phrase_start_rel=1.0,
        phrase_end_rel=2.5
    )

    assert detection is None, "El detector no debe dispararse con murmullo o ruido continuo"


def test_clap_located_before_phrase():
    """
    Verifica la detección de un aplauso que ocurre ANTES de que el hablante empiece la frase
    (ej. aplauso en t=0.6s, frase empieza en t=1.2s y termina en t=2.5s).
    """
    sr = 16000
    expected_clap = 0.60
    signal = generate_synthetic_clap(
        duration_s=4.0,
        sr=sr,
        clap_time_s=expected_clap,
        clap_amplitude=0.80
    )

    detection = detect_clap_in_signal(
        signal=signal,
        sr=sr,
        phrase_start_rel=1.2,
        phrase_end_rel=2.5
    )

    assert detection is not None
    assert abs(detection["timestamp_rel"] - expected_clap) < 0.02


def test_clap_located_after_phrase():
    """
    Verifica la detección de un aplauso que ocurre DESPUÉS de que el hablante termina la frase
    (ej. frase en t=1.0s a t=2.5s, aplauso en t=2.9s).
    """
    sr = 16000
    expected_clap = 2.90
    signal = generate_synthetic_clap(
        duration_s=4.5,
        sr=sr,
        clap_time_s=expected_clap,
        clap_amplitude=0.85
    )

    detection = detect_clap_in_signal(
        signal=signal,
        sr=sr,
        phrase_start_rel=1.0,
        phrase_end_rel=2.5
    )

    assert detection is not None
    assert abs(detection["timestamp_rel"] - expected_clap) < 0.02


def test_detect_claps_for_markers_end_to_end(tmp_path: Path):
    """
    Prueba integral de detect_claps_for_markers leyendo un archivo WAV real
    y procesando múltiples marcadores.
    """
    sr = 16000
    duration_s = 8.0
    clap_time = 4.25

    # Crear señal sintética y guardarla en WAV temporal
    audio_signal = generate_synthetic_clap(
        duration_s=duration_s,
        sr=sr,
        clap_time_s=clap_time,
        clap_amplitude=0.90
    )
    wav_path = tmp_path / "test_clap_audio.wav"
    sf.write(str(wav_path), audio_signal, sr)

    markers = [
        MarkerResult(
            bloque=1,
            toma=1,
            raw_text="Bloque 1, toma 1.",
            start_time=2.0,
            end_time=3.8,
            confidence=0.95
        )
    ]

    claps = detect_claps_for_markers(wav_path, markers)

    assert len(claps) == 1
    clap = claps[0]
    assert clap.bloque == 1
    assert clap.toma == 1
    assert clap.clap_detected is True
    assert abs(clap.clap_timestamp - clap_time) < 0.03
    assert clap.delta_phrase_clap > 0  # El aplauso fue después de la frase
    assert clap.clap_confidence >= 3.0


def test_fallback_when_no_clap_present(tmp_path: Path):
    """
    Prueba que si no existe ningún aplauso en la ventana, se activa el fallback
    con clap_detected=False y clap_timestamp=phrase_end.
    """
    sr = 16000
    total_samples = int(5.0 * sr)
    silent_signal = np.random.normal(0, 0.005, total_samples).astype(np.float32)

    wav_path = tmp_path / "test_silent.wav"
    sf.write(str(wav_path), silent_signal, sr)

    markers = [
        MarkerResult(
            bloque=2,
            toma=3,
            raw_text="Bloque 2, toma 3.",
            start_time=1.5,
            end_time=2.8,
            confidence=0.92
        )
    ]

    claps = detect_claps_for_markers(wav_path, markers)

    assert len(claps) == 1
    clap = claps[0]
    assert clap.clap_detected is False
    assert clap.clap_timestamp == 2.8  # Fallback a phrase_end
    assert clap.clap_confidence == 0.0
