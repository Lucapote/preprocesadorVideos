from types import SimpleNamespace
import pytest
from app.models.markers import ClapResult
from app.models.takes import DiscardReason, TakeStatus
from app.services.take_segmenter import (
    segment_takes, 
    generate_deterministic_take_id,
    detect_internal_silences
)


def make_dummy_word(text: str, start: float, end: float):
    return SimpleNamespace(word=text, start=start, end=end, probability=0.98)


def test_generate_deterministic_take_id():
    """Validar hash determinista SHA-256 de 12 caracteres."""
    id1 = generate_deterministic_take_id("video.mp4", 5.250)
    id2 = generate_deterministic_take_id("video.mp4", 5.250)
    id3 = generate_deterministic_take_id("video.mp4", 5.251)

    assert id1 == id2, "Mismos inputs deben producir exactamente el mismo hash"
    assert id1 != id3, "Diferentes timestamps deben producir hashes distintos"
    assert len(id1) == 12, "El hash debe tener exactamente 12 caracteres hex"


def test_margins_and_no_bleed_into_next_marker():
    """
    Validar que se apliquen 150 ms de entrada y 250 ms de salida
    sin exceder los límites entre marcadores.
    """
    total_duration = 30.0

    m1 = ClapResult(
        bloque=1, toma=1,
        phrase_start=2.0, phrase_end=3.5,
        clap_timestamp=4.0, clap_confidence=8.5,
        delta_phrase_clap=0.5, clap_detected=True,
        anchor_timestamp=4.0, anchor_type="clap"
    )
    m2 = ClapResult(
        bloque=2, toma=1,
        phrase_start=15.0, phrase_end=16.5,
        clap_timestamp=17.0, clap_confidence=9.0,
        delta_phrase_clap=0.5, clap_detected=True,
        anchor_timestamp=17.0, anchor_type="clap"
    )

    words = [
        # Toma 1: Primera palabra en 5.0, última en 14.85
        make_dummy_word("Inicio", 5.0, 5.5),
        make_dummy_word("del", 5.6, 5.9),
        make_dummy_word("contenido.", 14.0, 14.85),
        # Toma 2
        make_dummy_word("Siguiente", 18.0, 18.5)
    ]

    takes = segment_takes(
        file_name="clip.mp4",
        total_duration=total_duration,
        claps_or_markers=[m1, m2],
        all_words=words,
        lead_in_padding_s=0.150,
        lead_out_padding_s=0.250
    )

    take1 = takes[0]
    # cut_start = max(4.0, 5.0 - 0.150) = 4.85
    assert pytest.approx(take1.cut_start, 0.01) == 4.85
    # cut_end = min(fin_bruto=15.0, 14.85 + 0.250 = 15.10) => Debe ser acotado a 15.0 para no pisar el siguiente bloque
    assert pytest.approx(take1.cut_end, 0.01) == 15.0


def test_last_take_reaches_until_video_end():
    """Validar que la última toma se delimite correctamente hasta el final del video."""
    total_duration = 25.0
    m1 = ClapResult(
        bloque=1, toma=1,
        phrase_start=2.0, phrase_end=3.5,
        clap_timestamp=4.0, clap_confidence=8.0,
        delta_phrase_clap=0.5, clap_detected=True,
        anchor_timestamp=4.0, anchor_type="clap"
    )

    words = [
        make_dummy_word("Última", 5.0, 5.5),
        make_dummy_word("palabra.", 24.5, 24.8)
    ]

    takes = segment_takes(
        file_name="clip.mp4",
        total_duration=total_duration,
        claps_or_markers=[m1],
        all_words=words,
        lead_in_padding_s=0.150,
        lead_out_padding_s=0.250
    )

    take = takes[0]
    # cut_end no debe superar la duración total del video (25.0)
    assert pytest.approx(take.cut_end, 0.01) == 25.0


def test_detect_internal_silences_without_splitting():
    """Validar detección de pausas internas > 1.0s sin partir el archivo."""
    words = [
        make_dummy_word("Primera", 2.0, 2.5),
        # Silencio de 2.5s a 4.0s (1.5s > 1.0s)
        make_dummy_word("frase", 4.0, 4.5),
        # Silencio normal de 0.3s (no debe reportarse)
        make_dummy_word("continua.", 4.8, 5.2),
        # Silencio de 5.2s a 7.0s (1.8s > 1.0s)
        make_dummy_word("Final.", 7.0, 7.5)
    ]

    silences = detect_internal_silences(words, min_silence_duration_s=1.0)
    assert len(silences) == 2
    assert pytest.approx(silences[0].duration, 0.01) == 1.5
    assert pytest.approx(silences[1].duration, 0.01) == 1.8


def test_retakes_and_empty_takes():
    """Validar clasificación de repeticiones y tomas vacías."""
    m1 = ClapResult(bloque=1, toma=1, phrase_start=2.0, phrase_end=4.0, clap_timestamp=4.2, clap_confidence=8.0, delta_phrase_clap=0.2, clap_detected=True)
    m2 = ClapResult(bloque=1, toma=2, phrase_start=10.0, phrase_end=12.0, clap_timestamp=12.2, clap_confidence=8.0, delta_phrase_clap=0.2, clap_detected=True)
    m3 = ClapResult(bloque=2, toma=1, phrase_start=20.0, phrase_end=22.0, clap_timestamp=22.2, clap_confidence=8.0, delta_phrase_clap=0.2, clap_detected=True)

    words = [
        # Toma 1 de Bloque 1 (se descarta por repetición)
        make_dummy_word("Error", 5.0, 5.5),
        # Toma 2 de Bloque 1 (definitiva)
        make_dummy_word("Bien", 13.0, 13.5),
        # Bloque 2 sin palabras (toma vacía)
    ]

    takes = segment_takes("clip.mp4", 30.0, [m1, m2, m3], words)
    assert len(takes) == 3

    t1, t2, t3 = takes[0], takes[1], takes[2]
    assert t1.status == TakeStatus.DISCARD and t1.discard_reason == DiscardReason.RETAKE
    assert t2.status == TakeStatus.KEEP and t2.is_best_take and t2.discard_reason == DiscardReason.NONE
    assert t3.status == TakeStatus.DISCARD and t3.discard_reason == DiscardReason.EMPTY
