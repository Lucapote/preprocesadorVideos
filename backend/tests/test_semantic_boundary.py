import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import pytest

from app.models.markers import ClapResult
from app.models.takes import DiscardReason, SemanticBoundaryAnalysis, TakeStatus
from app.services.semantic_boundary import (
    _offline_fallback_boundary,
    analyze_semantic_boundary,
    analyze_take_boundary_with_llm,
    deterministic_boundary_fallback,
    format_indexed_words,
    resolve_take_boundary,
)
from app.services.take_segmenter import segment_takes


def _make_words(tuples_list):
    """Crea una lista de objetos palabra con SimpleNamespace(word, start, end)."""
    return [SimpleNamespace(word=t[0], start=t[1], end=t[2]) for t in tuples_list]


def test_format_indexed_words_never_leaks_timestamps():
    words = _make_words([
        ("¿Cuál", 1.12, 1.45),
        ("es", 1.48, 1.60),
        ("tu", 1.62, 1.75),
        ("comida?", 1.80, 2.30),
        ("Milanesas", 2.95, 3.60),
    ])
    formatted = format_indexed_words(words)
    assert formatted == "[0: ¿Cuál] [1: es] [2: tu] [3: comida?] [4: Milanesas]"
    assert "1.12" not in formatted
    assert "3.60" not in formatted


@patch("app.services.semantic_boundary.OpenAI")
def test_llm_ping_pong_standard(mock_openai_cls):
    """
    Escenario 1: Ping-pong estándar (pregunta en tokens 0-4, respuesta en 5-10,
    seguido de risa/comentario de cierre en 11-12 que se descarta).
    """
    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = json.dumps({
        "format_type": "ping_pong_dialogue",
        "status": "complete_exchange",
        "first_valid_word_index": 0,
        "last_valid_word_index": 10,
        "punchline_or_climax_detected": True,
        "confidence": 0.96,
        "reasoning": "Pregunta en índices 0-4 y respuesta rematada en índice 10."
    })
    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_resp
    mock_openai_cls.return_value = mock_client

    # Tokens 0..4: pregunta | Tokens 5..10: respuesta (con pausa natural de 1.4s en el medio) | Tokens 11..12: fuera de toma
    words = _make_words([
        ("¿Cuál", 2.00, 2.30),       # 0
        ("es", 2.35, 2.50),          # 1
        ("tu", 2.55, 2.70),          # 2
        ("plato", 2.75, 3.10),       # 3
        ("favorito?", 3.15, 3.70),   # 4
        # Pausa natural de pensamiento de 1.3s (5.00 - 3.70)
        ("Sin", 5.00, 5.30),         # 5
        ("duda", 5.35, 5.70),        # 6
        ("las", 5.75, 5.95),         # 7
        ("milanesas", 6.00, 6.60),   # 8
        ("con", 6.65, 6.85),         # 9
        ("puré.", 6.90, 7.40),       # 10 <- last_valid_word_index
        ("Listo", 8.50, 8.90),       # 11
        ("corte", 9.00, 9.40),       # 12
    ])

    with patch("app.core.config.settings.DEEPSEEK_API_KEY", "test_key"):
        analysis = analyze_semantic_boundary(words, use_llm=True)

    assert isinstance(analysis, SemanticBoundaryAnalysis)
    assert analysis.format_type == "ping_pong_dialogue"
    assert analysis.status == "complete_exchange"
    assert analysis.first_valid_word_index == 0
    assert analysis.last_valid_word_index == 10
    assert analysis.punchline_or_climax_detected is True

    # Verificar integración en take_segmenter con paddings (0.15s y 0.25s)
    marker = ClapResult(
        bloque=1, toma=1,
        phrase_start=0.5, phrase_end=1.2,
        clap_timestamp=1.5, clap_confidence=9.0,
        delta_phrase_clap=0.3, clap_detected=True
    )
    with patch("app.services.take_segmenter.analyze_semantic_boundary", return_value=analysis):
        takes = segment_takes(
            file_name="ping_pong.mp4",
            total_duration=15.0,
            claps_or_markers=[marker],
            all_words=words,
            lead_in_padding_s=0.15,
            lead_out_padding_s=0.25,
        )

    assert len(takes) == 1
    t = takes[0]
    assert t.usable is True
    assert t.status == TakeStatus.KEEP
    assert pytest.approx(t.cut_start, 0.01) == 2.00 - 0.15  # 1.85s
    assert pytest.approx(t.cut_end, 0.01) == 7.40 + 0.25    # 7.65s
    assert t.words_count == 11
    assert t.transcript_snippet.endswith("puré.")


@patch("app.services.semantic_boundary.OpenAI")
def test_llm_restart_inside_take(mock_openai_cls):
    """
    Escenario 2: Intento con reinicio interno (tropiezo en token 3,
    reformulación válida en tokens 8 a 15 sin repetir claqueta).
    """
    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = json.dumps({
        "format_type": "trend_reaction",
        "status": "complete_exchange",
        "first_valid_word_index": 8,
        "last_valid_word_index": 15,
        "punchline_or_climax_detected": True,
        "confidence": 0.94,
        "reasoning": "Se descarta el tropiezo inicial (0-7) y se conserva la reformulación limpia (8-15)."
    })
    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_resp
    mock_openai_cls.return_value = mock_client

    words = _make_words([
        ("Che", 3.0, 3.2),            # 0
        ("cuál", 3.25, 3.5),          # 1
        ("era", 3.55, 3.8),           # 2
        ("eh...", 3.9, 4.3),          # 3 (tropiezo)
        ("pará", 4.4, 4.7),           # 4
        ("va", 4.75, 4.9),            # 5
        ("de", 4.95, 5.1),            # 6
        ("nuevo.", 5.15, 5.5),        # 7
        ("¿Qué", 6.50, 6.80),         # 8 <- first_valid_word_index
        ("preferís", 6.85, 7.30),     # 9
        ("verano", 7.35, 7.75),       # 10
        ("o", 7.80, 7.90),            # 11
        ("invierno?", 7.95, 8.50),    # 12
        ("Invierno", 9.10, 9.60),     # 13
        ("toda", 9.65, 9.95),         # 14
        ("la_vida.", 10.00, 10.60),   # 15 <- last_valid_word_index
    ])

    with patch("app.core.config.settings.GROQ_API_KEY", "groq_test_key"):
        analysis = analyze_semantic_boundary(words, use_llm=True)

    assert analysis.first_valid_word_index == 8
    assert analysis.last_valid_word_index == 15

    marker = ClapResult(
        bloque=1, toma=1,
        phrase_start=1.0, phrase_end=2.0,
        clap_timestamp=2.2, clap_confidence=9.5,
        delta_phrase_clap=0.2, clap_detected=True
    )
    with patch("app.services.take_segmenter.analyze_semantic_boundary", return_value=analysis):
        takes = segment_takes(
            file_name="trend.mp4",
            total_duration=15.0,
            claps_or_markers=[marker],
            all_words=words,
            lead_in_padding_s=0.15,
            lead_out_padding_s=0.25,
        )

    t = takes[0]
    # El corte debe iniciar en el token 8 (6.50 - 0.15 = 6.35) y terminar en el token 15 (10.60 + 0.25 = 10.85)
    assert pytest.approx(t.cut_start, 0.01) == 6.35
    assert pytest.approx(t.cut_end, 0.01) == 10.85
    assert t.words_count == 8
    assert t.transcript_snippet.startswith("¿Qué preferís")


@patch("app.services.semantic_boundary.OpenAI")
def test_llm_monologue_standard(mock_openai_cls):
    """
    Escenario 3: Monólogo estándar simple de principio a fin.
    """
    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = json.dumps({
        "format_type": "monologue",
        "status": "complete_exchange",
        "first_valid_word_index": 0,
        "last_valid_word_index": 5,
        "punchline_or_climax_detected": True,
        "confidence": 0.99,
        "reasoning": "Monólogo directo a cámara sin interrupciones."
    })
    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_resp
    mock_openai_cls.return_value = mock_client

    words = _make_words([
        ("Tres", 4.0, 4.3),
        ("consejos", 4.35, 4.8),
        ("para", 4.85, 5.1),
        ("mejorar", 5.15, 5.6),
        ("tus", 5.65, 5.85),
        ("videos.", 5.9, 6.4),
    ])

    with patch("app.core.config.settings.DEEPSEEK_API_KEY", "test_key"):
        analysis = analyze_semantic_boundary(words, use_llm=True)

    assert analysis.format_type == "monologue"
    assert analysis.status == "complete_exchange"
    assert analysis.first_valid_word_index == 0
    assert analysis.last_valid_word_index == 5


@patch("app.services.semantic_boundary.OpenAI")
def test_llm_timeout_or_error_triggers_deterministic_fallback(mock_openai_cls):
    """
    Escenario 4: Si el LLM lanza error o timeout (> 6s), el sistema hace fallback
    algorítmico inmediato sin crashear el servidor (cortando antes de un silencio > 2.5s).
    """
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = TimeoutError("Request timed out after 6.0s")
    mock_openai_cls.return_value = mock_client

    words = _make_words([
        ("¿Café", 2.0, 2.4),          # 0
        ("o", 2.45, 2.55),            # 1
        ("mate?", 2.6, 3.0),          # 2
        ("Mate", 3.5, 3.9),           # 3
        ("amargo.", 3.95, 4.5),       # 4 <- última palabra antes de silencio de 3.2s
        # Silencio largo de 3.2s (> 2.5s)
        ("Che", 7.7, 8.0),            # 5
        ("quedó", 8.1, 8.5),          # 6
    ])

    with patch("app.core.config.settings.DEEPSEEK_API_KEY", "test_key"):
        analysis = analyze_semantic_boundary(words, use_llm=True)

    assert analysis.status == "complete_exchange"
    assert analysis.first_valid_word_index == 0
    assert analysis.last_valid_word_index == 4
    assert "silencio largo" in analysis.reasoning.lower()


def test_aborted_attempt_marks_take_unusable_without_crashing():
    """
    Verifica que si analysis.status == 'aborted_attempt', la toma se marque con
    usable=False y status=DISCARD sin romper el resto del video.
    """
    m1 = ClapResult(
        bloque=1, toma=1,
        phrase_start=1.0, phrase_end=2.0,
        clap_timestamp=2.3, clap_confidence=8.5,
        delta_phrase_clap=0.3, clap_detected=True
    )
    m2 = ClapResult(
        bloque=1, toma=2,
        phrase_start=10.0, phrase_end=11.0,
        clap_timestamp=11.3, clap_confidence=9.0,
        delta_phrase_clap=0.3, clap_detected=True
    )

    words = _make_words([
        # Toma 1: Intento abortado
        ("Ay", 3.0, 3.3),
        ("no", 3.4, 3.6),
        ("me", 3.7, 3.9),
        ("trabé", 4.0, 4.5),
        # Toma 2: Toma buena
        ("Hola", 12.0, 12.4),
        ("equipo", 12.5, 13.0),
    ])

    aborted_analysis = SemanticBoundaryAnalysis(
        format_type="ping_pong_dialogue",
        status="aborted_attempt",
        first_valid_word_index=0,
        last_valid_word_index=3,
        punchline_or_climax_detected=False,
        confidence=0.95,
        reasoning="El orador abortó la toma inmediatamente por bloqueo."
    )
    good_analysis = SemanticBoundaryAnalysis(
        format_type="ping_pong_dialogue",
        status="complete_exchange",
        first_valid_word_index=0,
        last_valid_word_index=1,
        punchline_or_climax_detected=True,
        confidence=0.98,
        reasoning="Interacción completa y limpia."
    )

    with patch(
        "app.services.take_segmenter.analyze_semantic_boundary",
        side_effect=[aborted_analysis, good_analysis]
    ):
        takes = segment_takes(
            file_name="session.mp4",
            total_duration=20.0,
            claps_or_markers=[m1, m2],
            all_words=words,
        )

    assert len(takes) == 2
    assert takes[0].usable is False
    assert takes[0].status == TakeStatus.DISCARD
    assert takes[0].discard_reason == DiscardReason.ABORTED
    assert "abortó" in takes[0].semantic_reasoning

    assert takes[1].usable is True
    assert takes[1].status == TakeStatus.KEEP
    assert takes[1].is_best_take is True


def test_offline_fallback_discards_meta_talk_at_end():
    words = _make_words([
        ("Vamos", 10.0, 10.4),
        ("a", 10.45, 10.55),
        ("hacer", 10.6, 10.9),
        ("este", 10.95, 11.2),
        ("truco", 11.25, 11.7),
        ("repetimos", 13.0, 13.6),
        ("la", 13.65, 13.8),
        ("intro", 13.85, 14.3),
    ])

    valid, end_t, has_meta, reason = _offline_fallback_boundary(words)

    assert has_meta is True
    assert "repetimos" in reason.lower()
    assert len(valid) == 5
    assert valid[-1].word == "truco"
    assert round(end_t, 2) == 11.7


def test_offline_fallback_cuts_before_acoustic_clap():
    words = _make_words([
        ("El", 1.0, 1.2),
        ("chef", 1.25, 1.6),
        ("nos", 1.65, 1.8),
        ("prepara", 1.85, 2.3),
        ("la", 2.35, 2.5),
        ("salsa", 2.55, 3.0),
        ("otra", 4.0, 4.3),
        ("vez", 4.35, 4.6),
    ])
    clap_timestamps = [3.5]

    valid, end_t, has_meta, reason = resolve_take_boundary(
        words, clap_timestamps=clap_timestamps, use_llm=False
    )

    assert has_meta is True
    assert len(valid) == 6
    assert valid[-1].word == "salsa"
    assert round(end_t, 2) == 3.0


def test_capa2_minilm_detects_paraphrased_blooper():
    """
    Capa 2 (MiniLM): Regla 1 de reglasinternas.md.
    Verifica que el modelo local de embeddings clasifique y pode un blooper
    parafroseado al final de la toma en modo offline sin requerir LLM de pago.
    """
    words = _make_words([
        ("Hoy", 1.0, 1.3),
        ("les", 1.35, 1.5),
        ("traigo", 1.55, 1.9),
        ("tres", 1.95, 2.2),
        ("recetas", 2.25, 2.7),
        ("increíbles.", 2.75, 3.4),
        # Cola de descarte / blooper
        ("corte", 4.0, 4.4),
        ("camarógrafo", 4.45, 5.1),
    ])

    analysis = deterministic_boundary_fallback(words)

    assert analysis.status == "complete_exchange"
    assert analysis.first_valid_word_index == 0
    # Debe podar el blooper en el índice 5 ('increíbles.')
    assert analysis.last_valid_word_index == 5
    assert analysis.confidence >= 0.58

