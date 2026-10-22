from types import SimpleNamespace
from unittest.mock import patch, MagicMock
from app.services.semantic_boundary import (
    _offline_fallback_boundary,
    resolve_take_boundary,
    analyze_take_boundary_with_llm
)


def _make_words(tuples_list):
    """Crea una lista de objetos palabra con SimpleNamespace(word, start, end)."""
    return [SimpleNamespace(word=t[0], start=t[1], end=t[2]) for t in tuples_list]


def test_offline_fallback_discards_meta_talk_at_end():
    words = _make_words([
        ("Vamos", 10.0, 10.4),
        ("a", 10.45, 10.55),
        ("hacer", 10.6, 10.9),
        ("este", 10.95, 11.2),
        ("truco", 11.25, 11.7),
        ("repetimos", 13.0, 13.6),
        ("la", 13.65, 13.8),
        ("intro", 13.85, 14.3)
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
        ("vez", 4.35, 4.6)
    ])
    clap_timestamps = [3.5]  # Aplauso de corte a los 3.5s

    valid, end_t, has_meta, reason = resolve_take_boundary(words, clap_timestamps=clap_timestamps, use_llm=False)

    assert has_meta is True
    assert len(valid) == 6
    assert valid[-1].word == "salsa"
    assert round(end_t, 2) == 3.0


def test_clean_scene_is_preserved_intact():
    words = _make_words([
        ("Hoy", 2.0, 2.3),
        ("el", 2.35, 2.5),
        ("dueño", 2.55, 3.0),
        ("va", 3.05, 3.2),
        ("a", 3.25, 3.35),
        ("probar", 3.4, 3.9),
        ("nuestra", 4.0, 4.4),
        ("hamburguesa", 4.45, 5.2)
    ])

    valid, end_t, has_meta, reason = resolve_take_boundary(words, use_llm=False)

    assert has_meta is False
    assert len(valid) == 8
    assert valid[-1].word == "hamburguesa"
    assert round(end_t, 2) == 5.2


@patch("app.services.semantic_boundary.OpenAI")
def test_deepseek_semantic_boundary_success(mock_openai_cls):
    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = '{"last_valid_index": 3, "has_meta_discourse": true, "reason": "Corte por instrucción off-camera"}'
    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_resp
    mock_openai_cls.return_value = mock_client

    words = _make_words([
        ("Pregunta", 1.0, 1.5),
        ("rápida", 1.55, 2.0),
        ("color", 2.05, 2.4),
        ("favorito", 2.45, 3.0),
        ("espera", 4.0, 4.4),
        ("que", 4.45, 4.6),
        ("paso", 4.65, 4.9),
        ("un", 4.95, 5.1),
        ("camion", 5.15, 5.6)
    ])

    with patch("app.core.config.settings.DEEPSEEK_API_KEY", "fake_test_key"):
        valid, end_t, has_meta, reason = analyze_take_boundary_with_llm(words)

    assert has_meta is True
    assert len(valid) == 4
    assert valid[-1].word == "favorito"
    assert round(end_t, 2) == 3.0
