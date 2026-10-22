import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models.matching import CandidateBlockContext, LLMMatcherResponse
from app.models.script import ClientScript
from app.models.takes import TakeSegment
from app.services.llm_matcher import arbitrate_ambiguous_take
from app.services.script_matcher import run_phase_3_matching


@pytest.fixture
def sample_script_dict():
    return {
        "cliente": "clinica_dental",
        "version": 1,
        "videos": [
            {
                "id": "V01",
                "titulo": "Implantes sin Dolor",
                "formato": "ping_pong",
                "duracion_objetivo_s": 30.0,
                "bloques": [
                    {
                        "id": "B01",
                        "objetivo": "Gancho inicial preguntando si duelen los implantes dentales",
                        "palabras_clave": ["implantes", "dolor", "anestesia", "dientes"],
                        "obligatorio": True,
                        "duracion_objetivo_s": 8.0,
                    },
                    {
                        "id": "B02",
                        "objetivo": "Explicar el escaneo digital 3D en consultorio",
                        "palabras_clave": ["escaneo", "digital", "3D", "tecnología"],
                        "obligatorio": True,
                        "duracion_objetivo_s": 12.0,
                    },
                    {
                        "id": "B03",
                        "objetivo": "Llamado a la acción invitando a agendar turno por WhatsApp",
                        "palabras_clave": ["turno", "agenda", "whatsapp", "consulta"],
                        "obligatorio": False,
                        "duracion_objetivo_s": 6.0,
                    },
                ],
            }
        ],
    }


def _write_script_file(tmp_path: Path, script_dict: dict) -> Path:
    cliente_dir = tmp_path / "clientes" / script_dict["cliente"]
    cliente_dir.mkdir(parents=True, exist_ok=True)
    script_file = cliente_dir / "guion.json"
    script_file.write_text(json.dumps(script_dict, ensure_ascii=False, indent=2), encoding="utf-8")
    return script_file


def test_layer1_direct_verbal_marker_assignment(tmp_path, sample_script_dict):
    """
    Escenario 1: Toma con marca verbal correcta (V01, B01) asignada por Capa 1
    sin necesidad de invocar embeddings ni LLM.
    """
    script_file = _write_script_file(tmp_path, sample_script_dict)

    t1 = TakeSegment(
        take_id="take_l1_b01",
        reel=1,
        bloque=1,
        toma=1,
        duration=7.5,
        transcript_snippet="¿De verdad duelen los implantes dentales?",
    )
    t2 = TakeSegment(
        take_id="take_l1_b02",
        reel=1,
        bloque=2,
        toma=1,
        duration=11.0,
        transcript_snippet="Con nuestro escaneo digital 3D planificamos todo al milímetro.",
    )

    with patch("app.services.script_matcher.compute_block_similarities") as mock_embed:
        report = run_phase_3_matching(
            cliente="clinica_dental",
            script_path=script_file,
            takes=[t1, t2],
        )
        # Capa 1 resuelve ambas tomas sin llamar a Capa 2
        mock_embed.assert_not_called()

    assert report.ready_for_export is True  # B01 y B02 son obligatorios; B03 es opcional
    assert report.orphan_takes == []
    assert report.pending_review_count == 0
    v01 = report.videos_coverage[0]
    assert v01.bloques_cubiertos == 2
    assert v01.missing_mandatory_blocks == []
    assert v01.blocks[0].assigned_take_ids == ["take_l1_b01"]
    assert v01.blocks[1].assigned_take_ids == ["take_l1_b02"]


def test_layer2_high_similarity_embedding_assignment(tmp_path, sample_script_dict):
    """
    Escenario 2: Toma sin marca verbal (bloque=0) con similitud de embedding >= 0.78
    asignada automáticamente por Capa 2 sin llamar al LLM.
    """
    script_file = _write_script_file(tmp_path, sample_script_dict)

    unmarked_take = TakeSegment(
        take_id="take_l2_unmarked",
        reel=0,
        bloque=0,
        toma=1,
        anchor_type="none",
        duration=8.2,
        transcript_snippet="Mucha gente pregunta si los implantes dentales dan dolor con la anestesia moderna.",
    )
    marked_b02 = TakeSegment(
        take_id="take_b02",
        reel=1,
        bloque=2,
        toma=1,
        duration=10.5,
        transcript_snippet="Escaneo digital 3D.",
    )

    mock_candidates = [
        CandidateBlockContext(
            video_id="V01",
            bloque_id="B01",
            objetivo=sample_script_dict["videos"][0]["bloques"][0]["objetivo"],
            palabras_clave=["implantes", "dolor", "anestesia", "dientes"],
            cosine_similarity=0.84,  # >= 0.78 -> Asignación directa Capa 2
        ),
        CandidateBlockContext(
            video_id="V01",
            bloque_id="B02",
            objetivo=sample_script_dict["videos"][0]["bloques"][1]["objetivo"],
            palabras_clave=["escaneo", "digital", "3D", "tecnología"],
            cosine_similarity=0.22,
        ),
    ]

    with (
        patch("app.services.script_matcher.compute_block_similarities", return_value=mock_candidates) as mock_embed,
        patch("app.services.script_matcher.arbitrate_ambiguous_take") as mock_llm,
    ):
        report = run_phase_3_matching(
            cliente="clinica_dental",
            script_path=script_file,
            takes=[unmarked_take, marked_b02],
        )
        mock_embed.assert_called_once()
        mock_llm.assert_not_called()

    assert report.ready_for_export is True
    assert report.orphan_takes == []
    assert report.videos_coverage[0].blocks[0].assigned_take_ids == ["take_l2_unmarked"]


def test_layer3_gray_zone_arbitrated_by_llm(tmp_path, sample_script_dict):
    """
    Escenario 3: Toma con marca rota (bloque=99 inexistente) y similitud en zona gris
    (0.45 <= sim < 0.78) derivada exitosamente a la Capa 3 (mock del LLM).
    """
    script_file = _write_script_file(tmp_path, sample_script_dict)

    gray_take = TakeSegment(
        take_id="take_l3_gray",
        reel=1,
        bloque=99,  # Marca rota que no existe en guion.json
        toma=1,
        duration=11.4,
        transcript_snippet="Acá mostramos la camarita que te copia la boca en la pantalla sin pasta.",
    )
    marked_b01 = TakeSegment(
        take_id="take_b01",
        reel=1,
        bloque=1,
        toma=1,
        duration=7.0,
        transcript_snippet="Implantes sin dolor.",
    )

    mock_candidates = [
        CandidateBlockContext(
            video_id="V01",
            bloque_id="B02",
            objetivo=sample_script_dict["videos"][0]["bloques"][1]["objetivo"],
            palabras_clave=["escaneo", "digital", "3D", "tecnología"],
            cosine_similarity=0.61,  # Zona gris (0.45 <= 0.61 < 0.78)
        ),
    ]

    mock_llm_response = LLMMatcherResponse(
        matched=True,
        matched_video_id="V01",
        matched_bloque_id="B02",
        confidence=0.88,
        reasoning="La 'camarita que copia la boca en pantalla sin pasta' describe el escaneo digital 3D (B02).",
        flag_for_human_review=False,
    )

    with (
        patch("app.services.script_matcher.compute_block_similarities", return_value=mock_candidates),
        patch("app.services.script_matcher.arbitrate_ambiguous_take", return_value=mock_llm_response) as mock_llm,
    ):
        report = run_phase_3_matching(
            cliente="clinica_dental",
            script_path=script_file,
            takes=[marked_b01, gray_take],
        )
        mock_llm.assert_called_once()

    assert report.ready_for_export is True
    assert report.orphan_takes == []
    assert report.videos_coverage[0].blocks[1].assigned_take_ids == ["take_l3_gray"]


def test_llm_failure_or_ambiguity_persists_to_decisiones_json(tmp_path, sample_script_dict):
    """
    Escenario 4: Toma ambigua en zona gris donde el LLM falla por timeout o conexión;
    se registra en `clientes/<cliente>/revision/decisiones.json` con `flag_for_human_review=True`
    y sin duplicar si se vuelve a ejecutar.
    """
    script_file = _write_script_file(tmp_path, sample_script_dict)

    ambiguous_take = TakeSegment(
        take_id="take_orphan_01",
        reel=0,
        bloque=0,
        toma=1,
        anchor_type="none",
        duration=5.8,
        transcript_snippet="Bueno gente, comenten acá abajo qué opinan de esto.",
    )

    mock_candidates = [
        CandidateBlockContext(
            video_id="V01",
            bloque_id="B03",
            objetivo=sample_script_dict["videos"][0]["bloques"][2]["objetivo"],
            palabras_clave=["turno", "agenda", "whatsapp", "consulta"],
            cosine_similarity=0.55,  # Zona gris -> va a Capa 3
        ),
    ]

    # Simular un cliente OpenAI que lanza TimeoutError
    failing_openai_client = MagicMock()
    failing_openai_client.chat.completions.create.side_effect = TimeoutError("API timed out after 6.0s")

    with patch("app.services.script_matcher.compute_block_similarities", return_value=mock_candidates):
        report = run_phase_3_matching(
            cliente="clinica_dental",
            script_path=script_file,
            takes=[ambiguous_take],
            llm_client=failing_openai_client,
        )
        # Segunda ejecución para comprobar idempotencia (sin duplicados en decisiones.json)
        report = run_phase_3_matching(
            cliente="clinica_dental",
            script_path=script_file,
            takes=[ambiguous_take],
            llm_client=failing_openai_client,
        )

    assert report.orphan_takes == ["take_orphan_01"]
    assert report.pending_review_count == 1

    decisiones_file = tmp_path / "clientes" / "clinica_dental" / "revision" / "decisiones.json"
    assert decisiones_file.exists()

    saved_decisions = json.loads(decisiones_file.read_text(encoding="utf-8"))
    assert saved_decisions["cliente"] == "clinica_dental"
    assert len(saved_decisions["pendientes"]) == 1

    item = saved_decisions["pendientes"][0]
    assert item["take_id"] == "take_orphan_01"
    assert item["flag_for_human_review"] is True
    assert item["suggested_video_id"] == "V01"
    assert item["suggested_bloque_id"] == "B03"
    assert item["cosine_similarity"] == 0.55


def test_missing_mandatory_block_marks_ready_for_export_false(tmp_path, sample_script_dict):
    """
    Escenario 5: Detección de bloque obligatorio faltante (B02 sin cubrir)
    marcando `ready_for_export=False` y listando B02 en `missing_mandatory_blocks`.
    """
    script_file = _write_script_file(tmp_path, sample_script_dict)

    only_b01 = TakeSegment(
        take_id="take_only_b01",
        reel=1,
        bloque=1,
        toma=1,
        duration=8.0,
        transcript_snippet="¿Duelen los implantes dentales?",
    )

    report = run_phase_3_matching(
        cliente="clinica_dental",
        script_path=script_file,
        takes=[only_b01],
    )

    assert report.ready_for_export is False
    v01 = report.videos_coverage[0]
    assert v01.bloques_cubiertos == 1
    assert v01.missing_mandatory_blocks == ["B02"]


def test_run_phase_3_endpoint_validation_and_execution(tmp_path, sample_script_dict):
    """
    Verifica que el endpoint POST /api/pipeline/run-phase-3 devuelva HTTP 422
    cuando `guion.json` es inválido y HTTP 200 con Phase3Report cuando es válido.
    """
    client = TestClient(app)

    # 1. Guion inválido (ID de bloque 'BLOQUE_1' no cumple pattern ^B\d{2}$)
    invalid_script = {
        "cliente": "cliente_invalido",
        "version": 1,
        "videos": [
            {
                "id": "V01",
                "titulo": "Test",
                "formato": "monologue",
                "duracion_objetivo_s": 20.0,
                "bloques": [
                    {
                        "id": "BLOQUE_1",  # Inválido
                        "objetivo": "Objetivo",
                        "palabras_clave": [],
                        "obligatorio": True,
                        "duracion_objetivo_s": 10.0,
                    }
                ],
            }
        ],
    }
    bad_file = _write_script_file(tmp_path, invalid_script)
    resp_422 = client.post(
        "/api/pipeline/run-phase-3",
        json={
            "cliente": "cliente_invalido",
            "script_path": str(bad_file),
            "takes": [],
        },
    )
    assert resp_422.status_code == 422

    # 2. Guion válido con todas las tomas obligatorias
    good_file = _write_script_file(tmp_path, sample_script_dict)
    resp_200 = client.post(
        "/api/pipeline/run-phase-3",
        json={
            "cliente": "clinica_dental",
            "script_path": str(good_file),
            "takes": [
                {
                    "take_id": "t_b01",
                    "reel": 1,
                    "bloque": 1,
                    "toma": 1,
                    "duration": 7.5,
                    "transcript_snippet": "Implantes sin dolor",
                },
                {
                    "take_id": "t_b02",
                    "reel": 1,
                    "bloque": 2,
                    "toma": 1,
                    "duration": 11.2,
                    "transcript_snippet": "Escaneo digital 3D",
                },
            ],
        },
    )
    assert resp_200.status_code == 200
    body = resp_200.json()
    assert body["cliente"] == "clinica_dental"
    assert body["ready_for_export"] is True
    assert body["videos_coverage"][0]["bloques_cubiertos"] == 2
