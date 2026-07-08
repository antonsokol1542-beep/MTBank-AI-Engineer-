"""
Integration tests for the FastAPI application.

These tests mock out the ASR service and LangGraph graph so they
run without GPU/model downloads.
"""

from __future__ import annotations

import io
import struct
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Mock data
# ---------------------------------------------------------------------------

MOCK_SEGMENTS = [
    {"speaker": "Оператор", "text": "Добрый день, МТБанк.", "start": 0.5, "end": 3.0},
    {"speaker": "Клиент", "text": "Хочу оформить кредит.", "start": 4.0, "end": 6.5},
]

MOCK_ANALYSIS_STATE = {
    "transcript_segments": MOCK_SEGMENTS,
    "full_text": "Добрый день, МТБанк. Хочу оформить кредит.",
    "classification": {
        "topic": "Потребительский кредит",
        "category": "Продажи",
        "subcategory": "Кредит наличными",
        "priority": "medium",
        "confidence": 0.9,
    },
    "quality_score": {
        "total": 75,
        "checklist": {
            "greeting": True,
            "need_detection": True,
            "solution_provided": True,
            "farewell": False,
        },
        "issues": ["Отсутствует прощание"],
        "recommendations": [],
    },
    "compliance": {
        "passed": True,
        "forbidden_phrases_found": [],
        "required_disclaimers": {
            "interest_rate_mentioned": False,
            "insurance_optional_mentioned": True,
            "personal_data_consent": True,
        },
        "issues": [],
    },
    "summary": "Клиент обратился по вопросу оформления кредита.",
    "action_items": ["[Оператор] Перезвонить клиенту"],
}


def _make_mock_asr(segments=None, duration=10.0) -> MagicMock:
    svc = MagicMock()
    svc.is_loaded = True
    svc.process = MagicMock(return_value=(segments or MOCK_SEGMENTS, duration))
    return svc


def _make_fake_wav() -> bytes:
    num_samples = 1
    sample_rate = 16000
    bits_per_sample = 16
    num_channels = 1
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    data_size = num_samples * block_align
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 36 + data_size, b"WAVE",
        b"fmt ", 16, 1, num_channels, sample_rate, byte_rate, block_align, bits_per_sample,
        b"data", data_size,
    )
    return header + b"\x00" * data_size


@pytest.fixture
def client():
    with (
        patch("main._asr_service", _make_mock_asr()),
        patch("main.run_analysis", new=AsyncMock(return_value=MOCK_ANALYSIS_STATE)),
    ):
        from main import app
        with TestClient(app, raise_server_exceptions=True) as c:
            yield c


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_health_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
        assert resp.json()["asr_loaded"] is True


# ---------------------------------------------------------------------------
# POST /analyze  (main ТЗ endpoint)
# ---------------------------------------------------------------------------

class TestAnalyze:
    def test_analyze_returns_full_structure(self, client):
        wav = _make_fake_wav()
        resp = client.post(
            "/analyze",
            files={"file": ("test.wav", io.BytesIO(wav), "audio/wav")},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "transcript" in body
        assert "classification" in body
        assert "quality_score" in body
        assert "compliance" in body
        assert "summary" in body
        assert "action_items" in body

    def test_analyze_summary_is_string(self, client):
        wav = _make_fake_wav()
        resp = client.post("/analyze", files={"file": ("test.wav", io.BytesIO(wav), "audio/wav")})
        body = resp.json()
        # ТЗ requires summary as a string at top level
        assert isinstance(body["summary"], str)
        # action_items at top level (not inside summary)
        assert isinstance(body["action_items"], list)

    def test_analyze_quality_score_structure(self, client):
        wav = _make_fake_wav()
        resp = client.post("/analyze", files={"file": ("test.wav", io.BytesIO(wav), "audio/wav")})
        qs = resp.json()["quality_score"]
        # ТЗ requires quality_score.total (not score)
        assert "total" in qs
        assert "checklist" in qs
        # ТЗ requires need_detection (not needs_identification)
        assert "need_detection" in qs["checklist"]
        assert 0 <= qs["total"] <= 100

    def test_analyze_transcript_speakers_in_russian(self, client):
        wav = _make_fake_wav()
        resp = client.post("/analyze", files={"file": ("test.wav", io.BytesIO(wav), "audio/wav")})
        speakers = {seg["speaker"] for seg in resp.json()["transcript"]}
        # Must use Russian speaker names per ТЗ
        assert speakers & {"Оператор", "Клиент"}

    def test_analyze_no_input_returns_400(self, client):
        resp = client.post("/analyze")
        assert resp.status_code == 400

    def test_analyze_priority_enum(self, client):
        wav = _make_fake_wav()
        resp = client.post("/analyze", files={"file": ("test.wav", io.BytesIO(wav), "audio/wav")})
        priority = resp.json()["classification"]["priority"]
        assert priority in ("low", "medium", "high", "critical")

    def test_analyze_compliance_structure(self, client):
        wav = _make_fake_wav()
        resp = client.post("/analyze", files={"file": ("test.wav", io.BytesIO(wav), "audio/wav")})
        comp = resp.json()["compliance"]
        assert "passed" in comp
        assert isinstance(comp["passed"], bool)
        assert "issues" in comp


# ---------------------------------------------------------------------------
# POST /api/v1/transcribe
# ---------------------------------------------------------------------------

class TestTranscribe:
    def test_transcribe_wav_upload(self, client):
        wav = _make_fake_wav()
        resp = client.post(
            "/api/v1/transcribe",
            files={"file": ("test.wav", io.BytesIO(wav), "audio/wav")},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "transcript" in body
        assert body["transcript"][0]["speaker"] == "Оператор"
        assert body["audio_duration"] == 10.0

    def test_transcribe_unsupported_format(self, client):
        resp = client.post(
            "/api/v1/transcribe",
            files={"file": ("audio.xyz", io.BytesIO(b"fake"), "application/octet-stream")},
        )
        assert resp.status_code == 422

    def test_transcribe_no_input_returns_400(self, client):
        resp = client.post("/api/v1/transcribe")
        assert resp.status_code == 400
