"""Unit tests for individual analytics agents."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from agents.classifier import ClassifierAgent
from agents.compliance_agent import ComplianceAgent
from agents.quality_agent import QualityAgent
from agents.summarizer import SummarizerAgent


def _make_mock_client(response_payload: dict) -> MagicMock:
    mock_message = MagicMock()
    mock_message.content = json.dumps(response_payload)

    mock_choice = MagicMock()
    mock_choice.message = mock_message

    mock_usage = MagicMock()
    mock_usage.total_tokens = 42

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage = mock_usage

    mock_completions = MagicMock()
    mock_completions.create = AsyncMock(return_value=mock_response)

    mock_chat = MagicMock()
    mock_chat.completions = mock_completions

    mock_client = MagicMock()
    mock_client.chat = mock_chat
    return mock_client


# ---------------------------------------------------------------------------
# ClassifierAgent
# ---------------------------------------------------------------------------

class TestClassifierAgent:
    @pytest.mark.asyncio
    async def test_run_returns_classification(self, sample_segments, sample_full_text, classification_response):
        client = _make_mock_client(classification_response)
        agent = ClassifierAgent(client, model="gpt-4o-mini")

        state = {"transcript_segments": sample_segments, "full_text": sample_full_text}
        result = await agent.run(state)

        assert "classification" in result
        clf = result["classification"]
        assert clf["topic"] == "Потребительский кредит наличными"
        assert clf["category"] == "Продажи"
        assert clf["priority"] == "medium"
        assert 0.0 <= clf["confidence"] <= 1.0

    @pytest.mark.asyncio
    async def test_run_handles_missing_fields(self, sample_segments, sample_full_text):
        client = _make_mock_client({})
        agent = ClassifierAgent(client, model="gpt-4o-mini")

        state = {"transcript_segments": sample_segments, "full_text": sample_full_text}
        result = await agent.run(state)

        assert "classification" in result
        assert result["classification"]["topic"] == "Неизвестно"
        assert result["classification"]["priority"] == "medium"


# ---------------------------------------------------------------------------
# QualityAgent
# ---------------------------------------------------------------------------

class TestQualityAgent:
    @pytest.mark.asyncio
    async def test_perfect_score(self, sample_segments, sample_full_text, quality_response):
        client = _make_mock_client(quality_response)
        agent = QualityAgent(client, model="gpt-4o-mini")

        state = {"transcript_segments": sample_segments, "full_text": sample_full_text}
        result = await agent.run(state)

        assert result["quality_score"]["total"] == 100
        assert result["quality_score"]["checklist"]["greeting"] is True
        assert result["quality_score"]["checklist"]["farewell"] is True
        assert result["quality_score"]["checklist"]["need_detection"] is True

    @pytest.mark.asyncio
    async def test_partial_score(self, sample_segments, sample_full_text):
        partial = {
            "total": 50,
            "checklist": {
                "greeting": True,
                "need_detection": True,
                "solution_provided": False,
                "farewell": False,
            },
            "issues": ["Оператор не попрощался"],
            "recommendations": [],
        }
        client = _make_mock_client(partial)
        agent = QualityAgent(client, model="gpt-4o-mini")

        state = {"transcript_segments": sample_segments, "full_text": sample_full_text}
        result = await agent.run(state)

        assert result["quality_score"]["checklist"]["farewell"] is False
        assert len(result["quality_score"]["issues"]) == 1

    @pytest.mark.asyncio
    async def test_uses_need_detection_key(self, sample_segments, sample_full_text, quality_response):
        client = _make_mock_client(quality_response)
        agent = QualityAgent(client, model="gpt-4o-mini")

        state = {"transcript_segments": sample_segments, "full_text": sample_full_text}
        result = await agent.run(state)

        # Must use need_detection (not needs_identification) per ТЗ spec
        checklist = result["quality_score"]["checklist"]
        assert "need_detection" in checklist
        assert "needs_identification" not in checklist


# ---------------------------------------------------------------------------
# ComplianceAgent
# ---------------------------------------------------------------------------

class TestComplianceAgent:
    @pytest.mark.asyncio
    async def test_passes_clean_call(self, sample_segments, sample_full_text, compliance_response):
        client = _make_mock_client(compliance_response)
        agent = ComplianceAgent(client, model="gpt-4o-mini")

        state = {"transcript_segments": sample_segments, "full_text": sample_full_text}
        result = await agent.run(state)

        assert result["compliance"]["passed"] is True
        assert result["compliance"]["forbidden_phrases_found"] == []

    @pytest.mark.asyncio
    async def test_fails_on_forbidden_phrase(self, sample_segments, sample_full_text):
        bad_response = {
            "passed": False,
            "forbidden_phrases_found": ["100% одобрим ваш кредит"],
            "required_disclaimers": {
                "interest_rate_mentioned": True,
                "insurance_optional_mentioned": True,
                "personal_data_consent": True,
            },
            "issues": ["Оператор гарантировал одобрение кредита"],
        }
        client = _make_mock_client(bad_response)
        agent = ComplianceAgent(client, model="gpt-4o-mini")

        state = {"transcript_segments": sample_segments, "full_text": sample_full_text}
        result = await agent.run(state)

        assert result["compliance"]["passed"] is False
        assert len(result["compliance"]["forbidden_phrases_found"]) == 1


# ---------------------------------------------------------------------------
# SummarizerAgent
# ---------------------------------------------------------------------------

class TestSummarizerAgent:
    @pytest.mark.asyncio
    async def test_returns_summary_string_and_action_items(
        self, sample_segments, sample_full_text, summary_response
    ):
        client = _make_mock_client(summary_response)
        agent = SummarizerAgent(client, model="gpt-4o-mini")

        state = {
            "transcript_segments": sample_segments,
            "full_text": sample_full_text,
            "classification": {"topic": "Кредит", "category": "Продажи", "priority": "medium"},
            "quality_score": {"total": 100, "issues": []},
            "compliance": {"passed": True, "issues": []},
        }
        result = await agent.run(state)

        # summary must be a string (not a nested object)
        assert isinstance(result["summary"], str)
        assert len(result["summary"]) > 10
        assert isinstance(result["action_items"], list)

    @pytest.mark.asyncio
    async def test_fallback_on_empty_llm_response(self, sample_segments, sample_full_text):
        client = _make_mock_client({})
        agent = SummarizerAgent(client, model="gpt-4o-mini")

        state = {
            "transcript_segments": sample_segments,
            "full_text": sample_full_text,
            "classification": {},
            "quality_score": {},
            "compliance": {},
        }
        result = await agent.run(state)

        assert "summary" in result
        assert isinstance(result["summary"], str)
        assert result["action_items"] == []
