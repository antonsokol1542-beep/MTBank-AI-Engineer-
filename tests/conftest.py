"""Shared pytest fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Add api/ to the path so tests can import application modules directly
API_DIR = Path(__file__).parent.parent / "api"
sys.path.insert(0, str(API_DIR))


# ---------------------------------------------------------------------------
# Sample data fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_segments() -> list[dict]:
    return [
        {"speaker": "Оператор", "text": "Добрый день, МТБанк, меня зовут Анна, чем могу помочь?", "start": 0.5, "end": 4.2},
        {"speaker": "Клиент", "text": "Здравствуйте, хочу узнать про кредит наличными.", "start": 5.0, "end": 8.1},
        {"speaker": "Оператор", "text": "Конечно, помогу разобраться. Какая сумма вас интересует?", "start": 8.5, "end": 11.3},
        {"speaker": "Клиент", "text": "Хотелось бы тысяч десять рублей на год.", "start": 12.0, "end": 14.5},
        {"speaker": "Оператор", "text": "Хорошо. Ставка от четырнадцати и девяти процентов годовых. Страхование жизни по желанию.", "start": 15.0, "end": 20.0},
        {"speaker": "Клиент", "text": "Понятно. А можно оформить онлайн?", "start": 21.0, "end": 23.0},
        {"speaker": "Оператор", "text": "Да, через мобильное приложение МТБанк. Отправить инструкцию на email?", "start": 23.5, "end": 27.0},
        {"speaker": "Клиент", "text": "Да, пожалуйста.", "start": 27.5, "end": 28.5},
        {"speaker": "Оператор", "text": "Отправила. Если есть вопросы — звоните. Всего доброго!", "start": 29.0, "end": 32.0},
        {"speaker": "Клиент", "text": "Спасибо, до свидания.", "start": 32.5, "end": 34.0},
    ]


@pytest.fixture
def sample_full_text(sample_segments) -> str:
    return " ".join(s["text"] for s in sample_segments)


@pytest.fixture
def classification_response() -> dict:
    return {
        "topic": "Потребительский кредит наличными",
        "category": "Продажи",
        "subcategory": "Кредит наличными",
        "priority": "medium",
        "confidence": 0.95,
    }


@pytest.fixture
def quality_response() -> dict:
    return {
        "total": 100,
        "checklist": {
            "greeting": True,
            "need_detection": True,
            "solution_provided": True,
            "farewell": True,
        },
        "issues": [],
        "recommendations": [],
    }


@pytest.fixture
def compliance_response() -> dict:
    return {
        "passed": True,
        "forbidden_phrases_found": [],
        "required_disclaimers": {
            "interest_rate_mentioned": True,
            "insurance_optional_mentioned": True,
            "personal_data_consent": False,
        },
        "issues": [],
    }


@pytest.fixture
def summary_response() -> dict:
    return {
        "summary": "Клиент обратился по вопросу оформления потребительского кредита на сумму 10 000 рублей сроком на год. Оператор предоставил информацию о ставке и условиях. Клиент решил оформить кредит через мобильное приложение.",
        "action_items": [
            "[Оператор] Отправить клиенту инструкцию по оформлению заявки на email",
            "[Клиент] Оформить заявку через мобильное приложение МТБанк",
        ],
    }
