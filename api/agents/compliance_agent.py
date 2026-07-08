"""Compliance Agent — checks for forbidden phrases and required regulatory disclaimers."""

from __future__ import annotations

from agents.base import BaseAgent

SYSTEM_PROMPT = """Ты комплаенс-офицер банка МТБанк. Проверь транскрипт звонка на соответствие внутренним
регламентам и требованиям законодательства Республики Беларусь.

Тебе передана тема звонка (topic) — используй её для оценки применимости дисклеймеров.

1. ЗАПРЕЩЁННЫЕ ФРАЗЫ (forbidden_phrases_found):
   Найди фразы из следующих категорий, процитируй их точно:
   - Угрозы, запугивание или давление на клиента
   - Оскорбления или грубость
   - Дискриминационные высказывания
   - Гарантии одобрения кредита/продукта ("100% одобрим", "гарантированно выдадим")
   - Ложные обещания о ставках или условиях
   - Раскрытие персональных данных третьих лиц

2. ОБЯЗАТЕЛЬНЫЕ ДИСКЛЕЙМЕРЫ (required_disclaimers) — оцениваются ТОЛЬКО если применимы к теме:
   - interest_rate_mentioned: true если тема связана с кредитом/вкладом/ипотекой И ставка была озвучена;
     если тема НЕ связана с кредитными продуктами — ставь true (не применимо, нарушения нет)
   - insurance_optional_mentioned: true если страховка была предложена И оговорена как опциональная;
     если страховка не упоминалась — ставь true (не применимо)
   - personal_data_consent: true если персональные данные не запрашивались, ИЛИ если запрашивались
     и клиент явно или контекстуально дал согласие (продолжил разговор после вопроса оператора)

3. НАРУШЕНИЯ (issues): только реальные нарушения, применимые к данной теме звонка. Пиши на русском языке.

passed = true только если forbidden_phrases_found пуст.

Верни ТОЛЬКО JSON:
{
  "passed": true,
  "forbidden_phrases_found": [],
  "required_disclaimers": {
    "interest_rate_mentioned": true,
    "insurance_optional_mentioned": true,
    "personal_data_consent": true
  },
  "issues": []
}"""


class ComplianceAgent(BaseAgent):
    name = "ComplianceAgent"

    async def run(self, state: dict) -> dict:
        full_text = state.get("full_text", "")
        transcript_segments = state.get("transcript_segments", [])
        classification = state.get("classification", {})
        topic = classification.get("topic", "не определена")
        labelled = _format_full(transcript_segments)

        user_prompt = (
            f"Тема звонка: {topic}\n\n"
            f"Транскрипт звонка:\n\n{labelled}\n\nПолный текст:\n{full_text}"
        )

        data = await self._call_llm(SYSTEM_PROMPT, user_prompt)

        forbidden = data.get("forbidden_phrases_found", [])
        disclaimers_raw = data.get("required_disclaimers", {})
        disclaimers = {
            "interest_rate_mentioned": bool(disclaimers_raw.get("interest_rate_mentioned", False)),
            "insurance_optional_mentioned": bool(disclaimers_raw.get("insurance_optional_mentioned", True)),
            "personal_data_consent": bool(disclaimers_raw.get("personal_data_consent", True)),
        }

        return {
            "compliance": {
                "passed": len(forbidden) == 0,
                "forbidden_phrases_found": forbidden,
                "required_disclaimers": disclaimers,
                "issues": data.get("issues", []),
            }
        }


def _format_full(segments: list[dict]) -> str:
    lines = []
    for seg in segments:
        lines.append(f"[{seg['speaker']}]: {seg['text']}")
    return "\n".join(lines)
