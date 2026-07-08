"""Summarizer Agent — generates a call summary and action items."""

from __future__ import annotations

from agents.base import BaseAgent

SYSTEM_PROMPT = """Ты аналитик контакт-центра банка МТБанк.
На основе транскрипта звонка и результатов анализа составь:

1. summary: краткое резюме разговора (3–5 предложений), включающее:
   - Суть обращения клиента
   - Что предложил/сделал оператор
   - Результат разговора

2. action_items: список конкретных действий для дальнейшей работы.
   Укажи исполнителя в формате "[Оператор]" или "[Клиент]" или "[Система]".
   Например: "[Оператор] Отправить клиенту на email инструкцию по оформлению заявки онлайн"

Верни ТОЛЬКО JSON:
{
  "summary": "...",
  "action_items": ["...", "..."]
}"""


class SummarizerAgent(BaseAgent):
    name = "SummarizerAgent"

    async def run(self, state: dict) -> dict:
        full_text = state.get("full_text", "")
        transcript_segments = state.get("transcript_segments", [])
        classification = state.get("classification", {})
        quality_score = state.get("quality_score", {})
        compliance = state.get("compliance", {})

        labelled = _format_full(transcript_segments)
        context = _build_context(classification, quality_score, compliance)

        user_prompt = (
            f"Транскрипт звонка:\n\n{labelled}\n\n"
            f"Результаты анализа:\n{context}\n\n"
            f"Полный текст:\n{full_text}"
        )

        data = await self._call_llm(SYSTEM_PROMPT, user_prompt)

        return {
            "summary": data.get("summary", "Резюме недоступно."),
            "action_items": data.get("action_items", []),
        }


def _format_full(segments: list[dict]) -> str:
    return "\n".join(f"[{s['speaker']}]: {s['text']}" for s in segments)


def _build_context(classification: dict, quality_score: dict, compliance: dict) -> str:
    parts = []
    if classification:
        parts.append(
            f"- Тема: {classification.get('topic', '–')}, "
            f"Категория: {classification.get('category', '–')}, "
            f"Приоритет: {classification.get('priority', '–')}"
        )
    if quality_score:
        issues = quality_score.get("issues", [])
        parts.append(f"- Качество: {quality_score.get('total', '–')}/100, нарушения: {issues or 'нет'}")
    if compliance:
        passed = compliance.get("passed", True)
        comp_issues = compliance.get("issues", [])
        parts.append(f"- Комплаенс: {'пройден' if passed else 'НЕ пройден'}, замечания: {comp_issues or 'нет'}")
    return "\n".join(parts)
