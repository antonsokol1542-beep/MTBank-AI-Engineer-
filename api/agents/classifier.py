"""Classifier Agent — determines call topic, category, and priority."""

from __future__ import annotations

from agents.base import BaseAgent

SYSTEM_PROMPT = """Ты аналитик контакт-центра банка МТБанк.
Проанализируй транскрипт звонка и определи:
1. Тему обращения (конкретная формулировка на русском, например: "Оформление потребительского кредита")
2. Категорию: одна из ["Продажи", "Обслуживание", "Жалоба", "Консультация", "Техподдержка"]
3. Подкатегорию (опционально, уточнение темы)
4. Приоритет: "low" | "medium" | "high" | "critical"
   - critical: жалобы, VIP-клиенты, угрозы судебных разбирательств
   - high: проблемы с картами/счётами, блокировки
   - medium: стандартные заявки на продукты
   - low: общие консультации, информационные запросы
5. Уверенность в классификации: от 0.0 до 1.0

Верни ТОЛЬКО JSON следующего формата (без пояснений):
{
  "topic": "...",
  "category": "...",
  "subcategory": "...",
  "priority": "low|medium|high|critical",
  "confidence": 0.0
}"""


class ClassifierAgent(BaseAgent):
    name = "ClassifierAgent"

    async def run(self, state: dict) -> dict:
        full_text = state.get("full_text", "")
        transcript_segments = state.get("transcript_segments", [])

        # Build labelled transcript for better context
        labelled = _format_transcript(transcript_segments)

        user_prompt = f"Транскрипт звонка:\n\n{labelled}\n\nПолный текст:\n{full_text}"

        data = await self._call_llm(SYSTEM_PROMPT, user_prompt)

        return {
            "classification": {
                "topic": data.get("topic", "Неизвестно"),
                "category": data.get("category", "Консультация"),
                "subcategory": data.get("subcategory"),
                "priority": data.get("priority", "medium"),
                "confidence": float(data.get("confidence", 0.5)),
            }
        }


def _format_transcript(segments: list[dict]) -> str:
    lines = []
    for seg in segments:
        lines.append(f"[{seg['speaker']}] ({seg['start']:.1f}s – {seg['end']:.1f}s): {seg['text']}")
    return "\n".join(lines)
