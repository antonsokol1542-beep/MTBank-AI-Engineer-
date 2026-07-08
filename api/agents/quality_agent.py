"""Quality Agent — evaluates operator performance against a service checklist."""

from __future__ import annotations

from agents.base import BaseAgent

SYSTEM_PROMPT = """Ты эксперт по качеству обслуживания контакт-центра банка МТБанк.
Оцени работу оператора по стандартному чеклисту из 4 пунктов.

Чеклист (каждый пункт = 25 баллов):
1. greeting — Оператор поприветствовал клиента и представился по имени.
2. need_detection — Оператор выяснил цель обращения и потребность клиента.
3. solution_provided — Оператор предоставил решение, информацию или передал обращение куда нужно.
4. farewell — Оператор вежливо завершил разговор: сказал "до свидания", "всего доброго", "хорошего дня"
   или аналогичную прощальную фразу. Уточнение про дополнительные вопросы желательно, но не обязательно.

Для каждого пункта: true (выполнено) или false (не выполнено).
Итоговый total = количество выполненных пунктов × 25.

Также перечисли:
- issues: конкретные нарушения стандартов (список строк)
- recommendations: рекомендации по улучшению (список строк)

Верни ТОЛЬКО JSON:
{
  "total": 0,
  "checklist": {
    "greeting": true,
    "need_detection": true,
    "solution_provided": true,
    "farewell": true
  },
  "issues": [],
  "recommendations": []
}"""


class QualityAgent(BaseAgent):
    name = "QualityAgent"

    async def run(self, state: dict) -> dict:
        full_text = state.get("full_text", "")
        transcript_segments = state.get("transcript_segments", [])
        labelled = _format_operator_focus(transcript_segments)

        user_prompt = (
            f"Транскрипт звонка (разбивка по репликам):\n\n{labelled}\n\n"
            f"Полный текст:\n{full_text}"
        )

        data = await self._call_llm(SYSTEM_PROMPT, user_prompt)

        checklist_raw = data.get("checklist", {})
        checklist = {
            "greeting": bool(checklist_raw.get("greeting", False)),
            "need_detection": bool(checklist_raw.get("need_detection", False)),
            "solution_provided": bool(checklist_raw.get("solution_provided", False)),
            "farewell": bool(checklist_raw.get("farewell", False)),
        }
        total = sum(1 for v in checklist.values() if v) * 25

        return {
            "quality_score": {
                "total": data.get("total", total),
                "checklist": checklist,
                "issues": data.get("issues", []),
                "recommendations": data.get("recommendations", []),
            }
        }


def _format_operator_focus(segments: list[dict]) -> str:
    lines = []
    for seg in segments:
        prefix = ">>> ОПЕРАТОР" if seg["speaker"] == "Оператор" else "    КЛИЕНТ   "
        lines.append(f"{prefix} ({seg['start']:.1f}s): {seg['text']}")
    return "\n".join(lines)
