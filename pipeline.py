"""
MTBank Speech Analytics — Основной OpenWebUI Pipeline

Единый Pipeline, который:
1. Принимает аудиофайл или URL из чата OpenWebUI
2. Транскрибирует речь через faster-whisper (ASR)
3. Выполняет диаризацию (Оператор / Клиент)
4. Запускает 4 LLM-агента через LangGraph:
   - Классификатор темы и приоритета
   - Агент качества оператора
   - Комплаенс-агент
   - Суммаризатор
5. Возвращает форматированный Markdown-отчёт в чат

Оркестрация: LangGraph (последовательный pipeline).
Выбор LangGraph обоснован в README: явный граф состояний, трассируемость,
простое добавление новых агентов без переписывания оркестратора.

Установка:
  Admin Panel → Pipelines → Upload file: pipeline.py
  Настройте Valves (LLM_BASE_URL, LLM_MODEL, WHISPER_MODEL, HF_TOKEN)
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Generator, Iterator, List, Union

from pydantic import BaseModel

PRIORITY_EMOJI = {"low": "🟢", "medium": "🟡", "high": "🟠", "critical": "🔴"}


class Pipeline:
    class Valves(BaseModel):
        LLM_BASE_URL: str = "http://api:8000/v1"
        LLM_MODEL: str = "gpt-4o-mini"
        WHISPER_MODEL: str = "medium"
        WHISPER_DEVICE: str = "cpu"
        WHISPER_COMPUTE_TYPE: str = "int8"
        WHISPER_LANGUAGE: str = "ru"
        HF_TOKEN: str = ""
        # If set — delegate to FastAPI backend instead of running locally
        API_BASE_URL: str = "http://api:8000"

    def __init__(self) -> None:
        self.name = "MTBank Analytics — Full Call Analysis"
        self.valves = self.Valves(
            **{
                k: os.environ.get(k, v.default)
                for k, v in self.Valves.model_fields.items()
            }
        )
        self._transcriber = None
        self._graph = None

    async def on_startup(self) -> None:
        print(f"[MTBank Pipeline] Starting up...")
        self._transcriber = self._init_transcriber()
        self._graph = self._init_graph()
        print("[MTBank Pipeline] Ready.")

    async def on_shutdown(self) -> None:
        print("[MTBank Pipeline] Shutdown.")

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def pipe(
        self,
        body: dict,
        __user__: dict | None = None,
        user_message: str = "",
        model_id: str = "",
        messages: List[dict] | None = None,
    ) -> Union[str, Generator, Iterator]:
        messages = messages or body.get("messages", [])
        last = messages[-1] if messages else {}
        content = last.get("content", "")

        # --- Try base64 audio in structured content ---
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "audio":
                    data = item.get("data") or item.get("url", "")
                    if data.startswith("data:"):
                        _, encoded = data.split(",", 1)
                        return self._run_sync(self._analyze_bytes(base64.b64decode(encoded), ".wav"))
                    if data.startswith("http"):
                        return self._run_sync(self._analyze_url(data))

        # --- Try URL in text ---
        text = user_message or (content if isinstance(content, str) else "")
        url_match = re.search(r"https?://\S+", text.strip())
        if url_match:
            return self._run_sync(self._analyze_url(url_match.group(0)))

        return (
            "📎 Прикрепите аудиофайл (WAV/MP3/OGG) или вставьте URL для анализа.\n\n"
            "_Пример: https://example.com/call.wav_"
        )

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    async def _analyze_url(self, url: str) -> str:
        import httpx
        try:
            async with httpx.AsyncClient(timeout=180.0) as client:
                resp = await client.post(
                    f"{self.valves.API_BASE_URL}/analyze",
                    data={"url": url},
                )
                resp.raise_for_status()
                return self._format_report(resp.json())
        except Exception as exc:
            return f"❌ Ошибка анализа: {exc}"

    async def _analyze_bytes(self, audio_bytes: bytes, suffix: str = ".wav") -> str:
        import httpx
        tmp_path = ""
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name

            async with httpx.AsyncClient(timeout=180.0) as client:
                with open(tmp_path, "rb") as f:
                    resp = await client.post(
                        f"{self.valves.API_BASE_URL}/analyze",
                        files={"file": (f"audio{suffix}", f, "audio/wav")},
                    )
                resp.raise_for_status()
                return self._format_report(resp.json())
        except Exception as exc:
            return f"❌ Ошибка анализа: {exc}"
        finally:
            if tmp_path:
                Path(tmp_path).unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # Transcriber / Graph init (used if running locally without API)
    # ------------------------------------------------------------------

    def _init_transcriber(self):
        try:
            from faster_whisper import WhisperModel
            return WhisperModel(
                self.valves.WHISPER_MODEL,
                device=self.valves.WHISPER_DEVICE,
                compute_type=self.valves.WHISPER_COMPUTE_TYPE,
            )
        except Exception as exc:
            print(f"[MTBank Pipeline] Whisper not available locally: {exc}")
            return None

    def _init_graph(self):
        try:
            from openai import AsyncOpenAI
            from langgraph.graph import END, START, StateGraph
            from typing import TypedDict

            client = AsyncOpenAI(
                api_key=os.environ.get("OPENAI_API_KEY", ""),
                base_url=self.valves.LLM_BASE_URL,
            )
            # Lazy import agents to avoid import errors if not available
            return None  # Full graph is run via API backend
        except Exception as exc:
            print(f"[MTBank Pipeline] LangGraph not available locally: {exc}")
            return None

    # ------------------------------------------------------------------
    # Report formatting
    # ------------------------------------------------------------------

    @staticmethod
    def _format_report(data: dict) -> str:
        duration = data.get("audio_duration", 0)
        proc_time = data.get("processing_time", 0)
        clf = data.get("classification", {})
        qual = data.get("quality_score", {})
        comp = data.get("compliance", {})
        summary = data.get("summary", "")
        action_items = data.get("action_items", [])
        segments = data.get("transcript", [])

        priority = clf.get("priority", "medium")
        priority_emoji = PRIORITY_EMOJI.get(priority, "⚪")
        total = qual.get("total", 0)
        quality_bar = "🟩" * (total // 10) + "⬜" * (10 - total // 10)
        passed = comp.get("passed", True)
        comp_icon = "✅" if passed else "❌"

        lines = [
            f"# 📊 Аналитика звонка",
            f"**Длительность:** {duration:.1f} с | **Обработано за:** {proc_time:.1f} с",
            "",
            "## 🏷️ Классификация",
            f"| Параметр | Значение |",
            f"|---|---|",
            f"| Тема | {clf.get('topic', '–')} |",
            f"| Категория | {clf.get('category', '–')} |",
            f"| Приоритет | {priority_emoji} {priority.upper()} |",
            f"| Уверенность | {clf.get('confidence', 0):.0%} |",
            "",
            "## ✅ Качество обслуживания",
            f"**Итоговый балл:** {total}/100 {quality_bar}",
            "",
            "| Критерий | Статус |",
            "|---|---|",
        ]

        checklist = qual.get("checklist", {})
        criteria = {
            "greeting": "Приветствие и представление",
            "need_detection": "Выявление потребности",
            "solution_provided": "Предоставление решения",
            "farewell": "Прощание с клиентом",
        }
        for key, label in criteria.items():
            icon = "✅" if checklist.get(key) else "❌"
            lines.append(f"| {label} | {icon} |")

        if qual.get("issues"):
            lines += ["", "**Нарушения:**"]
            for issue in qual["issues"]:
                lines.append(f"- ⚠️ {issue}")

        lines += ["", f"## {comp_icon} Комплаенс"]
        if not passed and comp.get("forbidden_phrases_found"):
            lines += ["", "**Запрещённые фразы:**"]
            for phrase in comp["forbidden_phrases_found"]:
                lines.append(f"- 🚫 *«{phrase}»*")
        if comp.get("issues"):
            for issue in comp["issues"]:
                lines.append(f"- ⚠️ {issue}")

        lines += ["", "## 📝 Резюме", summary or "–"]

        if action_items:
            lines += ["", "**Список действий:**"]
            for item in action_items:
                lines.append(f"- [ ] {item}")

        lines += ["", "## 🎙️ Транскрипт", "<details><summary>Показать</summary>", ""]
        current_speaker = None
        for seg in segments:
            speaker = seg.get("speaker", "")
            if speaker != current_speaker:
                current_speaker = speaker
                icon = "👤" if "Оператор" in speaker else "🧑"
                lines.append(f"\n**{icon} {speaker}** `[{seg.get('start', 0):.1f}s]`")
            lines.append(f"> {seg.get('text', '')}")
        lines.append("</details>")

        return "\n".join(lines)

    @staticmethod
    def _run_sync(coro) -> str:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(asyncio.run, coro)
                    return future.result(timeout=180)
            return loop.run_until_complete(coro)
        except Exception as exc:
            return f"❌ Ошибка: {exc}"
