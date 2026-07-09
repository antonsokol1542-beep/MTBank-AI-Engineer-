"""
OpenWebUI Pipeline: Full Call Analytics (ASR + Multi-Agent Analysis)
"""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from typing import Generator, Iterator, List, Union

import httpx
from pydantic import BaseModel

PRIORITY_EMOJI = {
    "low": "🟢",
    "medium": "🟡",
    "high": "🟠",
    "critical": "🔴",
}

AUDIO_EXTENSIONS = {".wav", ".mp3", ".ogg", ".flac", ".m4a", ".webm"}
OPENWEBUI_BASE_URL = "http://openwebui:8080"


class Pipeline:
    class Valves(BaseModel):
        API_BASE_URL: str = "http://api:8000"
        REQUEST_TIMEOUT: int = 180
        OPENWEBUI_BASE_URL: str = "http://openwebui:8080"
        OPENWEBUI_API_KEY: str = ""

    def __init__(self) -> None:
        self.name = "MTBank Analytics — Full Call Analysis"
        self.valves = self.Valves(
            **{
                k: os.environ.get(k, v.default)
                for k, v in self.Valves.model_fields.items()
            }
        )
        self._last_error = ""

    async def on_startup(self) -> None:
        print(f"[Analytics Pipeline] Startup. API: {self.valves.API_BASE_URL}")

    async def on_shutdown(self) -> None:
        print("[Analytics Pipeline] Shutdown.")

    def pipe(
        self,
        user_message: str,
        model_id: str,
        messages: List[dict],
        body: dict,
    ) -> Union[str, Generator, Iterator]:

        self._last_error = ""

        # Try to extract audio URL or bytes from the request
        audio_url, audio_bytes, audio_suffix = self._extract_audio(body, messages, user_message)

        if audio_url:
            return self._analyze_url(audio_url)
        if audio_bytes:
            return self._analyze_bytes(audio_bytes, audio_suffix)

        # A file was attached but could not be fetched — report why instead of
        # pretending nothing was attached.
        if self._last_error:
            return (
                "❌ Файл прикреплён, но скачать его из OpenWebUI не удалось.\n\n"
                f"`{self._last_error}`\n\n"
                "Проверьте валвы `OPENWEBUI_BASE_URL` и `OPENWEBUI_API_KEY` — "
                "они обязательны, когда контейнеры не делят общий том."
            )

        return (
            "📎 Прикрепите аудиофайл (WAV/MP3/OGG) или вставьте ссылку для анализа.\n\n"
            "_Нажмите значок скрепки рядом с полем ввода, чтобы прикрепить файл._"
        )

    # ------------------------------------------------------------------
    # Audio extraction — handles all OpenWebUI file-upload formats
    # ------------------------------------------------------------------

    def _extract_audio(
        self, body: dict, messages: list, user_message: str
    ) -> tuple[str | None, bytes | None, str]:
        """Returns (url, bytes, suffix). One of url/bytes will be non-None."""

        token = (body.get("user") or {}).get("token", "") or self.valves.OPENWEBUI_API_KEY

        # 1. <attached_files> XML in message content (actual OpenWebUI format)
        last = messages[-1] if messages else {}
        content = last.get("content", "")
        if isinstance(content, str) and "<attached_files>" in content:
            result = self._parse_attached_files(content, token)
            if result:
                return result

        # 2. body["files"] list (older OpenWebUI versions)
        for file_item in body.get("files", []):
            result = self._try_file_item(file_item, token)
            if result:
                return result + ("",)

        # 3. Message content as list
        if isinstance(content, list):
            for item in content:
                if not isinstance(item, dict):
                    continue
                item_type = item.get("type", "")
                if item_type == "file":
                    result = self._try_file_item(item, token)
                    if result:
                        return result + ("",)
                if item_type == "audio":
                    import base64
                    data = item.get("data") or item.get("url", "")
                    if data.startswith("http"):
                        return data, None, ""
                    if data.startswith("data:"):
                        _, encoded = data.split(",", 1)
                        return None, base64.b64decode(encoded), ".wav"

        # 4. URL in user message text
        text = (user_message or "").strip()
        url_match = re.search(r"https?://\S+\.(wav|mp3|ogg|flac)(\?[^\s]*)?", text, re.I)
        if url_match:
            return url_match.group(0), None, ""

        return None, None, ""

    def _parse_attached_files(self, content: str, token: str) -> tuple | None:
        """Parse <file ... url="UUID" content_type="audio/..." name="..."/> from message."""
        matches = re.findall(
            r'<file[^>]+url="([^"]+)"[^>]+content_type="([^"]+)"[^>]+name="([^"]+)"',
            content,
        )
        # Also try reversed attribute order
        if not matches:
            matches = re.findall(
                r'<file[^>]+name="([^"]+)"[^>]+content_type="([^"]+)"[^>]+url="([^"]+)"',
                content,
            )
            matches = [(m[2], m[1], m[0]) for m in matches]  # reorder to (url, ct, name)

        for file_id, content_type, name in matches:
            is_audio = content_type.startswith("audio/") or any(
                name.lower().endswith(ext) for ext in AUDIO_EXTENSIONS
            )
            if not is_audio:
                continue

            # Download file bytes from OpenWebUI internal API
            audio_bytes, suffix = self._download_from_openwebui(file_id, name, token)
            if audio_bytes:
                return None, audio_bytes, suffix

        return None

    def _download_from_openwebui(self, file_id: str, name: str, token: str) -> tuple[bytes | None, str]:
        """Download file content from OpenWebUI by file UUID."""
        suffix = Path(name).suffix or ".wav"

        # 1. Read directly from mounted volume (avoids HTTP auth issues)
        import glob as _glob
        for pattern in [
            f"/openwebui_data/uploads/{file_id}_*",   # most common: {uuid}_{original_name}
            f"/openwebui_data/uploads/{file_id}{suffix}",
            f"/openwebui_data/uploads/{file_id}",
            f"/openwebui_data/uploads/{file_id}.*",
        ]:
            candidates = _glob.glob(pattern) if "*" in pattern else ([pattern] if Path(pattern).exists() else [])
            for path in sorted(candidates):  # sort for determinism
                try:
                    data = Path(path).read_bytes()
                    if data and _is_audio_bytes(data):
                        print(f"[Analytics Pipeline] Read {len(data)} bytes from {path}")
                        return data, suffix
                    elif data:
                        print(f"[Analytics Pipeline] Skipping non-audio file {path} ({data[:8]})")
                except Exception as exc:
                    print(f"[Analytics Pipeline] Disk read failed {path}: {exc}")

        # 2. HTTP fallback with auth token (required when there is no shared volume)
        base = (self.valves.OPENWEBUI_BASE_URL or OPENWEBUI_BASE_URL).rstrip("/")
        url = f"{base}/api/v1/files/{file_id}/content"
        try:
            with httpx.Client(timeout=60, follow_redirects=True) as client:
                headers = {"Authorization": f"Bearer {token}"} if token else {}
                resp = client.get(url, headers=headers)
                resp.raise_for_status()
                return resp.content, suffix
        except Exception as exc:
            self._last_error = f"GET {url} → {type(exc).__name__}: {exc}"
            print(f"[Analytics Pipeline] HTTP download failed {url}: {exc}")
            return None, suffix

    def _try_file_item(self, item: dict, token: str = "") -> tuple[str, None] | None:
        f = item.get("file", item)
        name = f.get("name", "") or f.get("filename", "")
        url = f.get("url", "")
        meta = f.get("meta", {})
        content_type = meta.get("content_type", "") or f.get("content_type", "")

        is_audio = content_type.startswith("audio/") or any(
            name.lower().endswith(ext) for ext in AUDIO_EXTENSIONS
        )
        if not is_audio:
            return None

        if url.startswith("/"):
            url = f"{OPENWEBUI_BASE_URL}{url}"
        if url:
            return url, None
        return None

    # ------------------------------------------------------------------
    # Analysis calls to FastAPI backend
    # ------------------------------------------------------------------

    def _analyze_url(self, url: str) -> str:
        try:
            with httpx.Client(timeout=self.valves.REQUEST_TIMEOUT) as client:
                resp = client.post(
                    f"{self.valves.API_BASE_URL}/analyze",
                    data={"url": url},
                )
                resp.raise_for_status()
                return self._format_report(resp.json())
        except Exception as exc:
            return f"❌ Ошибка анализа: {exc}"

    def _analyze_bytes(self, audio_bytes: bytes, suffix: str = ".wav") -> str:
        tmp_path = ""
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name

            with httpx.Client(timeout=self.valves.REQUEST_TIMEOUT) as client:
                with open(tmp_path, "rb") as f:
                    resp = client.post(
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
    # If the URL requires OpenWebUI auth — download first, then upload
    # ------------------------------------------------------------------

    def _download_and_analyze(self, url: str) -> str:
        try:
            with httpx.Client(timeout=60) as client:
                resp = client.get(url)
                resp.raise_for_status()
            suffix = Path(url.split("?")[0]).suffix or ".wav"
            return self._analyze_bytes(resp.content, suffix)
        except Exception as exc:
            return f"❌ Ошибка загрузки файла: {exc}"

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
            "# 📊 Аналитика звонка",
            f"**Длительность:** {duration:.1f} с | **Обработано за:** {proc_time:.1f} с",
            "",
            "## 🏷️ Классификация",
            "| Параметр | Значение |",
            "|---|---|",
            f"| Тема | {clf.get('topic', '–')} |",
            f"| Категория | {clf.get('category', '–')} |",
            f"| Подкатегория | {clf.get('subcategory') or '–'} |",
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

        if qual.get("recommendations"):
            lines += ["", "**Рекомендации:**"]
            for rec in qual["recommendations"]:
                lines.append(f"- 💡 {rec}")

        lines += ["", f"## {comp_icon} Комплаенс"]
        if not passed and comp.get("forbidden_phrases_found"):
            lines += ["", "**Запрещённые фразы:**"]
            for phrase in comp["forbidden_phrases_found"]:
                lines.append(f"- 🚫 *«{phrase}»*")

        disclaimers = comp.get("required_disclaimers", {})
        disc_labels = {
            "interest_rate_mentioned": "Процентная ставка озвучена",
            "insurance_optional_mentioned": "Опциональность страховки упомянута",
            "personal_data_consent": "Согласие на обработку данных",
        }
        if disclaimers:
            lines += ["", "**Обязательные дисклеймеры:**"]
            for key, label in disc_labels.items():
                icon = "✅" if disclaimers.get(key, False) else "❌"
                lines.append(f"- {icon} {label}")

        if comp.get("issues"):
            lines += ["", "**Замечания:**"]
            for issue in comp["issues"]:
                lines.append(f"- ⚠️ {issue}")

        lines += ["", "## 📝 Резюме", summary or "–"]

        if action_items:
            lines += ["", "**Список действий:**"]
            for item in action_items:
                lines.append(f"- [ ] {item}")

        lines += ["", "## 🎙️ Транскрипт", "<details><summary>Показать транскрипт</summary>", ""]
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


def _is_audio_bytes(data: bytes) -> bool:
    """Check audio magic bytes to avoid reading metadata/JSON files as audio."""
    if len(data) < 4:
        return False
    magic = data[:4]
    return magic in (
        b"RIFF",   # WAV
        b"OggS",   # OGG
        b"fLaC",   # FLAC
        b"ID3\x03", b"ID3\x04",  # MP3 with ID3v2
    ) or (magic[:2] == b"\xff\xfb" or magic[:2] == b"\xff\xf3")  # MP3 sync


def _score_bar(score: int) -> str:
    filled = score // 10
    return "🟩" * filled + "⬜" * (10 - filled)
