"""
OpenWebUI Pipeline: ASR (Speech-to-Text with Speaker Diarization)
"""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from typing import Generator, Iterator, List, Union

import httpx
from pydantic import BaseModel

AUDIO_EXTENSIONS = {".wav", ".mp3", ".ogg", ".flac", ".m4a", ".webm"}
OPENWEBUI_BASE_URL = "http://openwebui:8080"


class Pipeline:
    class Valves(BaseModel):
        API_BASE_URL: str = "http://api:8000"
        REQUEST_TIMEOUT: int = 120
        OPENWEBUI_BASE_URL: str = "http://openwebui:8080"
        OPENWEBUI_API_KEY: str = ""

    def __init__(self) -> None:
        self.name = "MTBank ASR — Speech to Text"
        self.valves = self.Valves(
            **{
                k: os.environ.get(k, v.default)
                for k, v in self.Valves.model_fields.items()
            }
        )

    async def on_startup(self) -> None:
        print(f"[ASR Pipeline] Startup. API: {self.valves.API_BASE_URL}")

    async def on_shutdown(self) -> None:
        print("[ASR Pipeline] Shutdown.")

    def pipe(
        self,
        user_message: str,
        model_id: str,
        messages: List[dict],
        body: dict,
    ) -> Union[str, Generator, Iterator]:

        audio_url, audio_bytes, audio_suffix = self._extract_audio(body, messages, user_message)

        if audio_url:
            return self._transcribe_url(audio_url)
        if audio_bytes:
            return self._transcribe_bytes(audio_bytes, audio_suffix)

        return (
            "📎 Прикрепите аудиофайл (WAV/MP3/OGG) или вставьте URL для транскрипции.\n\n"
            "_Как прикрепить: нажмите значок скрепки рядом с полем ввода_"
        )

    def _extract_audio(
        self, body: dict, messages: list, user_message: str
    ) -> tuple[str | None, bytes | None, str]:

        token = (body.get("user") or {}).get("token", "") or self.valves.OPENWEBUI_API_KEY

        # 1. <attached_files> XML in message content (actual OpenWebUI format)
        last = messages[-1] if messages else {}
        content = last.get("content", "")
        if isinstance(content, str) and "<attached_files>" in content:
            result = self._parse_attached_files(content, token)
            if result:
                return result

        # 2. body["files"] (older OpenWebUI versions)
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

        # 4. URL in text
        text = (user_message or "").strip()
        url_match = re.search(r"https?://\S+\.(wav|mp3|ogg|flac)(\?[^\s]*)?", text, re.I)
        if url_match:
            return url_match.group(0), None, ""

        return None, None, ""

    def _parse_attached_files(self, content: str, token: str) -> tuple | None:
        matches = re.findall(
            r'<file[^>]+url="([^"]+)"[^>]+content_type="([^"]+)"[^>]+name="([^"]+)"',
            content,
        )
        if not matches:
            matches = re.findall(
                r'<file[^>]+name="([^"]+)"[^>]+content_type="([^"]+)"[^>]+url="([^"]+)"',
                content,
            )
            matches = [(m[2], m[1], m[0]) for m in matches]

        for file_id, content_type, name in matches:
            is_audio = content_type.startswith("audio/") or any(
                name.lower().endswith(ext) for ext in AUDIO_EXTENSIONS
            )
            if not is_audio:
                continue
            audio_bytes, suffix = self._download_from_openwebui(file_id, name, token)
            if audio_bytes:
                return None, audio_bytes, suffix
        return None

    def _download_from_openwebui(self, file_id: str, name: str, token: str) -> tuple[bytes | None, str]:
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
            for path in sorted(candidates):
                try:
                    data = Path(path).read_bytes()
                    if data and _is_audio_bytes(data):
                        print(f"[ASR Pipeline] Read {len(data)} bytes from {path}")
                        return data, suffix
                    elif data:
                        print(f"[ASR Pipeline] Skipping non-audio file {path} ({data[:8]})")
                except Exception as exc:
                    print(f"[ASR Pipeline] Disk read failed {path}: {exc}")

        # 2. HTTP fallback with auth token
        base = self.valves.OPENWEBUI_BASE_URL or OPENWEBUI_BASE_URL
        url = f"{base}/api/v1/files/{file_id}/content"
        try:
            with httpx.Client(timeout=60) as client:
                headers = {"Authorization": f"Bearer {token}"} if token else {}
                resp = client.get(url, headers=headers)
                if resp.status_code == 401 and token:
                    resp = client.get(url)
                resp.raise_for_status()
                return resp.content, suffix
        except Exception as exc:
            print(f"[ASR Pipeline] HTTP download failed {url}: {exc}")
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

    def _transcribe_url(self, url: str) -> str:
        try:
            with httpx.Client(timeout=self.valves.REQUEST_TIMEOUT) as client:
                resp = client.post(
                    f"{self.valves.API_BASE_URL}/api/v1/transcribe",
                    data={"audio_url": url},
                )
                resp.raise_for_status()
                return self._format_result(resp.json())
        except Exception as exc:
            return f"❌ Ошибка транскрипции: {exc}"

    def _transcribe_bytes(self, audio_bytes: bytes, suffix: str = ".wav") -> str:
        tmp_path = ""
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name

            with httpx.Client(timeout=self.valves.REQUEST_TIMEOUT) as client:
                with open(tmp_path, "rb") as f:
                    resp = client.post(
                        f"{self.valves.API_BASE_URL}/api/v1/transcribe",
                        files={"file": (f"audio{suffix}", f, "audio/wav")},
                    )
                resp.raise_for_status()
                return self._format_result(resp.json())
        except Exception as exc:
            return f"❌ Ошибка транскрипции: {exc}"
        finally:
            if tmp_path:
                Path(tmp_path).unlink(missing_ok=True)

    @staticmethod
    def _format_result(data: dict) -> str:
        segments = data.get("transcript", [])
        duration = data.get("audio_duration", 0)
        proc_time = data.get("processing_time", 0)

        lines = [
            "## 🎙️ Транскрипт звонка",
            f"**Длительность:** {duration:.1f} с | **Время обработки:** {proc_time:.1f} с",
            "",
        ]

        current_speaker = None
        for seg in segments:
            speaker = seg.get("speaker", "Unknown")
            text = seg.get("text", "")
            start = seg.get("start", 0)

            if speaker != current_speaker:
                current_speaker = speaker
                icon = "👤" if "Оператор" in speaker else "🧑"
                lines.append(f"\n**{icon} {speaker}** `[{start:.1f}s]`")
            lines.append(f"> {text}")

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
