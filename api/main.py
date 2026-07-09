"""
MTBank Speech Analytics API — FastAPI application.

Endpoints:
  POST /analyze             — audio → full analysis (ТЗ-compatible)
  POST /api/v1/transcribe   — audio → transcript only (ASR)
  POST /api/v1/analyze      — audio → full analysis (versioned alias)
  GET  /health              — health check
  GET  /docs                — Swagger UI
"""

from __future__ import annotations

import asyncio
import tempfile
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware

from config import get_settings
from models.schemas import AnalysisResult, HealthResponse, TranscribeResult, TranscriptSegment
from orchestrator.graph import run_analysis
from services.asr import ASRService
from utils.logging import configure_logging, get_logger

logger = get_logger(__name__)

_asr_service: ASRService | None = None
_asr_ready = threading.Event()  # set once the (heavy) model load finishes, success or fail
SUPPORTED_AUDIO = {".wav", ".mp3", ".ogg", ".flac", ".m4a"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    settings = get_settings()

    global _asr_service
    _asr_service = ASRService(
        model_name=settings.whisper_model,
        device=settings.whisper_device,
        compute_type=settings.whisper_compute_type,
        language=settings.whisper_language,
        hf_token=settings.hf_token,
    )

    # Load the Whisper/pyannote models in a background thread so the ASGI server
    # starts accepting connections immediately. Otherwise a blocking load (~1.5 GB
    # model download on cold start) delays /health and can trip a platform
    # healthcheck / readiness probe, marking the deploy unhealthy.
    def _load_model() -> None:
        try:
            _asr_service.load()
            logger.info("ASR model loaded (background)")
        except Exception:
            logger.exception("ASR model failed to load")
        finally:
            _asr_ready.set()

    threading.Thread(target=_load_model, name="asr-loader", daemon=True).start()
    logger.info("Application started (ASR model loading in background)")

    yield

    logger.info("Application shutting down")


app = FastAPI(
    title="MTBank Speech Analytics API",
    version="1.0.0",
    description="AI-powered call center speech analytics: ASR + multi-agent analysis",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        asr_loaded=_asr_service is not None and _asr_service.is_loaded,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _save_upload(file: UploadFile) -> str:
    suffix = Path(file.filename or "audio.wav").suffix.lower()
    if suffix not in SUPPORTED_AUDIO:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unsupported audio format '{suffix}'. Supported: {SUPPORTED_AUDIO}",
        )
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    content = await file.read()
    tmp.write(content)
    tmp.flush()
    return tmp.name


async def _download_audio(url: str) -> str:
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.get(url, follow_redirects=True)
        resp.raise_for_status()

    content_type = resp.headers.get("content-type", "")
    suffix = _suffix_from_content_type(content_type) or Path(url).suffix.lower() or ".wav"

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(resp.content)
    tmp.flush()
    return tmp.name


def _suffix_from_content_type(ct: str) -> str:
    mapping = {
        "audio/wav": ".wav",
        "audio/x-wav": ".wav",
        "audio/mpeg": ".mp3",
        "audio/ogg": ".ogg",
        "audio/flac": ".flac",
    }
    for key, val in mapping.items():
        if key in ct:
            return val
    return ""


def _build_full_text(segments: list[dict]) -> str:
    return " ".join(s["text"] for s in segments)


async def _process_audio(file: UploadFile | None, audio_url: str | None) -> tuple[list, float, str]:
    """Shared audio processing: returns (segments, duration, full_text)."""
    if not _asr_service:
        raise HTTPException(status_code=503, detail="ASR service not initialised")

    # On a cold start the model may still be loading in the background — wait for it
    # (up to 5 min) instead of failing immediately, so the first request succeeds.
    if not _asr_service.is_loaded:
        await asyncio.to_thread(_asr_ready.wait, 300)
        if not _asr_service.is_loaded:
            raise HTTPException(status_code=503, detail="ASR model still loading or failed to load")

    if file is None and audio_url is None:
        raise HTTPException(status_code=400, detail="Provide 'file' or 'url'")

    audio_path = await _save_upload(file) if file else await _download_audio(audio_url)

    try:
        segments, duration = _asr_service.process(audio_path)
    finally:
        Path(audio_path).unlink(missing_ok=True)

    return segments, duration, _build_full_text(segments)


# ---------------------------------------------------------------------------
# POST /analyze  (ТЗ-compatible — matches the spec exactly)
# ---------------------------------------------------------------------------

@app.post(
    "/analyze",
    response_model=AnalysisResult,
    summary="Full call analytics (ASR + classification + quality + compliance + summary)",
    tags=["Analytics"],
)
async def analyze(
    file: UploadFile | None = File(default=None, description="Audio file (WAV/MP3/OGG)"),
    url: str | None = Form(default=None, description="URL to audio file"),
) -> AnalysisResult:
    t0 = time.perf_counter()
    segments, duration, full_text = await _process_audio(file, url)
    state = await run_analysis(segments, full_text)
    transcript = [TranscriptSegment(**s) for s in segments]

    return AnalysisResult(
        audio_duration=round(duration, 2),
        transcript=transcript,
        full_text=full_text,
        classification=state["classification"],
        quality_score=state["quality_score"],
        compliance=state["compliance"],
        summary=state["summary"],
        action_items=state["action_items"],
        processing_time=round(time.perf_counter() - t0, 3),
    )


# ---------------------------------------------------------------------------
# POST /api/v1/transcribe  (ASR only)
# ---------------------------------------------------------------------------

@app.post(
    "/api/v1/transcribe",
    response_model=TranscribeResult,
    summary="Transcribe audio with speaker diarization",
    tags=["ASR"],
)
async def transcribe(
    file: UploadFile | None = File(default=None),
    audio_url: str | None = Form(default=None),
) -> TranscribeResult:
    t0 = time.perf_counter()
    segments, duration, full_text = await _process_audio(file, audio_url)
    transcript = [TranscriptSegment(**s) for s in segments]

    return TranscribeResult(
        audio_duration=round(duration, 2),
        transcript=transcript,
        full_text=full_text,
        processing_time=round(time.perf_counter() - t0, 3),
    )


# ---------------------------------------------------------------------------
# POST /api/v1/analyze  (versioned alias)
# ---------------------------------------------------------------------------

@app.post(
    "/api/v1/analyze",
    response_model=AnalysisResult,
    summary="Full call analytics (versioned alias of /analyze)",
    tags=["Analytics"],
)
async def analyze_v1(
    file: UploadFile | None = File(default=None),
    audio_url: str | None = Form(default=None),
) -> AnalysisResult:
    t0 = time.perf_counter()
    segments, duration, full_text = await _process_audio(file, audio_url)
    state = await run_analysis(segments, full_text)
    transcript = [TranscriptSegment(**s) for s in segments]

    return AnalysisResult(
        audio_duration=round(duration, 2),
        transcript=transcript,
        full_text=full_text,
        classification=state["classification"],
        quality_score=state["quality_score"],
        compliance=state["compliance"],
        summary=state["summary"],
        action_items=state["action_items"],
        processing_time=round(time.perf_counter() - t0, 3),
    )
