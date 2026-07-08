"""
ASR Service: audio transcription via faster-whisper + speaker diarization.

Speaker diarization strategy:
  1. If HF_TOKEN is set → pyannote/speaker-diarization-3.1 (proper diarization).
  2. Fallback → energy/pause-based heuristic: first speaker is Оператор,
     switches at silences > SPEAKER_CHANGE_GAP_S seconds.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Optional

from utils.logging import get_logger

logger = get_logger(__name__)

SPEAKER_OPERATOR = "Оператор"
SPEAKER_CLIENT = "Клиент"
SPEAKER_CHANGE_GAP_S = 0.3   # seconds of silence to trigger speaker switch
SPEAKER_CHANGE_DURATION_S = 6.0  # force speaker switch if one speaker runs longer than this


def _split_long_segment(seg: dict, max_duration: float) -> list[dict]:
    """Recursively split a segment at the sentence boundary nearest to midpoint."""
    import re as _re
    duration = seg["end"] - seg["start"]
    if duration <= max_duration:
        return [seg]

    text = seg["text"].strip()
    # Find sentence boundaries
    boundaries: list[int] = [m.end() for m in _re.finditer(r'[.!?]+\s*', text)]
    if not boundaries or boundaries[-1] >= len(text):
        boundaries = boundaries[:-1]  # drop trailing boundary

    mid_char = len(text) / 2
    if boundaries:
        # Choose the sentence boundary closest to the character midpoint
        split_at = min(boundaries, key=lambda b: abs(b - mid_char))
    else:
        # No sentence boundary found — split at word midpoint
        words = text.split()
        half = max(1, len(words) // 2)
        split_at = len(" ".join(words[:half]))

    first_text = text[:split_at].strip()
    second_text = text[split_at:].strip()
    if not first_text or not second_text:
        return [seg]

    mid_t = seg["start"] + duration * (split_at / len(text))
    first = {"start": seg["start"], "end": mid_t, "text": first_text}
    second = {"start": mid_t, "end": seg["end"], "text": second_text}

    return _split_long_segment(first, max_duration) + _split_long_segment(second, max_duration)


class ASRService:
    def __init__(
        self,
        model_name: str = "medium",
        device: str = "cpu",
        compute_type: str = "int8",
        language: str = "ru",
        hf_token: str = "",
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.hf_token = hf_token

        self._whisper: Optional[object] = None
        self._diarizer: Optional[object] = None
        self._loaded = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        from faster_whisper import WhisperModel

        logger.info(
            "Loading Whisper model",
            model=self.model_name,
            device=self.device,
            compute_type=self.compute_type,
        )
        self._whisper = WhisperModel(
            self.model_name,
            device=self.device,
            compute_type=self.compute_type,
        )

        if self.hf_token:
            self._load_diarizer()

        self._loaded = True
        logger.info("ASR service ready", diarizer=self._diarizer is not None)

    def _load_diarizer(self) -> None:
        try:
            # Patch huggingface_hub: newer versions removed `use_auth_token` parameter
            import huggingface_hub
            for _fn_name in ("hf_hub_download", "snapshot_download"):
                _orig = getattr(huggingface_hub, _fn_name, None)
                if _orig is None:
                    continue
                import inspect
                if "use_auth_token" not in inspect.signature(_orig).parameters:
                    import functools
                    @functools.wraps(_orig)
                    def _patched(*args, use_auth_token=None, token=None, _fn=_orig, **kwargs):
                        if use_auth_token and not token:
                            token = use_auth_token
                        return _fn(*args, token=token, **kwargs)
                    setattr(huggingface_hub, _fn_name, _patched)

            from pyannote.audio import Pipeline as PyannotePipeline

            # Set token in env so pyannote/HF Hub can pick it up
            os.environ.setdefault("HF_TOKEN", self.hf_token)
            os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", self.hf_token)

            logger.info("Loading pyannote diarization pipeline")
            try:
                self._diarizer = PyannotePipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    token=self.hf_token,
                )
            except TypeError:
                # Older pyannote versions use `use_auth_token` parameter
                self._diarizer = PyannotePipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=self.hf_token,
                )
            logger.info("Diarization pipeline loaded")
        except Exception as exc:
            logger.warning("Diarization pipeline failed to load, using fallback", error=str(exc))
            self._diarizer = None

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, audio_path: str) -> tuple[list[dict], float]:
        """Transcribe audio and assign speakers.

        Returns:
            (segments, duration_seconds)
            Each segment: {"speaker": str, "text": str, "start": float, "end": float}
        """
        if not self._loaded:
            raise RuntimeError("ASR service not loaded. Call load() first.")

        raw_segments, duration = self._transcribe(audio_path)

        if self._diarizer:
            diarization = self._diarize_pyannote(audio_path)
            result = self._align(raw_segments, diarization)
        else:
            result = self._assign_speakers_simple(raw_segments)

        return result, duration

    # ------------------------------------------------------------------
    # Transcription
    # ------------------------------------------------------------------

    def _transcribe(self, audio_path: str) -> tuple[list[dict], float]:
        logger.info("Transcribing audio", path=audio_path)
        segments_iter, info = self._whisper.transcribe(
            audio_path,
            language=self.language,
            beam_size=1,
            word_timestamps=False,
            condition_on_previous_text=True,
            max_new_tokens=150,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 300},
        )

        segments = []
        for seg in segments_iter:
            segments.append(
                {
                    "start": round(seg.start, 3),
                    "end": round(seg.end, 3),
                    "text": seg.text.strip(),
                }
            )

        duration = info.duration
        logger.info("Transcription done", segments=len(segments), duration=round(duration, 1))
        return segments, duration

    # ------------------------------------------------------------------
    # Diarization — pyannote
    # ------------------------------------------------------------------

    def _diarize_pyannote(self, audio_path: str) -> list[dict]:
        logger.info("Running pyannote diarization")
        diarization = self._diarizer(audio_path)

        speaker_map: dict[str, str] = {}
        counter = 0
        diar_segments = []

        for turn, _, speaker in diarization.itertracks(yield_label=True):
            if speaker not in speaker_map:
                speaker_map[speaker] = SPEAKER_OPERATOR if counter == 0 else SPEAKER_CLIENT
                counter += 1
            diar_segments.append(
                {
                    "start": round(turn.start, 3),
                    "end": round(turn.end, 3),
                    "speaker": speaker_map[speaker],
                }
            )

        logger.info("Diarization done", speaker_segments=len(diar_segments))
        return diar_segments

    # ------------------------------------------------------------------
    # Diarization — simple heuristic fallback
    # ------------------------------------------------------------------

    def _assign_speakers_simple(self, segments: list[dict]) -> list[dict]:
        """Assign speakers based on pause gaps and segment duration."""
        expanded: list[dict] = []
        for seg in segments:
            expanded.extend(_split_long_segment(seg, SPEAKER_CHANGE_DURATION_S))

        result = []
        current_speaker = SPEAKER_OPERATOR
        last_end = 0.0
        continuous_s = 0.0  # time current speaker has spoken without a break

        def _switch(spk: str) -> str:
            return SPEAKER_CLIENT if spk == SPEAKER_OPERATOR else SPEAKER_OPERATOR

        for seg in expanded:
            gap = seg["start"] - last_end
            seg_dur = seg["end"] - seg["start"]

            if last_end > 0 and gap > SPEAKER_CHANGE_GAP_S:
                # Pause detected → switch speaker
                current_speaker = _switch(current_speaker)
                continuous_s = 0.0
            elif continuous_s > SPEAKER_CHANGE_DURATION_S:
                # One speaker talking too long without pause → force switch
                current_speaker = _switch(current_speaker)
                continuous_s = 0.0

            continuous_s += seg_dur
            result.append(
                {
                    "speaker": current_speaker,
                    "text": seg["text"],
                    "start": seg["start"],
                    "end": seg["end"],
                }
            )
            last_end = seg["end"]

        return result

    # ------------------------------------------------------------------
    # Alignment: merge whisper text with pyannote speaker labels
    # ------------------------------------------------------------------

    def _align(self, whisper_segments: list[dict], diar_segments: list[dict]) -> list[dict]:
        result = []
        for ws in whisper_segments:
            speaker = self._majority_speaker(ws["start"], ws["end"], diar_segments)
            result.append(
                {
                    "speaker": speaker,
                    "text": ws["text"],
                    "start": ws["start"],
                    "end": ws["end"],
                }
            )
        return result

    @staticmethod
    def _majority_speaker(start: float, end: float, diar_segments: list[dict]) -> str:
        overlap: dict[str, float] = {}
        for ds in diar_segments:
            ov_start = max(start, ds["start"])
            ov_end = min(end, ds["end"])
            ov = max(0.0, ov_end - ov_start)
            if ov > 0:
                overlap[ds["speaker"]] = overlap.get(ds["speaker"], 0.0) + ov

        if not overlap:
            return "Неизвестно"
        return max(overlap, key=overlap.get)
