"""Unit tests for ASR service logic (speaker assignment, alignment)."""

from __future__ import annotations

import pytest

from services.asr import ASRService, SPEAKER_CHANGE_GAP_S, SPEAKER_OPERATOR, SPEAKER_CLIENT


class TestAssignSpeakersSimple:
    def _make_service(self) -> ASRService:
        svc = ASRService.__new__(ASRService)
        svc._diarizer = None
        return svc

    def test_first_segment_is_operator(self):
        svc = self._make_service()
        segs = [{"start": 0.0, "end": 3.0, "text": "Привет"}]
        result = svc._assign_speakers_simple(segs)
        assert result[0]["speaker"] == SPEAKER_OPERATOR

    def test_speaker_names_are_in_russian(self):
        svc = self._make_service()
        gap = SPEAKER_CHANGE_GAP_S + 1.0
        segs = [
            {"start": 0.0, "end": 2.0, "text": "A"},
            {"start": 2.0 + gap, "end": 4.0, "text": "B"},
        ]
        result = svc._assign_speakers_simple(segs)
        assert result[0]["speaker"] == "Оператор"
        assert result[1]["speaker"] == "Клиент"

    def test_switch_on_long_gap(self):
        svc = self._make_service()
        gap = SPEAKER_CHANGE_GAP_S + 0.5
        segs = [
            {"start": 0.0, "end": 2.0, "text": "Оператор говорит"},
            {"start": 2.0 + gap, "end": 5.0, "text": "Клиент отвечает"},
        ]
        result = svc._assign_speakers_simple(segs)
        assert result[0]["speaker"] == SPEAKER_OPERATOR
        assert result[1]["speaker"] == SPEAKER_CLIENT

    def test_no_switch_on_short_gap(self):
        svc = self._make_service()
        segs = [
            {"start": 0.0, "end": 2.0, "text": "Первая фраза"},
            {"start": 2.2, "end": 4.0, "text": "Продолжение"},
        ]
        result = svc._assign_speakers_simple(segs)
        assert result[0]["speaker"] == result[1]["speaker"] == SPEAKER_OPERATOR

    def test_alternates_back_to_operator(self):
        svc = self._make_service()
        gap = SPEAKER_CHANGE_GAP_S + 1.0
        segs = [
            {"start": 0.0, "end": 2.0, "text": "A"},
            {"start": 2.0 + gap, "end": 4.0, "text": "B"},
            {"start": 4.0 + gap, "end": 6.0, "text": "C"},
        ]
        result = svc._assign_speakers_simple(segs)
        assert result[0]["speaker"] == SPEAKER_OPERATOR
        assert result[1]["speaker"] == SPEAKER_CLIENT
        assert result[2]["speaker"] == SPEAKER_OPERATOR

    def test_empty_input(self):
        svc = self._make_service()
        assert svc._assign_speakers_simple([]) == []


class TestMajoritySpeaker:
    def test_exact_overlap(self):
        diar = [{"start": 0.0, "end": 5.0, "speaker": SPEAKER_OPERATOR}]
        assert ASRService._majority_speaker(0.0, 5.0, diar) == SPEAKER_OPERATOR

    def test_partial_overlap_picks_majority(self):
        diar = [
            {"start": 0.0, "end": 3.0, "speaker": SPEAKER_OPERATOR},
            {"start": 3.0, "end": 5.0, "speaker": SPEAKER_CLIENT},
        ]
        assert ASRService._majority_speaker(0.0, 5.0, diar) == SPEAKER_OPERATOR

    def test_no_overlap_returns_unknown(self):
        diar = [{"start": 10.0, "end": 15.0, "speaker": SPEAKER_CLIENT}]
        result = ASRService._majority_speaker(0.0, 5.0, diar)
        assert result == "Неизвестно"

    def test_client_majority(self):
        diar = [
            {"start": 0.0, "end": 1.0, "speaker": SPEAKER_OPERATOR},
            {"start": 1.0, "end": 5.0, "speaker": SPEAKER_CLIENT},
        ]
        assert ASRService._majority_speaker(0.0, 5.0, diar) == SPEAKER_CLIENT


class TestAlign:
    def _make_service(self) -> ASRService:
        return ASRService.__new__(ASRService)

    def test_align_assigns_correct_speakers(self):
        svc = self._make_service()
        whisper = [
            {"start": 0.0, "end": 4.0, "text": "Hello"},
            {"start": 5.0, "end": 9.0, "text": "World"},
        ]
        diar = [
            {"start": 0.0, "end": 4.0, "speaker": SPEAKER_OPERATOR},
            {"start": 5.0, "end": 9.0, "speaker": SPEAKER_CLIENT},
        ]
        result = svc._align(whisper, diar)
        assert result[0]["speaker"] == SPEAKER_OPERATOR
        assert result[1]["speaker"] == SPEAKER_CLIENT

    def test_align_preserves_text_and_timestamps(self):
        svc = self._make_service()
        whisper = [{"start": 1.0, "end": 3.0, "text": "Test"}]
        diar = [{"start": 0.0, "end": 5.0, "speaker": SPEAKER_OPERATOR}]
        result = svc._align(whisper, diar)
        assert result[0]["text"] == "Test"
        assert result[0]["start"] == 1.0
        assert result[0]["end"] == 3.0
