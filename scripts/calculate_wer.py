"""
Calculate WER (Word Error Rate) for all test_data/ audio files.

Requirements:
  pip install jiwer faster-whisper

Usage:
  python scripts/calculate_wer.py
  python scripts/calculate_wer.py --model large-v3
"""

from __future__ import annotations

import argparse
from pathlib import Path

TEST_DATA = Path(__file__).parent.parent / "test_data"


def load_reference(txt_path: Path) -> str:
    text = txt_path.read_text(encoding="utf-8")
    # Normalize: lowercase, strip punctuation for fair comparison
    import re
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = " ".join(text.split())
    return text


def transcribe(wav_path: Path, model_name: str) -> str:
    from faster_whisper import WhisperModel
    model = WhisperModel(model_name, device="cpu", compute_type="int8")
    # Same params as the production ASR service (api/services/asr.py) so the
    # WER numbers here match what the deployed system actually produces.
    segments, _ = model.transcribe(
        str(wav_path),
        language="ru",
        beam_size=1,
        word_timestamps=False,
        condition_on_previous_text=True,
        max_new_tokens=150,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 300},
    )
    text = " ".join(seg.text.strip() for seg in segments)

    import re
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = " ".join(text.split())
    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="medium", help="Whisper model name")
    args = parser.parse_args()

    try:
        from jiwer import wer
    except ImportError:
        print("jiwer not installed. Run: pip install jiwer")
        return

    wav_files = sorted(TEST_DATA.glob("*.wav"))
    if not wav_files:
        print(f"No WAV files found in {TEST_DATA}")
        print("Run: python scripts/generate_audio.py")
        return

    print(f"\nWER Calculation — Model: {args.model}")
    print("=" * 70)
    print(f"{'File':<30} {'WER':>8}  {'Reference words':>15}  Notes")
    print("-" * 70)

    total_wer = 0.0
    count = 0

    for wav in wav_files:
        txt = wav.with_suffix(".txt")
        if not txt.exists():
            print(f"{wav.name:<30} {'N/A':>8}  (no reference transcript)")
            continue

        reference = load_reference(txt)
        hypothesis = transcribe(wav, args.model)

        error = wer(reference, hypothesis)
        ref_words = len(reference.split())
        notes = "8kHz телефония" if "card_issue" in wav.name else ""

        print(f"{wav.name:<30} {error:>7.1%}  {ref_words:>15}  {notes}")
        total_wer += error
        count += 1

    if count:
        print("-" * 70)
        print(f"{'Average WER':<30} {total_wer / count:>7.1%}")

    print()


if __name__ == "__main__":
    main()
