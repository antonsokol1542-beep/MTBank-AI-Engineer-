"""
Calculate WER by calling the running FastAPI transcription endpoint.
Usage: python scripts/run_wer_api.py [--api http://localhost:8000]
"""
from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

TEST_DATA = Path(__file__).parent.parent / "test_data"


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return " ".join(text.split())


def transcribe_via_api(wav_path: Path, api_base: str) -> tuple[str, float]:
    import requests

    with open(wav_path, "rb") as f:
        resp = requests.post(
            f"{api_base}/api/v1/transcribe",
            files={"file": (wav_path.name, f, "audio/wav")},
            timeout=600,
        )
    resp.raise_for_status()
    data = resp.json()
    segs = data.get("transcript", [])
    text = normalize(" ".join(s["text"] for s in segs))
    duration = data.get("audio_duration", 0.0)
    return text, duration


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default="http://localhost:8000")
    args = parser.parse_args()

    try:
        from jiwer import wer
    except ImportError:
        print("jiwer not installed: pip install jiwer")
        return

    wav_files = sorted(TEST_DATA.glob("*.wav"))
    if not wav_files:
        print(f"No WAV files in {TEST_DATA}")
        return

    print(f"\nWER via API ({args.api})")
    print("=" * 80)
    fmt = "{:<28} {:>7}  {:>8}  {:>9}  {}"
    print(fmt.format("File", "WER", "Duration", "Ref words", "Notes"))
    print("-" * 80)

    results = []
    for wav in wav_files:
        txt = wav.with_suffix(".txt")
        if not txt.exists():
            continue
        reference = normalize(txt.read_text(encoding="utf-8"))
        notes = "8kHz телефония" if "card_issue" in wav.name else "16kHz TTS"

        t0 = time.time()
        try:
            hypothesis, duration = transcribe_via_api(wav, args.api)
        except Exception as exc:
            print(fmt.format(wav.name, "ERROR", "-", "-", str(exc)[:40]))
            continue

        elapsed = time.time() - t0
        error = wer(reference, hypothesis)
        ref_words = len(reference.split())
        results.append((wav.name, error, duration, ref_words, notes))
        print(fmt.format(wav.name, f"{error:.1%}", f"{duration:.1f}s", str(ref_words), notes))
        print(f"  processed in {elapsed:.0f}s")
        print(f"  HYP: {hypothesis[:100]}")
        print(f"  REF: {reference[:100]}")

    if results:
        avg_wer = sum(r[1] for r in results) / len(results)
        total_dur = sum(r[2] for r in results)
        print("-" * 80)
        print(fmt.format("Average / Total", f"{avg_wer:.1%}", f"{total_dur:.0f}s", "", ""))
        print()
        print("Markdown table:")
        print("| Файл | Длительность | WER (medium) | Sample Rate | Примечание |")
        print("|---|---|---|---|---|")
        for name, err, dur, rw, notes in results:
            sr = "8kHz" if "card_issue" in name else "16kHz"
            print(f"| `{name}` | {dur:.0f} с | **{err:.1%}** | {sr} | {notes} |")
        print(f"| **Среднее** | **{total_dur:.0f} с** | **{avg_wer:.1%}** | | |")


if __name__ == "__main__":
    main()
