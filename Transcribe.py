"""
Transcription pipeline: accepts a WAV/MP3 file, normalizes it,
transcribes it with faster-whisper, and outputs structured JSON.

Usage:
    python transcribe.py path/to/audio.mp3
    python transcribe.py path/to/audio.wav --model small --out result.json
    python transcribe.py path/to/audio.wav --mock
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

SUPPORTED_FORMATS = {".wav", ".mp3"}
MAX_FILE_MB = 100


class PipelineError(Exception):
    """Raised for any expected failure in the pipeline."""


# ---------- Step 1: Validate input ----------
def validate(path: str) -> None:
    if not os.path.isfile(path):
        raise PipelineError(f"File not found: {path}")
    ext = os.path.splitext(path)[1].lower()
    if ext not in SUPPORTED_FORMATS:
        raise PipelineError(f"Unsupported format '{ext}'. Use WAV or MP3.")
    size_mb = os.path.getsize(path) / (1024 * 1024)
    if size_mb > MAX_FILE_MB:
        raise PipelineError(f"File too large ({size_mb:.1f} MB > {MAX_FILE_MB} MB).")
    if shutil.which("ffmpeg") is None:
        raise PipelineError("ffmpeg is not installed or not on PATH.")


# ---------- Step 2: Normalize audio ----------
def normalize(path: str, workdir: str) -> str:
    """Convert to 16 kHz mono WAV, the format speech models expect."""
    out = os.path.join(workdir, "normalized.wav")
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", path, "-ac", "1", "-ar", "16000", out],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise PipelineError("Could not decode audio (file may be corrupted).")
    return out


# ---------- Step 3: Transcribe ----------
class WhisperTranscriber:
    """Wraps faster-whisper. Swap this class to use another engine or API."""

    def __init__(self, model_size: str = "base"):
        from faster_whisper import WhisperModel
        self.model = WhisperModel(model_size, device="cpu", compute_type="int8")

    def transcribe(self, wav_path: str) -> dict:
        segments, info = self.model.transcribe(wav_path, vad_filter=True)
        return {
            "language": info.language,
            "language_probability": round(info.language_probability, 3),
            "duration_sec": round(info.duration, 2),
            "segments": [
                {"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text}
                for s in segments  # generator: transcription runs here
            ],
        }


class MockTranscriber:
    """Fake engine for tests or demos without downloading a model."""

    def transcribe(self, wav_path: str) -> dict:
        return {
            "language": "en",
            "language_probability": 1.0,
            "duration_sec": 3.0,
            "segments": [
                {"start": 0.0, "end": 1.5, "text": "  Hello   world. "},
                {"start": 1.5, "end": 3.0, "text": "This is a test."},
            ],
        }


# ---------- Step 4: Post-process ----------
def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def postprocess(raw: dict, source: str) -> dict:
    segments = [
        {**s, "text": clean(s["text"])} for s in raw["segments"] if clean(s["text"])
    ]
    full_text = " ".join(s["text"] for s in segments)
    return {
        "source_file": os.path.basename(source),
        "language": raw["language"],
        "language_probability": raw["language_probability"],
        "duration_sec": raw["duration_sec"],
        "word_count": len(full_text.split()),
        "text": full_text,
        "segments": segments,
    }


# ---------- Pipeline ----------
def run_pipeline(path: str, transcriber) -> dict:
    validate(path)
    with tempfile.TemporaryDirectory() as workdir:  # auto-cleans temp files
        wav = normalize(path, workdir)
        raw = transcriber.transcribe(wav)
    return postprocess(raw, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audio transcription pipeline")
    parser.add_argument("audio", help="Path to a WAV or MP3 file")
    parser.add_argument("--model", default="base",
                        help="Whisper size: tiny, base, small, medium, large-v3")
    parser.add_argument("--mock", action="store_true",
                        help="Use the mock engine (no model download)")
    parser.add_argument("--out", help="Save JSON result to this file")
    args = parser.parse_args()

    try:
        engine = MockTranscriber() if args.mock else WhisperTranscriber(args.model)
        result = run_pipeline(args.audio, engine)
    except PipelineError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    output = json.dumps(result, indent=2, ensure_ascii=False)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(output)
        print(f"Saved to {args.out}")
    else:
        print(output)


if __name__ == "__main__":
    main()