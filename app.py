"""
Transcription service: upload WAV/MP3, get text with per-segment timestamps.
Reuses the same pipeline as the CLI (transcribe.py).

Run:  uvicorn app:app --reload
Test: curl -F "file=@samples/sample.mp3" http://localhost:8000/transcribe
Docs: http://localhost:8000/docs
"""
import os
import tempfile

from fastapi import FastAPI, File, HTTPException, UploadFile

from transcribe import SUPPORTED_FORMATS, PipelineError, WhisperTranscriber, run_pipeline

app = FastAPI(title="Transcription Service")
engine = WhisperTranscriber("base")  # load model once at startup


def fmt(seconds: float) -> str:
    """Format seconds as HH:MM:SS.mmm"""
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


@app.post("/transcribe")
def transcribe(file: UploadFile = File(...)):
    # Sync endpoint: FastAPI runs it in a thread pool, so slow transcription
    # doesn't block the server's event loop.
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in SUPPORTED_FORMATS:
        raise HTTPException(400, "Only WAV or MP3 files are supported.")

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        shutil_copy(file, tmp)
        path = tmp.name

    try:
        result = run_pipeline(path, engine)
    except PipelineError as e:
        raise HTTPException(422, str(e))
    finally:
        os.remove(path)

    result["source_file"] = file.filename
    for i, seg in enumerate(result["segments"]):
        seg["id"] = i
        seg["start_time"] = fmt(seg["start"])
        seg["end_time"] = fmt(seg["end"])
    return result


def shutil_copy(upload: UploadFile, dest) -> None:
    """Stream the upload to disk in chunks instead of loading it all into memory."""
    while chunk := upload.file.read(1024 * 1024):
        dest.write(chunk)


@app.get("/health")
def health():
    return {"status": "ok"}