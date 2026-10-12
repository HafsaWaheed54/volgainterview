# Transcription Pipeline

Converts a WAV/MP3 file into clean, timestamped text, available as a CLI script and a REST API.

## Setup
1. Install Python 3.10+ and ffmpeg (`brew install ffmpeg` / `sudo apt install ffmpeg`)
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Run

**CLI**
```
python transcribe.py samples/sample.mp3
python transcribe.py samples/sample.mp3 --model small --out result.json
python transcribe.py samples/sample.mp3 --mock     # no model download, for testing
```

**API**
```
uvicorn app:app --reload
curl -F "file=@samples/sample.mp3" http://localhost:8000/transcribe
```
Interactive docs: http://localhost:8000/docs

The first real run downloads the Whisper model (~150 MB for `base`).

## Example output
```json
{
  "source_file": "sample.mp3",
  "language": "en",
  "language_probability": 0.98,
  "duration_sec": 6.8,
  "word_count": 13,
  "text": "Hello, thanks for joining the call today. Let's go over the project timeline.",
  "segments": [
    {"id": 0, "start": 0.0, "end": 3.2, "start_time": "00:00:00.000",
     "end_time": "00:00:03.200", "text": "Hello, thanks for joining the call today."},
    {"id": 1, "start": 3.2, "end": 6.8, "start_time": "00:00:03.200",
     "end_time": "00:00:06.800", "text": "Let's go over the project timeline."}
  ]
}
```
(`id`, `start_time` and `end_time` are added by the API.)

## How it works
```
Validate → Normalize (ffmpeg) → Transcribe (faster-whisper) → Post-process → JSON
```
1. **Validate**: file exists, is WAV/MP3, is under 100 MB, and ffmpeg is available.
2. **Normalize**: convert to 16 kHz mono WAV in a temp directory (auto-deleted).
3. **Transcribe**: faster-whisper with voice activity detection; language auto-detected.
4. **Post-process**: clean whitespace, drop empty segments, add metadata (language, duration, word count).

## Design decisions
- **faster-whisper**: open source, accurate, runs locally on CPU, and returns segment timestamps.
  No per-minute API cost, and audio never leaves the machine. Trade-off: needs local compute.
- **Normalization first**: every input becomes the same format, so the engine behaves the
  same regardless of the source file. Supporting a new format is mostly a config change.
- **Swappable engine**: any class with a `transcribe(wav_path)` method works, such as a cloud
  API or the included `MockTranscriber` used for testing without a model.
- **VAD filter**: skips silence, which is faster and reduces hallucinated text.
- **Model loaded once**: the API loads the model at startup, not per request.
- **One pipeline, two entry points**: the CLI and the API share `run_pipeline()`,
  so behavior is identical.
- **Clear errors**: missing file, unsupported format, oversize file, missing ffmpeg,
  or corrupted audio each return a clear message (HTTP 400/422 in the API).

## Handling long audio
For long files, transcription should run in the background: the API returns a job ID
immediately and a worker processes the file. The audio is split into chunks at silences
(using VAD), chunks are transcribed in parallel, and timestamps are shifted by each chunk's
offset when merging. A failed chunk is retried on its own.

## System design (Part 2)
- **Concurrent uploads**: the API only saves the file and enqueues a job (Redis + Celery or SQS),
  returning a job ID. Workers process jobs; scale by adding workers. Rate limits and size limits
  protect the system.
- **Storage**: audio in object storage (S3); job metadata and transcripts in PostgreSQL
  (segments in JSONB). Raw audio is deleted after a retention period; data encrypted at rest.
- **Retries and recovery**: job states (queued → processing → completed/failed). Transient
  failures retry with exponential backoff (max 3), then go to a dead-letter queue.
  Corrupted or unsupported files fail fast without retry. Original audio stays in storage,
  so any job can be reprocessed.
- **API**: async REST, versioned under `/v1`:
  - `POST /v1/transcriptions`: upload, returns job ID
  - `GET /v1/transcriptions/{id}`: status and result (optional `?format=srt`)
  - `DELETE /v1/transcriptions/{id}`: remove audio and transcript
  - Optional webhook on completion, API key auth, rate limiting, OpenAPI docs.

## Future improvements
- Background job queue and chunking (described above)
- Speaker diarization (pyannote)
- GPU support
- Unit tests using `MockTranscriber`