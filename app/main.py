"""
app/main.py
──────────────────────────────────────────────────────────────────────────
MedVoice ASR  –  FastAPI Application

Endpoints:
  GET  /          → Swagger redirect / health check
  GET  /health    → Model status
  POST /transcribe → Audio → transcript + medical NER entities
"""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.pipeline import load_models, get_asr, get_ner
from app.schemas import TranscribeResponse, HealthResponse

# ──────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("medvoice")

# ──────────────────────────────────────────────────────────────────────
# Application lifecycle
# ──────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup; clean up on shutdown."""
    logger.info("=== MedVoice ASR starting up ===")
    load_models()
    logger.info("=== Models ready – serving requests ===")
    yield
    logger.info("=== MedVoice ASR shutting down ===")


# ──────────────────────────────────────────────────────────────────────
# FastAPI app
# ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="MedVoice ASR API",
    description=(
        "Real-time medical speech recognition pipeline.\n\n"
        "- **POST /transcribe** – Upload an audio file; receive a transcript "
        "and named entities (diseases, drugs, symptoms).\n"
        "- **GET /health** – Service and model status."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────────────────────────────
# Allowed audio MIME types
# ──────────────────────────────────────────────────────────────────────
ALLOWED_CONTENT_TYPES = {
    "audio/wav", "audio/wave", "audio/x-wav",
    "audio/mpeg", "audio/mp3",
    "audio/flac", "audio/x-flac",
    "audio/ogg",
    "audio/mp4", "audio/m4a",
    "application/octet-stream",   # generic fallback
}
MAX_FILE_SIZE_MB = 25


# ──────────────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root():
    return JSONResponse({"message": "MedVoice ASR API – visit /docs for the interactive UI"})


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """Return the service health and loaded model names."""
    asr = get_asr()
    ner = get_ner()
    return HealthResponse(
        status="ok",
        asr_model=asr.model_name if asr else "not loaded",
        ner_model=ner.model_name if ner else "not loaded",
        version="1.0.0",
    )


@app.post(
    "/transcribe",
    response_model=TranscribeResponse,
    status_code=status.HTTP_200_OK,
    tags=["ASR"],
    summary="Transcribe audio and extract medical entities",
    response_description="Transcript text and list of medical named entities",
)
async def transcribe(
    file: UploadFile = File(
        ...,
        description="Audio file (WAV, MP3, FLAC, OGG, M4A). Max 25 MB.",
    ),
):
    """
    **Upload** a doctor-patient audio recording.

    Returns:
    - `transcript` – Full text transcription
    - `entities`   – List of `{text, label, start, end, score}` objects
      where `label` ∈ {Disease, Chemical, Symptom}
    - `duration_seconds` – Audio length
    - `model_info` – Which model variants were used
    """
    # ── Validate file ────────────────────────────────────────────────
    content_type = (file.content_type or "").lower().split(";")[0].strip()
    if content_type and content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported media type '{content_type}'. "
                   f"Accepted: WAV, MP3, FLAC, OGG, M4A.",
        )

    audio_bytes = await file.read()
    size_mb = len(audio_bytes) / 1e6
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large ({size_mb:.1f} MB). Max is {MAX_FILE_SIZE_MB} MB.",
        )
    if not audio_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty file received.",
        )

    asr = get_asr()
    ner = get_ner()

    if asr is None or ner is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Models are still loading. Please retry in a moment.",
        )

    # ── ASR ─────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    try:
        transcript, duration = asr.transcribe(audio_bytes, filename=file.filename or "audio.wav")
    except Exception as exc:
        logger.exception("ASR failed")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Audio transcription failed: {exc}",
        )
    asr_ms = (time.perf_counter() - t0) * 1000

    # ── NER ─────────────────────────────────────────────────────────
    t1 = time.perf_counter()
    try:
        entities = ner.extract_entities(transcript)
    except Exception as exc:
        logger.exception("NER failed")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Named entity extraction failed: {exc}",
        )
    ner_ms = (time.perf_counter() - t1) * 1000

    logger.info(
        "Transcribed %.1fs audio in %.0fms ASR + %.0fms NER | %d entities",
        duration, asr_ms, ner_ms, len(entities),
    )

    return TranscribeResponse(
        transcript=transcript,
        language="en",
        duration_seconds=round(duration, 3),
        entities=entities,
        model_info={
            "asr": asr.model_name,
            "ner": ner.model_name,
            "asr_latency_ms": round(asr_ms, 1),
            "ner_latency_ms": round(ner_ms, 1),
        },
    )
