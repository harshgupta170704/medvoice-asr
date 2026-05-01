"""
app/pipeline.py
──────────────────────────────────────────────────────────────────────────
Core inference pipeline:
  1. ASR  – Whisper (ONNX quantized or PyTorch fallback)
  2. NER  – BioBERT token classifier

Both models are loaded once at startup and reused across requests.
"""

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import librosa

from app.schemas import Entity

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Configuration (overridable via environment variables)
# ──────────────────────────────────────────────────────────────────────
WHISPER_MODEL_DIR  = os.getenv("WHISPER_MODEL_DIR",  "models/whisper")
WHISPER_ONNX_DIR   = os.getenv("WHISPER_ONNX_DIR",   "models/whisper_onnx")
BIOBERT_MODEL_DIR  = os.getenv("BIOBERT_MODEL_DIR",  "models/biobert")
USE_ONNX           = os.getenv("USE_ONNX", "true").lower() == "true"
SAMPLE_RATE        = 16_000
NER_BATCH_SIZE     = 32
MAX_NER_LENGTH     = 512


# ──────────────────────────────────────────────────────────────────────
# ASR: Whisper
# ──────────────────────────────────────────────────────────────────────

class WhisperASR:
    """Wraps Whisper inference (ONNX-first, PyTorch fallback)."""

    def __init__(self):
        self._use_onnx = False
        self._model = None
        self._processor = None
        self._load()

    def _load(self):
        # ── Try ONNX via Optimum ──────────────────────────────────────
        if USE_ONNX and Path(WHISPER_ONNX_DIR).exists():
            try:
                from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
                from transformers import WhisperProcessor

                logger.info("Loading Whisper ONNX model from %s", WHISPER_ONNX_DIR)
                self._model = ORTModelForSpeechSeq2Seq.from_pretrained(
                    WHISPER_ONNX_DIR,
                    provider="CPUExecutionProvider",
                )
                self._processor = WhisperProcessor.from_pretrained(WHISPER_ONNX_DIR)
                self._use_onnx = True
                logger.info("Whisper ONNX model loaded ✓")
                return
            except Exception as e:
                logger.warning("ONNX load failed (%s), falling back to PyTorch.", e)

        # ── PyTorch fallback ─────────────────────────────────────────
        model_dir = WHISPER_MODEL_DIR
        if not (Path(model_dir) / "preprocessor_config.json").exists():
            # Use HuggingFace hub pretrained model if no local checkpoint
            model_dir = "openai/whisper-small"
            logger.info("No local Whisper checkpoint found; using %s", model_dir)

        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        logger.info("Loading Whisper PyTorch model from %s", model_dir)
        self._processor = WhisperProcessor.from_pretrained(model_dir)
        self._model = WhisperForConditionalGeneration.from_pretrained(model_dir)
        self._model.eval()
        logger.info("Whisper PyTorch model loaded ✓")

    def transcribe(self, audio_bytes: bytes, filename: str = "audio.wav") -> Tuple[str, float]:
        """
        Transcribe raw audio bytes.
        Returns (transcript_text, duration_seconds).
        """
        # Write to tmp file so librosa can decode any format
        suffix = Path(filename).suffix or ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        try:
            waveform, _ = librosa.load(tmp_path, sr=SAMPLE_RATE, mono=True)
        finally:
            os.unlink(tmp_path)

        duration = len(waveform) / SAMPLE_RATE

        inputs = self._processor(
            waveform, sampling_rate=SAMPLE_RATE, return_tensors="pt"
        )

        if self._use_onnx:
            # Optimum ORTModel has same generate() API
            generated_ids = self._model.generate(inputs.input_features)
        else:
            import torch
            with torch.no_grad():
                generated_ids = self._model.generate(
                    inputs.input_features,
                    language="english",
                    task="transcribe",
                )

        transcript = self._processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0].strip()

        return transcript, duration

    @property
    def model_name(self) -> str:
        return f"whisper-small ({'onnx-int8' if self._use_onnx else 'pytorch'})"


# ──────────────────────────────────────────────────────────────────────
# NER: BioBERT
# ──────────────────────────────────────────────────────────────────────

class BioBERTNER:
    """Token-classification NER using fine-tuned BioBERT."""

    DEFAULT_LABEL2ID = {
        "O": 0,
        "B-Disease": 1, "I-Disease": 2,
        "B-Chemical": 3, "I-Chemical": 4,
        "B-Symptom": 5,  "I-Symptom": 6,
    }

    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._id2label: Dict[int, str] = {}
        self._load()

    def _load(self):
        model_dir = BIOBERT_MODEL_DIR
        fallback = "dmis-lab/biobert-base-cased-v1.2"

        if not (Path(model_dir) / "config.json").exists():
            logger.info("No local BioBERT checkpoint; loading pretrained %s (no NER head)", fallback)
            model_dir = fallback

        from transformers import (
            AutoTokenizer,
            AutoModelForTokenClassification,
            pipeline,
        )

        logger.info("Loading BioBERT from %s", model_dir)
        self._tokenizer = AutoTokenizer.from_pretrained(model_dir)

        # Load label map
        label_map_path = Path(BIOBERT_MODEL_DIR) / "label_map.json"
        if label_map_path.exists():
            with open(label_map_path) as f:
                lm = json.load(f)
            id2label = {int(k): v for k, v in lm.get("id2label", {}).items()}
            label2id = lm.get("label2id", self.DEFAULT_LABEL2ID)
        else:
            label2id = self.DEFAULT_LABEL2ID
            id2label = {v: k for k, v in label2id.items()}

        self._id2label = id2label

        try:
            self._model = AutoModelForTokenClassification.from_pretrained(
                model_dir,
                num_labels=len(label2id),
                id2label=id2label,
                label2id=label2id,
                ignore_mismatched_sizes=True,
            )
            self._model.eval()
        except Exception as e:
            logger.warning("Could not load NER head weights (%s). Using random init.", e)
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(
                model_dir,
                num_labels=len(label2id),
                id2label=id2label,
                label2id=label2id,
            )
            self._model = AutoModelForTokenClassification.from_config(config)
            self._model.eval()

        logger.info("BioBERT NER model loaded ✓")

    def extract_entities(self, text: str) -> List[Entity]:
        """
        Run NER on text and return a list of Entity objects with
        character-level spans.
        """
        if not text.strip():
            return []

        import torch
        from transformers import pipeline as hf_pipeline

        ner_pipeline = hf_pipeline(
            "ner",
            model=self._model,
            tokenizer=self._tokenizer,
            aggregation_strategy="simple",
            device=-1,           # CPU; change to 0 for GPU
        )

        raw_entities = ner_pipeline(text)

        entities: List[Entity] = []
        for ent in raw_entities:
            label = ent["entity_group"]
            if label == "O":
                continue
            entities.append(Entity(
                text=ent["word"],
                label=label,
                start=ent["start"],
                end=ent["end"],
                score=round(float(ent["score"]), 4),
            ))

        return entities

    @property
    def model_name(self) -> str:
        return "biobert-base-cased-v1.2-finetuned"


# ──────────────────────────────────────────────────────────────────────
# Singleton loader (called once at FastAPI startup)
# ──────────────────────────────────────────────────────────────────────

_whisper_asr: WhisperASR = None
_biobert_ner: BioBERTNER = None


def load_models() -> None:
    global _whisper_asr, _biobert_ner
    logger.info("Initialising ASR model …")
    _whisper_asr = WhisperASR()
    logger.info("Initialising NER model …")
    _biobert_ner = BioBERTNER()
    logger.info("All models ready.")


def get_asr() -> WhisperASR:
    return _whisper_asr


def get_ner() -> BioBERTNER:
    return _biobert_ner
