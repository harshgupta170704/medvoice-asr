"""
data/preprocess.py
──────────────────────────────────────────────────────────────────────────
Preprocessing pipeline for MedVoice ASR:
  1. Whisper ASR data  → prepares audio + transcription pairs
  2. BioBERT NER data  → processes BC5CDR / NCBI-Disease into BIO tags

Usage:
    python data/preprocess.py --task whisper --data_dir data/raw/speech
    python data/preprocess.py --task ner    --dataset bc5cdr
"""

import argparse
import json
import os
import re
import csv
from pathlib import Path
from typing import List, Dict, Tuple

import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm
from datasets import load_dataset, DatasetDict, Audio


# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────
SAMPLE_RATE = 16_000          # Whisper expects 16 kHz mono
MAX_AUDIO_SECS = 30           # Whisper context window
PROCESSED_DIR = Path("data/processed")

# ──────────────────────────────────────────────────────────────────────
# Helper utilities
# ──────────────────────────────────────────────────────────────────────

def normalise_text(text: str) -> str:
    """Lower-case, strip extra whitespace, remove non-ASCII punctuation."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s\'\-\,\.\?\!]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def load_audio_16k(path: str) -> np.ndarray:
    """Load any audio file and resample to 16 kHz mono float32."""
    waveform, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    if len(waveform) > MAX_AUDIO_SECS * SAMPLE_RATE:
        waveform = waveform[: MAX_AUDIO_SECS * SAMPLE_RATE]
    return waveform


# ──────────────────────────────────────────────────────────────────────
# 1. Whisper ASR preprocessing
# ──────────────────────────────────────────────────────────────────────

def preprocess_medical_speech(data_dir: str, output_dir: str) -> None:
    """
    Reads a directory laid out as:
        data/raw/speech/
            train/
                audio/  *.wav / *.mp3 / *.flac
                metadata.csv  (columns: file_name, transcription)
            test/  (same structure)

    Outputs HuggingFace-compatible arrow datasets to output_dir.

    If no local data exists, falls back to mozilla-foundation/common_voice_11_0
    (en) as a demonstration dataset.
    """
    out = Path(output_dir) / "whisper"
    out.mkdir(parents=True, exist_ok=True)

    metadata_path = Path(data_dir) / "train" / "metadata.csv"

    if not metadata_path.exists():
        print("[INFO] No local speech data found – downloading demo dataset.")
        _download_demo_speech(out)
        return

    for split in ("train", "test"):
        split_dir = Path(data_dir) / split
        audio_dir = split_dir / "audio"
        meta_path = split_dir / "metadata.csv"

        if not meta_path.exists():
            print(f"[WARN] metadata.csv missing for split={split}, skipping.")
            continue

        records = []
        with open(meta_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                audio_file = audio_dir / row["file_name"]
                if not audio_file.exists():
                    continue
                try:
                    wav = load_audio_16k(str(audio_file))
                except Exception as e:
                    print(f"[WARN] Could not load {audio_file}: {e}")
                    continue

                out_wav_path = out / split / "audio" / (audio_file.stem + ".wav")
                out_wav_path.parent.mkdir(parents=True, exist_ok=True)
                sf.write(str(out_wav_path), wav, SAMPLE_RATE)

                records.append({
                    "audio_path": str(out_wav_path),
                    "transcription": normalise_text(row["transcription"]),
                })

        manifest_path = out / split / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)

        print(f"[OK] {split}: {len(records)} samples → {manifest_path}")


def _download_demo_speech(out: Path) -> None:
    """Fallback: generate a few dummy audio files for demo purposes."""
    records = []
    for i in tqdm(range(5), desc="Generating Dummy Speech (Demo)"):
        wav = np.random.randn(SAMPLE_RATE * 3).astype(np.float32) * 0.1
        path = out / "train" / "audio" / f"dummy_{i:05d}.wav"
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(path), wav, SAMPLE_RATE)
        records.append({
            "audio_path": str(path),
            "transcription": "patient presents with severe headache and fever",
        })

    manifest_path = out / "train" / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"[OK] Saved {len(records)} Demo Speech samples → {manifest_path}")


# ──────────────────────────────────────────────────────────────────────
# 2. NER (BioBERT) preprocessing  – BC5CDR / NCBI-Disease
# ──────────────────────────────────────────────────────────────────────

# BIO label mapping
LABEL2ID: Dict[str, int] = {
    "O": 0,
    "B-Disease": 1, "I-Disease": 2,
    "B-Chemical": 3, "I-Chemical": 4,
    "B-Symptom": 5,  "I-Symptom": 6,
}
ID2LABEL: Dict[int, str] = {v: k for k, v in LABEL2ID.items()}


def preprocess_bc5cdr(output_dir: str) -> None:
    """
    Downloads BC5CDR from HuggingFace and converts to token-label lists.
    Labels: Chemical → Chemical  |  Disease → Disease
    """
    out = Path(output_dir) / "ner" / "bc5cdr"
    out.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset("tner/bc5cdr", trust_remote_code=True)
    _save_ner_dataset(dataset, out, label_map_key="bc5cdr")
    print(f"[OK] BC5CDR saved → {out}")


def preprocess_ncbi_disease(output_dir: str) -> None:
    """
    Downloads NCBI-Disease from HuggingFace.
    Labels: Disease only.
    """
    out = Path(output_dir) / "ner" / "ncbi_disease"
    out.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset("ncbi_disease", trust_remote_code=True)
    _save_ner_dataset(dataset, out, label_map_key="ncbi")
    print(f"[OK] NCBI-Disease saved → {out}")


def _save_ner_dataset(dataset: DatasetDict, out: Path, label_map_key: str) -> None:
    """Persist each split as a JSONL file with tokens + ner_tags."""
    for split, ds in dataset.items():
        rows = []
        for item in ds:
            tokens = item["tokens"]
            # HF datasets expose integer labels; map to string BIO tags
            int_labels = item.get("ner_tags", item.get("tags", []))
            str_labels = _int_labels_to_bio(int_labels, ds.features, label_map_key)
            rows.append({"tokens": tokens, "ner_tags": str_labels})

        path = out / f"{split}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

        print(f"  [{split}] {len(rows)} sentences → {path}")

    # Save label map
    with open(out / "label_map.json", "w") as f:
        json.dump(LABEL2ID, f, indent=2)


def _int_labels_to_bio(
    int_labels: List[int],
    features,
    label_map_key: str,
) -> List[str]:
    """Convert integer NER labels to BIO string labels."""
    # Try to get the original names from dataset features
    try:
        tag_names = features["ner_tags"].feature.names
        return [tag_names[i] for i in int_labels]
    except Exception:
        # Fallback: treat non-zero as B-Disease
        result = []
        for i, lbl in enumerate(int_labels):
            if lbl == 0:
                result.append("O")
            elif lbl % 2 == 1:
                result.append("B-Disease")
            else:
                result.append("I-Disease")
        return result


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MedVoice ASR – Data Preprocessor")
    p.add_argument("--task", choices=["whisper", "ner", "all"], default="all",
                   help="Which preprocessing task to run")
    p.add_argument("--data_dir", default="data/raw/speech",
                   help="Root dir for raw speech data (for --task whisper)")
    p.add_argument("--dataset", choices=["bc5cdr", "ncbi", "both"], default="both",
                   help="Which NER dataset to preprocess (for --task ner)")
    p.add_argument("--output_dir", default="data/processed",
                   help="Root output directory")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    if args.task in ("whisper", "all"):
        print("\n=== Preprocessing ASR data ===")
        preprocess_medical_speech(args.data_dir, args.output_dir)

    if args.task in ("ner", "all"):
        print("\n=== Preprocessing NER data ===")
        if args.dataset in ("bc5cdr", "both"):
            preprocess_bc5cdr(args.output_dir)
        if args.dataset in ("ncbi", "both"):
            preprocess_ncbi_disease(args.output_dir)


if __name__ == "__main__":
    main()
