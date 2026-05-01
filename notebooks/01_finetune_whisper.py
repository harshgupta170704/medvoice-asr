#!/usr/bin/env python
"""
notebooks/01_finetune_whisper.py
──────────────────────────────────────────────────────────────────────────
Self-contained notebook-style script for Whisper fine-tuning.
Run cell-by-cell in Jupyter (percent-format) or as a plain script.

  jupyter nbconvert --to notebook --execute notebooks/01_finetune_whisper.py
  # or simply:
  python notebooks/01_finetune_whisper.py
"""

# %% [markdown]
# # Fine-Tuning Whisper (small) for Medical ASR
#
# This notebook walks through:
# 1. Dataset loading and exploration
# 2. Feature extraction
# 3. Training loop with WER monitoring
# 4. Saving the best checkpoint

# %% [markdown]
# ## 1. Imports & Configuration

# %%
import sys
sys.path.insert(0, "..")   # allow imports from project root

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from IPython.display import Audio, display

import torch
from datasets import load_dataset, Audio as HFAudio
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
import evaluate

# ── Config ────────────────────────────────────────────────────────────
MODEL_ID       = "openai/whisper-small"
LANGUAGE       = "english"
TASK           = "transcribe"
SAMPLE_RATE    = 16_000
OUTPUT_DIR     = "../models/whisper"
MANIFEST_DIR   = "../data/processed/whisper"
NUM_EPOCHS     = 5          # reduce to 1 for a quick smoke-test
BATCH_SIZE     = 8
GRAD_ACCUM     = 2
LR             = 1e-5
WARMUP_STEPS   = 500
USE_GPU        = torch.cuda.is_available()

print(f"GPU available: {USE_GPU}")
print(f"Device: {'cuda' if USE_GPU else 'cpu'}")


# %% [markdown]
# ## 2. Load Processor

# %%
feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_ID)
tokenizer         = WhisperTokenizer.from_pretrained(MODEL_ID, language=LANGUAGE, task=TASK)
processor         = WhisperProcessor.from_pretrained(MODEL_ID, language=LANGUAGE, task=TASK)
print("Processor loaded ✓")


# %% [markdown]
# ## 3. Explore the Dataset

# %%
# Load a small Common Voice slice for exploration when local data is absent
DEMO_DATASET = True
if DEMO_DATASET:
    demo_ds = load_dataset(
        "mozilla-foundation/common_voice_11_0",
        "en",
        split="train[:50]",
        trust_remote_code=True,
    ).cast_column("audio", HFAudio(sampling_rate=SAMPLE_RATE))
    print(f"Demo dataset: {len(demo_ds)} samples")
    sample = demo_ds[0]
    print("Sample keys:", list(sample.keys()))
    print("Transcript:", sample["sentence"])

    # Listen (works in Jupyter)
    display(Audio(sample["audio"]["array"], rate=SAMPLE_RATE))


# %% [markdown]
# ## 4. Build Training Dataset from Manifests

# %%
# Import training utilities from project scripts
import importlib, sys
sys.path.insert(0, "..")
from scripts.finetune_whisper import build_hf_dataset, DataCollatorSpeechSeq2SeqWithPadding

manifest_path = Path(MANIFEST_DIR) / "train" / "manifest.json"
if manifest_path.exists():
    dataset = build_hf_dataset(MANIFEST_DIR, processor)
    print("Train size:", len(dataset.get("train", [])))
    print("Test  size:", len(dataset.get("test",  [])))
else:
    print(f"[INFO] {manifest_path} not found – run data/preprocess.py first.")
    print("       Continuing with demo dataset for illustration.")
    dataset = None


# %% [markdown]
# ## 5. Configure Data Collator

# %%
model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)
model.generation_config.language = LANGUAGE
model.generation_config.task = TASK
model.generation_config.forced_decoder_ids = None
model.config.use_cache = False

data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)
print("Model loaded, params:", sum(p.numel() for p in model.parameters()) / 1e6, "M")


# %% [markdown]
# ## 6. WER Metric

# %%
wer_metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids  = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str  = processor.tokenizer.batch_decode(pred_ids,  skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": round(wer, 4)}


# %% [markdown]
# ## 7. Training

# %%
if dataset is not None:
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        warmup_steps=WARMUP_STEPS,
        num_train_epochs=NUM_EPOCHS,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        predict_with_generate=True,
        generation_max_length=225,
        fp16=USE_GPU,
        logging_steps=25,
        report_to="none",
        push_to_hub=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("test"),
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print(f"\n✅ Model saved to {OUTPUT_DIR}")
else:
    print("[SKIP] No dataset loaded – skipping training.")


# %% [markdown]
# ## 8. Quick Inference Test

# %%
from transformers import pipeline as hf_pipeline

asr_pipe = hf_pipeline(
    "automatic-speech-recognition",
    model=OUTPUT_DIR if Path(OUTPUT_DIR).exists() else MODEL_ID,
    device=0 if USE_GPU else -1,
)

# Test with demo audio
if DEMO_DATASET:
    test_sample = demo_ds[1]
    result = asr_pipe(test_sample["audio"]["array"].astype(np.float32))
    print("Predicted:", result["text"])
    print("Reference:", test_sample["sentence"])
