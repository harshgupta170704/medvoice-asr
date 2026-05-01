"""
scripts/finetune_whisper.py
──────────────────────────────────────────────────────────────────────────
Fine-tunes OpenAI Whisper (small) on a medical speech dataset.

Usage (single GPU):
    python scripts/finetune_whisper.py \
        --manifest_dir data/processed/whisper \
        --output_dir   models/whisper \
        --epochs        5 \
        --batch_size    8

Usage (multi-GPU via accelerate):
    accelerate launch scripts/finetune_whisper.py ...
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import torch
import evaluate
from datasets import Dataset, DatasetDict
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

MODEL_ID = "openai/whisper-small"
SAMPLE_RATE = 16_000
LANGUAGE = "english"
TASK = "transcribe"


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def load_manifest(path: Path) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_hf_dataset(manifest_dir: str, processor: WhisperProcessor) -> DatasetDict:
    manifest_dir = Path(manifest_dir)
    splits = {}

    for split in ("train", "test"):
        manifest_path = manifest_dir / split / "manifest.json"
        if not manifest_path.exists():
            print(f"[WARN] {manifest_path} not found, skipping split {split}")
            continue

        records = load_manifest(manifest_path)

        def process_record(record):
            import librosa
            waveform, _ = librosa.load(record["audio_path"], sr=SAMPLE_RATE, mono=True)
            input_features = processor(waveform, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_features[0]
            labels = processor.tokenizer(record["transcription"], return_tensors="pt").input_ids[0]
            return {"input_features": input_features.numpy(), "labels": labels.numpy().tolist()}

        processed = [process_record(r) for r in records]
        splits[split] = Dataset.from_list(processed)
        print(f"[OK] {split}: {len(processed)} samples")

    return DatasetDict(splits)


def build_compute_metrics(processor: WhisperProcessor):
    wer_metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": round(wer, 4)}

    return compute_metrics


def train(args: argparse.Namespace) -> None:
    print(f"[INFO] Loading model: {MODEL_ID}")
    processor = WhisperProcessor.from_pretrained(MODEL_ID, language=LANGUAGE, task=TASK)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)
    model.generation_config.language = LANGUAGE
    model.generation_config.task = TASK
    model.generation_config.forced_decoder_ids = None
    model.config.use_cache = False

    print("[INFO] Building datasets …")
    dataset = build_hf_dataset(args.manifest_dir, processor)

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.epochs,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        predict_with_generate=True,
        generation_max_length=225,
        fp16=torch.cuda.is_available(),
        logging_steps=25,
        report_to="none",
        push_to_hub=False,
        dataloader_num_workers=2,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset.get("train"),
        eval_dataset=dataset.get("test"),
        data_collator=data_collator,
        compute_metrics=build_compute_metrics(processor),
        tokenizer=processor.feature_extractor,
    )

    print("[INFO] Starting training …")
    trainer.train()
    trainer.save_model(str(output_dir))
    processor.save_pretrained(str(output_dir))
    print(f"[OK] Best model saved to {output_dir}")


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune Whisper for Medical ASR")
    p.add_argument("--manifest_dir", default="data/processed/whisper")
    p.add_argument("--output_dir",   default="models/whisper")
    p.add_argument("--epochs",       type=int,   default=5)
    p.add_argument("--batch_size",   type=int,   default=8)
    p.add_argument("--grad_accum",   type=int,   default=2)
    p.add_argument("--lr",           type=float, default=1e-5)
    p.add_argument("--warmup_steps", type=int,   default=500)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
