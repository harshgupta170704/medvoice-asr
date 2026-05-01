"""
scripts/finetune_biobert.py
──────────────────────────────────────────────────────────────────────────
Fine-tunes BioBERT for Named Entity Recognition (NER) on BC5CDR / NCBI-Disease.
Extracts: Diseases, Chemicals (drugs), and Symptoms.

Usage:
    python scripts/finetune_biobert.py \
        --data_dir   data/processed/ner/bc5cdr \
        --output_dir models/biobert \
        --epochs     5 \
        --batch_size 16
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import evaluate
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)

MODEL_ID = "dmis-lab/biobert-base-cased-v1.2"

LABEL2ID: Dict[str, int] = {
    "O": 0,
    "B-Disease": 1, "I-Disease": 2,
    "B-Chemical": 3, "I-Chemical": 4,
    "B-Symptom": 5,  "I-Symptom": 6,
}
ID2LABEL: Dict[int, str] = {v: k for k, v in LABEL2ID.items()}


# ──────────────────────────────────────────────────────────────────────
# Load JSONL datasets produced by data/preprocess.py
# ──────────────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def load_ner_datasets(data_dir: str) -> DatasetDict:
    data_dir = Path(data_dir)
    splits = {}
    for split in ("train", "validation", "test"):
        path = data_dir / f"{split}.jsonl"
        if path.exists():
            splits[split] = Dataset.from_list(load_jsonl(path))
            print(f"[OK] {split}: {len(splits[split])} sentences")
    if not splits:
        raise FileNotFoundError(
            f"No JSONL files found in {data_dir}. "
            "Run data/preprocess.py first."
        )
    return DatasetDict(splits)


# ──────────────────────────────────────────────────────────────────────
# Tokenise + align labels
# ──────────────────────────────────────────────────────────────────────

def tokenise_and_align(examples: Dict, tokenizer, label2id: Dict[str, int]) -> Dict:
    tokenized = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        max_length=512,
    )

    all_labels = []
    for i, ner_tags in enumerate(examples["ner_tags"]):
        word_ids = tokenized.word_ids(batch_index=i)
        prev_word_id = None
        label_ids = []
        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)
            elif word_id != prev_word_id:
                tag = ner_tags[word_id] if isinstance(ner_tags[word_id], str) else ID2LABEL.get(ner_tags[word_id], "O")
                label_ids.append(label2id.get(tag, 0))
            else:
                # Sub-word: use I- version of the label
                tag = ner_tags[word_id] if isinstance(ner_tags[word_id], str) else ID2LABEL.get(ner_tags[word_id], "O")
                if tag.startswith("B-"):
                    tag = "I-" + tag[2:]
                label_ids.append(label2id.get(tag, 0))
            prev_word_id = word_id
        all_labels.append(label_ids)

    tokenized["labels"] = all_labels
    return tokenized


# ──────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────

def build_compute_metrics(id2label: Dict[int, str]):
    seqeval = evaluate.load("seqeval")

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)
        true_labels = [
            [id2label[l] for l in label if l != -100]
            for label in labels
        ]
        true_preds = [
            [id2label[p] for (p, l) in zip(pred, label) if l != -100]
            for pred, label in zip(predictions, labels)
        ]
        results = seqeval.compute(predictions=true_preds, references=true_labels)
        return {
            "precision": round(results["overall_precision"], 4),
            "recall":    round(results["overall_recall"],    4),
            "f1":        round(results["overall_f1"],        4),
            "accuracy":  round(results["overall_accuracy"],  4),
        }

    return compute_metrics


# ──────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading tokenizer from {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    print("[INFO] Loading datasets …")
    raw_datasets = load_ner_datasets(args.data_dir)

    label2id = LABEL2ID
    id2label = ID2LABEL

    # Override with dataset-specific labels if available
    label_map_path = Path(args.data_dir) / "label_map.json"
    if label_map_path.exists():
        with open(label_map_path) as f:
            label2id = json.load(f)
        id2label = {int(v): k for k, v in label2id.items()}

    print("[INFO] Tokenising …")
    tokenized_datasets = raw_datasets.map(
        lambda ex: tokenise_and_align(ex, tokenizer, label2id),
        batched=True,
        remove_columns=raw_datasets[next(iter(raw_datasets))].column_names,
    )

    print(f"[INFO] Loading model: {MODEL_ID}")
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_ID,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        logging_steps=50,
        report_to="none",
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets.get("train"),
        eval_dataset=tokenized_datasets.get("validation", tokenized_datasets.get("test")),
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=build_compute_metrics(id2label),
    )

    print("[INFO] Starting training …")
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # Save label maps for inference
    with open(output_dir / "label_map.json", "w") as f:
        json.dump({"label2id": label2id, "id2label": {str(k): v for k, v in id2label.items()}}, f, indent=2)

    print(f"[OK] Best NER model saved to {output_dir}")


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune BioBERT for Medical NER")
    p.add_argument("--data_dir",   default="data/processed/ner/bc5cdr")
    p.add_argument("--output_dir", default="models/biobert")
    p.add_argument("--epochs",     type=int,   default=5)
    p.add_argument("--batch_size", type=int,   default=16)
    p.add_argument("--lr",         type=float, default=2e-5)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
