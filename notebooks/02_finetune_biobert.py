#!/usr/bin/env python
"""
notebooks/02_finetune_biobert.py
──────────────────────────────────────────────────────────────────────────
Self-contained notebook-style script for BioBERT NER fine-tuning.
Run cell-by-cell in Jupyter (percent-format) or as a plain script.

  jupyter nbconvert --to notebook --execute notebooks/02_finetune_biobert.py
  # or simply:
  python notebooks/02_finetune_biobert.py
"""

# %% [markdown]
# # Fine-Tuning BioBERT for Medical NER
#
# Extracts: Diseases, Chemicals (drugs), and Symptoms.
# Uses BC5CDR / NCBI-Disease datasets.

# %% [markdown]
# ## 1. Imports & Configuration

# %%
import sys
sys.path.insert(0, "..")

import json
import numpy as np
from pathlib import Path

import torch
import evaluate
from datasets import Dataset, load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    pipeline,
)

MODEL_ID    = "dmis-lab/biobert-base-cased-v1.2"
DATA_DIR    = "../data/processed/ner/bc5cdr"
OUTPUT_DIR  = "../models/biobert"

EPOCHS      = 3
BATCH_SIZE  = 16
LR          = 2e-5
USE_GPU     = torch.cuda.is_available()

print(f"GPU available: {USE_GPU}")


# %% [markdown]
# ## 2. Load Datasets
# If local preprocessed JSONL files don't exist, we fall back to HF datasets.

# %%
dataset_path = Path(DATA_DIR)

if dataset_path.exists():
    from scripts.finetune_biobert import load_ner_datasets
    raw_datasets = load_ner_datasets(str(dataset_path))
    print(f"Loaded local data from {DATA_DIR}")
else:
    print(f"[WARN] Local data not found at {DATA_DIR}. Loading NCBI Disease directly from HF...")
    raw_datasets = load_dataset("ncbi_disease", trust_remote_code=True)
    # The dataset feature for NER tags uses the following names:
    # 0:'O', 1:'B-Disease', 2:'I-Disease'
    
print("Train size:", len(raw_datasets["train"]))


# %% [markdown]
# ## 3. Define Label Maps

# %%
# Standard BIO tagging
LABEL2ID = {
    "O": 0,
    "B-Disease": 1, "I-Disease": 2,
    "B-Chemical": 3, "I-Chemical": 4,
    "B-Symptom": 5,  "I-Symptom": 6,
}

if not dataset_path.exists():
    # If using default NCBI fallback, only disease is labeled
    LABEL2ID = {"O": 0, "B-Disease": 1, "I-Disease": 2}

ID2LABEL = {v: k for k, v in LABEL2ID.items()}
print("Labels:", ID2LABEL)


# %% [markdown]
# ## 4. Tokenization & Alignment

# %%
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

def tokenize_and_align(examples):
    tokenized = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True, max_length=512
    )
    all_labels = []
    
    for i, tags in enumerate(examples.get("ner_tags", examples.get("tags", []))):
        word_ids = tokenized.word_ids(batch_index=i)
        prev_word_id = None
        label_ids = []
        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)
            elif word_id != prev_word_id:
                # If int, keep it (assuming mapped), if string, convert
                tag = tags[word_id]
                if isinstance(tag, str):
                    label_ids.append(LABEL2ID.get(tag, 0))
                else:
                    # Int label from NCBI fallback
                    label_ids.append(tag)
            else:
                tag = tags[word_id]
                if isinstance(tag, str):
                    # subword I- tag mapping logic
                    t = tag
                    if t.startswith("B-"): t = "I-" + t[2:]
                    label_ids.append(LABEL2ID.get(t, 0))
                else:
                    # Int label, convert B(1) to I(2)
                    t = tag
                    if t == 1: t = 2
                    label_ids.append(t)
            prev_word_id = word_id
        all_labels.append(label_ids)

    tokenized["labels"] = all_labels
    return tokenized

tokenized_datasets = raw_datasets.map(
    tokenize_and_align, 
    batched=True,
    remove_columns=raw_datasets["train"].column_names
)
print("Tokenization complete.")


# %% [markdown]
# ## 5. Metrics & Model

# %%
model = AutoModelForTokenClassification.from_pretrained(
    MODEL_ID,
    num_labels=len(LABEL2ID),
    id2label=ID2LABEL,
    label2id=LABEL2ID,
    ignore_mismatched_sizes=True,
)

seqeval = evaluate.load("seqeval")
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    
    true_labels = [[ID2LABEL[l] for l in label if l != -100] for label in labels]
    true_preds = [
        [ID2LABEL[p] for (p, l) in zip(pred, label) if l != -100]
        for pred, label in zip(predictions, labels)
    ]
    
    results = seqeval.compute(predictions=true_preds, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)


# %% [markdown]
# ## 6. Training

# %%
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy="epoch",
    learning_rate=LR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    fp16=USE_GPU,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets.get("validation", tokenized_datasets.get("test")),
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

print("Starting training...")
trainer.train()

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# Save map
with open(Path(OUTPUT_DIR) / "label_map.json", "w") as f:
    json.dump({"label2id": LABEL2ID, "id2label": {str(k): v for k, v in ID2LABEL.items()}}, f, indent=2)

print(f"✅ NER Model saved to {OUTPUT_DIR}")


# %% [markdown]
# ## 7. Quick Inference Test

# %%
ner_pipe = pipeline("ner", model=OUTPUT_DIR, tokenizer=OUTPUT_DIR, aggregation_strategy="simple", device=0 if USE_GPU else -1)

text = "The patient presents with severe headache and fever, and was prescribed Ibuprofen and Amoxicillin."
print(f"\nText: {text}\nEntities:")
for ent in ner_pipe(text):
    print(f" - {ent['entity_group']}: {ent['word']} (Confidence: {ent['score']:.2f})")
