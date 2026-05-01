import os
from transformers import WhisperProcessor, WhisperForConditionalGeneration, AutoTokenizer, AutoModelForTokenClassification

def predownload():
    print("Downloading Whisper-tiny (~150MB) during build to prevent Render startup timeouts...")
    WhisperProcessor.from_pretrained("openai/whisper-tiny")
    WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")

    print("Downloading BioBERT (~400MB)...")
    AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.2")
    AutoModelForTokenClassification.from_pretrained("dmis-lab/biobert-base-cased-v1.2")

    print("Pre-download complete!")

if __name__ == "__main__":
    predownload()
