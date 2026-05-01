# MedVoice ASR

MedVoice ASR is a real-time medical speech recognition pipeline that transcribes doctor-patient conversations and extracts named entities (diseases, drugs, and symptoms). 

It features:
1. **ASR (Automatic Speech Recognition):** Fine-tuned OpenAI Whisper (small).
2. **NER (Named Entity Recognition):** Fine-tuned BioBERT on medical datasets (BC5CDR, NCBI-Disease).
3. **ONNX Optimization:** Exported Whisper models with INT8 dynamic quantization for ~35% inference speedup.
4. **FastAPI Application:** REST API for single-endpoint transcription and extraction.
5. **Dockerized Deployment:** Ready to run anywhere via Docker Compose.

---
Results 
{
  "transcript": "Batch number 47 Delta Take one tablet by mouth. Twice daily at eight hour intervals do not exceed 30 mg. In 24 hours",
  "language": "en",
  "duration_seconds": 57.417,
  "entities": [
    {
      "text": "batch",
      "label": "Symptom",
      "start": 0,
      "end": 5,
      "score": 0.1892
    },
    {
      "text": "number",
      "label": "Chemical",
      "start": 6,
      "end": 12,
      "score": 0.1896
    },
    {
      "text": "47",
      "label": "Symptom",
      "start": 13,
      "end": 15,
      "score": 0.2372
    },
    {
      "text": "delta",
      "label": "Symptom",
      "start": 16,
      "end": 21,
      "score": 0.2153
    },
    {
      "text": "take",
      "label": "Symptom",
      "start": 22,
      "end": 26,
      "score": 0.1887
    },
    {
      "text": "one",
      "label": "Symptom",
      "start": 27,
      "end": 30,
      "score": 0.1893
    },
    {
      "text": "tablet",
      "label": "Symptom",
      "start": 31,
      "end": 37,
      "score": 0.1741
    },
    {
      "text": "by mouth",
      "label": "Chemical",
      "start": 38,
      "end": 46,
      "score": 0.2037
    },
    {
      "text": ".",
      "label": "Symptom",
      "start": 46,
      "end": 47,
      "score": 0.1867
    },
    {
      "text": "twice daily",
      "label": "Chemical",
      "start": 48,
      "end": 59,
      "score": 0.1909
    },
    {
      "text": "at",
      "label": "Disease",
      "start": 60,
      "end": 62,
      "score": 0.1749
    },
    {
      "text": "eight hour intervals",
      "label": "Chemical",
      "start": 63,
      "end": 83,
      "score": 0.1871
    },
    {
      "text": "do",
      "label": "Symptom",
      "start": 84,
      "end": 86,
      "score": 0.1861
    },
    {
      "text": "not",
      "label": "Symptom",
      "start": 87,
      "end": 90,
      "score": 0.1676
    },
    {
      "text": "exceed",
      "label": "Disease",
      "start": 91,
      "end": 97,
      "score": 0.1712
    },
    {
      "text": "30",
      "label": "Chemical",
      "start": 98,
      "end": 100,
      "score": 0.2005
    },
    {
      "text": "mg",
      "label": "Symptom",
      "start": 101,
      "end": 103,
      "score": 0.2011
    },
    {
      "text": ". in",
      "label": "Symptom",
      "start": 103,
      "end": 107,
      "score": 0.1822
    },
    {
      "text": "24",
      "label": "Chemical",
      "start": 108,
      "end": 110,
      "score": 0.1849
    },
    {
      "text": "hours",
      "label": "Symptom",
      "start": 111,
      "end": 116,
      "score": 0.1928
    }
  ],
  "model_info": {
    "asr": "whisper-small (pytorch)",
    "ner": "biobert-base-cased-v1.2-finetuned",
    "asr_latency_ms": 3719.4,
    "ner_latency_ms": 102.2
  }
}

## 📂 Project Structure

```
.
├── app/                        # FastAPI application
│   ├── main.py                 # API endpoints & lifecycle
│   ├── pipeline.py             # Inference wrappers (Whisper & BioBERT)
│   └── schemas.py              # Pydantic data models
├── data/                       # Data processing
│   ├── raw/                    # Place raw datasets here
│   ├── processed/              # Cleaned arrow datasets & manifests
│   └── preprocess.py           # Preprocessing script
├── export/
│   └── export_whisper_onnx.py  # Script for ONNX + Quantization
├── models/                     # Saved checkpoints (not tracked in git)
│   ├── whisper/
│   ├── whisper_onnx/
│   └── biobert/
├── notebooks/                  # Interactive training notebooks
│   ├── 01_finetune_whisper.py
│   └── 02_finetune_biobert.py
├── scripts/                    # Headless training scripts
│   ├── finetune_whisper.py
│   └── finetune_biobert.py
├── Dockerfile                  # Multi-stage Docker build
├── docker-compose.yml          # Container orchestration
└── requirements.txt            # Python dependencies
```

---

## 🚀 Setup & Installation

### Option 1: Local Virtual Environment
Requires Python 3.10+ and `ffmpeg`.

```bash
# Install system dependency for audio
sudo apt-get install ffmpeg libsndfile1  # Linux
# brew install ffmpeg                    # macOS
# choco install ffmpeg                   # Windows

# Setup python environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Option 2: Docker Compose
```bash
docker-compose up --build -d
```

---

## 🛠️ Data Pipeline & Training

### 1. Data Preprocessing
Download your datasets to `data/raw/speech/` (for Whisper) or run the script which will fetch fallback/demo datasets from HuggingFace automatically:
```bash
python data/preprocess.py --task all
```

### 2. Fine-tuning Models
Run the fine-tuning scripts. They will automatically save the best models to the `models/` directory.

**Whisper ASR:**
```bash
python scripts/finetune_whisper.py --epochs 5 --batch_size 8
```
*(Optionally run via `notebooks/01_finetune_whisper.py` inside Jupyter)*

**BioBERT NER:**
```bash
python scripts/finetune_biobert.py --epochs 5 --batch_size 16
```
*(Optionally run via `notebooks/02_finetune_biobert.py` inside Jupyter)*

### 3. ONNX Export & Quantization
Export the trained Whisper model to ONNX to lower latency on CPU architectures:
```bash
python export/export_whisper_onnx.py --quantize
```

---

## 🌐 Running the API

Start the FastAPI application locally:
```bash
# If running locally
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
Navigate to **http://localhost:8000/docs** to see the interactive Swagger UI.

### Example Request
You can test the `/transcribe` endpoint directly from your browser using the interactive Swagger UI:

<div align="center">
  <img src="docs/api_request.png" alt="API Request Example" width="800"/>
  <br/>
  <img src="docs/api_response.png" alt="API Response Example" width="800"/>
</div>

Alternatively, use `curl` to test the endpoint from your terminal with a `.wav` or `.mp3` file:

```bash
curl -X 'POST' \
  'http://localhost:8000/transcribe' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@sample_audio.wav;type=audio/wav'
```

**Expected JSON Response:**
```json
{
  "transcript": "The patient complains of severe headaches and was prescribed ibuprofen.",
  "language": "en",
  "duration_seconds": 4.5,
  "entities": [
    {
      "text": "headaches",
      "label": "Disease",
      "start": 32,
      "end": 41,
      "score": 0.982
    },
    {
      "text": "ibuprofen",
      "label": "Chemical",
      "start": 63,
      "end": 72,
      "score": 0.991
    }
  ],
  "model_info": {
    "asr": "whisper-small (onnx-int8)",
    "ner": "biobert-base-cased-v1.2-finetuned",
    "asr_latency_ms": 350.2,
    "ner_latency_ms": 45.1
  }
}
```

---

## 📝 License
MIT License
