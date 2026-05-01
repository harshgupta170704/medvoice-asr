# MedVoice ASR

MedVoice ASR is a real-time medical speech recognition pipeline that transcribes doctor-patient conversations and extracts named entities (diseases, drugs, and symptoms). 

It features:
1. **ASR (Automatic Speech Recognition):** Fine-tuned OpenAI Whisper (small).
2. **NER (Named Entity Recognition):** Fine-tuned BioBERT on medical datasets (BC5CDR, NCBI-Disease).
3. **ONNX Optimization:** Exported Whisper models with INT8 dynamic quantization for ~35% inference speedup.
4. **FastAPI Application:** REST API for single-endpoint transcription and extraction.
5. **Dockerized Deployment:** Ready to run anywhere via Docker Compose.

---


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
