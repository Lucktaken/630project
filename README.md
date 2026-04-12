# Financial Risk Alert System

A multi-stage NLP pipeline that detects financial risks from news articles and generates reliable alerts with extracted organization entities.

## 🚀 Features

- **SBERT-based Clickbait Filter**: Discards articles where the headline doesn't match the body using cosine similarity.
- **FinBERT Risk Classifier**: Fine-tuned on Financial PhraseBank to classify news into High Risk, Neutral, or Low Risk.
- **CRF Named Entity Recognition**: Extracts organization names (ORG) from high-risk news using a CRF model trained on CoNLL-2003.
- **Template-based Alert Generation**: Produces structured, human-readable alerts without hallucination.

## 📁 Repository Structure
```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── assets/                 # Images and training curves
├── models/                 # Small serialized models (e.g., CRF)
│   └── crf_org_extractor.pkl
├── notebooks/              # Jupyter notebooks for training & experiments
├── scripts/
│   ├── run_demo.py         # Quick demo script
│   └── download_data.py    # (Optional) script to download datasets
└── src/                    # Core Python modules
    ├── sbert_filter.py
    ├── finbert_classifier.py
    ├── crf_extractor.py
    ├── pipeline.py
    └── utils.py
```

## 📦 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/financial-risk-alert-system.git
   cd financial-risk-alert-system
   ```
2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   venv\Scripts\activate   # On Windows
   source venv/bin/activate  # On macOS/Linux
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 🤖 Models
- FinBERT Risk Classifier: Hosted on Hugging Face Hub at `xuyifei1234/finbert-risk-classifier`.
- CRF Entity Extractor: Saved locally in `models/crf_org_extractor.pkl` (small file, included in the repo).
- SBERT: Uses `all-MiniLM-L6-v2` from sentence-transformers (downloaded automatically).

## 📊 Datasets Used
The following datasets were used for training/calibration. They are not included in this repository but can be downloaded via the provided scripts or directly from their sources:
- Webis Clickbait-17 – For SBERT threshold calibration.
- Financial PhraseBank – For fine-tuning FinBERT.
- CoNLL-2003 – For training the CRF entity extractor.

## 🧪 Quick Demo

Run the end-to-end pipeline with a sample news article:
```bash
python scripts/run_demo.py
```
The script will automatically download the fine-tuned FinBERT model from Hugging Face Hub on the first run.

### Expected Output:
```text
============================================================
RISK ALERT DEMO
============================================================
Title: Apple faces regulatory pressure in China
Body: Apple may face significant regulatory challenges...
SBERT Similarity: 0.783 (Passed: True)
Risk: High Risk (conf: 0.948)
Alert Triggered: True

ALERT MESSAGE:
ALERT: High Risk (conf=0.948, high-risk prob=0.948). Affected: Apple.
============================================================
```

## ⚙️ Configuration
To suppress verbose warnings and progress bars during model loading (especially useful for demos), the following environment settings are applied in `scripts/run_demo.py:
```python
warnings.filterwarnings("ignore")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TQDM_DISABLE"] = "1"
os.environ["SENTENCE_TRANSFORMERS_SILENT"] = "1"
```
Please comment out these lines if you encounter any issues with model downloading or loading.
They are intended only for a cleaner demo output and do not affect the underlying functionality.