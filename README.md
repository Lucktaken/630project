# Financial Risk Alert System

A multi-stage NLP pipeline that detects financial risks from news articles and generates reliable alerts with extracted organization entities.

## 🚀 Features

- **SBERT-based Clickbait Filter**: Discards articles where the headline doesn't match the body using cosine similarity.
- **FinBERT Risk Classifier**: Fine-tuned on Financial PhraseBank and augmented with Yahoo Finance data (Macro-F1: 0.9656+).
- **CRF Named Entity Recognition**: Extracts organization names (ORG) from high-risk news using a CRF model trained on CoNLL-2003.
- **LLM-powered Alert Generation**: Uses GitHub Models (Qwen) to produce natural, context-aware risk alerts without hallucination.
- **Web Article Parser**: Automatically extracts title and body from news URLs (with fallback to manual input).

## 📦 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/xuyifei1234/financial-risk-alert-system.git
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

4. Set up API keys (optional, for LLM alert generation):
   - Place your GitHub Personal Access Token in `github_token.txt`
   - Alternatively, set environment variable `GITHUB_TOKEN`

## 🧪 Quick Demo

Run the interactive pipeline:

```bash
python scripts/run_demo.py
```

You will be prompted to enter a news URL. The system will automatically extract the content, analyze it, and output a risk alert.

### Sample Output

```bash
$ python scripts/run_demo.py
Loading models...
Models loaded.

Enter news URL (or press Enter to input manually): https://www.cnn.com/2026/04/11/business/high-inflation-rate-problem

✅ Successfully parsed:
Title: Uncomfortably high inflation is a real problem and it's not going away anytime soon
Body (first 200 chars): Here's something no American wants to hear: Prices are surging again, and uncomfortably high inflation could be with us for quite some time.

Inflation has been a thorn in the US economy's side since ...

============================================================
RISK ALERT DEMO
============================================================
Title: Uncomfortably high inflation is a real problem and it's not going away anytime soon
Body: Here's something no American wants to hear: Prices are surging again, and uncomfortably high inflati...
SBERT Similarity: 0.619 (Passed: True)
Risk: High Risk (conf: 0.986)
Alert Triggered: True

ALERT MESSAGE:
**Risk Alert: High Risk**
**Confidence: 0.99**
**Affected Organizations:** Commerce Department, PNC Financial Services Group, CNN, Federal Reserve Chair Jerome Powell, Navy Federal Credit Union, Strait of Hormuz, Purdue University

**Alert:** Uncomfortably high inflation is persisting in the US economy, with prices surging and no immediate resolution in sight. This poses significant risks to the financial stability of the affected organizations.
============================================================
```

## 📓 Notebooks

This repository now includes two notebooks:

- `notebooks/Training.ipynb` - training and experimentation for the individual models
- `notebooks/Pipeline_Error_Analysis.ipynb` - end-to-end system architecture and pipeline error analysis without retraining

The new error-analysis notebook documents the deployed pipeline:

`raw news -> denoising -> classification -> NER -> alert template`

It reuses the trained models and traces representative articles through each stage to answer practical system questions such as:

- Does a denoising mistake suppress a valid alert before classification?
- Does a classifier error create a false alert or a missed alert?
- Does an NER failure degrade the final alert by missing affected organizations?

In the current curated examples, the clearest observed propagated error is at the denoising stage: a false negative in the SBERT filter prevents the article from reaching classification and therefore causes a missed downstream alert.

## ⚙️ Configuration

To suppress verbose warnings and progress bars during model loading, the following environment settings are applied in `scripts/run_demo.py`:

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

**Please comment out these lines if you encounter any issues with model downloading or loading.**  
They are intended only for a cleaner demo output and do not affect the underlying functionality.

## 📁 Repository Structure

```
.
├── README.md
├── requirements.txt
├── .gitignore
├── assets/                 # Images and training curves
├── models/                 # Small serialized models (e.g., CRF)
│   └── crf_org_extractor.pkl
├── notebooks/              # Jupyter notebooks for training & experiments
│   ├── Training.ipynb
│   └── Pipeline_Error_Analysis.ipynb
├── scripts/
│   ├── run_demo.py         # Interactive demo script
│   └── evaluate_models.py  # Model comparison script
└── src/                    # Core Python modules
    ├── sbert_filter.py
    ├── finbert_classifier.py
    ├── crf_extractor.py
    ├── pipeline.py
    ├── web_parser.py
    ├── llm_alert_generator.py
    └── utils.py
```

## 🤖 Models

- **FinBERT Risk Classifier**: Hosted on Hugging Face Hub at [`xuyifei1234/finbert-risk-classifier-v2`](https://huggingface.co/xuyifei1234/finbert-risk-classifier-v2).  
- **CRF Entity Extractor**: Saved locally in `models/crf_org_extractor.pkl` (small file, included in the repo).  
- **SBERT**: Uses `all-MiniLM-L6-v2` from sentence-transformers (downloaded automatically).  
- **LLM Alert Generator**: Uses GitHub Models (Qwen) via free API, with fallback to template-based generation.

## 📊 Datasets Used

The following datasets were used for training/calibration. They are not included in this repository but can be downloaded via the provided scripts or directly from their sources:

- **Webis Clickbait-17** – For SBERT threshold calibration.
- **Financial PhraseBank** – For fine-tuning FinBERT.
- **Yahoo Finance News** – Additional training data for improved generalization.
- **CoNLL-2003** – For training the CRF entity extractor.
- **CNN / Common Pile News** – For domain-adaptive pretraining (DAPT).

## 📄 License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [Sentence-Transformers](https://github.com/UKPLab/sentence-transformers)
- [sklearn-crfsuite](https://github.com/TeamHG-Memex/sklearn-crfsuite)
- [newspaper3k](https://github.com/codelucas/newspaper)
- [GitHub Models](https://github.com/marketplace/models)
