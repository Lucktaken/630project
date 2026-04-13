#!/usr/bin/env python3
import os
import sys
import warnings

## Please comment the following part if you encounter parameter download problems
warnings.filterwarnings("ignore")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TQDM_DISABLE"] = "1"
os.environ["SENTENCE_TRANSFORMERS_SILENT"] = "1"

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import RiskAlertPipeline
from src.crf_extractor import CRFExtractor
from src.web_parser import WebParser

def main():
    # title = "Apple faces regulatory pressure in China"
    # body = "Apple may face significant regulatory challenges in China next quarter as authorities tighten market rules."
    # title = "JPMorgan Chase beats profit estimates on strong consumer banking"
    # body = "JPMorgan Chase reported quarterly earnings that exceeded Wall Street expectations, driven by robust performance in its consumer and community banking division. Higher interest rates boosted net interest income, though CEO Jamie Dimon warned of persistent inflation and geopolitical risks ahead."
    if not os.getenv("GITHUB_TOKEN"):
        token_file = "github_token.txt"
        if os.path.exists(token_file):
            with open(token_file, 'r') as f:
                os.environ["GITHUB_TOKEN"] = f.read().strip()
        else:
            print("Warning: GITHUB_TOKEN not found. Alert generation will fallback to template.")

    crf_extractor = CRFExtractor(crf_path="models/crf_org_extractor.pkl")
    pipeline = RiskAlertPipeline(
        sbert_threshold=0.54,
        finbert_model_path="xuyifei1234/finbert-risk-classifier-v2",
        crf_model=crf_extractor.crf
    )

    parser = WebParser()
    url = input("Enter news URL (or press Enter to input manually): ").strip()
    if not url:
        title = input("Enter title: ").strip()
        body = input("Enter body: ").strip()
    else:
        try:
            title, body = parser.parse(url)
            print(f"\n✅ Successfully parsed:")
            print(f"Title: {title}")
            print(f"Body (first 200 chars): {body[:200]}...\n")
        except Exception as e:
            print(f"\n❌ Automatic extraction failed: {e}")
            print("Please enter the title and body manually.\n")
            title = input("Enter title: ").strip()
            body = input("Enter body: ").strip()
    
    try:
        title, body = parser.parse(url)
        print(f"Title: {title}")
        print(f"Body (first 200 chars): {body[:200]}...")
        result = pipeline.run(title, body)
        print("\n" + "="*60)
        print("RISK ALERT DEMO")
        print("="*60)
        print(f"Title: {title}")
        print(f"Body: {body[:100]}...")
        print(f"SBERT Similarity: {result['similarity_score']:.3f} (Passed: {result['passed_sbert']})")
        print(f"Risk: {result['risk_label']} (conf: {result['risk_confidence']:.3f})")
        print(f"Alert Triggered: {result['alert_triggered']}")
        print(f"\nALERT MESSAGE:\n{result['alert_message']}")
        print("="*60)
        
    except Exception as e:
        print(f"Error: {e}")
    

if __name__ == "__main__":
    main()