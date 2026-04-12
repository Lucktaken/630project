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

def main():
    # 示例输入
    title = "Apple faces regulatory pressure in China"
    body = "Apple may face significant regulatory challenges in China next quarter as authorities tighten market rules."

    crf_extractor = CRFExtractor(crf_path="models/crf_org_extractor.pkl")
    pipeline = RiskAlertPipeline(
        sbert_threshold=0.54,
        finbert_model_path="xuyifei1234/finbert-risk-classifier",
        crf_model=crf_extractor.crf
    )

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

if __name__ == "__main__":
    main()