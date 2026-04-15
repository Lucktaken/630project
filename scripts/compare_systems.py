#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import time
import warnings
import requests
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

warnings.filterwarnings("ignore")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TQDM_DISABLE"] = "1"
os.environ["SENTENCE_TRANSFORMERS_SILENT"] = "1"

import logging
logging.basicConfig(level=logging.ERROR)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import RiskAlertPipeline
from src.crf_extractor import CRFExtractor
from src.sbert_filter import ClickbaitFilter
from src.finbert_classifier import RiskClassifier
from sklearn.metrics import f1_score

# ==================== Configuration ====================
TEST_DATA_PATH = "test_articles_2026.json"
OUTPUT_DIR = "assets"
os.makedirs(OUTPUT_DIR, exist_ok=True)

OLD_MODEL_ID = "xuyifei1234/finbert-risk-classifier"
NEW_MODEL_ID = "xuyifei1234/finbert-risk-classifier-v2"
SBERT_THRESHOLD = 0.54

LABEL_MAP = {0: "High Risk", 1: "Neutral", 2: "Low Risk"}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

# ==================== Load Test Data ====================
def load_test_data(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} test URLs.")
    return data

# ==================== System 1: GitHub Models API (OpenAI) ====================
class GitHubModelsSystem:
    def __init__(self, model="gpt-4o-mini"):
        self.api_key = os.getenv("GITHUB_TOKEN")
        if not self.api_key:
            token_file = "github_token.txt"
            if os.path.exists(token_file):
                with open(token_file, "r") as f:
                    self.api_key = f.read().strip()
        if not self.api_key:
            raise ValueError("GITHUB_TOKEN not found in environment or github_token.txt")
        
        self.model = model
        self.api_url = "https://models.inference.ai.azure.com/chat/completions"

    def run(self, title: str, body: str) -> Dict:
        prompt = f"""Analyze the following financial news article and return ONLY a valid JSON object with exactly these keys:
- "risk": one of "High Risk", "Neutral", "Low Risk"
- "confidence": a float between 0 and 1
- "organizations": a list of company/organization names mentioned

Title: {title}
Body: {body[:2000]}

Output the JSON only, no other text."""
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a financial risk analyst. Always respond with valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 300
        }

        try:
            response = requests.post(self.api_url, json=data, headers=headers, timeout=30)
            response.raise_for_status()
            content = response.json()['choices'][0]['message']['content'].strip()
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
            else:
                result = json.loads(content)
            return {
                "risk_label": result.get("risk", "Neutral"),
                "risk_confidence": result.get("confidence", 0.5),
                "orgs": result.get("organizations", []),
                "alert_triggered": result.get("risk", "Neutral") == "High Risk"
            }
        except Exception as e:
            print(f"GitHub API error: {e}")
            return {"risk_label": "Neutral", "risk_confidence": 0.0, "orgs": [], "alert_triggered": False}

# ==================== Systems 2 & 3: Our Modular Pipeline ====================
class ModularSystem:
    def __init__(self, finbert_model_id: str):
        self.sbert_filter = ClickbaitFilter(threshold=SBERT_THRESHOLD)
        self.risk_classifier = RiskClassifier(model_path=finbert_model_id)
        self.crf_extractor = CRFExtractor(crf_path="models/crf_org_extractor.pkl")
        self.high_risk_threshold = 0.30

    def run(self, title: str, body: str) -> Dict:
        passed, sim = self.sbert_filter.check_similarity(title, body)
        if not passed:
            return {"risk_label": "Filtered", "risk_confidence": 0.0, "orgs": [], "alert_triggered": False}

        text = title + " " + body
        risk = self.risk_classifier.predict(text)
        label = risk["label"]
        conf = risk["confidence"]
        high_prob = float(risk["probabilities"][0])

        triggered = (label == "High Risk") or (high_prob >= self.high_risk_threshold)

        orgs = []
        if triggered and self.crf_extractor.crf is not None:
            try:
                orgs = self.crf_extractor.extract_orgs(text)
            except:
                pass

        return {
            "risk_label": label,
            "risk_confidence": conf,
            "orgs": orgs,
            "alert_triggered": triggered
        }

# ==================== Evaluation Metrics ====================
def compute_metrics(true_labels: List[str], pred_labels: List[str], 
                    true_orgs_list: List[List[str]], pred_orgs_list: List[List[str]]) -> Dict:
    y_true = [INV_LABEL_MAP[l] for l in true_labels]
    y_pred = [INV_LABEL_MAP[p] if p in INV_LABEL_MAP else 1 for p in pred_labels]
    
    accuracy = np.mean(np.array(y_true) == np.array(y_pred))
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    
    tp_org, fp_org, fn_org = 0, 0, 0
    for t_orgs, p_orgs in zip(true_orgs_list, pred_orgs_list):
        t_set = set(t_orgs)
        p_set = set(p_orgs)
        tp_org += len(t_set & p_set)
        fp_org += len(p_set - t_set)
        fn_org += len(t_set - p_set)
    
    org_precision = tp_org / (tp_org + fp_org) if (tp_org + fp_org) > 0 else 0.0
    org_recall = tp_org / (tp_org + fn_org) if (tp_org + fn_org) > 0 else 0.0
    org_f1 = 2 * org_precision * org_recall / (org_precision + org_recall) if (org_precision + org_recall) > 0 else 0.0
    
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "org_precision": org_precision,
        "org_recall": org_recall,
        "org_f1": org_f1
    }

# ==================== Main Function ====================
def main():
    test_data = load_test_data(TEST_DATA_PATH)
    
    titles = []
    bodies = []
    true_risks = []
    true_orgs = []
    
    print("Loading test articles...")
    for i, item in enumerate(test_data):
        title = item["title"]
        body = item["body"]
        titles.append(title)
        bodies.append(body)
        true_risks.append(item["true_risk"])
        true_orgs.append(item.get("true_orgs", []))
        print(f"  [{i+1}/{len(test_data)}] Loaded: {title[:50]}...")
    
    print(f"Successfully loaded {len(titles)} articles.")

    print("Initializing systems...")
    api_sys = GitHubModelsSystem()
    old_sys = ModularSystem(OLD_MODEL_ID)
    new_sys = ModularSystem(NEW_MODEL_ID)

    systems = {
        "GitHub Models (OpenAI)": api_sys,
        "Our System (FinBERT v1)": old_sys,
        "Our System (FinBERT v2 + DAPT)": new_sys
    }

    results = {}
    for name, sys in systems.items():
        print(f"\nRunning {name}...")
        pred_risks = []
        pred_orgs = []
        for title, body in zip(titles, bodies):
            out = sys.run(title, body)
            pred_risks.append(out["risk_label"])
            pred_orgs.append(out["orgs"])
            time.sleep(0.5)
        results[name] = {"risks": pred_risks, "orgs": pred_orgs}

    metrics = {}
    for name, res in results.items():
        print(f"\nEvaluating {name}...")
        m = compute_metrics(true_risks, res["risks"], true_orgs, res["orgs"])
        metrics[name] = m
        print(f"  Accuracy: {m['accuracy']:.4f}")
        print(f"  Macro-F1: {m['macro_f1']:.4f}")
        print(f"  ORG Precision: {m['org_precision']:.4f}, Recall: {m['org_recall']:.4f}, F1: {m['org_f1']:.4f}")

    sys_names = list(metrics.keys())
    accs = [metrics[n]["accuracy"] for n in sys_names]
    f1s = [metrics[n]["macro_f1"] for n in sys_names]
    org_f1s = [metrics[n]["org_f1"] for n in sys_names]

    x = np.arange(len(sys_names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width, accs, width, label='Accuracy', color='#1f77b4')
    bars2 = ax.bar(x, f1s, width, label='Macro-F1', color='#2ca02c')
    bars3 = ax.bar(x + width, org_f1s, width, label='ORG F1', color='#ff7f0e')

    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    for bar in bars3:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

    total_samples = len(test_data)
    success_counts = []
    for name in sys_names:
        pred_risks = results[name]["risks"]
        valid_count = sum(1 for r in pred_risks if r != "Filtered")
        success_counts.append(valid_count)

    ax.set_ylabel('Score')
    ax.set_title('System Comparison on End-to-End Test Set')
    ax.set_xticks(x)
    xtick_labels = [f"{name}\n({success_counts[i]}/{len(titles)} passed)" for i, name in enumerate(sys_names)]
    ax.set_xticklabels(xtick_labels, rotation=0, ha='center')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "system_comparison_bar.png"), dpi=150)
    plt.show()

    print(f"\n✅ Comparison complete. Charts saved to {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()