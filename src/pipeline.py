# src/pipeline.py
from .sbert_filter import ClickbaitFilter
from .finbert_classifier import RiskClassifier
from .crf_extractor import CRFExtractor
from .llm_alert_generator import LLMAlertGenerator

class RiskAlertPipeline:
    def __init__(self, sbert_threshold=0.54, finbert_model_path="xuyifei1234/finbert-risk-classifier-v2", crf_model=None):
        self.sbert_filter = ClickbaitFilter(threshold=sbert_threshold)
        self.risk_classifier = RiskClassifier(model_path=finbert_model_path)
        self.crf_extractor = CRFExtractor()
        self.llm_generator = LLMAlertGenerator()
        self.high_risk_threshold = 0.30

    def run(self, title, body):
        result = {
            "title": title,
            "body": body,
            "passed_sbert": False,
            "similarity_score": None,
            "risk_label": None,
            "risk_confidence": None,
            "high_risk_prob": None,
            "alert_triggered": False,
            "orgs": [],
            "alert_message": None
        }

        # Step 1: SBERT filter
        passed, sim_score = self.sbert_filter.check_similarity(title, body)
        result["passed_sbert"] = passed
        result["similarity_score"] = sim_score

        if not passed:
            result["alert_message"] = "Filtered: headline-body mismatch."
            return result

        # Step 2: Risk classification
        text_for_risk = title + " " + body
        risk_result = self.risk_classifier.predict(text_for_risk)
        result["risk_label"] = risk_result["label"]
        result["risk_confidence"] = risk_result["confidence"]
        result["high_risk_prob"] = float(risk_result["probabilities"][0])

        # Step 3: Alert trigger
        triggered = (risk_result["label"] == "High Risk") or (result["high_risk_prob"] >= self.high_risk_threshold)
        result["alert_triggered"] = triggered

        # Step 4: ORG extraction if triggered
        if triggered and self.crf_extractor.crf is not None:
            try:
                orgs = self.crf_extractor.extract_orgs(text_for_risk)
                result["orgs"] = orgs
            except Exception as e:
                result["orgs"] = [f"NER Error: {e}"]
        elif triggered:
            result["orgs"] = []  # CRF not available

        # Step 5: Generate alert message
        if result["alert_triggered"]:
            body_snippet = body[:200] + "..." if len(body) > 200 else body
            try:
                result["alert_message"] = self.llm_generator.generate(
                    title=title,
                    risk_label=result["risk_label"],
                    confidence=result["risk_confidence"],
                    orgs=result["orgs"],
                    body_snippet=body_snippet
                )
            except Exception as e:
                # Fallback to template if LLM fails
                org_str = ", ".join(result["orgs"]) if result["orgs"] else "No organization identified"
                result["alert_message"] = (f"ALERT: {result['risk_label']} "
                                           f"(conf={result['risk_confidence']:.3f}, "
                                           f"high-risk prob={result['high_risk_prob']:.3f}). "
                                           f"Affected: {org_str}.")
        elif not result["passed_sbert"]:
            pass  # already set
        else:
            result["alert_message"] = (f"No alert. Risk: {result['risk_label']} "
                                       f"(conf={result['risk_confidence']:.3f})")

        return result