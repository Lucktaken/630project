from .sbert_filter import ClickbaitFilter
from .finbert_classifier import RiskClassifier
from .crf_extractor import CRFExtractor

class RiskAlertPipeline:
    def __init__(self, sbert_threshold=0.54, finbert_model_path="ProsusAI/finbert", crf_model=None):
        self.sbert_filter = ClickbaitFilter(threshold=sbert_threshold)
        self.risk_classifier = RiskClassifier(model_path=finbert_model_path)
        self.crf_extractor = CRFExtractor()
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

        # Step 1: SBERT Filter
        passed, sim_score = self.sbert_filter.check_similarity(title, body)
        result["passed_sbert"] = passed
        result["similarity_score"] = sim_score
        if not passed:
            result["alert_message"] = "Filtered: headline-body mismatch."
            return result

        # Step 2: Risk Classification
        text = title + " " + body
        risk = self.risk_classifier.predict(text)
        result["risk_label"] = risk["label"]
        result["risk_confidence"] = risk["confidence"]
        result["high_risk_prob"] = float(risk["probabilities"][0])

        # Step 3: Alert Trigger
        triggered = (risk["label"] == "High Risk") or (result["high_risk_prob"] >= self.high_risk_threshold)
        result["alert_triggered"] = triggered

        # Step 4: ORG Extraction if triggered
        if triggered and self.crf_extractor.crf is not None:
            try:
                orgs = self.crf_extractor.extract_orgs(text)
                result["orgs"] = orgs
            except Exception as e:
                result["orgs"] = [f"NER Error: {e}"]

        # Step 5: Build Alert Message
        result["alert_message"] = self._build_message(result)
        return result

    def _build_message(self, result):
        if not result["passed_sbert"]:
            return result["alert_message"]
        if not result["alert_triggered"]:
            return (f"No alert. Risk: {result['risk_label']} "
                    f"(conf={result['risk_confidence']:.3f})")
        org_str = ", ".join(result["orgs"]) if result["orgs"] else "No organization identified"
        return (f"ALERT: {result['risk_label']} "
                f"(conf={result['risk_confidence']:.3f}, "
                f"high-risk prob={result['high_risk_prob']:.3f}). "
                f"Affected: {org_str}.")