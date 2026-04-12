import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class RiskClassifier:
    def __init__(self, model_path="xuyifei1234/finbert-risk-classifier", use_local=False):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=use_local)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=use_local)
        self.model.to(self.device)
        self.model.eval()
        self.id2label = {0: "High Risk", 1: "Neutral", 2: "Low Risk"}

    def predict(self, text):
        inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=512,
            padding="max_length",
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1)
            pred_id = torch.argmax(probs, dim=-1).item()
            confidence = probs[0, pred_id].item()
        return {
            "label_id": pred_id,
            "label": self.id2label[pred_id],
            "confidence": confidence,
            "probabilities": probs.cpu().numpy()[0]
        }