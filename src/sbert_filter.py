import torch
from sentence_transformers import SentenceTransformer, util

class ClickbaitFilter:
    def __init__(self, model_name='all-MiniLM-L6-v2', threshold=0.54):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = SentenceTransformer(model_name, device=self.device)
        self.optimal_threshold = threshold

    def check_similarity(self, title, body):
        title_emb = self.model.encode([title], convert_to_tensor=True)
        body_emb = self.model.encode([body], convert_to_tensor=True)
        score = util.pairwise_cos_sim(title_emb, body_emb).item()
        passed = score >= self.optimal_threshold
        return passed, score