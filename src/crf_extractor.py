import re
import sklearn_crfsuite
import os
import joblib

class CRFExtractor:
    def __init__(self, crf_path="models/crf_org_extractor.pkl"):
        if os.path.exists(crf_path):
            self.crf = joblib.load(crf_path)
            # print(f"✅ CRF model loaded from {crf_path} ")
        else:
            self.crf = None
            print(f"⚠️ {crf_path} not found, Event Extractor invalid")

    @staticmethod
    def simple_tokenize(text):
        return re.findall(r"\w+|[^\w\s]", text)

    @staticmethod
    def word2features(sent, i):
        word = sent[i]
        features = {
            'bias': 1.0,
            'word.lower()': word.lower(),
            'word[-3:]': word[-3:],
            'word[-2:]': word[-2:],
            'word.isupper()': word.isupper(),
            'word.istitle()': word.istitle(),
            'word.isdigit()': word.isdigit(),
        }
        if i > 0:
            prev_word = sent[i-1]
            features.update({
                '-1:word.lower()': prev_word.lower(),
                '-1:word.istitle()': prev_word.istitle(),
                '-1:word.isupper()': prev_word.isupper(),
            })
        else:
            features['BOS'] = True
        if i < len(sent)-1:
            next_word = sent[i+1]
            features.update({
                '+1:word.lower()': next_word.lower(),
                '+1:word.istitle()': next_word.istitle(),
                '+1:word.isupper()': next_word.isupper(),
            })
        else:
            features['EOS'] = True
        return features

    def sent2features(self, sent):
        return [self.word2features(sent, i) for i in range(len(sent))]

    def extract_orgs(self, text):
        if self.crf is None:
            raise ValueError("CRF model not loaded. Please train or provide a model.")
        tokens = self.simple_tokenize(text)
        features = self.sent2features(tokens)
        tags = self.crf.predict_single(features)
        orgs = self._extract_org_entities(tokens, tags)
        return orgs

    @staticmethod
    def _extract_org_entities(tokens, tags):
        entities = []
        current = []
        for token, tag in zip(tokens, tags):
            if tag == "B-ORG":
                if current:
                    entities.append(" ".join(current))
                current = [token]
            elif tag == "I-ORG":
                if current:
                    current.append(token)
            else:
                if current:
                    entities.append(" ".join(current))
                    current = []
        if current:
            entities.append(" ".join(current))
        # 去重并保持顺序
        seen = set()
        deduped = []
        for ent in entities:
            if ent not in seen:
                deduped.append(ent)
                seen.add(ent)
        return deduped