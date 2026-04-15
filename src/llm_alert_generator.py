# src/llm_alert_generator.py
import os
import requests

class LLMAlertGenerator:
    def __init__(self, model="gpt-4o-mini"):
        """
        Generate alert messages using GitHub Models API.
        You need a GitHub Personal Access Token (PAT), which is usually passed in through the environment variable 'GITHUB-TOKEN'.
        """
        self.api_key = os.getenv("GITHUB_TOKEN")
        if not self.api_key:
            raise ValueError("Environment variable GITHUB_TOKEN is required.")
        
        self.model = model
        self.api_url = "https://models.inference.ai.azure.com/chat/completions"

    def generate(self, title, risk_label, confidence, orgs, body_snippet):
        """message generator."""
        prompt = self._build_prompt(title, risk_label, confidence, orgs, body_snippet)
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a financial risk alert system. Generate concise, factual alerts in English."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 150
        }

        try:
            response = requests.post(self.api_url, json=data, headers=headers)
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content'].strip()
        except Exception as e:
            print(f"LLM API Error: {e}. Falling back to template.")
            return self._fallback_template(title, risk_label, confidence, orgs)

    def _build_prompt(self, title, risk_label, confidence, orgs, body_snippet):
        """Templated Prompt."""
        org_list = ", ".join(orgs) if orgs else "no specific organizations identified"
        return f"""
Based on the following financial news, generate a concise risk alert message in English.

Risk Level: {risk_label}
Confidence: {confidence:.2f}
Affected Organizations: {org_list}
Headline: {title}
News Snippet: {body_snippet}...

The alert should mention the risk level, the organizations involved, and a brief reason if possible.
Alert Message:
"""

    def _fallback_template(self, title, risk_label, confidence, orgs):
        """Fall back when fails."""
        org_str = ", ".join(orgs) if orgs else "No specific organizations identified"
        return (f"ALERT: {risk_label} (conf={confidence:.2f}). "
                f"Affected: {org_str}. Headline: {title}")