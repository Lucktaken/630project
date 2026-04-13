# src/web_parser.py
import requests
from bs4 import BeautifulSoup
from newspaper import Article, ArticleException
from readability import Document
import re

class WebParser:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

    def parse(self, url):
        if not self._is_valid_url(url):
            raise ValueError("Invalid URL format.")
        try:
            title, body = self._parse_with_newspaper(url)
            if title and body:
                return title, body
        except Exception as e:
            print(f"Newspaper3k parsing failed: {e}")
        try:
            title, body = self._parse_with_readability(url)
            if title and body:
                return title, body
        except Exception as e:
            print(f"Readability parsing failed: {e}")
        raise RuntimeError(f"Failed to extract content from {url}. The page may not be a news article.")

    def _is_valid_url(self, url):
        regex = re.compile(
            r'^(?:http|ftp)s?://'
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'
            r'localhost|'
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'
            r'(?::\d+)?'
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
        return re.match(regex, url) is not None

    def _parse_with_newspaper(self, url):
        article = Article(url, language='en')
        article.download()
        article.parse()
        title = article.title
        body = article.text
        if len(title) < 5 or len(body) < 200:
            raise ValueError("Content too short, probably not a news article.")
        return title, body

    def _parse_with_readability(self, url):
        response = requests.get(url, headers=self.headers, timeout=10)
        response.raise_for_status()
        doc = Document(response.text)
        title = doc.title()
        soup = BeautifulSoup(doc.summary(), 'html.parser')
        body = soup.get_text(separator='\n').strip()
        if len(title) < 5 or len(body) < 200:
            raise ValueError("Content too short, probably not a news article.")
        return title, body