import requests
from newsapi import NewsApiClient
import json

class NewsCollectionAgent:
    def __init__(self, api_key=None):
        self.newsapi = NewsApiClient(api_key=api_key) if api_key else None

    def collect_from_api(self, query='technology', sources='bbc-news'):
        if not self.newsapi:
            raise ValueError("API key required for NewsAPI")
        articles = self.newsapi.get_everything(q=query, sources=sources, language='en')
        return [article['title'] + "\n" + article['description'] + "\n" + article['content'] for article in articles['articles']]

    def collect_from_json(self, file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
        return [item['title'] + "\n" + item['content'] for item in data]