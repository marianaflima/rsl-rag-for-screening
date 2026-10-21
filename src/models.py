from dataclasses import dataclass, field
from datetime import date
from typing import List
import requests
from langchain_core.embeddings import Embeddings

class OpenRouterEmbeddings(Embeddings):
    def __init__(self, model: str, api_key: str):
        self.model = model
        self.api_key = api_key
        self.url = "https://openrouter.ai/api/v1/embeddings"

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Envia um por um ou em lote dependendo do que o modelo aceita
        return [self._get_embedding(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._get_embedding(text)

    def _get_embedding(self, text: str) -> List[float]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "input": text  # Garantindo que vai como string simples, se necessário
        }
        response = requests.post(self.url, json=payload, headers=headers)
        if response.status_code != 200:
            raise Exception(f"Erro OpenRouter API: {response.text}")
        
        data = response.json()
        return data["data"][0]["embedding"]

@dataclass
class AcademicRecord:
    source: str
    doi: str | None
    title: str
    abstract: str
    keywords: list[str]
    topics: list[str] | None
    authors: list[str]
    pub_date: date | None
    pdf_url: str | None
    article_type: str | None
