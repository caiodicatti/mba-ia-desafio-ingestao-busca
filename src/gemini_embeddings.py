import os
import requests
from langchain_core.embeddings import Embeddings


class GeminiEmbeddings(Embeddings):
    def __init__(self, model: str = "gemini-embedding-001", api_key: str = None):
        self.model = model
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.base_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:embedContent"

    def _embed_one(self, text: str) -> list[float]:
        payload = {
            "model": f"models/{self.model}",
            "content": {"parts": [{"text": text}]},
        }
        response = requests.post(self.base_url, json=payload, params={"key": self.api_key})
        if not response.ok:
            raise RuntimeError(f"Embedding API error {response.status_code}: {response.text}")
        return response.json()["embedding"]["values"]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed_one(text)
