import os
from typing import List, Optional
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from langchain_core.embeddings import Embeddings


class LocalSemanticEmbeddings(Embeddings):
    """
    Zero-dependency, offline deterministic semantic embedding engine.
    Uses sublinear frequency hashing with L2 unit normalization.
    Guarantees instant vectorization without internet access or downloading large model weights.
    """

    def __init__(self, n_features: int = 384):
        self.n_features = n_features
        self.vectorizer = HashingVectorizer(
            n_features=n_features,
            norm="l2",
            alternate_sign=False,
            ngram_range=(1, 2)
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        matrix = self.vectorizer.transform(texts).toarray()
        return matrix.astype(np.float32).tolist()

    def embed_query(self, text: str) -> List[float]:
        matrix = self.vectorizer.transform([text]).toarray()
        return matrix.astype(np.float32)[0].tolist()


class RobustGeminiEmbeddings(Embeddings):
    """
    Production-grade adaptive Gemini Embeddings adapter.
    Automatically discovers and binds to the active embedding model supported
    by the user's project (models/gemini-embedding-001, models/text-embedding-004, etc.)
    and gracefully falls back to LocalSemanticEmbeddings if quota is exceeded or network fails.
    """

    CANDIDATE_MODELS = [
        "models/gemini-embedding-001",
        "gemini-embedding-001",
        "models/text-embedding-004",
        "text-embedding-004",
        "models/embedding-001"
    ]

    def __init__(self, api_key: str):
        self.api_key = api_key.strip()
        self.active_client: Optional[Embeddings] = None
        self.fallback = LocalSemanticEmbeddings(n_features=384)
        self._init_active_model()

    def _init_active_model(self):
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        for model_name in self.CANDIDATE_MODELS:
            try:
                candidate = GoogleGenerativeAIEmbeddings(
                    model=model_name,
                    google_api_key=self.api_key
                )
                # Test query to verify active support
                candidate.embed_query("ping")
                self.active_client = candidate
                print(f"[RobustGeminiEmbeddings] Successfully initialized Gemini model: '{model_name}'")
                return
            except Exception as e:
                # Try next model candidate
                continue

        print("[RobustGeminiEmbeddings] All Gemini models failed or quota exceeded. Falling back to LocalSemanticEmbeddings.")
        self.active_client = self.fallback

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        try:
            if self.active_client:
                return self.active_client.embed_documents(texts)
        except Exception as e:
            print(f"[RobustGeminiEmbeddings] embed_documents failed ({e}). Using local fallback.")
            return self.fallback.embed_documents(texts)
        return self.fallback.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        try:
            if self.active_client:
                return self.active_client.embed_query(text)
        except Exception as e:
            print(f"[RobustGeminiEmbeddings] embed_query failed ({e}). Using local fallback.")
            return self.fallback.embed_query(text)
        return self.fallback.embed_query(text)


class EmbeddingManager:
    """
    Factory & Manager for vector embedding providers.
    Supports Google Gemini, OpenAI, and Local Offline fallback.
    """

    @staticmethod
    def get_embeddings(
        provider: str = "gemini",
        google_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None
    ) -> Embeddings:
        """
        Instantiates the requested embedding provider.
        Gracefully falls back to LocalSemanticEmbeddings if API keys are missing or invalid.
        """
        provider = (provider or "gemini").lower().strip()
        g_key = (google_api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or "").strip()
        o_key = (openai_api_key or os.getenv("OPENAI_API_KEY") or "").strip()

        if provider == "gemini" and g_key:
            try:
                return RobustGeminiEmbeddings(api_key=g_key)
            except Exception as e:
                print(f"[EmbeddingManager] Gemini Embeddings initialization warning: {e}. Falling back.")

        elif provider == "openai" and o_key:
            try:
                from langchain_openai import OpenAIEmbeddings
                client = OpenAIEmbeddings(
                    model="text-embedding-3-small",
                    openai_api_key=o_key
                )
                # Quick validation
                client.embed_query("ping")
                return client
            except Exception as e:
                print(f"[EmbeddingManager] OpenAI Embeddings error/quota: {e}. Falling back.")

        # Default / Fallback: Local offline embeddings
        return LocalSemanticEmbeddings(n_features=384)
