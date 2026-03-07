"""
This module provides wrappers for embedding models to make them compatible with
LangChain's Embeddings interface.

To be compatible with LangChain tools (like SemanticChunker or VectorStores),
an embedding class must implement the following interface:

1. embed_documents(texts: List[str]) -> List[List[float]]
   - Takes a list of strings and returns a list of embeddings.
2. embed_query(text: str) -> List[float]
   - Takes a single string and returns its embedding.
"""

from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer

class SentenceTransformerWrapper:
    """
    Wrapper for sentence-transformers models.
    Compatible with LangChain's Embeddings interface.
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: Optional[str] = None):
        self.model = SentenceTransformer(model_name, device=device)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        embeddings = self.model.encode(texts)
        if isinstance(embeddings, np.ndarray):
            return embeddings.tolist()
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        embedding = self.model.encode([text])[0]
        if isinstance(embedding, np.ndarray):
            return embedding.tolist()
        return embedding

class FastEmbedWrapper:
    """
    Wrapper for FastEmbed (Qdrant) models for lightweight, fast CPU inference.
    """
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        try:
            from fastembed import TextEmbedding
        except ImportError:
            raise ImportError("Please install fastembed to use this wrapper: pip install fastembed, or uv add fastembed")
        
        self.model = TextEmbedding(model_name=model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        # FastEmbed returns a generator of lists
        return [e.tolist() for e in self.model.embed(texts)]

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        # FastEmbed returns a generator, we take the first item
        return next(self.model.embed([text])).tolist()
    