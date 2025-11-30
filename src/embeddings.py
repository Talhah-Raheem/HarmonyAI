"""
Local text embedding generation for HarmonyAI.

This module provides offline text embeddings using HashingVectorizer
as a lightweight alternative to SBERT for fully offline operation.
"""

from typing import List

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize


# Global vectorizer instance for consistent embeddings
_vectorizer = None


def get_vectorizer() -> HashingVectorizer:
    """
    Get or create the global HashingVectorizer instance.

    Returns:
        Configured HashingVectorizer
    """
    global _vectorizer

    if _vectorizer is None:
        _vectorizer = HashingVectorizer(
            n_features=512,
            alternate_sign=False,
            norm=None,  # We'll normalize manually
            ngram_range=(1, 2)
        )

    return _vectorizer


def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Generate embeddings for a list of texts using HashingVectorizer.

    This provides a lightweight, deterministic, fully-offline alternative
    to transformer-based embeddings like SBERT. While not as semantically
    rich, it maintains consistency and works well for lexical matching.

    Args:
        texts: List of text strings to embed

    Returns:
        2D numpy array of shape (len(texts), 512) containing L2-normalized embeddings
    """
    if not texts:
        return np.array([]).reshape(0, 512)

    # Get vectorizer
    vectorizer = get_vectorizer()

    # Transform texts to sparse matrix
    sparse_embeddings = vectorizer.transform(texts)

    # Convert to dense
    dense_embeddings = sparse_embeddings.toarray()

    # L2 normalize each embedding vector
    normalized_embeddings = normalize(dense_embeddings, norm='l2', axis=1)

    return normalized_embeddings


def embed_single(text: str) -> np.ndarray:
    """
    Generate embedding for a single text.

    Args:
        text: Text string to embed

    Returns:
        1D numpy array of length 512
    """
    embeddings = embed_texts([text])
    return embeddings[0]
