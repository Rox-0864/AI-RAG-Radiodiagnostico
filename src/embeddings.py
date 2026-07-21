"""
Embedding model wrapper using HuggingFace sentence-transformers.

Uses all-MiniLM-L6-v2: 90MB model, 384-dimension vectors,
runs efficiently on CPU without GPU.
"""

import logging
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

# Default embedding model — lightweight, fast, CPU-friendly
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Singleton instance to avoid reloading the model
_embedding_model = None


def get_embeddings(model_name: str = DEFAULT_MODEL) -> HuggingFaceEmbeddings:
    """
    Get or create the HuggingFace embedding model.

    The model is loaded once and reused. On first call it downloads
    ~90MB from HuggingFace Hub (cached locally for subsequent runs).

    Args:
        model_name: HuggingFace model identifier.

    Returns:
        Configured HuggingFaceEmbeddings instance.
    """
    global _embedding_model

    if _embedding_model is not None:
        return _embedding_model

    logger.info(f"Loading embedding model: {model_name}")
    _embedding_model = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    logger.info("Embedding model loaded successfully")
    return _embedding_model
