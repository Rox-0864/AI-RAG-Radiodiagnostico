"""
LLM interface using Ollama.

Connects to a local Ollama instance running llama3.2:3b.
The model runs entirely on CPU with ~3-4GB RAM usage.
"""

import logging
from langchain_ollama import OllamaLLM

logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3.2:3b"

# Singleton to avoid reconnecting
_llm_instance = None


def get_llm(
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    temperature: float = 0.2,
) -> OllamaLLM:
    """
    Get or create the Ollama LLM instance.

    Low temperature (0.2) ensures factual, consistent answers
    based on the retrieved context rather than creative generation.

    Args:
        model: Ollama model name (must be pulled first with 'ollama pull').
        base_url: Ollama server URL.
        temperature: Controls randomness (0.0 = deterministic, 1.0 = creative).

    Returns:
        Configured OllamaLLM instance.
    """
    global _llm_instance

    if _llm_instance is not None:
        return _llm_instance

    logger.info(f"Connecting to Ollama: {base_url} (model: {model})")
    _llm_instance = OllamaLLM(
        model=model,
        base_url=base_url,
        temperature=temperature,
        num_ctx=4096,       # Context window: 4K tokens (enough for chunks + history)
        num_predict=1024,   # Max output tokens
    )
    logger.info("Ollama LLM ready")
    return _llm_instance
