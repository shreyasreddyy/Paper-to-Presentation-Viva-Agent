"""Ollama client initialization, health checks, and model discovery."""

import urllib.request
import json
from langchain_ollama import ChatOllama, OllamaEmbeddings
from core.config import (
    DEFAULT_CHAT_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_OLLAMA_HOST,
    DEFAULT_NUM_CTX,
    DEFAULT_TEMPERATURE,
)


def check_ollama_health(host: str = DEFAULT_OLLAMA_HOST) -> dict:
    """Check whether local Ollama is reachable and return status info."""
    endpoint = f"{host.rstrip('/')}/api/tags"
    try:
        req = urllib.request.Request(endpoint, headers={"User-Agent": "PaperVivaAgent/1.0"})
        with urllib.request.urlopen(req, timeout=3) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode("utf-8"))
                models = [m.get("name", "") for m in data.get("models", [])]
                return {
                    "connected": True,
                    "models": models,
                    "error": None,
                }
            return {
                "connected": False,
                "models": [],
                "error": f"Ollama returned HTTP {resp.status}",
            }
    except Exception as e:
        return {
            "connected": False,
            "models": [],
            "error": str(e),
        }


def get_llm(
    model_name: str = DEFAULT_CHAT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    num_ctx: int = DEFAULT_NUM_CTX,
    base_url: str = DEFAULT_OLLAMA_HOST,
) -> ChatOllama:
    """Instantiate a ChatOllama client with bounded context window for fast local CPU/GPU execution."""
    return ChatOllama(
        model=model_name,
        temperature=temperature,
        num_ctx=num_ctx,
        base_url=base_url,
    )


def get_embedder(
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    base_url: str = DEFAULT_OLLAMA_HOST,
) -> OllamaEmbeddings:
    """Instantiate OllamaEmbeddings with specified model and host."""
    return OllamaEmbeddings(
        model=model_name,
        base_url=base_url,
    )
