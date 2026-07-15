"""LLM factory using Ollama (local, no API key)."""

from langchain_ollama import ChatOllama
from .config import LLM_MODEL, LLM_TEMPERATURE

_llm = None


def get_llm(temperature: float | None = None):
    global _llm
    if _llm is None:
        _llm = ChatOllama(
            model=LLM_MODEL,
            temperature=temperature if temperature is not None else LLM_TEMPERATURE,
        )
    return _llm


def invoke_llm(messages: list[tuple[str, str]], temperature: float | None = None) -> str:
    llm = get_llm(temperature)
    response = llm.invoke(messages)
    return response.content
