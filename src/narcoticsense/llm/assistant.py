from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ScientificAssistant:
    """Retrieval-grounded assistant interface.

    Production implementations should connect this class to a vetted LLM provider,
    a document retriever, citation tracking, and safety filters.
    """

    system_prompt: str = "You are a cautious spectroscopy research assistant."

    def answer(self, question: str, context: str) -> str:
        return (
            "LLM backend not configured. Question received: "
            f"{question!r}. Retrieved context length: {len(context)} characters."
        )
