# safety.py
from typing import Final

INJECTION_KEYWORDS: Final[list[str]] = [
    "ignore previous instructions",
    "disregard previous instructions",
    "forget previous instructions",
    "forget all previous instructions",
    "you are now",
    "from now on you are",
    "change your rules",
    "redefine your system prompt",
    "act as a different assistant",
    "pretend you are not",
]

def looks_like_prompt_injection(text: str) -> bool:
    lower = text.lower()
    return any(kw in lower for kw in INJECTION_KEYWORDS)
