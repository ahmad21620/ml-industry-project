# agents/user_memory_agent.py

from __future__ import annotations

import json
import re
from typing import List, Optional

from config import llm
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ValidationError


class MemoryUpdateDecision(BaseModel):
    action: str  # "add" | "update" | "delete" | "none"
    old_fact: Optional[str] = None
    new_fact: Optional[str] = None
    reasoning: str


class UserMemoryAgent:
    SYSTEM_PROMPT = (
        "You are a user memory curator for an AWS billing assistant. "
        "Your task is to maintain DURABLE user facts that can assist in various questions. "
        "NEVER store transient info (questions, emotions, ticket details). "
        "Only act on explicit, stable statements. "
        "the new added fact should be like this: user lives in India "
        "Respond ONLY with a valid JSON object matching this schema:\n"
        '{"action": "add|update|delete|none", "old_fact": "...", "new_fact": "...", "reasoning": "..."}\n'
        "Do NOT add markdown, explanations, or extra text. ONLY JSON."
    )

    def __init__(self, llm_client: ChatOpenAI = llm):
        # ⚠️ Do NOT use with_structured_output — your LLM doesn't support it
        self.llm = llm_client.bind(temperature=0.0)

    def _extract_clean_json(self, text: str) -> dict:
        """Extract JSON from LLM output, even if wrapped in markdown or extra text."""
        # Remove common markdown fences
        text = re.sub(r"```(?:json)?", "", text)
        text = text.strip()

        # Try direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to find {...} block
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        raise ValueError("No valid JSON found")

    def propose_update(
        self,
        user_message: str,
        existing_facts: List[str]
    ) -> Optional[MemoryUpdateDecision]:
        facts_text = "\n".join(f"- {f}" for f in existing_facts) if existing_facts else "None"

        messages = [
            SystemMessage(content=self.SYSTEM_PROMPT),
            HumanMessage(content=(
                f"Existing user facts:\n{facts_text}\n\n"
                f"Latest user message:\n{user_message}\n\n"
                "What memory update should be made? Respond ONLY with JSON."
            ))
        ]

        try:
            response = self.llm.invoke(messages)
            raw_text = response.content.strip()
            print(raw_text)

            # Debug: print raw LLM output
            # print("LLM RAW OUTPUT:", repr(raw_text))

            json_dict = self._extract_clean_json(raw_text)
            return MemoryUpdateDecision(**json_dict)

        except (json.JSONDecodeError, ValidationError, ValueError, KeyError) as e:
            # Optionally log: print(f"MemoryAgent parse error: {e}")
            return None

    def apply_update(
        self,
        existing_facts: List[str],
        decision: MemoryUpdateDecision
    ) -> List[str]:
        facts = [f.strip() for f in existing_facts if f.strip()]

        if decision.action == "none":
            return facts

        if decision.action == "add" and decision.new_fact:
            if decision.new_fact not in facts:
                facts.append(decision.new_fact)

        elif decision.action == "update" and decision.old_fact and decision.new_fact:
            facts = [f for f in facts if f != decision.old_fact]
            if decision.new_fact not in facts:
                facts.append(decision.new_fact)

        elif decision.action == "delete" and decision.old_fact:
            facts = [f for f in facts if f != decision.old_fact]

        return sorted(set(facts))

    def update_memory(
        self,
        user_message: str,
        existing_facts: List[str]
    ) -> List[str]:
        decision = self.propose_update(user_message, existing_facts)
        if decision is None:
            return existing_facts
        return self.apply_update(existing_facts, decision)