# agents/user_memory_agent.py

from __future__ import annotations

import json
import re
from typing import List, Optional

from config import llm
from db import UserMemory
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ValidationError
from sqlmodel import Session


class MemoryUpdateInstruction(BaseModel):
    action: str  # "add" | "update" | "delete" | "none"
    index: Optional[int] = None  # 1-based index of existing fact (for update/delete)
    new_fact: Optional[str] = None  # for add/update
    reasoning: str


class UserMemoryAgent:
    SYSTEM_PROMPT = (
        "You are a user memory curator for an AWS billing assistant. "
        "You will receive a numbered list of the user's current memory facts (1, 2, 3...). "
        "Analyze the new user message and decide how to update the memory.\n\n"

        "RULES:\n"
        "- Only store DURABLE facts: location, language, job role, company, timezone.\n"
        "- NEVER store: questions, emotions, temporary info, or ticket details.\n"
        "- Be conservative: only act on explicit, stable statements.\n"
        "- Each fact should be a short, clear sentence like: 'user is located in India'.\n\n"

        "ACTIONS:\n"
        '- "add": provide "new_fact"\n'
        '- "update": provide "index" (of fact to replace) and "new_fact"\n'
        '- "delete": provide "index" (of fact to remove)\n'
        '- "none": do nothing (omit other fields)\n\n'

        "Output a JSON ARRAY of instruction objects. "
        "Use 1-based indices from the provided list. "
        "Respond ONLY with valid JSON. NO markdown, NO extra text."
            "Example of one element(valid):\n"
    '{"action": "add", "new_fact": "user is located in UK", "reasoning": "User stated they are in the UK."}\n\n'

    )

    def __init__(self, llm_client: ChatOpenAI = llm):
        self.llm = llm_client.bind(temperature=0.0)

    def _extract_clean_json_array(self, text: str) -> List[dict]:
        text = re.sub(r"```(?:json)?", "", text).strip()
        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, list) else [parsed]
        except json.JSONDecodeError:
            pass
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        raise ValueError("No valid JSON array found")

    def _parse_instructions(
        self,
        user_message: str,
        existing_memories: List[UserMemory]
    ) -> Optional[List[MemoryUpdateInstruction]]:
        # Format numbered list: "1: user lives in India"
        if existing_memories:
            facts_numbered = "\n".join(
                f"{i+1}: {mem.fact}" for i, mem in enumerate(existing_memories)
            )
        else:
            facts_numbered = "None"

        messages = [
            SystemMessage(content=self.SYSTEM_PROMPT),
            HumanMessage(content=(
                f"Current user memory (numbered):\n{facts_numbered}\n\n"
                f"New user message:\n{user_message}\n\n"
                "What memory updates should be made? Respond ONLY with a JSON array."
            ))
        ]

        # try:
        response = self.llm.invoke(messages)
        print(response)
        raw_text = response.content.strip()
        json_list = self._extract_clean_json_array(raw_text)
        return [MemoryUpdateInstruction(**item) for item in json_list]
        # except (json.JSONDecodeError, ValidationError, ValueError, KeyError, TypeError):
            # return None

    def update_memory_in_db(
        self,
        user_id: int,
        user_message: str,
        session: Session
    ) -> bool:
        """
        Full memory update cycle:
        1. Fetch current memories from DB
        2. Get LLM instructions
        3. Apply changes to DB
        4. Commit transaction
        Returns True if changes were made, False otherwise.
        """
        # Fetch current memories (ordered by id for stable indexing)
        current_memories = session.query(UserMemory).filter(
            UserMemory.user_id == user_id
        ).order_by(UserMemory.id).all()

        instructions = self._parse_instructions(user_message, current_memories)
        if not instructions:
            return False  # No valid instructions → no change

        changed = False
        # Track which memory objects to delete (to avoid modifying list during iteration)
        to_delete = []

        for instr in instructions:
            if instr.action == "none":
                continue

            elif instr.action == "add":
                if instr.new_fact:
                    # Avoid duplicates
                    if not any(m.fact == instr.new_fact for m in current_memories):
                        new_mem = UserMemory(user_id=user_id, fact=instr.new_fact.strip())
                        session.add(new_mem)
                        current_memories.append(new_mem)
                        changed = True

            elif instr.action == "update":
                if instr.index is not None and instr.new_fact:
                    idx = instr.index - 1  # Convert 1-based to 0-based
                    if 0 <= idx < len(current_memories):
                        old_mem = current_memories[idx]
                        if old_mem.fact != instr.new_fact:  # Avoid no-op updates
                            old_mem.fact = instr.new_fact.strip()
                            session.add(old_mem)
                            changed = True

            elif instr.action == "delete":
                if instr.index is not None:
                    idx = instr.index - 1
                    if 0 <= idx < len(current_memories):
                        mem_to_del = current_memories.pop(idx)
                        to_delete.append(mem_to_del)
                        changed = True

        # Perform deletions after loop
        for mem in to_delete:
            session.delete(mem)

        if changed:
            session.commit()
        return changed