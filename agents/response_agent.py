from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from config import llm
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from rag.faiss_store import RAGAgent, RetrievedChunk


@dataclass
class SourceCitation:
    source_file: str
    section_heading: str
    chunk_id: int
    score: float


@dataclass
class Answer:
    answer_text: str
    citations: List[SourceCitation]


class ResponseAgent:
    """
    Uses:
    - RAGAgent to retrieve relevant context from AWS billing docs.
    - LLM (ChatOpenAI-compatible endpoint) to generate the answer.
    - Optional user memory for personalization.
    """

    SYSTEM_PROMPT = (
        "You are an AWS billing support assistant. "
        "Answer user questions using ONLY the provided context AND the known user information. "
        "Personalize your response based on the user's profile if relevant. "
        "If the context is not sufficient to answer reliably, "
        "say that you don't know and recommend contacting AWS Support "
        "or checking the AWS Billing and Cost Management documentation.\n\n"
        "Guidelines:\n"
        "- Be concise and precise.\n"
        "- Do not invent policies, features, or user details not present in the context or user memory.\n"
        "- If multiple possibilities exist, explain them clearly.\n"
        "- Use the user's known context (e.g., location, language preference) to tailor answers when appropriate."
    )

    def __init__(self, rag_agent: RAGAgent, llm_client: ChatOpenAI = llm) -> None:
        self.rag_agent = rag_agent
        self.llm = llm_client

    def answer(
        self,
        question: str,
        k: int = 5,
        chat_history: Optional[List[Tuple[str, str]]] = None,
        user_memory: str = "",  # ← NEW PARAMETER
    ) -> Answer:
        """
        High-level call:
        - retrieve context with RAG
        - call LLM with system + user memory + history + context + question
        - return answer text + structured citations
        """
        retrieved_chunks: List[RetrievedChunk] = self.rag_agent.retrieve(
            query=question, k=k
        )

        context_block = self._format_context(retrieved_chunks)

        # Format recent conversation history (short-term memory)
        history_text = ""
        if chat_history:
            history_lines = []
            for role, content in chat_history:
                prefix = "User" if role == "user" else "Assistant"
                history_lines.append(f"{prefix}: {content}")
            history_text = "\n".join(history_lines)

        user_prompt_parts = [
            "User question:",
            question,
            "",
        ]

        # Inject user memory if available
        if user_memory.strip():
            user_prompt_parts.extend([
                "Known information about this user:",
                user_memory,
                "",
            ])

        if history_text:
            user_prompt_parts.extend([
                "Recent conversation history:",
                history_text,
                "",
            ])

        user_prompt_parts.extend([
            "Context from AWS documentation and past tickets:",
            context_block,
            "",
            "Use ONLY this context, user memory, and conversation history to answer.",
        ])

        messages = [
            SystemMessage(content=self.SYSTEM_PROMPT),
            HumanMessage(content="\n".join(user_prompt_parts)),
        ]

        llm_response = self.llm.invoke(messages)

        citations = [
            SourceCitation(
                source_file=chunk.source_file,
                section_heading=chunk.section_heading,
                chunk_id=chunk.chunk_id,
                score=chunk.score,
            )
            for chunk in retrieved_chunks
        ]

        return Answer(
            answer_text=str(llm_response.content).strip(),
            citations=citations
        )

    @staticmethod
    def _format_context(chunks: List[RetrievedChunk]) -> str:
        """
        Prepare a readable, labeled context section for the LLM.
        """
        if not chunks:
            return "No relevant documentation found."

        parts = []
        for idx, chunk in enumerate(chunks):
            header = (
                f"[DOC {idx}] file={chunk.source_file}, "
                f"section='{chunk.section_heading}', "
                f"score={chunk.score:.3f}"
            )
            parts.append(header)
            parts.append(chunk.content)
            parts.append("\n---\n")
        return "\n".join(parts)