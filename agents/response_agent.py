from __future__ import annotations

from dataclasses import dataclass
from typing import List

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

from rag.faiss_store import RAGAgent, RetrievedChunk
from config import llm


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
    """

    SYSTEM_PROMPT = (
        "You are an AWS billing support assistant. "
        "Answer user questions using ONLY the provided context. "
        "If the context is not sufficient to answer reliably, "
        "say that you don't know and recommend contacting AWS Support "
        "or checking the AWS Billing and Cost Management documentation.\n\n"
        "Guidelines:\n"
        "- Be concise and precise.\n"
        "- Do not invent policies or features not present in the context.\n"
        "- If multiple possibilities exist, explain them clearly."
    )

    def __init__(self, rag_agent: RAGAgent, llm_client: ChatOpenAI = llm) -> None:
        self.rag_agent = rag_agent
        self.llm = llm_client

    def answer(self, question: str, k: int = 5) -> Answer:
        """
        High-level call:
        - retrieve context with RAG
        - call LLM with system + context + question
        - return answer text + structured citations
        """
        retrieved_chunks: List[RetrievedChunk] = self.rag_agent.retrieve(
            query=question, k=k
        )

        context_block = self._format_context(retrieved_chunks)

        messages = [
            SystemMessage(content=self.SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    "User question:\n"
                    f"{question}\n\n"
                    "Context from AWS documentation and past tickets:\n"
                    f"{context_block}\n\n"
                    "Use only this context to answer."
                )
            ),
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

        return Answer(answer_text=llm_response.content, citations=citations)

    @staticmethod
    def _format_context(chunks: List[RetrievedChunk]) -> str:
        """
        Prepare a readable, labeled context section for the LLM.
        """
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
