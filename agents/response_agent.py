from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from config import llm
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from rag.faiss_store import RAGAgent, RetrievedChunk
from rag.knowledge_graph_agent import KnowledgeGraphAgent, KGRetrievedChunk


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
    - KnowledgeGraphAgent to retrieve context from Neo4j when RAG is weak/unsure.
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

    # If the best RAG similarity score is below this, we consider RAG "weak"
    RAG_MIN_SCORE = 0.8

    def __init__(
        self,
        rag_agent: RAGAgent,
        kg_agent: Optional[KnowledgeGraphAgent] = None,
        llm_client: ChatOpenAI = llm,
    ) -> None:
        self.rag_agent = rag_agent
        self.kg_agent = kg_agent
        self.llm = llm_client

    def answer(
        self,
        question: str,
        k: int = 5,
        chat_history: Optional[List[Tuple[str, str]]] = None,
        user_memory: str = "",
    ) -> Answer:
        """
        High-level call:
        - retrieve context (RAG first, optionally fall back to KG)
        - call LLM with system + user memory + history + context + question
        - return answer text + structured citations
        """

        # Decide context source & build citations
        context_block, citations = self._get_best_context_and_citations(
            question=question,
            k=k,
        )

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
            user_prompt_parts.extend(
                [
                    "Known information about this user:",
                    user_memory,
                    "",
                ]
            )

        if history_text:
            user_prompt_parts.extend(
                [
                    "Recent conversation history:",
                    history_text,
                    "",
                ]
            )

        user_prompt_parts.extend(
            [
                "Context from AWS documentation and/or knowledge graph:",
                context_block,
                "",
                "Use ONLY this context, user memory, and conversation history to answer.",
            ]
        )

        print("=== FULL PROMPT (user memory only for debug) ===")
        print(user_memory)
        print("=== END PROMPT ===")

        messages = [
            SystemMessage(content=self.SYSTEM_PROMPT),
            HumanMessage(content="\n".join(user_prompt_parts)),
        ]

        llm_response = self.llm.invoke(messages)

        return Answer(
            answer_text=str(llm_response.content).strip(),
            citations=citations,
        )

    # --------- CONTEXT SELECTION LOGIC ---------

    def _get_best_context_and_citations(
        self,
        question: str,
        k: int,
    ) -> Tuple[str, List[SourceCitation]]:
        """
        1) Try RAG (FAISS) first.
        2) If RAG is empty or the best score is below threshold AND we have a KG agent,
           try KnowledgeGraphAgent.
        3) If KG also fails or is not available, fall back to whatever RAG returned.
        Returns:
            - context_block: str
            - citations: List[SourceCitation]
        """

        # ---- Step 1: RAG retrieval (primary) ----
        rag_chunks: List[RetrievedChunk] = self.rag_agent.retrieve(
            query=question,
            k=k,
        )

        best_rag_score = max((c.score for c in rag_chunks), default=0.0)
        rag_strong_enough = bool(rag_chunks) and best_rag_score >= self.RAG_MIN_SCORE
        # If RAG is clearly good or we don't have KG at all, just use RAG
        if rag_strong_enough or self.kg_agent is None:
            print("I am in the RAG")
            context_block = self._format_rag_context(rag_chunks)
            citations = [
                SourceCitation(
                    source_file=chunk.source_file,
                    section_heading=chunk.section_heading,
                    chunk_id=chunk.chunk_id,
                    score=chunk.score,
                )
                for chunk in rag_chunks
            ]
            return context_block, citations

        # ---- Step 2: RAG is weak → try KG if available ----
        kg_chunks: List[KGRetrievedChunk] = self.kg_agent.retrieve(question, k=k)
        if kg_chunks:
            # Use KG result as main context
            context_block = self._format_kg_context(kg_chunks)
            citations = [
                SourceCitation(
                    source_file=chunk.source,
                    section_heading="",  # KG doesn't track section headings by default
                    chunk_id=chunk.index,
                    score=(chunk.enhanced_score or chunk.semantic_score or 0.0),
                )
                for chunk in kg_chunks
            ]
            return context_block, citations

        # ---- Step 3: KG also failed → fall back to whatever RAG gave ----
        context_block = self._format_rag_context(rag_chunks)
        citations = [
            SourceCitation(
                source_file=chunk.source_file,
                section_heading=chunk.section_heading,
                chunk_id=chunk.chunk_id,
                score=chunk.score,
            )
            for chunk in rag_chunks
        ]
        return context_block, citations

    # --------- CONTEXT FORMATTERS ---------

    @staticmethod
    def _format_rag_context(chunks: List[RetrievedChunk]) -> str:
        """
        Prepare a readable, labeled context section for the LLM (FAISS / RAG).
        """
        if not chunks:
            return "No relevant documentation found in the FAISS index."

        parts = []
        for idx, chunk in enumerate(chunks):
            header = (
                f"[RAG DOC {idx}] file={chunk.source_file}, "
                f"section='{chunk.section_heading}', "
                f"score={chunk.score:.3f}"
            )
            parts.append(header)
            parts.append(chunk.content)
            parts.append("\n---\n")
        return "\n".join(parts)

    @staticmethod
    def _format_kg_context(chunks: List[KGRetrievedChunk]) -> str:
        """
        Prepare a readable, labeled context section for the LLM (Knowledge Graph).
        """
        if not chunks:
            return "No relevant information found in the knowledge graph."

        parts = []
        for idx, chunk in enumerate(chunks):
            score = chunk.enhanced_score if chunk.enhanced_score is not None else (
                chunk.semantic_score or 0.0
            )
            header = (
                f"[KG DOC {idx}] source={chunk.source}, "
                f"index={chunk.index}, "
                f"score={score:.3f}"
            )
            parts.append(header)
            parts.append(chunk.text)

            if chunk.context_preview:
                parts.append(f"Related context: {chunk.context_preview}")

            parts.append("\n---\n")
        return "\n".join(parts)
