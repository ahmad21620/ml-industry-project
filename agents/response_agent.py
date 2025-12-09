from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import logging

from config import llm
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from rag.faiss_store import RAGAgent, RetrievedChunk
from rag.knowledge_graph_agent import KnowledgeGraphAgent, KGRetrievedChunk

from datetime import datetime
import re

from tools import (
    CurrencyFXTool,
    CurrencyCalculatorTool,
    FXRateResult,
    FXAPIError,
)

from tracing import TraceBuilder

logger = logging.getLogger(__name__)


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


@dataclass
class CurrencyConversionResult:
    amount: float
    source_currency: str
    target_currency: str
    rate: float
    converted_amount: float
    rate_source: str
    rate_timestamp: datetime
    # Optional: user-supplied implied rate, never used for the calculation itself.
    user_supplied_rate: Optional[float] = None

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
        "- Use the user's known context (e.g., location, language preference) to tailor answers when appropriate.\n"
        "- For any currency conversion, always use the system's currency tools to obtain exchange rates; "
        "never rely on user-provided exchange rates for the actual calculation.\n"
        "- If the user provides an exchange rate, you may mention it and compare it to the tool-based rate, "
        "but the computation must always use the rate returned by the currency tools.\n"
        "- If live exchange rates are unavailable or an error occurs, clearly state that you cannot safely perform "
        "the conversion instead of guessing or using user-provided rates."
    )

    # If the best RAG similarity score is below this, we consider RAG "weak"
    RAG_MIN_SCORE = 0.8

    def __init__(
        self,
        rag_agent: RAGAgent,
        kg_agent: Optional[KnowledgeGraphAgent] = None,
        llm_client: ChatOpenAI = llm,
        currency_fx_tool: Optional[CurrencyFXTool] = None,
        currency_calculator_tool: Optional[CurrencyCalculatorTool] = None,
    ) -> None:
        self.rag_agent = rag_agent
        self.kg_agent = kg_agent
        self.llm = llm_client

        # Optional tools for currency conversion; if not provided,
        # currency-specific logic will be skipped.
        self.currency_fx_tool = currency_fx_tool
        self.currency_calculator_tool = currency_calculator_tool

    def answer(
        self,
        question: str,
        k: int = 5,
        chat_history: Optional[List[Tuple[str, str]]] = None,
        user_memory: str = "",
        trace_builder: Optional[TraceBuilder] = None,
    ) -> Answer:
        """
        High-level call:
        - retrieve context (RAG first, optionally fall back to KG)
        - call LLM with system + user memory + history + context + question
        - optionally update a TraceBuilder with metadata about the tools/agents used
        - return answer text + structured citations
        """

        # First, check if this is a direct currency conversion request that can be
        # handled entirely by the currency tools. If so, bypass RAG/KG.
        try:
            conversion_result = self._handle_currency_conversion(question)
        except FXAPIError as exc:
            # The question looks like a currency conversion request, but we could
            # not obtain a reliable live FX rate from the external API.
            if trace_builder is not None:
                trace_builder.add_error(
                    component="CurrencyFXTool",
                    type="FXAPIError",
                    message=str(exc) or "Failed to obtain live FX rate.",
                )

            error_message_lines = [
                "You asked for a currency conversion, but live exchange rates "
                "are temporarily unavailable.",
                "Because I cannot obtain a reliable rate from the currency tool, "
                "I cannot safely perform this conversion right now.",
            ]
            return Answer(
                answer_text="\n".join(error_message_lines),
                citations=[],
            )

        if conversion_result is not None:
            logger.info(
                "Direct currency conversion handled in ResponseAgent.answer: "
                "%s %s -> %s (rate=%f, source=%s)",
                conversion_result.amount,
                conversion_result.source_currency,
                conversion_result.target_currency,
                conversion_result.rate,
                conversion_result.rate_source,
            )

            # Record this in the trace, if available.
            if trace_builder is not None:
                trace_builder.set_currency_conversion(
                    mode="direct",
                    amount=conversion_result.amount,
                    source_currency=conversion_result.source_currency,
                    target_currency=conversion_result.target_currency,
                    rate=conversion_result.rate,
                    rate_source=conversion_result.rate_source,
                    rate_timestamp_utc=conversion_result.rate_timestamp,
                    converted_amount=conversion_result.converted_amount,
                    user_supplied_rate=conversion_result.user_supplied_rate,
                    fx_error=False,
                )
                # Mark ResponseAgent as invoked, with no RAG/KG context.
                trace_builder.mark_response_context(
                    source="NONE",
                    rag_used=False,
                    kg_used=False,
                    notes="Direct currency conversion; no RAG/KG retrieval used.",
                )

            # Build a direct answer using the conversion result.
            rate_str = f"{conversion_result.rate:.6f}"
            amount_str = f"{conversion_result.amount:.2f}"
            converted_str = f"{conversion_result.converted_amount:.2f}"
            rate_time_str = conversion_result.rate_timestamp.isoformat()

            answer_lines = [
                (
                    "Using a live exchange rate of "
                    f"1 {conversion_result.source_currency} = "
                    f"{rate_str} {conversion_result.target_currency} "
                    f"(source: {conversion_result.rate_source}, "
                    f"fetched at {rate_time_str} UTC),"
                ),
                (
                    f"{amount_str} {conversion_result.source_currency} "
                    f"is approximately {converted_str} "
                    f"{conversion_result.target_currency}."
                ),
            ]

            # Optionally mention a user-supplied rate, but never use it
            # for the actual computation.
            if conversion_result.user_supplied_rate is not None:
                user_rate_str = f"{conversion_result.user_supplied_rate:.6f}"
                logger.info(
                    "User-supplied rate detected and ignored for computation: "
                    "user_rate=%s (tool_rate=%s)",
                    user_rate_str,
                    rate_str,
                )
                answer_lines.append(
                    "Note: you mentioned an exchange rate of "
                    f"{user_rate_str} {conversion_result.target_currency} "
                    f"per 1 {conversion_result.source_currency}, but the "
                    "calculation above uses the official rate from the "
                    "currency tool."
                )

            return Answer(
                answer_text="\n".join(answer_lines),
                citations=[],
            )

        # Decide context source & build citations
        context_block, citations = self._get_best_context_and_citations(
            question=question,
            k=k,
            trace_builder=trace_builder,
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

        raw_answer_text = str(llm_response.content).strip()
        final_answer_text = self._maybe_enhance_answer_with_currency_conversion(
            question=question,
            answer_text=raw_answer_text,
            trace_builder=trace_builder,
        )

        return Answer(
            answer_text=final_answer_text,
            citations=citations,
        )

    def _maybe_enhance_answer_with_currency_conversion(
        self,
        question: str,
        answer_text: str,
        trace_builder: Optional[TraceBuilder] = None,
    ) -> str:
        """Append a currency conversion to a normal AWS billing answer when requested.

        This is used for questions like:
            - "Show my AWS bill for last month in EUR"
        where the main answer is produced via RAG/KG, and we then convert a single
        amount from the answer into the requested currency.
        """
        # Tools must be available.
        if not self.currency_fx_tool or not self.currency_calculator_tool:
            return answer_text

        if not question or not answer_text:
            return answer_text

        # Detect a target currency in the question, e.g. "in EUR" or "to SAR".
        target_pattern = r"(?:in|to)\s+(?P<code>[A-Za-z]{3})\b"
        target_match = re.search(target_pattern, question, flags=re.IGNORECASE)
        if not target_match:
            return answer_text

        target_currency = target_match.group("code").upper()

        # Guard: direct conversion requests like "100 USD to EUR" are already fully
        # handled earlier in answer(), so we avoid duplicating anything here.
        direct_pattern = r"\d+(?:\.\d+)?\s*[A-Za-z]{3}\s*(?:to|in)\s*[A-Za-z]{3}"
        if re.search(direct_pattern, question, flags=re.IGNORECASE):
            return answer_text

        # Extract exactly one amount + currency pair from the answer text,
        # such as "123.45 USD". We avoid guessing if multiple amounts appear.
        amount_currency_pattern = (
            r"(?P<amount>\d[\d,]*(?:\.\d+)?)\s*"
            r"(?P<code>[A-Za-z]{3})\b"
        )
        matches = list(
            re.finditer(amount_currency_pattern, answer_text, flags=re.IGNORECASE)
        )
        if len(matches) != 1:
            # Either none or multiple; do not guess which one to convert.
            return answer_text

        m = matches[0]
        amount_str_raw = m.group("amount")
        source_currency = m.group("code").upper()

        # If the source currency already equals the requested target, no conversion.
        if source_currency == target_currency:
            return answer_text

        # Normalize amount (remove thousands separators) and validate.
        try:
            amount_str_clean = amount_str_raw.replace(",", "")
            amount = float(amount_str_clean)
        except ValueError:
            return answer_text

        if amount <= 0:
            return answer_text

        # Attempt to fetch live FX rate and compute converted amount.
        try:
            fx_result = self.currency_fx_tool.get_rate(
                source_currency,
                target_currency,
            )
            converted_amount = self.currency_calculator_tool.convert_amount(
                amount=amount,
                rate=fx_result.rate,
            )
        except FXAPIError as exc:
            # Record FX API error in the trace (non-fatal for the AWS answer).
            if trace_builder is not None:
                trace_builder.add_error(
                    component="CurrencyFXTool",
                    type="FXAPIError",
                    message=str(exc) or "Failed to obtain live FX rate.",
                )
            return answer_text
        except ValueError:
            # Numerical issue; do not modify the original answer.
            return answer_text

        rate_str = f"{fx_result.rate:.6f}"
        amount_str_fmt = f"{amount:.2f}"
        converted_str = f"{converted_amount:.2f}"
        rate_time_str = fx_result.fetched_at.isoformat()

        extra_lines = [
            "",
            "",
            "Currency conversion (based on live exchange rates):",
            (
                "Using a live exchange rate of "
                f"1 {source_currency} = {rate_str} {target_currency} "
                f"(source: {fx_result.source}, fetched at {rate_time_str} UTC),"
            ),
            (
                f"your total of {amount_str_fmt} {source_currency} is approximately "
                f"{converted_str} {target_currency}."
            ),
        ]

        logger.info(
            "AWS bill answer enhanced with currency conversion: "
            "%s %s -> %s (rate=%f, source=%s)",
            amount,
            source_currency,
            target_currency,
            fx_result.rate,
            fx_result.source,
        )

        # Record this enhancement in the trace, if available.
        if trace_builder is not None:
            trace_builder.set_currency_conversion(
                mode="aws_bill_enhancement",
                amount=amount,
                source_currency=source_currency,
                target_currency=target_currency,
                rate=fx_result.rate,
                rate_source=fx_result.source,
                rate_timestamp_utc=fx_result.fetched_at,
                converted_amount=converted_amount,
                user_supplied_rate=None,
                fx_error=False,
            )

        return answer_text + "\n".join(extra_lines)

    def _handle_currency_conversion(
        self,
        question: str,
    ) -> Optional[CurrencyConversionResult]:
        """Try to interpret the question as a currency conversion request.

        If successful and tools are available, use the FX API tool and the
        calculator tool to compute the converted amount.

        Returns:
            CurrencyConversionResult on success, or None if the question is
            not a supported conversion query.

        Raises:
            FXAPIError: if the question is a conversion request but obtaining
                a reliable FX rate from the external API fails.
        """
        # If tools are not wired, skip conversion handling.
        if not self.currency_fx_tool or not self.currency_calculator_tool:
            return None

        parsed = self._parse_currency_conversion_request(question)
        if not parsed:
            return None

        amount, source_currency, target_currency, user_supplied_rate = parsed

        # Let FXAPIError propagate to the caller so it can decide how to
        # inform the user about live FX unavailability.
        fx_result: FXRateResult = self.currency_fx_tool.get_rate(
            source_currency,
            target_currency,
        )

        converted_amount = self.currency_calculator_tool.convert_amount(
            amount=amount,
            rate=fx_result.rate,
        )

        return CurrencyConversionResult(
            amount=amount,
            source_currency=source_currency,
            target_currency=target_currency,
            rate=fx_result.rate,
            converted_amount=converted_amount,
            rate_source=fx_result.source,
            rate_timestamp=fx_result.fetched_at,
            user_supplied_rate=user_supplied_rate,
        )


    @staticmethod
    def _parse_currency_conversion_request(
        question: str,
    ) -> Optional[Tuple[float, str, str, Optional[float]]]:
        """Attempt to parse a simple currency conversion request from text.

        Supported pattern examples:
            "Convert 120 USD to EUR"
            "How much is 500 sar in usd?"

        Returns:
            (amount, source_currency, target_currency, user_supplied_rate)
            or None if parsing fails.
        """
        if not question:
            return None

        text = question.strip()

        # Basic pattern: "<amount> <SRC> to <DST>" or "<amount> <SRC> in <DST>"
        amount_pattern = (
            r"(?P<amount>\d+(?:\.\d+)?)\s*"
            r"(?P<src>[A-Za-z]{3})\s*"
            r"(?:to|in)\s*"
            r"(?P<dst>[A-Za-z]{3})"
        )
        match = re.search(amount_pattern, text, flags=re.IGNORECASE)
        if not match:
            return None

        try:
            amount = float(match.group("amount"))
        except ValueError:
            return None

        source_currency = match.group("src").upper()
        target_currency = match.group("dst").upper()

        user_supplied_rate: Optional[float] = None

        # Optional pattern for user-supplied rate, e.g. "1 USD = 5 SAR".
        rate_pattern = (
            r"(?P<amount1>\d+(?:\.\d+)?)\s*"
            r"(?P<code1>[A-Za-z]{3})\s*=\s*"
            r"(?P<amount2>\d+(?:\.\d+)?)\s*"
            r"(?P<code2>[A-Za-z]{3})"
        )
        rate_match = re.search(rate_pattern, text, flags=re.IGNORECASE)
        if rate_match:
            try:
                amount1 = float(rate_match.group("amount1"))
                amount2 = float(rate_match.group("amount2"))
                code1 = rate_match.group("code1").upper()
                code2 = rate_match.group("code2").upper()

                # Only record the user-supplied rate if the pair matches the
                # parsed source/target currencies. This rate is NEVER used
                # for the actual computation, only for transparency later.
                if amount1 > 0 and code1 == source_currency and code2 == target_currency:
                    user_supplied_rate = amount2 / amount1
                elif amount2 > 0 and code1 == target_currency and code2 == source_currency:
                    user_supplied_rate = amount1 / amount2
            except ValueError:
                user_supplied_rate = None

        return amount, source_currency, target_currency, user_supplied_rate

    # --------- CONTEXT SELECTION LOGIC ---------

    def _get_best_context_and_citations(
        self,
        question: str,
        k: int,
        trace_builder: Optional[TraceBuilder] = None,
    ) -> Tuple[str, List[SourceCitation]]:
        """
        1) Try RAG (FAISS) first.
        2) If RAG is empty or the best score is below threshold AND we have a KG agent,
           try KnowledgeGraphAgent.
        3) If KG also fails or is not available, fall back to whatever RAG returned.

        When a TraceBuilder is provided, this method also records:
            - which context source was effectively used (RAG / KG / RAG+KG / NONE)
            - basic RAG/KG scores and k values
            - which tools were used (RAG_FAISS, KnowledgeGraph_Neo4j)
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

            if trace_builder is not None:
                trace_builder.add_tool("RAG_FAISS")
                trace_builder.mark_response_context(
                    source="RAG" if rag_chunks else "NONE",
                    rag_used=bool(rag_chunks),
                    rag_k=k,
                    rag_top_score=best_rag_score if rag_chunks else None,
                    kg_used=False,
                    notes=(
                        "Using RAG only; KG not configured."
                        if self.kg_agent is None
                        else "Using RAG only; RAG score above threshold."
                    ),
                )
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

            if trace_builder is not None:
                # We did call RAG first, even if its content is not used as context.
                kg_best_score = max(
                    (
                        (chunk.enhanced_score or chunk.semantic_score or 0.0)
                        for chunk in kg_chunks
                    ),
                    default=0.0,
                )
                # Decide label: if RAG had chunks, we say RAG+KG; otherwise KG only.
                if rag_chunks:
                    source_label = "RAG+KG"
                else:
                    source_label = "KG"

                trace_builder.add_tool("RAG_FAISS")
                trace_builder.add_tool("KnowledgeGraph_Neo4j")
                trace_builder.mark_response_context(
                    source=source_label,
                    rag_used=bool(rag_chunks),
                    rag_k=k,
                    rag_top_score=best_rag_score if rag_chunks else None,
                    kg_used=True,
                    kg_k=k,
                    kg_top_score=kg_best_score,
                    notes="RAG score below threshold; KG used as main context.",
                )
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

        if trace_builder is not None:
            trace_builder.add_tool("RAG_FAISS")
            trace_builder.mark_response_context(
                source="RAG" if rag_chunks else "NONE",
                rag_used=bool(rag_chunks),
                rag_k=k,
                rag_top_score=best_rag_score if rag_chunks else None,
                kg_used=False,
                notes="KG returned no results; falling back to RAG context.",
            )

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
