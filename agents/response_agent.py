from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import List, Optional, Tuple

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from config import llm
from rag.faiss_store import RAGAgent, RetrievedChunk
from rag.knowledge_graph_agent import KGRetrievedChunk, KnowledgeGraphAgent
from tools import CurrencyCalculatorTool, CurrencyFXTool, FXAPIError, FXRateResult
from tracing import TraceBuilder
from agents.tool_planner_agent import ToolPlannerAgent, ToolPlan, ToolCall


logger = logging.getLogger(__name__)


class SourceCitation(BaseModel):
    source_file: str
    section_heading: str
    chunk_id: int
    score: float


class Answer(BaseModel):
    answer_text: str
    citations: List[SourceCitation]


class CurrencyConversionResult(BaseModel):
    amount: float
    source_currency: str
    target_currency: str
    rate: float
    converted_amount: float
    rate_source: str
    rate_timestamp: datetime
    # Optional: user-supplied implied rate, never used for the calculation itself.
    user_supplied_rate: Optional[float] = None


class AnswerWithReasoning(BaseModel):
    reasoning: str
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
        "You are an AWS billing support assistant. You help users with ANY billing-related "
        "questions, including checking balance, understanding charges, analyzing anomalies, "
        "navigating the Billing Console, credit usage, invoices, budgets, payments, refunds, "
        "and cost management tools.\n\n"

        "Use ONLY the provided context + user memory + conversation history to answer. "
        "If the context does not contain the required information, you MUST say so clearly.\n\n"

        "You MUST return TWO sections in your final output:\n"
        "1) Reasoning: — a detailed chain-of-thought explaining step-by-step how you reached the answer.\n"
        "2) Final Answer: — a clean answer intended for the user.\n\n"

        "The user explicitly wants reasoning, so do NOT hide chain-of-thought.\n"
        "Do NOT hallucinate AWS features or policies.\n"
        "If multiple interpretations exist, explain them.\n\n"

        "============================================================\n"
        "FEW-SHOT EXAMPLES\n"
        "============================================================\n\n"

        "EXAMPLE 1 — General navigation question\n"
        "User: How can I check my AWS account balance?\n"
        "Assistant:\n"
        "Reasoning:\n"
        "- The question is general and relates to checking charges.\n"
        "- AWS Billing Console → Bills page contains this info.\n"
        "Final Answer:\n"
        "Open the AWS Console → Billing → Bills. This page shows your month-to-date charges.\n\n"

        "EXAMPLE 2 — Sudden increase\n"
        "User: My bill increased by $50. What happened?\n"
        "Assistant:\n"
        "Reasoning:\n"
        "- Without exact service breakdown, I must generalize.\n"
        "- Common causes: EC2 usage, NAT Gateway, S3 requests, data transfer.\n"
        "- Must instruct user how to confirm.\n"
        "Final Answer:\n"
        "A $50 jump is typically driven by EC2 runtime, NAT Gateway traffic, or S3 activity. "
        "Check Billing → Cost Explorer → Service view to identify the exact source.\n\n"

        "EXAMPLE 3 — Missing context\n"
        "User: Why am I being charged for AWS Backup?\n"
        "Assistant:\n"
        "Reasoning:\n"
        "- If context lacks backup details, I must clearly state that.\n"
        "Final Answer:\n"
        "The provided context does not include AWS Backup usage details. "
        "Check Billing → Cost Explorer → Service view or the AWS Backup dashboard.\n\n"

        "============================================================\n"
        "END FEW-SHOT EXAMPLES\n"
        "============================================================\n\n"

        "Follow the few-shot behavior EXACTLY. Respond with:\n"
        "Reasoning: <detailed chain-of-thought>\n"
        "Final Answer: <short answer>"
    )


    # Minimal alias map for common natural-language currency mentions.
    # (Extend anytime you want.)
    CURRENCY_ALIASES = {
        # AED
        "aed": "AED",
        "aeds": "AED",
        "dirham": "AED",
        "dirhams": "AED",
        "uaedirham": "AED",
        "uaedirhams": "AED",

        # RUB
        "rub": "RUB",
        "ruble": "RUB",
        "rubles": "RUB",
        "rouble": "RUB",
        "roubles": "RUB",

        # a few common ones (optional but helpful)
        "usd": "USD",
        "dollar": "USD",
        "dollars": "USD",
        "eur": "EUR",
        "euro": "EUR",
        "euros": "EUR",
        "jpy": "JPY",
        "yen": "JPY",
        "gbp": "GBP",
        "pound": "GBP",
        "pounds": "GBP",
    }

    @classmethod
    def _normalize_currency_token(cls, token: str) -> Optional[str]:
        if token is None:
            return None

        t = token.strip().lower()
        # keep only letters
        t = re.sub(r"[^a-z]", "", t)

        if not t:
            return None

        # ISO code
        if len(t) == 3 and t.isalpha():
            return t.upper()

        # pluralized ISO like "aeds" -> "aed"
        if len(t) == 4 and t.endswith("s") and t[:3].isalpha():
            return t[:3].upper()

        # alias map
        if t in cls.CURRENCY_ALIASES:
            return cls.CURRENCY_ALIASES[t]

        # plural alias fallback
        if t.endswith("s") and t[:-1] in cls.CURRENCY_ALIASES:
            return cls.CURRENCY_ALIASES[t[:-1]]

        return None

    # If the best RAG similarity score is below this, we consider RAG "weak"
    RAG_MIN_SCORE = 0.8

    def __init__(
        self,
        rag_agent: RAGAgent,
        kg_agent: Optional[KnowledgeGraphAgent] = None,
        llm_client: ChatOpenAI = llm,
        currency_fx_tool: Optional[CurrencyFXTool] = None,
        currency_calculator_tool: Optional[CurrencyCalculatorTool] = None,
        tool_planner: Optional[ToolPlannerAgent] = None,
    ) -> None:
        self.rag_agent = rag_agent
        self.kg_agent = kg_agent
        self.llm = llm_client

        # Optional tools for currency conversion; if not provided,
        # currency-specific logic will be skipped.
        self.currency_fx_tool = currency_fx_tool
        self.currency_calculator_tool = currency_calculator_tool

        # Optional tool planner; if None, the agent will fall back to
        # its built-in heuristic behavior (until we fully migrate).
        self.tool_planner = tool_planner

    def _maybe_plan_tools(
        self,
        question: str,
        chat_history: Optional[List[Tuple[str, str]]],
        user_memory: str,
    ) -> Optional[ToolPlan]:
        """
        Call the ToolPlannerAgent (if configured) to obtain a ToolPlan.

        For now this helper only returns the plan; actual execution and full
        wiring into the answer flow will be added in later steps.
        """
        if self.tool_planner is None:
            return None

        try:
            plan = self.tool_planner.plan(
                question=question,
                chat_history=chat_history,
                user_memory_summary=user_memory or None,
            )
            return plan
        except Exception as exc:
            logger.exception("ToolPlannerAgent.plan failed: %s", exc)
            return None

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
        - handle direct currency conversion with tools (and tracing) when possible
        - otherwise retrieve context (RAG first, optionally fall back to KG)
        - call LLM with a chain-of-thought (CoT) style prompt and few-shot format
        - parse reasoning + final answer from the LLM output
        - optionally update a TraceBuilder with metadata about the tools/agents used
        - return final answer text + structured citations
        """

        # Ask the (optional) tool planner which tools it would use for this question.
        # The rest of the method (currency logic, RAG/KG, etc.) will respect this
        # plan when deciding which tools to actually invoke.
        tool_plan: Optional[ToolPlan] = self._maybe_plan_tools(
            question=question,
            chat_history=chat_history,
            user_memory=user_memory,
        )

        # Log whether we are relying on the planner or falling back to the
        # built-in heuristic behavior for tool decisions.
        if tool_plan is None:
            logger.info(
                "ResponseAgent.answer: no tool plan available; using heuristic "
                "fallback for tool decisions."
            )
        else:
            logger.info(
                "ResponseAgent.answer: tool plan obtained with %d call(s).",
                len(tool_plan.calls),
            )

        # Record the raw tool plan (planning != execution).
        # tools_used will be populated ONLY when a tool is actually executed.
        if trace_builder is not None:
            trace_builder.set_tool_plan(tool_plan.dict() if tool_plan is not None else None)

        # 1) Direct currency conversion fast-path (no RAG/KG)

        # Decide whether we should attempt direct currency conversion based
        # on the tool plan. If there is no planner (or it failed), we keep
        # the previous behavior and always try. If there is a plan, we only
        # try when it explicitly chose the currency_conversion tool.
        conversion_result: Optional[CurrencyConversionResult] = None

        should_try_direct_currency = False
        if tool_plan is None:
            # No planner configured or planner failed → preserve old heuristic.
            should_try_direct_currency = True
        else:
            # Planner is present: only attempt conversion if it requested it.
            for call in tool_plan.calls:
                if call.name == "currency_conversion":
                    should_try_direct_currency = True
                    break

        if should_try_direct_currency:
            planned_call = None
            if tool_plan is not None:
                planned_call = next(
                    (c for c in tool_plan.calls if c.name == "currency_conversion"),
                    None,
                )

            try:
                # 1) If planner provided args, execute conversion from args (most flexible).
                if planned_call is not None and isinstance(planned_call.args, dict):
                    amt_raw = planned_call.args.get("amount")
                    src_raw = planned_call.args.get("source_currency")
                    dst_raw = planned_call.args.get("target_currency")

                    if amt_raw is not None and src_raw and dst_raw:
                        amount = float(amt_raw)
                        src = self._normalize_currency_token(str(src_raw))
                        dst = self._normalize_currency_token(str(dst_raw))

                        if src and dst:
                            if trace_builder is not None:
                                trace_builder.add_tool("currency_conversion")
                                trace_builder.add_tool_execution(
                                    tool_name="currency_conversion",
                                    args={
                                        "amount": amount,
                                        "source_currency": src,
                                        "target_currency": dst,
                                    },
                                    success=True,
                                    error=None,
                                )

                            conversion_result = self._run_currency_conversion_tool(
                                amount=amount,
                                source_currency=src,
                                target_currency=dst,
                                user_supplied_rate=None,
                                propagate_fx_errors=True,
                            )
                        else:
                            # Planner chose the tool but args were not usable → fall back to parsing question.
                            conversion_result = self._handle_currency_conversion(question)
                    else:
                        # Planner chose the tool but didn't provide args → fall back to parsing question.
                        conversion_result = self._handle_currency_conversion(question)
                else:
                    # 2) No usable planner call → fallback to heuristic parsing.
                    conversion_result = self._handle_currency_conversion(question)

            except FXAPIError as exc:
                if trace_builder is not None:
                    trace_builder.add_tool("currency_conversion")
                    trace_builder.add_tool_execution(
                        tool_name="currency_conversion",
                        args={"question": question},
                        success=False,
                        error=str(exc) or "FXAPIError",
                    )
                    trace_builder.add_error(
                        component="CurrencyFXTool",
                        type="FXAPIError",
                        message=str(exc) or "Failed to obtain live FX rate.",
                    )

                return Answer(
                    answer_text=(
                        "You asked for a currency conversion, but live exchange rates "
                        "are temporarily unavailable.\n"
                        "Because I cannot obtain a reliable rate from the currency tool, "
                        "I cannot safely perform this conversion right now."
                    ),
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

        # 2) Retrieve context (RAG/KG), honoring the tool planner where possible.

        # Decide whether we should perform any RAG/KG retrieval at all. If there is
        # no planner (or it failed), we preserve the previous behavior and always
        # try to retrieve context. If there is a plan, we only retrieve when it
        # explicitly chose rag_retrieval and/or kg_retrieval.
        use_retrieval = False
        if tool_plan is None:
            # No planner configured or planner failed → keep old behavior.
            use_retrieval = True
        else:
            for call in tool_plan.calls:
                if call.name in ("rag_retrieval", "kg_retrieval"):
                    use_retrieval = True
                    break

        # Defaults if we end up not retrieving.
        context_block: str = ""
        citations: List[SourceCitation] = []

        if use_retrieval:
            # Existing helper that decides between RAG and KG using internal
            # heuristics (e.g., similarity thresholds). In later steps we can
            # refine this to look more closely at the planner's choices.
            context_block, citations = self._get_best_context_and_citations(
                question=question,
                k=k,
                trace_builder=trace_builder,
            )
        else:
            # No retrieval requested by the planner: explicitly record this in the
            # trace so downstream components understand that no RAG/KG context was used.
            if trace_builder is not None:
                trace_builder.mark_response_context(
                    source="NONE",
                    rag_used=False,
                    kg_used=False,
                    notes="Tool planner did not request RAG/KG retrieval.",
                )

        # 3) Format recent conversation history (short-term memory)

        history_text = ""
        if chat_history:
            history_lines: List[str] = []
            for role, content in chat_history:
                prefix = "User" if role == "user" else "Assistant"
                history_lines.append(f"{prefix}: {content}")
            history_text = "\n".join(history_lines)

        # 4) Build CoT-style user prompt parts (from version 1)
        user_prompt_parts: List[str] = [
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
                # Keep the strong CoT/few-shot formatting instruction:
                "Use ONLY this context, user memory, and conversation history to answer.",
                "Now respond EXACTLY in the format shown in the few-shot examples.",
            ]
        )

        messages = [
            SystemMessage(content=self.SYSTEM_PROMPT),
            HumanMessage(content="\n".join(user_prompt_parts)),
        ]

        llm_response = self.llm.invoke(messages)
        full_output = str(llm_response.content).strip()

        # 5) Parse reasoning + final answer from the LLM output (from version 1)
        reasoning, final_answer = self._parse_reasoning_and_answer(full_output)

        # 6) Optionally enhance the *final answer* with a currency conversion.
        #
        # We make this dependent on the tool plan:
        # - If there is no planner (or it failed → tool_plan is None), we preserve
        #   the previous behavior and always attempt the enhancement.
        # - If there is a valid plan, we only attempt enhancement when it explicitly
        #   includes the currency_conversion tool.
        enhanced_final_answer = final_answer.strip()

        should_try_enhancement = False
        if tool_plan is None:
            # No planner configured or planner failed → keep old behavior.
            should_try_enhancement = True
        else:
            for call in tool_plan.calls:
                if call.name == "currency_conversion":
                    should_try_enhancement = True
                    break

        if should_try_enhancement:
            enhanced_final_answer = self._maybe_enhance_answer_with_currency_conversion(
                question=question,
                answer_text=enhanced_final_answer,
                trace_builder=trace_builder,
            )

        # For now we ignore the reasoning in the HTTP response, but we could
        # log it or add it to traces in the future.

        return AnswerWithReasoning(
            reasoning=reasoning.strip(),
            answer_text=enhanced_final_answer.strip(),
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
                a reliable FX rate from the external API fails or if the
                currency conversion tools are not properly configured.
        """
        if not question:
            return None

        # Attempt to parse the question into a structured conversion request.
        parsed = self._parse_currency_conversion_request(question)
        if not parsed:
            # Not a supported conversion request pattern.
            return None

        amount, source_currency, target_currency, user_supplied_rate = parsed

        # Delegate the actual tool invocation to the shared helper.
        result = self._run_currency_conversion_tool(
            amount=amount,
            source_currency=source_currency,
            target_currency=target_currency,
            user_supplied_rate=user_supplied_rate,
            # For the direct conversion fast-path we want FXAPIError to
            # propagate so the caller can show a clear message.
            propagate_fx_errors=True,
        )
        return result

    def _run_currency_conversion_tool(
        self,
        amount: float,
        source_currency: str,
        target_currency: str,
        user_supplied_rate: Optional[float] = None,
        propagate_fx_errors: bool = True,
    ) -> Optional[CurrencyConversionResult]:
        """Central helper that calls the FX API tool and calculator.

        Args:
            amount: Amount of money in the source currency.
            source_currency: Three-letter source currency code (e.g. 'USD').
            target_currency: Three-letter target currency code (e.g. 'EUR').
            user_supplied_rate: Optional implied rate parsed from the question.
            propagate_fx_errors:
                - If True, any FXAPIError from the FX tool is propagated to the
                  caller (used by the direct conversion fast-path).
                - If False, FXAPIError is swallowed and this method returns None.

        Returns:
            A CurrencyConversionResult on success, or None when conversion
            cannot be performed and errors are not propagated.
        """
        # Sanity checks on amount.
        if amount is None or amount <= 0:
            return None

        # Tools must be available.
        if not self.currency_fx_tool or not self.currency_calculator_tool:
            if propagate_fx_errors:
                raise FXAPIError(
                    "Currency conversion tools are not configured; cannot perform conversion."
                )
            return None

        try:
            fx_result: FXRateResult = self.currency_fx_tool.get_rate(
                source_currency,
                target_currency,
            )
            converted_amount = self.currency_calculator_tool.convert_amount(
                amount=amount,
                rate=fx_result.rate,
            )
        except FXAPIError:
            if propagate_fx_errors:
                # Let the caller decide how to explain this to the user.
                raise
            return None

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
        self,
        question: str,
    ) -> Optional[Tuple[float, str, str, Optional[float]]]:
        """Attempt to parse a simple currency conversion request from text.

        Supports ISO codes and some common currency-name aliases, e.g.:
            "Convert 120 USD to EUR"
            "How much is 500 sar in usd?"
            "translate 1000 aeds to rubles"
        """
        if not question:
            return None

        text = question.strip()

        # "<amount> <SRC> to|in <DST>" where SRC/DST can be code or name token
        amount_pattern = (
            r"(?P<amount>\d+(?:\.\d+)?)\s*"
            r"(?P<src>[A-Za-z]{3,20})\s*"
            r"(?:to|in)\s*"
            r"(?P<dst>[A-Za-z]{3,20})\b"
        )
        match = re.search(amount_pattern, text, flags=re.IGNORECASE)
        if not match:
            return None

        try:
            amount = float(match.group("amount"))
        except ValueError:
            return None

        src_token = match.group("src")
        dst_token = match.group("dst")

        source_currency = self._normalize_currency_token(src_token)
        target_currency = self._normalize_currency_token(dst_token)

        if not source_currency or not target_currency:
            return None

        user_supplied_rate: Optional[float] = None

        # Optional: "1 USD = 5 SAR" (still expects ISO codes here)
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
        try:
            kg_chunks: List[KGRetrievedChunk] = self.kg_agent.retrieve(question, k=k)
        except Exception as e:
            logger.warning("KG retrieval failed; falling back to RAG. Error: %s", e)
            kg_chunks = []

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

    @staticmethod
    def _parse_reasoning_and_answer(output: str) -> Tuple[str, str]:
        """
        Parses LLM output into reasoning and final answer.
        Handles formats like:
          Reasoning: ...\nFinal Answer: ...
        Even if labels appear on separate lines.
        """
        output = re.sub(r'\r\n?', '\n', output).strip()

        # Normalize labels (allow optional newlines after colon)
        reasoning_match = re.search(r'^Reasoning:\s*(.*?)(?=^Final Answer:|\Z)', output, re.DOTALL | re.MULTILINE)
        final_match = re.search(r'^Final Answer:\s*(.*)', output, re.DOTALL | re.MULTILINE)

        reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
        final_answer = final_match.group(1).strip() if final_match else output

        # Fallback: if both missing, treat as final answer
        if not reasoning and not final_answer:
            final_answer = output

        return reasoning, final_answer