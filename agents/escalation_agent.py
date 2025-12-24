from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, List, Tuple
import json


from config import llm
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from tracing import TraceBuilder

EscalationPriority = Literal["P0", "P1", "P2", "NONE"]


@dataclass
class EscalationDecision:
    """
    Final, structured decision about escalation for a single turn.

    Fields:
    - escalate: whether a human handoff should be triggered.
    - priority: P0 / P1 / P2 / NONE (NONE only when escalate == False).
    - reason: short internal rationale (for logs and debugging).
    - human_summary: concise summary to send to a human (Slack/email/etc).
    """

    escalate: bool
    priority: EscalationPriority
    reason: str
    human_summary: str


class EscalationDecisionModel(BaseModel):
    """
    Pydantic model used to validate the LLM's JSON output.
    """

    escalate: bool = Field(
        description="True if a human handoff is needed, False otherwise."
    )
    priority: EscalationPriority = Field(
        description='Escalation priority: "P0", "P1", "P2", or "NONE" when no escalation is needed.'
    )
    reason: str = Field(
        description="Short explanation of why this decision was made, for internal use."
    )
    human_summary: str = Field(
        description="Concise summary of the situation suitable for a human support engineer."
    )


class EscalationAgent:
    """
    EscalationAgent: decides whether to escalate a conversation and at which severity.

    It is a pure decision-making agent:
    - It does NOT call Slack/email directly.
    - It only returns a structured decision that the app can act upon.

    Severity definitions (P0/P1/P2):

    - P0 (Critical):
        * User cannot access their AWS account or billing console,
          OR payment/billing is blocked in a way that stops critical workloads.
        * Large or unexpected charges that may cause serious financial damage.
        * Signs of fraud, account takeover, or security/compliance risk.
        * User clearly indicates urgent crisis, e.g., "system down", "production blocked",
          "I will lose a lot of money", "fraudulent charges", etc.
        => Requires immediate human response.

    - P1 (High):
        * Significant billing confusion or misconfiguration affecting business planning,
          but not an immediate outage or catastrophe.
        * Repeated failed answers from the assistant on the same issue.
        * User frustration is high, or user explicitly asks for a human multiple times.
        => Needs human follow-up soon, but not an emergency.

    - P2 (Normal/Low):
        * Routine or niche billing questions where the assistant cannot give
          a confident answer, but there is no urgency or risk.
        * User is calm and can wait for a normal support workflow.
        => Human review is useful but not time-critical.

    - NONE:
        * No escalation needed. The assistant answered well enough,
          or the issue is simple and clearly resolved.
    """

    SYSTEM_PROMPT = """
You are an escalation analysis agent for an AWS Billing support assistant.

Inputs:
- latest_user_message
- conversation_history (lines starting with "User:" or "Assistant:")
- assistant_answer
- metadata (e.g. sentiment, num_failed_attempts, user_requested_human_flag)

Goal: decide if this needs escalation to a human, and at what priority.

CORE RULES:
1) No billing issue → NEVER escalate.
   - Pure greetings/tests like "hi", "hello", "hey", "good morning", "test", "ping", "are you there?"
   - Chit-chat with no AWS/billing/account/payment/charges/invoice/support issue.
   For these, you MUST return:
   {
     "escalate": false,
     "priority": "NONE",
     "reason": "No billing-related issue; greeting or test only.",
     "human_summary": "User sent a greeting or test message with no AWS billing issue."
   }

2) "User requested a human" only if you see it in a USER message.
   - Only treat lines starting with "User:" or latest_user_message as user speech.
   - Ignore "Assistant:" lines as evidence; they may be wrong.
   - You may treat metadata.user_requested_human_flag as a HINT ONLY.
   - You may claim the user requested a human ONLY if some user message clearly combines:
       * a reference to a human/agent/support person (e.g. "human", "agent", "support", "real person"), AND
       * a request phrase (e.g. "I want", "I need", "can I", "please", "I'd like", "could you").
   - If you cannot quote such a user message, you MUST act as if the user did NOT request a human and MUST NOT say that they did.

3) No invented facts.
   - Do not claim previous escalations unless explicitly stated.
   - Do not change currencies, regions, or other details.
   - Do not exaggerate sentiment; only call the user frustrated/angry if their wording or metadata clearly shows it.
   - If unsure, omit the detail.

SEVERITY LEVELS:
- P0 (Critical): billing/account/payment issue clearly blocking critical workloads or showing fraud/account takeover/major financial damage, or explicit crisis language ("prod is down", "we will lose a lot of money", "fraudulent charges", "my account was hacked").
- P1 (High): significant billing confusion or misconfiguration affecting business planning; repeated failed answers; or clearly strong frustration about an unresolved billing issue; OR a clear user request for a human (as defined above).
- P2 (Normal/Low): non-urgent billing question where the assistant's answer seems incomplete/uncertain and a human review would help, but no crisis.
- NONE: no escalation needed (answer is adequate) OR no billing issue at all.

DECISION LOGIC:
1) If the conversation so far contains NO AWS billing/account/payment/charges/invoice-related issue:
   - Set "escalate": false, "priority": "NONE".
2) Otherwise, set "escalate" to true ONLY if at least one of:
   - P0 conditions (critical block / fraud / major urgent financial risk).
   - P1 conditions (explicit human request as per rule 2, repeated failures, or strong frustration on a billing issue).
   - P2 conditions (non-urgent billing issue where assistant is unsure/incomplete and a human would help).
3) If none of those conditions apply:
   - Set "escalate": false, "priority": "NONE".

OUTPUT FORMAT (STRICT):
Return ONLY a JSON object:

{
  "escalate": boolean,
  "priority": "P0" | "P1" | "P2" | "NONE",
  "reason": string,
  "human_summary": string
}

- If "escalate" is false, "priority" MUST be "NONE".
- If "escalate" is true, "priority" MUST be "P0", "P1", or "P2".
- "reason": short internal explanation (e.g. "greeting only, no billing issue", "user requested human, and repeated confusion").
- "human_summary": short description for a human engineer, using ONLY facts supported by user messages or metadata.
"""

    def __init__(self, llm_client: ChatOpenAI | None = None) -> None:
        # We accept a client for testability; default to the global llm.
        self.llm: ChatOpenAI = llm_client or llm

    def analyze(
        self,
        *,
        latest_user_message: str,
        conversation_history: List[Tuple[str, str]],
        assistant_answer: str,
        sentiment_label: Optional[str] = None,
        sentiment_score: Optional[float] = None,
        num_failed_attempts: int = 0,
        user_explicitly_requested_human: bool = False,
        rag_top_score: Optional[float] = None,
        rag_no_results: bool = False,
        trace_builder: Optional[TraceBuilder] = None,
    ) -> EscalationDecision:

        """
        Main public method: decide whether to escalate and at which P-level.

        Parameters:
        - latest_user_message: most recent user message text.
        - conversation_history: list of (role, content) for recent turns.
          role is "user" or "assistant".
        - assistant_answer: the assistant's latest answer text.
        - sentiment_label: e.g. "positive", "neutral", "negative", if available.
        - sentiment_score: numeric sentiment score, if available.
        - num_failed_attempts: how many times the assistant has failed to answer
          this issue in the current conversation (approximate count).
        - user_explicitly_requested_human: True if the user asked for a human.
        - rag_top_score: similarity score of the top retrieved chunk, if available.
        - rag_no_results: True if RAG returned no meaningful results.

        Returns:
        - EscalationDecision dataclass instance.
        """
        history_text = self._format_history(conversation_history)
        metadata_text = self._format_metadata(
            sentiment_label=sentiment_label,
            sentiment_score=sentiment_score,
            num_failed_attempts=num_failed_attempts,
            user_requested_human=user_explicitly_requested_human,
            rag_top_score=rag_top_score,
            rag_no_results=rag_no_results,
        )

        user_prompt_parts = [
            "You will receive the following information:",
            "",
            "1) Latest user message:",
            latest_user_message,
            "",
            "2) Recent conversation history (oldest to newest):",
            history_text if history_text else "(no prior history)",
            "",
            "3) Assistant's latest answer:",
            assistant_answer if assistant_answer else "(no answer yet)",
            "",
            "4) Metadata about this situation:",
            metadata_text if metadata_text else "(no extra metadata)",
            "",
            "Based on all of this, decide whether to escalate and at which severity.",
            "Remember to follow the P0/P1/P2/NONE definitions from the system prompt.",
            "",
            "Output ONLY a single JSON object with the fields:",
            '{ "escalate": boolean, "priority": "P0" | "P1" | "P2" | "NONE", "reason": string, "human_summary": string }',
        ]

        messages = [
            SystemMessage(content=self.SYSTEM_PROMPT),
            HumanMessage(content="\n".join(user_prompt_parts)),
        ]

        try:
            llm_response = self.llm.invoke(messages)
            raw_text = str(llm_response.content)
        except Exception as e:
            fallback_reason = f"Escalation LLM call failed: {e}"
            decision = EscalationDecision(
                escalate=False,
                priority="NONE",
                reason=fallback_reason,
                human_summary="No escalation triggered due to internal LLM error.",
            )
            if trace_builder is not None:
                trace_builder.add_error(
                    component="EscalationAgent",
                    type="LLMInvokeError",
                    message=str(e) or repr(e),
                )
                trace_builder.set_escalation(
                    escalate=decision.escalate,
                    priority=decision.priority,
                    reason=decision.reason,
                    human_summary=decision.human_summary,
                )
            return decision


        try:
            model_obj = self._parse_llm_output(raw_text)
        except Exception as e:
            # Fallback: no escalation if parsing fails
            fallback_reason = f"Failed to parse escalation decision from LLM output: {e}"
            decision = EscalationDecision(
                escalate=False,
                priority="NONE",
                reason=fallback_reason,
                human_summary="No escalation triggered due to internal parsing fallback.",
            )
            if trace_builder is not None:
                trace_builder.set_escalation(
                    escalate=decision.escalate,
                    priority=decision.priority,
                    reason=decision.reason,
                    human_summary=decision.human_summary,
                )
            return decision

        # Enforce consistency: if escalate is False, priority must be NONE.
        escalate = bool(model_obj.escalate)
        priority: EscalationPriority = model_obj.priority
        if not escalate:
            priority = "NONE"
        elif priority == "NONE":
            # If the model says escalate but priority is NONE, downgrade to P2 by default.
            priority = "P2"

        decision = EscalationDecision(
            escalate=escalate,
            priority=priority,
            reason=model_obj.reason.strip(),
            human_summary=model_obj.human_summary.strip(),
        )

        if trace_builder is not None:
            trace_builder.set_escalation(
                escalate=decision.escalate,
                priority=decision.priority,
                reason=decision.reason,
                human_summary=decision.human_summary,
            )

        return decision

    def _format_history(self, conversation_history: List[Tuple[str, str]]) -> str:
        """
        Convert recent (role, content) pairs into a readable text block.
        """
        if not conversation_history:
            return ""

        lines = []
        for role, content in conversation_history:
            prefix = "User" if role == "user" else "Assistant"
            lines.append(f"{prefix}: {content}")
        return "\n".join(lines)

    def _format_metadata(
        self,
        *,
        sentiment_label: Optional[str],
        sentiment_score: Optional[float],
        num_failed_attempts: int,
        user_requested_human: bool,
        rag_top_score: Optional[float],
        rag_no_results: bool,
    ) -> str:
        """
        Create a compact metadata description for the prompt.
        """
        parts = []

        if sentiment_label is not None:
            if sentiment_score is not None:
                parts.append(
                    f"Sentiment: {sentiment_label} (score={sentiment_score:.3f})"
                )
            else:
                parts.append(f"Sentiment: {sentiment_label}")

        if num_failed_attempts > 0:
            parts.append(f"Estimated failed attempts on this issue: {num_failed_attempts}")

        if user_requested_human:
            parts.append("metadata.user_requested_human_flag = true")
        else:
            parts.append("metadata.user_requested_human_flag = false")

        if rag_no_results:
            parts.append("RAG: no relevant documentation was found for this query.")
        elif rag_top_score is not None:
            parts.append(f"RAG: top similarity score = {rag_top_score:.3f}")

        if not parts:
            return ""

        return " | ".join(parts)

    def _parse_llm_output(self, raw_text: str) -> EscalationDecisionModel:
        """
        Parse the LLM's raw text into an EscalationDecisionModel instance.

        Handles common cases like extra text around the JSON or fenced code blocks.
        """
        text = raw_text.strip()

        # Strip markdown code fences if present
        if text.startswith("```"):
            # Remove leading ``` or ```json
            lines = text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            # Remove trailing ``` if present
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        # Try direct JSON parse
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Fallback: find first '{' and last '}' and try that slice
            start = text.find("{")
            end = text.rfind("}")
            if start == -1 or end == -1 or end <= start:
                raise ValueError(f"Could not find a JSON object in: {text!r}")
            snippet = text[start : end + 1]
            data = json.loads(snippet)

        return EscalationDecisionModel.parse_obj(data)

