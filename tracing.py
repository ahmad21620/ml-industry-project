from __future__ import annotations

from datetime import datetime
from typing import List, Optional, Literal, Dict, Any

from pydantic import BaseModel, Field


# ----- Nested structures for clarity -----


class UserMemoryAgentTrace(BaseModel):
    invoked: bool = False
    action: Literal["update", "none"] = "none"
    summary: Optional[str] = None


class ResponseAgentTrace(BaseModel):
    invoked: bool = False
    context_source: Literal["RAG", "KG", "RAG+KG", "NONE"] = "NONE"
    notes: Optional[str] = None


class EscalationAgentTrace(BaseModel):
    invoked: bool = False
    escalate: bool = False
    priority: Literal["P0", "P1", "P2", "NONE"] = "NONE"
    reason: Optional[str] = None


class AgentsTrace(BaseModel):
    UserMemoryAgent: UserMemoryAgentTrace = Field(
        default_factory=UserMemoryAgentTrace
    )
    ResponseAgent: ResponseAgentTrace = Field(
        default_factory=ResponseAgentTrace
    )
    EscalationAgent: EscalationAgentTrace = Field(
        default_factory=EscalationAgentTrace
    )


class RAGContextTrace(BaseModel):
    used: bool = False
    k: Optional[int] = None
    top_score: Optional[float] = None


class KGContextTrace(BaseModel):
    used: bool = False
    k: Optional[int] = None
    top_score: Optional[float] = None


class ContextTrace(BaseModel):
    source: Literal["RAG", "KG", "RAG+KG", "NONE"] = "NONE"
    rag: RAGContextTrace = Field(default_factory=RAGContextTrace)
    kg: KGContextTrace = Field(default_factory=KGContextTrace)


class CurrencyConversionTrace(BaseModel):
    mode: Literal["direct", "aws_bill_enhancement"]
    amount: float
    source_currency: str
    target_currency: str
    rate: float
    rate_source: str
    rate_timestamp_utc: datetime
    converted_amount: float
    user_supplied_rate: Optional[float] = None
    fx_error: bool = False


class EscalationTrace(BaseModel):
    escalate: bool = False
    priority: Literal["P0", "P1", "P2", "NONE"] = "NONE"
    human_summary: Optional[str] = None


class ErrorEventTrace(BaseModel):
    component: str
    type: str
    message: str


# ----- Top-level payload -----


class MessageTracePayload(BaseModel):
    """Structured trace payload stored as JSON in MessageTrace.trace_json."""

    version: str = "1.0"

    conversation_id: int
    message_id: int
    user_id: int
    timestamp_utc: datetime

    agents: AgentsTrace = Field(default_factory=AgentsTrace)
    tools_used: List[str] = Field(default_factory=list)
    context: ContextTrace = Field(default_factory=ContextTrace)
    currency_conversion: Optional[CurrencyConversionTrace] = None
    escalation: EscalationTrace = Field(default_factory=EscalationTrace)
    errors: List[ErrorEventTrace] = Field(default_factory=list)

    # Full tool-planning and execution details (optional).
    tool_plan: Optional[Dict[str, Any]] = None
    tool_executions: List[Dict[str, Any]] = Field(default_factory=list)

    pipeline_summary: Optional[str] = None
    extra_metadata: Dict[str, Any] = Field(default_factory=dict)


class TraceBuilder:
    """Convenience wrapper around MessageTracePayload for building traces.

    The /chat endpoint will:
        - create a TraceBuilder via TraceBuilder.create(...)
        - pass it into agents that need to log their behavior
        - persist builder.payload.json() into MessageTrace.trace_json
    """

    def __init__(self, payload: MessageTracePayload) -> None:
        self.payload = payload

    @classmethod
    def create(
        cls,
        *,
        conversation_id: int,
        message_id: int,
        user_id: int,
        timestamp_utc: Optional[datetime] = None,
    ) -> "TraceBuilder":
        ts = timestamp_utc or datetime.utcnow()
        payload = MessageTracePayload(
            conversation_id=conversation_id,
            message_id=message_id,
            user_id=user_id,
            timestamp_utc=ts,
        )
        return cls(payload)

    # ----- Helper methods for agents/tools to update trace -----

    def set_tool_plan(self, plan: Optional[Dict[str, Any]]) -> None:
        """Record the raw tool plan (if any) produced by the ToolPlannerAgent."""
        self.payload.tool_plan = plan

    def add_tool_execution(
        self,
        *,
        tool_name: str,
        args: Optional[Dict[str, Any]] = None,
        success: bool,
        error: Optional[str] = None,
    ) -> None:
        """Append a single tool execution record."""
        entry: Dict[str, Any] = {
            "tool_name": tool_name,
            "args": args or {},
            "success": success,
            "error": error,
        }
        self.payload.tool_executions.append(entry)

    def add_tool(self, tool_name: str) -> None:
        if tool_name and tool_name not in self.payload.tools_used:
            self.payload.tools_used.append(tool_name)

    def mark_user_memory(
        self,
        *,
        invoked: bool = True,
        action: Literal["update", "none"] = "update",
        summary: Optional[str] = None,
    ) -> None:
        um = self.payload.agents.UserMemoryAgent
        um.invoked = invoked
        um.action = action
        um.summary = summary

    def mark_response_context(
        self,
        *,
        source: Literal["RAG", "KG", "RAG+KG", "NONE"],
        rag_used: Optional[bool] = None,
        rag_k: Optional[int] = None,
        rag_top_score: Optional[float] = None,
        kg_used: Optional[bool] = None,
        kg_k: Optional[int] = None,
        kg_top_score: Optional[float] = None,
        notes: Optional[str] = None,
    ) -> None:
        # Top-level context source
        self.payload.context.source = source

        # ResponseAgent-level view
        ra = self.payload.agents.ResponseAgent
        ra.invoked = True
        ra.context_source = source
        ra.notes = notes

        # RAG details
        if rag_used is not None:
            self.payload.context.rag.used = rag_used
        if rag_k is not None:
            self.payload.context.rag.k = rag_k
        if rag_top_score is not None:
            self.payload.context.rag.top_score = rag_top_score

        # KG details
        if kg_used is not None:
            self.payload.context.kg.used = kg_used
        if kg_k is not None:
            self.payload.context.kg.k = kg_k
        if kg_top_score is not None:
            self.payload.context.kg.top_score = kg_top_score

    def set_currency_conversion(
        self,
        mode: str,
        amount: float,
        source_currency: str,
        target_currency: str,
        rate: float,
        rate_source: str,
        rate_timestamp_utc: datetime,
        converted_amount: float,
        user_supplied_rate: Optional[float],
        fx_error: bool,
    ) -> None:
        """
        Record a currency conversion attempt/result and mark the currency tools
        as used for this message.
        """
        # Store structured currency-conversion info on the payload
        self.payload.currency_conversion = CurrencyConversionTrace(
            mode=mode,
            amount=amount,
            source_currency=source_currency,
            target_currency=target_currency,
            rate=rate,
            rate_source=rate_source,
            rate_timestamp_utc=rate_timestamp_utc,
            converted_amount=converted_amount,
            user_supplied_rate=user_supplied_rate,
            fx_error=fx_error,
        )

        # Make sure the tools used are explicitly visible in tools_used.
        # (Avoid duplicates if the method is called more than once.)
        if "CurrencyFXTool" not in self.payload.tools_used:
            self.payload.tools_used.append("CurrencyFXTool")
        if "CurrencyCalculatorTool" not in self.payload.tools_used:
            self.payload.tools_used.append("CurrencyCalculatorTool")

    def set_escalation(
        self,
        *,
        escalate: bool,
        priority: Literal["P0", "P1", "P2", "NONE"],
        reason: Optional[str] = None,
        human_summary: Optional[str] = None,
    ) -> None:
        # Agent-level info
        ea = self.payload.agents.EscalationAgent
        ea.invoked = True
        ea.escalate = escalate
        ea.priority = priority
        ea.reason = reason

        # Top-level escalation section
        self.payload.escalation.escalate = escalate
        self.payload.escalation.priority = priority
        self.payload.escalation.human_summary = human_summary or reason

    def add_error(self, component: str, type: str, message: str) -> None:
        self.payload.errors.append(
            ErrorEventTrace(
                component=component,
                type=type,
                message=message,
            )
        )

    def build_pipeline_summary(self) -> None:
        """Generate a short human-readable summary for admin view."""
        parts: List[str] = []

        # UserMemoryAgent
        um = self.payload.agents.UserMemoryAgent
        if um.invoked:
            parts.append(f"UserMemory {um.action}")

        # ResponseAgent / context
        ra = self.payload.agents.ResponseAgent
        if ra.invoked:
            parts.append(f"ResponseAgent ctx={ra.context_source}")

        # Currency conversion
        cc = self.payload.currency_conversion
        if cc is not None:
            parts.append(
                f"FX {cc.mode} {cc.source_currency}->{cc.target_currency}"
            )

        # Escalation
        esc = self.payload.escalation
        if esc.priority != "NONE" or esc.escalate:
            parts.append(f"Escalation {esc.priority}")
        else:
            parts.append("Escalation NONE")

        if parts:
            self.payload.pipeline_summary = "; ".join(parts)

    def to_json(self) -> str:
        """Serialize the underlying payload to JSON."""
        return self.payload.json()
