from __future__ import annotations

import json
from typing import Any, Dict, List, Literal, Optional, Tuple

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from config import llm


# The set of tools that the planner can choose between.
ToolName = Literal[
    "rag_retrieval",
    "kg_retrieval",
    "currency_conversion",
    "no_tool",
]


class ToolArgumentSpec(BaseModel):
    """
    Specification of a single argument for a tool.

    This is used only for documentation and prompt construction for the planner.
    """

    name: str = Field(..., description="Argument name as it will appear in args.")
    description: str = Field(..., description="Human-readable description.")
    type: str = Field(..., description="Logical type (e.g. string, integer, float).")
    required: bool = Field(
        default=True,
        description="Whether this argument is required for the tool to work.",
    )


class ToolSpec(BaseModel):
    """
    Specification of a tool that the planner may choose to call.
    """

    name: ToolName = Field(..., description="Canonical tool name.")
    description: str = Field(..., description="What the tool does and when to use it.")
    args: List[ToolArgumentSpec] = Field(
        default_factory=list,
        description="Arguments this tool expects in ToolCall.args.",
    )


# Registry of all tools known to the planner.
TOOL_SPECS: Dict[ToolName, ToolSpec] = {
    "rag_retrieval": ToolSpec(
        name="rag_retrieval",
        description=(
            "Retrieve relevant chunks from the AWS billing documentation via the "
            "FAISS vector store. Use this when the user is asking about AWS billing, "
            "Cost and Usage Reports, or anything that is likely answered in the "
            "official docs."
        ),
        args=[
            ToolArgumentSpec(
                name="question",
                description="The user's question or a focused sub-question.",
                type="string",
                required=True,
            ),
            ToolArgumentSpec(
                name="k",
                description="Maximum number of chunks to retrieve (top-k).",
                type="integer",
                required=False,
            ),
        ],
    ),
    "kg_retrieval": ToolSpec(
        name="kg_retrieval",
        description=(
            "Retrieve context from the Neo4j knowledge graph that was built from "
            "the AWS billing documentation. Use this if RAG retrieval seems weak "
            "or you want a graph-aware view of related content."
        ),
        args=[
            ToolArgumentSpec(
                name="question",
                description="The user's question or a focused sub-question.",
                type="string",
                required=True,
            ),
            ToolArgumentSpec(
                name="k",
                description="Maximum number of chunks or nodes to retrieve (top-k).",
                type="integer",
                required=False,
            ),
        ],
    ),
    "currency_conversion": ToolSpec(
        name="currency_conversion",
        description=(
            "Convert a numerical amount from one currency to another using a live "
            "FX rate from the external API and a precise decimal calculator. Use "
            "this when the user explicitly asks for a monetary amount in a different "
            "currency (e.g., USD to EUR)."
        ),
        args=[
            ToolArgumentSpec(
                name="amount",
                description="The numeric amount of money to convert.",
                type="float",
                required=True,
            ),
            ToolArgumentSpec(
                name="source_currency",
                description="Three-letter source currency code, e.g. 'USD'.",
                type="string",
                required=True,
            ),
            ToolArgumentSpec(
                name="target_currency",
                description="Three-letter target currency code, e.g. 'EUR'.",
                type="string",
                required=True,
            ),
        ],
    ),
    "no_tool": ToolSpec(
        name="no_tool",
        description=(
            "Do not call any tools. Use this when the question can be answered "
            "directly from the conversation history and user memory, or when the "
            "user is only chit-chatting and tools are unnecessary."
        ),
        args=[],
    ),
}


def build_tool_prompt_block() -> str:
    """
    Build a human-readable description of all tools and their arguments.

    This string will be included in the planner's system prompt so the LLM
    understands what tools exist and how to use them.
    """
    lines: List[str] = []
    for spec in TOOL_SPECS.values():
        lines.append(f"Tool name: {spec.name}")
        lines.append(f"Description: {spec.description}")
        if spec.args:
            lines.append("Arguments:")
            for arg in spec.args:
                required_str = "required" if arg.required else "optional"
                lines.append(
                    f"  - {arg.name} ({arg.type}, {required_str}): {arg.description}"
                )
        else:
            lines.append("Arguments: none")
        lines.append("")  # blank line between tools
    return "\n".join(lines).strip()


class ToolCall(BaseModel):
    """
    A single tool invocation decided by the planner.
    """

    name: ToolName = Field(
        ...,
        description=(
            "Name of the tool to call "
            "(e.g. rag_retrieval, kg_retrieval, currency_conversion, no_tool)."
        ),
    )
    args: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arguments to pass to the selected tool.",
    )


class ToolPlan(BaseModel):
    """
    High-level plan for which tools to call (and in what order).
    """

    calls: List[ToolCall] = Field(
        default_factory=list,
        description="Ordered list of tool calls to execute.",
    )
    rationale: str = Field(
        default="",
        description="Short natural-language explanation of why these tools were chosen.",
    )


class ToolPlannerAgent:
    """
    LLM-based planner that decides which tools the ResponseAgent should use.

    Responsibilities:
    - Inspect the user question, conversation history, and optional user memory.
    - Decide whether to call RAG, KG, currency conversion, no tools, or a combination.
    - Output a structured ToolPlan (list of ToolCall + rationale).
    """

    def __init__(
        self,
        llm_client: Optional[ChatOpenAI] = None,
        tool_specs: Optional[Dict[ToolName, ToolSpec]] = None,
    ) -> None:
        # Reference to the LLM client used for planning.
        self.llm: ChatOpenAI = llm_client or llm
        # Tool specs known to this planner instance.
        self.tool_specs: Dict[ToolName, ToolSpec] = tool_specs or TOOL_SPECS

    def _build_system_prompt(self) -> str:
        """
        Build the system prompt that explains the available tools and the JSON schema.
        """
        tools_block = build_tool_prompt_block()
        return (
            "You are a tool-planning assistant for an AWS Billing LLM system.\n"
            "Your job is to decide which tools the main assistant should call to "
            "answer the user's question.\n\n"
            "Available tools:\n"
            f"{tools_block}\n\n"
            "Output format (very important):\n"
            "You MUST respond with a single JSON object matching this schema:\n"
            "{\n"
            '  "calls": [\n'
            "    {\n"
            '      "name": "<tool_name>",\n'
            '      "args": { /* key-value arguments appropriate for that tool */ }\n'
            "    }\n"
            "    // zero or more tool calls\n"
            "  ],\n"
            '  "rationale": "Short explanation of why you chose these tools."\n'
            "}\n\n"
            "Rules:\n"
            "- Prefer rag_retrieval for AWS billing and documentation-related questions.\n"
            "- Consider kg_retrieval if the question is complex or RAG alone might miss "
            "relationships (we may chain it after RAG later).\n"
            "- Use currency_conversion when the user explicitly wants an amount in a "
            "different currency (e.g., USD -> EUR).\n"
            "- Use no_tool when the question can be answered from general reasoning, "
            "conversation history, or user memory alone (e.g., chit-chat).\n"
            "- Do NOT explain your reasoning outside the JSON. Put reasoning only in "
            'the "rationale" field of the JSON.\n'
        )

    def _build_human_prompt(
        self,
        question: str,
        chat_history: Optional[List[Tuple[str, str]]],
        user_memory_summary: Optional[str],
    ) -> str:
        """
        Build the human message that gives the planner the concrete situation.
        """
        parts: List[str] = []
        parts.append("User question:")
        parts.append(question)

        if chat_history:
            parts.append("\nRecent conversation history (oldest first):")
            for role, content in chat_history:
                parts.append(f"{role}: {content}")

        if user_memory_summary:
            parts.append("\nUser memory summary:")
            parts.append(user_memory_summary)

        parts.append(
            "\nBased on the above, decide which tools to call and return ONLY the JSON "
            "object as specified."
        )

        return "\n".join(parts)

    @staticmethod
    def _safe_extract_json(text: str) -> Optional[Dict[str, Any]]:
        """
        Try to extract a JSON object from the LLM output in a robust way.

        The planner is instructed to return pure JSON, but this helper defends
        against minor deviations (e.g. extra text before/after).
        """
        text = text.strip()
        if not text:
            return None

        # Try direct parse first.
        try:
            return json.loads(text)
        except Exception:
            pass

        # Fallback: look for the first '{' and last '}' and try that slice.
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None

        candidate = text[start : end + 1]
        try:
            return json.loads(candidate)
        except Exception:
            return None

    def plan(
        self,
        question: str,
        chat_history: Optional[List[Tuple[str, str]]] = None,
        user_memory_summary: Optional[str] = None,
    ) -> ToolPlan:
        """
        Decide which tools to call for this question using the LLM.

        On any parsing or LLM error, fall back to a safe default plan that chooses
        `no_tool`, so the system keeps working.
        """
        system_prompt = self._build_system_prompt()
        human_prompt = self._build_human_prompt(
            question=question,
            chat_history=chat_history,
            user_memory_summary=user_memory_summary,
        )

        try:
            result = self.llm.invoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=human_prompt),
                ]
            )

            raw_content = result.content
            if isinstance(raw_content, str):
                raw_text = raw_content
            else:
                # Fallback: stringify non-string content.
                raw_text = str(raw_content)

            data = self._safe_extract_json(raw_text)
            if data is None or not isinstance(data, dict):
                raise ValueError("Planner output was not valid JSON")

            # Validate against the ToolPlan schema.
            plan = ToolPlan(**data)
            return plan

        except Exception as e:
            # Fallback: safe default so the system does not crash if planning fails.
            return ToolPlan(
                calls=[ToolCall(name="no_tool", args={})],
                rationale=f"Planner fallback: failed to produce a valid plan ({e}).",
            )
