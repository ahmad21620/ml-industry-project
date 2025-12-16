from __future__ import annotations

import secrets
from datetime import datetime
from typing import List, Literal, Optional
import json


from tracing import TraceBuilder

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from sqlmodel import Session

from agents.escalation_agent import EscalationAgent, EscalationDecision
from agents.memory_agent import UserMemoryAgent
from agents.response_agent import ResponseAgent
from agents.tool_planner_agent import ToolPlannerAgent
from tools import CurrencyFXTool, CurrencyCalculatorTool
from config import ADMIN_TOKEN, DOCS_DIR, FAISS_INDEX_DIR

from db import (
    Conversation,
    EscalationEvent,
    Message,
    MessageTrace,
    User,
    UserMemory,
    create_db_and_tables,
    get_session,
    select,
)
from rag.faiss_store import RAGAgent
from rag.knowledge_graph_agent import KnowledgeGraphAgent


from typing import Optional
from fastapi import Header, HTTPException
import logging

logger = logging.getLogger(__name__)

# ---------- APP SETUP ----------

app = FastAPI(
    title="ML-Industry AWS Billing Chat API",
    version="0.1.0",
)

templates = Jinja2Templates(directory="templates")

security = HTTPBearer()

# Global agent instances (initialized on startup or lazily)
rag_agent: Optional[RAGAgent] = None
response_agent: Optional[ResponseAgent] = None
user_memory_agent: Optional[UserMemoryAgent] = None
escalation_agent: Optional[EscalationAgent] = None
knowledge_graph_agent: Optional[KnowledgeGraphAgent] = None
tool_planner_agent: Optional[ToolPlannerAgent] = None

MAX_HISTORY_MESSAGES = 10


def initialize_agents_if_needed() -> None:
    global rag_agent, response_agent, user_memory_agent, escalation_agent, knowledge_graph_agent, tool_planner_agent

    if (
        rag_agent is None
        or response_agent is None
        or user_memory_agent is None
        or escalation_agent is None
        or knowledge_graph_agent is None
        or tool_planner_agent is None
    ):
        local_rag = RAGAgent(docs_dir=DOCS_DIR, index_dir=FAISS_INDEX_DIR)
        local_rag.build_or_load_index()

        local_kg = KnowledgeGraphAgent()
        if local_kg.is_graph_empty():
            for pdf_file in DOCS_DIR.glob("*.pdf"):
                local_kg.index_pdf(pdf_file)

        # Instantiate shared tools and the tool planner.
        local_fx_tool = CurrencyFXTool()
        local_calculator_tool = CurrencyCalculatorTool()
        local_tool_planner = ToolPlannerAgent()

        local_response_agent = ResponseAgent(
            rag_agent=local_rag,
            kg_agent=local_kg,
            currency_fx_tool=local_fx_tool,
            currency_calculator_tool=local_calculator_tool,
            tool_planner=local_tool_planner,
        )
        local_memory_agent = UserMemoryAgent()
        local_escalation_agent = EscalationAgent()

        rag_agent = local_rag
        knowledge_graph_agent = local_kg
        response_agent = local_response_agent
        user_memory_agent = local_memory_agent
        escalation_agent = local_escalation_agent
        tool_planner_agent = local_tool_planner

def user_requested_human_explicitly(message: str) -> bool:
    """
    Simple heuristic to detect if the user explicitly asks for a human / escalation.

    This is a best-effort string check; the EscalationAgent will still see the full text.
    """
    text = message.lower()
    keywords = [
        "talk to a human",
        "talk to human",
        "human agent",
        "live agent",
        "real person",
        "support agent",
        "escalate",
        "escalation",
        "speak to a person",
        "speak to someone",
    ]
    return any(k in text for k in text.split()) or any(k in text for k in keywords)


# ---------- AUTH HELPERS ----------

def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    session: Session = Depends(get_session),
) -> User:
    token = credentials.credentials
    user = session.exec(
        select(User).where(User.api_key == token, User.is_active == True)
    ).first()
    if not user:
        raise HTTPException(status_code=401, detail="Invalid or inactive API token")
    return user


from typing import Optional
from fastapi import Header, HTTPException
import logging

logger = logging.getLogger(__name__)

def verify_admin_token(
    x_admin_token: Optional[str] = Header(None, alias="X-Admin-Token"),
) -> None:
    """
    Verify the admin token passed from the admin UI.

    Currently this is in *permissive* mode so you can use the admin panel
    even if there is a mismatch. The strict check is left commented out
    so you can re-enable it when everything matches.
    """
    expected = ADMIN_TOKEN or "0000"

    logger.info(
        "verify_admin_token: received X-Admin-Token=%r, expected=%r",
        x_admin_token,
        expected,
    )

    # STRICT MODE (re-enable once you have confirmed the values match):
    #
    # if x_admin_token is None:
    #     raise HTTPException(
    #         status_code=401,
    #         detail="Missing admin token in X-Admin-Token header",
    #     )
    #
    # if x_admin_token != expected:
    #     raise HTTPException(status_code=401, detail="Invalid admin token")

    # Permissive mode: always allow for now
    return

# ---------- Pydantic MODELS (API SCHEMAS) ----------

class SourceCitationModel(BaseModel):
    source_file: str
    section_heading: str
    chunk_id: int
    score: float


class ChatMessageModel(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    created_at: datetime


class ChatRequest(BaseModel):
    message: str
    new_conversation: bool = False
    top_k: int = 5

class EscalationInfo(BaseModel):
    escalate: bool
    priority: str
    reason: str
    human_summary: str

class ChatResponse(BaseModel):
    answer: str
    conversation_id: int
    messages: List[ChatMessageModel]
    sources: List[SourceCitationModel]
    escalation: Optional[EscalationInfo] = None
    # NEW: long-term memory facts for this user, as plain strings
    user_memory_facts: List[str] = []

class AdminMessageTraceModel(BaseModel):
    message_id: int
    role: str
    content: str
    created_at: datetime
    trace: Optional[dict]

class EscalationEventModel(BaseModel):
    id: int
    conversation_id: int
    user_id: int
    level: str
    human_summary: str
    created_at: datetime
    acknowledged: bool

class UserCreate(BaseModel):
    name: str
    api_key: Optional[str] = None  # if not provided, auto-generated


class UserUpdate(BaseModel):
    name: Optional[str] = None
    is_active: Optional[bool] = None


class UserRead(BaseModel):
    id: int
    name: str
    api_key: str
    is_active: bool
    created_at: datetime


# ---------- STARTUP ----------

@app.on_event("startup")
def on_startup() -> None:
    # Ensure DB tables exist
    create_db_and_tables()

    # Initialize agents (idempotent)
    initialize_agents_if_needed()

    print(f"RAG initialized with docs from: {DOCS_DIR}")
    print(f"FAISS index directory: {FAISS_INDEX_DIR}")


@app.get("/", include_in_schema=False)
def chat_page(request: Request):
    """
    User-facing chat UI.
    """
    return templates.TemplateResponse("chat.html", {"request": request})


@app.get("/admin-ui", include_in_schema=False)
def admin_page(request: Request):
    """
    Admin-facing UI for managing users and API keys.
    """
    return templates.TemplateResponse("admin.html", {"request": request})

# ---------- CHAT ENDPOINT ----------

@app.post("/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    initialize_agents_if_needed()
    if response_agent is None:
        raise HTTPException(
            status_code=500,
            detail="Response agent could not be initialized",
        )

    # 1) Find or create conversation for this user
    conversation: Optional[Conversation] = None
    if not request.new_conversation:
        conversation = session.exec(
            select(Conversation)
            .where(Conversation.user_id == user.id)
            .order_by(Conversation.last_activity_at.desc())
        ).first()

    if conversation is None:
        conversation = Conversation(user_id=user.id)
        session.add(conversation)
        session.commit()
        session.refresh(conversation)

    # 2) Load conversation history for short-term memory (without current message)
    history_messages: List[Message] = session.exec(
        select(Message)
        .where(Message.conversation_id == conversation.id)
        .order_by(Message.created_at)
    ).all()

    recent_history = history_messages[-MAX_HISTORY_MESSAGES:]
    chat_history = [(m.role, m.content) for m in recent_history]

    # 3) Store the new user message
    user_msg = Message(
        conversation_id=conversation.id,
        role="user",
        content=request.message,
    )
    session.add(user_msg)
    session.commit()
    session.refresh(user_msg)

    # 4) Initialize a per-message trace builder for admin observability.
    trace_builder = TraceBuilder.create(
        conversation_id=conversation.id,
        message_id=user_msg.id,
        user_id=user.id,
    )

    # 5) Long-term memory update based on the new user message
    changes = user_memory_agent.update_memory_in_db(
        user_id=user.id,
        user_message=request.message,
        session=session,
        trace_builder=trace_builder,
    )
    print("DB for Memory changed? ", changes)

    # 6) Fetch user memory (after potential update)
    user_memories = session.exec(
        select(UserMemory).where(UserMemory.user_id == user.id)
    ).all()
    user_memory_text = (
        "\n".join(f"- {m.fact}" for m in user_memories) if user_memories else ""
    )
    # NEW: for UI display (simple list of strings)
    user_memory_facts = [m.fact for m in user_memories]

    # 7) Call RAG + ResponseAgent
    # Pass previous history (without the current message) to the agent
    answer_obj = response_agent.answer(
        question=request.message,
        k=request.top_k,
        chat_history=chat_history,
        user_memory=user_memory_text,
        trace_builder=trace_builder,
    )

    # 8) Store assistant message with initial text
    final_answer_text = answer_obj.answer_text
    reasoning = answer_obj.reasoning
    assistant_msg = Message(
        conversation_id=conversation.id,
        role="assistant",
        content=final_answer_text,
    )
    session.add(assistant_msg)

    conversation.last_activity_at = datetime.utcnow()
    session.add(conversation)
    session.commit()
    session.refresh(assistant_msg)

    # 9) Prepare history for response and escalation
    updated_messages = history_messages + [user_msg, assistant_msg]

    # 10) Escalation analysis (P0/P1/P2/NONE)
    escalation_info: Optional[EscalationInfo] = None

    # Make sure you have this somewhere before the escalation block:
    # final_answer_text = answer_obj.answer_text

    if escalation_agent is not None:
        # Prepare conversation history for the escalation agent
        convo_for_escalation = [
            (m.role, m.content) for m in updated_messages[-MAX_HISTORY_MESSAGES:]
        ]

        # Simple heuristics for metadata flags
        user_requested_human_flag = user_requested_human_explicitly(request.message)
        rag_no_results = len(answer_obj.citations) == 0
        rag_top_score: Optional[float] = None
        if not rag_no_results:
            try:
                rag_top_score = max(c.score for c in answer_obj.citations)
            except Exception:
                rag_top_score = None

        # For now we do not have sentiment or failed-attempt tracking wired in,
        # so we pass defaults for those fields.
        decision = escalation_agent.analyze(
            latest_user_message=request.message,
            conversation_history=convo_for_escalation,
            assistant_answer=answer_obj.answer_text,
            sentiment_label=None,
            sentiment_score=None,
            num_failed_attempts=0,
            user_explicitly_requested_human=user_requested_human_flag,
            rag_top_score=rag_top_score,
            rag_no_results=rag_no_results,
            trace_builder=trace_builder,
        )

        if decision.escalate:
            # 10a) Persist escalation info on the conversation
            conversation.escalation_level = decision.priority
            if conversation.escalated_at is None:
                conversation.escalated_at = datetime.utcnow()
            conversation.escalation_reason = decision.reason
            session.add(conversation)

            # 10b) Create an escalation event (notification) for the admin
            escalation_event = EscalationEvent(
                conversation_id=conversation.id,
                user_id=user.id,
                level=decision.priority,
                human_summary=decision.human_summary,
            )
            session.add(escalation_event)

            session.commit()

            # 10c) Only show human-support notice for P0
            if decision.priority == "P0":
                support_notice = (
                    "I have notified our human AWS Billing support team about your issue. "
                    "They will review your case and contact you as soon as possible."
                )
                final_answer_text = f"{answer_obj.answer_text}\n\n---\n\n{support_notice}"

                # Update the stored assistant message to include the notice
                assistant_msg.content = final_answer_text
                session.add(assistant_msg)
                session.commit()

        escalation_info = EscalationInfo(
            escalate=decision.escalate,
            priority=decision.priority,
            reason=decision.reason,
            human_summary=decision.human_summary,
        )

    # 11) Build and persist per-message trace
    if trace_builder is not None:
        trace_builder.build_pipeline_summary()
        trace_row = MessageTrace(
            conversation_id=conversation.id,
            message_id=user_msg.id,
            user_id=user.id,
            trace_json=trace_builder.to_json(),
        )
        session.add(trace_row)
        session.commit()

    # 12) Build response messages and sources from updated state
    # Ensure we use the (possibly updated) assistant_msg content
    updated_messages[-1] = assistant_msg
    recent_for_response = updated_messages[-MAX_HISTORY_MESSAGES:]

    response_messages = [
        ChatMessageModel(
            role=m.role,
            content=m.content,
            created_at=m.created_at,
        )
        for m in recent_for_response
    ]

    sources = [
        SourceCitationModel(
            source_file=c.source_file,
            section_heading=c.section_heading,
            chunk_id=c.chunk_id,
            score=c.score,
        )
        for c in answer_obj.citations
    ]

    return ChatResponse(
        answer=final_answer_text,
        conversation_id=conversation.id,
        messages=response_messages,
        sources=sources,
        escalation=escalation_info,
        user_memory_facts=user_memory_facts,
    )

# ---------- ADMIN: USER MANAGEMENT ----------

@app.post(
    "/admin/users",
    response_model=UserRead,
    dependencies=[Depends(verify_admin_token)],
)
def create_user(
    user_in: UserCreate,
    session: Session = Depends(get_session),
):
    api_key = user_in.api_key or secrets.token_urlsafe(32)

    user = User(
        name=user_in.name,
        api_key=api_key,
        is_active=True,
    )
    session.add(user)
    session.commit()
    session.refresh(user)
    return user


@app.get(
    "/admin/users",
    response_model=List[UserRead],
    dependencies=[Depends(verify_admin_token)],
)
def list_users(
    session: Session = Depends(get_session),
):
    users = session.exec(select(User).order_by(User.created_at)).all()
    return users


@app.patch(
    "/admin/users/{user_id}",
    response_model=UserRead,
    dependencies=[Depends(verify_admin_token)],
)
def update_user(
    user_id: int,
    user_in: UserUpdate,
    session: Session = Depends(get_session),
):
    user = session.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if user_in.name is not None:
        user.name = user_in.name
    if user_in.is_active is not None:
        user.is_active = user_in.is_active

    session.add(user)
    session.commit()
    session.refresh(user)
    return user


@app.delete(
    "/admin/users/{user_id}",
    dependencies=[Depends(verify_admin_token)],
)
def delete_user(
    user_id: int,
    session: Session = Depends(get_session),
):
    user = session.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    session.delete(user)
    session.commit()
    return {"detail": "User deleted"}

@app.get("/admin/escalations", response_model=List[EscalationEventModel])
def list_escalations(
    only_unacknowledged: bool = True,
    _: None = Depends(verify_admin_token),
    session: Session = Depends(get_session),
):
    """
    List escalation events for the admin.

    By default, only returns events that have not been acknowledged yet.
    Set `only_unacknowledged=false` to see all.
    """
    stmt = select(EscalationEvent).order_by(EscalationEvent.created_at.desc())
    if only_unacknowledged:
        stmt = stmt.where(EscalationEvent.acknowledged == False)  # noqa: E712

    events = session.exec(stmt).all()
    return [
        EscalationEventModel(
            id=e.id,
            conversation_id=e.conversation_id,
            user_id=e.user_id,
            level=e.level,
            human_summary=e.human_summary,
            created_at=e.created_at,
            acknowledged=e.acknowledged,
        )
        for e in events
    ]


@app.post("/admin/escalations/{event_id}/ack", response_model=EscalationEventModel)
def acknowledge_escalation(
    event_id: int,
    _: None = Depends(verify_admin_token),
    session: Session = Depends(get_session),
):
    """
    Mark an escalation event as acknowledged/handled by the admin.
    """
    event = session.get(EscalationEvent, event_id)
    if event is None:
        raise HTTPException(status_code=404, detail="Escalation event not found")

    event.acknowledged = True
    session.add(event)
    session.commit()
    session.refresh(event)

    return EscalationEventModel(
        id=event.id,
        conversation_id=event.conversation_id,
        user_id=event.user_id,
        level=event.level,
        human_summary=event.human_summary,
        created_at=event.created_at,
        acknowledged=event.acknowledged,
    )

@app.get(
    "/admin/conversations/{conversation_id}/traces",
    response_model=List[AdminMessageTraceModel],
)
def get_conversation_traces(
    conversation_id: int,
    _: None = Depends(verify_admin_token),
    session: Session = Depends(get_session),
):
    """
    Return all messages of a conversation with their trace payloads.
    Used by the admin UI to inspect which agents/tools were used.
    """
    # Load all messages for this conversation, ordered by creation time.
    messages = (
        session.query(Message)
        .filter(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.asc())
        .all()
    )

    if not messages:
        return []

    message_ids = [m.id for m in messages]

    # Load all traces for these messages in a single query.
    traces = (
        session.query(MessageTrace)
        .filter(
            MessageTrace.conversation_id == conversation_id,
            MessageTrace.message_id.in_(message_ids),
        )
        .all()
    )
    traces_by_message_id = {t.message_id: t for t in traces}

    results: List[AdminMessageTraceModel] = []

    for msg in messages:
        trace_row = traces_by_message_id.get(msg.id)
        if trace_row is not None:
            try:
                trace_data = json.loads(trace_row.trace_json)
            except Exception:
                # If parsing fails, expose the raw string as a best-effort.
                trace_data = {"_raw": trace_row.trace_json}
        else:
            trace_data = None

        results.append(
            AdminMessageTraceModel(
                message_id=msg.id,
                role=msg.role,
                content=msg.content,
                created_at=msg.created_at,
                trace=trace_data,
            )
        )

    return results


