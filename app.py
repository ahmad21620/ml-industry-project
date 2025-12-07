from __future__ import annotations

import secrets
from datetime import datetime
from typing import List, Literal, Optional

from agents.memory_agent import UserMemoryAgent
from agents.response_agent import ResponseAgent
from config import ADMIN_TOKEN, DOCS_DIR, FAISS_INDEX_DIR
from db import (
    Conversation,
    Message,
    User,
    UserMemory,
    create_db_and_tables,
    get_session,
    select,
)
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from rag.faiss_store import RAGAgent
from sqlmodel import Session

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

MAX_HISTORY_MESSAGES = 10


def initialize_agents_if_needed() -> None:
    """
    Ensure RAGAgent and ResponseAgent are initialized.

    This is called on startup and also lazily in the /chat endpoint,
    so that the system still works even if the startup event was skipped
    or failed previously.
    """
    global rag_agent, response_agent, user_memory_agent

    if rag_agent is None or response_agent is None:
        local_rag = RAGAgent(docs_dir=DOCS_DIR, index_dir=FAISS_INDEX_DIR)
        local_rag.build_or_load_index()

        local_response_agent = ResponseAgent(rag_agent=local_rag)
        local_user_memory_agent = UserMemoryAgent()

        rag_agent = local_rag
        response_agent = local_response_agent
        user_memory_agent = local_user_memory_agent

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


def verify_admin_token(
    x_admin_token: str = Header(..., alias="X-Admin-Token"),
) -> None:
    if x_admin_token != ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid admin token")


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


class ChatResponse(BaseModel):
    answer: str
    conversation_id: int
    messages: List[ChatMessageModel]
    sources: List[SourceCitationModel]


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
    # Fetch current facts
    current_memories = session.exec(
        select(UserMemory).where(UserMemory.user_id == user.id)
    ).all()
    current_facts = [m.fact for m in current_memories]
    # Propose & apply update
    changes = user_memory_agent.update_memory_in_db(
        user_id=user.id,
        user_message=request.message,
        session=session

    )
    print("DB for Memory changed? ", changes)
    # print("new facts are", new_facts)
    # # Sync DB: delete old, insert new
    # if set(new_facts) != set(current_facts):
    #     # Delete all old memory entries
    #     for m in current_memories:
    #         session.delete(m)
    #     # Add new ones
    #     for fact in new_facts:
    #         session.add(UserMemory(user_id=user.id, fact=fact))
    #     session.commit()

    # 2) Load conversation history for short-term memory
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

    # 4) Call RAG + ResponseAgent
    # Pass previous history (without the current message) to the agent
# Fetch user memory
    user_memories = session.exec(
        select(UserMemory).where(UserMemory.user_id == user.id)
    ).all()
    user_memory_text = "\n".join(f"- {m.fact}" for m in user_memories) if user_memories else ""

    # Call agent with memory
    answer_obj = response_agent.answer(
        question=request.message,
        k=request.top_k,
        chat_history=chat_history,
        user_memory=user_memory_text,
    )

    # 5) Store assistant reply
    assistant_msg = Message(
        conversation_id=conversation.id,
        role="assistant",
        content=answer_obj.answer_text,
    )
    session.add(assistant_msg)

    conversation.last_activity_at = datetime.utcnow()
    session.add(conversation)
    session.commit()
    session.refresh(assistant_msg)

    # 6) Build response: recent messages + new turn
    updated_messages = history_messages + [user_msg, assistant_msg]
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
        answer=answer_obj.answer_text,
        conversation_id=conversation.id,
        messages=response_messages,
        sources=sources,
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
