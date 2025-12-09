from __future__ import annotations

from datetime import datetime
from typing import Optional

from config import DB_PATH
from sqlmodel import Field, Session, SQLModel, create_engine, select

# ---------- ENGINE ----------

engine = create_engine(f"sqlite:///{DB_PATH}", echo=False)


# ---------- MODELS ----------

class User(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    api_key: str = Field(index=True, unique=True)
    is_active: bool = Field(default=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Conversation(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="user.id")
    started_at: datetime = Field(default_factory=datetime.utcnow)
    last_activity_at: datetime = Field(default_factory=datetime.utcnow)

    # Escalation-related fields
    escalation_level: Optional[str] = Field(
        default=None,
        index=True,
        description='Escalation severity: "P0", "P1", "P2", or None if not escalated.',
    )
    escalated_at: Optional[datetime] = Field(
        default=None,
        description="Timestamp when the conversation was first escalated.",
    )
    escalation_reason: Optional[str] = Field(
        default=None,
        description="Short internal reason explaining why the conversation was escalated.",
    )



class Message(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    conversation_id: int = Field(foreign_key="conversation.id")
    role: str  # "user" or "assistant"
    content: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

class UserMemory(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="user.id", index=True)
    fact: str = Field(max_length=500)  # free-text memory entry
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class EscalationEvent(SQLModel, table=True):
    """
    A notification to the admin about an escalated conversation.

    This is the record that represents:
    - which user and conversation were escalated,
    - at what level (P0/P1/P2),
    - with which human-readable summary,
    - when it happened,
    - and whether it has been acknowledged/handled.
    """
    id: Optional[int] = Field(default=None, primary_key=True)
    conversation_id: int = Field(foreign_key="conversation.id")
    user_id: int = Field(foreign_key="user.id")
    level: str = Field(
        description='Escalation level: "P0", "P1", or "P2".',
    )
    human_summary: str = Field(
        description="Short description of the issue suitable for the admin."
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)
    acknowledged: bool = Field(
        default=False,
        description="True when an admin has seen/handled this escalation.",
    )

class MessageTrace(SQLModel, table=True):
    """Per-message trace for admin observability.

    Stores a JSON-serialized trace document describing which agents and tools
    were used for a given user message, along with context, currency, and
    escalation metadata.
    """

    id: Optional[int] = Field(default=None, primary_key=True)

    conversation_id: int = Field(foreign_key="conversation.id")
    message_id: int = Field(foreign_key="message.id")
    user_id: int = Field(foreign_key="user.id")

    # When this trace was recorded (UTC).
    timestamp_utc: datetime = Field(default_factory=datetime.utcnow)

    # JSON string containing the structured trace payload.
    trace_json: str
# ---------- DB HELPERS ----------

def create_db_and_tables() -> None:
    SQLModel.metadata.create_all(engine)


def get_session():
    with Session(engine) as session:
        yield session


# Re-export select so callers can import from db
__all__ = [
    "User",
    "Conversation",
    "Message",
    "UserMemory",
    "engine",
    "create_db_and_tables",
    "get_session",
    "select",
]