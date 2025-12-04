from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlmodel import SQLModel, Field, create_engine, Session, select

from config import DB_PATH


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


class Message(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    conversation_id: int = Field(foreign_key="conversation.id")
    role: str  # "user" or "assistant"
    content: str
    created_at: datetime = Field(default_factory=datetime.utcnow)


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
    "engine",
    "create_db_and_tables",
    "get_session",
    "select",
]