import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

load_dotenv()
# ---------- PATHS ----------

# Folder where you will manually put downloaded AWS billing docs
DOCS_DIR = Path(os.getenv("DOCS_DIR", "./data"))

# Where to persist the FAISS index
FAISS_INDEX_DIR = Path(os.getenv("FAISS_INDEX_DIR", "./data/faiss_index"))

# Make sure directories exist
DOCS_DIR.mkdir(parents=True, exist_ok=True)
FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)

# ---------- DATABASE / LOGGING ----------

DB_PATH = Path(os.getenv("DB_PATH", "./data/chat_logs.db"))
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# Admin token for managing users via /admin endpoints
# Set this as an environment variable in real deployments.
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "0000")                    #Repalce here

# ---------- NEO4J CONFIG ----------

# Load from environment variables OR fall back to defaults (for local testing)
NEO4J_URI = os.getenv("NEO4J_URI", "neo4j+s://14d5e4dd.databases.neo4j.io")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "ajKgAyzSjCkjtYfRSpjr1TmHOS39pZMvdAvqaDxA7Fc")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

# ---------- LLM PROVIDER ----------

BASE_URL = os.getenv("LLM_BASE_URL", "http://a6k2.dgx:34000/v1")
API_KEY = os.getenv("LLM_API_KEY", "your-default-or-fail-safe-key")
MODEL_NAME = os.getenv("LLM_MODEL_NAME", "qwen3-32b")

llm = ChatOpenAI(
    base_url=BASE_URL,
    api_key=API_KEY,
    model=MODEL_NAME,
    temperature=0.0,      # deterministic for support answers
)

# ---------- CURRENCY FX API CONFIG ----------

# Base URL for the currency exchange API.
# Default uses the free Frankfurter API (no API key required).
FX_API_BASE_URL = os.getenv(
    "FX_API_BASE_URL",
    "https://api.frankfurter.dev/v1",
)

# Optional API key; many free endpoints like Frankfurter do not require it,
# but this allows you to switch providers without touching code.
FX_API_KEY = os.getenv("FX_API_KEY", "")

# Network configuration for FX API calls.
FX_API_TIMEOUT_SECONDS = float(os.getenv("FX_API_TIMEOUT_SECONDS", "5.0"))
FX_API_MAX_RETRIES = int(os.getenv("FX_API_MAX_RETRIES", "2"))

# Feature flag: allows disabling FX functionality cleanly if needed.
FX_API_ENABLED = os.getenv("FX_API_ENABLED", "true").lower() == "true"

# ---------- EMBEDDINGS (LOCAL, CPU) ----------

# Local sentence-transformers model; runs on CPU.
# You can change this to any HF sentence-transformers model you prefer.
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
)

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs={"device": "cpu"},
)

