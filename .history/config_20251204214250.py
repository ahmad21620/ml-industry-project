import os
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings


# ---------- PATHS ----------

# Folder where you will manually put downloaded AWS billing docs
DOCS_DIR = Path(os.getenv("DOCS_DIR", "./data"))

# Where to persist the FAISS index
FAISS_INDEX_DIR = Path(os.getenv("FAISS_INDEX_DIR", "./data/faiss_index"))

# Make sure directories exist
DOCS_DIR.mkdir(parents=True, exist_ok=True)
FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)

# ---------- LLM PROVIDER ----------

BASE_URL = "http://a6k2.dgx:34000/v1"
API_KEY = "sk-0LicA8eVoVLcwZMJrw4lJQ"
MODEL_NAME = "qwen3-32b"

llm = ChatOpenAI(
    base_url=BASE_URL,
    api_key=API_KEY,
    model=MODEL_NAME,
    temperature=0.0,      # deterministic for support answers
)

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

