import os
from pathlib import Path
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# ---------- PATHS ----------

# Folder where you will manually put downloaded AWS billing docs
DOCS_DIR = Path(os.getenv("DOCS_DIR", "./data/aws_billing_docs"))

# Where to persist the FAISS index
FAISS_INDEX_DIR = Path(os.getenv("FAISS_INDEX_DIR", "./data/faiss_index"))

# Make sure directories exist
DOCS_DIR.mkdir(parents=True, exist_ok=True)
FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)

# ---------- LLM PROVIDER ----------

BASE_URL = os.getenv("LLM_BASE_URL", "http://a6k2.dgx:34000/v1")
API_KEY = os.getenv("LLM_API_KEY", "REPLACE_ME")  # do NOT hardcode real key in repo
MODEL_NAME = os.getenv("LLM_MODEL_NAME", "qwen3-32b")

llm = ChatOpenAI(
    base_url=BASE_URL,
    api_key=API_KEY,
    model=MODEL_NAME,
    temperature=0.0,      # deterministic for support answers
)

# ---------- EMBEDDINGS ----------

# Adjust model name to match what your endpoint actually exposes for embeddings.
# If your provider does not expose embeddings, switch to a local HF model here.
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-3-large")

embeddings = OpenAIEmbeddings(
    base_url=BASE_URL,
    api_key=API_KEY,
    model=EMBEDDING_MODEL_NAME,
)
