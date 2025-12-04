from rag.faiss_store import RAGAgent
from config import DOCS_DIR, FAISS_INDEX_DIR

if __name__ == "__main__":
    print(f"Building FAISS index from docs in: {DOCS_DIR}")
    rag_agent = RAGAgent(docs_dir=DOCS_DIR, index_dir=FAISS_INDEX_DIR)
    rag_agent.build_or_load_index()
    print(f"Index ready at: {FAISS_INDEX_DIR}")
