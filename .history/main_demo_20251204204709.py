from rag.faiss_store import RAGAgent
from agents.response_agent import ResponseAgent
from config import DOCS_DIR, FAISS_INDEX_DIR

def main():
    rag_agent = RAGAgent(docs_dir=DOCS_DIR, index_dir=FAISS_INDEX_DIR)
    rag_agent.build_or_load_index()

    response_agent = ResponseAgent(rag_agent=rag_agent)

    question = input("Ask an AWS billing question: ")
    answer = response_agent.answer(question, k=5)

    print("\n=== ANSWER ===")
    print(answer.answer_text)

    print("\n=== SOURCES ===")
    for src in answer.citations:
        print(
            f"- file: {src.source_file} | section: '{src.section_heading}' "
            f"| chunk_id: {src.chunk_id} | score: {src.score:.3f}"
        )

if __name__ == "__main__":
    main()
