from rag.faiss_store import RAGAgent
from agents.response_agent import ResponseAgent
from agents.tool_planner_agent import ToolPlannerAgent
from tools import CurrencyFXTool, CurrencyCalculatorTool
from config import DOCS_DIR, FAISS_INDEX_DIR


def main():
    print(f"Using docs from: {DOCS_DIR}")
    print(f"Using FAISS index in: {FAISS_INDEX_DIR}")

    rag_agent = RAGAgent(docs_dir=DOCS_DIR, index_dir=FAISS_INDEX_DIR)
    rag_agent.build_or_load_index()

    tool_planner = ToolPlannerAgent()
    fx_tool = CurrencyFXTool()
    calculator_tool = CurrencyCalculatorTool()

    response_agent = ResponseAgent(
        rag_agent=rag_agent,
        kg_agent=None,
        currency_fx_tool=fx_tool,
        currency_calculator_tool=calculator_tool,
        tool_planner=tool_planner,
    )

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
