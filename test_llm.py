# test_llm.py

from langchain_openai import ChatOpenAI

BASE_URL = "http://a6k2.dgx:34000/v1"
API_KEY = "sk-0LicA8eVoVLcwZMJrw4lJQ"
MODEL_NAME = "qwen3-32b"

def main():
    llm = ChatOpenAI(
        base_url=BASE_URL,
        api_key=API_KEY,
        model=MODEL_NAME,
        temperature=0.0,
    )

    question = "What are the primary colors?"
    response = llm.invoke(question)

    # For ChatOpenAI, response is a ChatMessage object
    print("Question:", question)
    print("Answer:", response.content)

if __name__ == "__main__":
    main()
