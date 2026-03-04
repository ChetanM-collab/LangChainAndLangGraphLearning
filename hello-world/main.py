import os
from dotenv import load_dotenv
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate


load_dotenv()


def main():
    print("Hello from langchain-course!")
    # llm = OllamaLLM(model="llama3.2", temperature=0.0)
    llm = OllamaLLM(model="gemma3:latest", temperature=0.0)
    prompt = PromptTemplate(
        input_variables=["product"],
        template="What is the best name for a company that makes {product}?",
    )
    # chain = LLMChain(llm=llm, prompt=prompt)
    chain = prompt | llm  # creates RunnableInterface object that can be invoked.
    result = chain.invoke({"product": "socks"})
    print(result)


if __name__ == "__main__":
    main()

