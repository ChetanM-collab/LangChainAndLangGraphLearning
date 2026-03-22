from operator import itemgetter
import os
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
load_dotenv()

embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
vectorstore = PineconeVectorStore(index_name=os.environ["INDEX_NAME"], embedding=embeddings)   
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

prompt = ChatPromptTemplate.from_template(
    """
    Answer the question based only on the following context:
    {context}
    Question: {question}
    Provide a Detailed and long Answer:
    """
)

def format_docs(docs):
    """Concatenate LangChain ``Document`` objects into one context block.

    Args:
        docs: Iterable of documents (e.g. from the Pinecone retriever), each with
            a ``page_content`` attribute.

    Returns:
        A single string with document bodies joined by double newlines.
    """
    return "\n\n".join(doc.page_content for doc in docs)

def retrieve_and_answer_without_lcel(query: str):
    """Answer a question using RAG without composing an LCEL pipeline.

    Retrieves top-k chunks from Pinecone, builds the chat prompt with
    ``context`` and ``question``, and invokes the LLM directly.

    Args:
        query: User question string.

    Returns:
        An ``AIMessage`` from ``ChatOpenAI.invoke`` (not plain text).
    """
    docs = retriever.invoke(query)
    context = format_docs(docs)
    messages = prompt.format_messages(context=context, question=query)
    return llm.invoke(messages)

def create_retriever_chain():
    """Return an LCEL RAG chain: retrieve context, prompt, model, then string output.

    Pipeline:
        1. ``RunnablePassthrough.assign`` — takes ``{"question": ...}``, adds
           ``context`` by piping ``question`` through the retriever and
           ``format_docs``.
        2. ``ChatPromptTemplate`` — fills ``{context}`` and ``{question}``.
        3. ``ChatOpenAI`` — generates the answer.
        4. ``StrOutputParser`` — returns the assistant text as a string.

    Returns:
        A runnable that expects ``invoke({"question": "<user question>"})`` and
        returns the final answer string.
    """
    retriever_chain = (
        RunnablePassthrough.assign(
            context=itemgetter("question") | retriever | format_docs,
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    return retriever_chain

if __name__ == "__main__":
    query = "What is a vector database?"
    
    results = retrieve_and_answer_without_lcel(query)
    print("-"*100)
    print("Implementation 1 - Without lcel: \n")
    print("-"*100)
    print(results)

    retriever_chain = create_retriever_chain()
    results = retriever_chain.invoke({"question": query})
    print("-"*100)
    print("Implememntation 2 - With lcel: \n")
    print("-"*100)
    print(results)
