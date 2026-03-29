"""
Natural Health RAG - Retrieval Chain
Builds the full RAG pipeline: retrieve → format → generate
"""

import os
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from embeddings.vectorstore import get_retriever


# ── System Prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a Natural Health Research Assistant with access to a 
curated knowledge base of evidence-based medical literature, including:
- NIH Office of Dietary Supplements fact sheets
- NCCIH (National Center for Complementary and Integrative Health) herb profiles  
- PubMed clinical research abstracts
- ClinicalTrials.gov study summaries
- WHO & EMA herbal medicine monographs

Your role is to bridge the gap between ancient herbal wisdom and modern clinical evidence.

GUIDELINES:
1. Always distinguish between "traditional use" and "clinical evidence"
2. Cite the source type (e.g., "According to a PubMed study..." or "NIH ODS notes...")
3. Be honest about evidence quality — note when studies are small or preliminary
4. Never recommend replacing prescribed medication without consulting a doctor
5. If evidence is conflicting, present both sides
6. Flag safety concerns (drug interactions, contraindications) when relevant

CONTEXT FROM KNOWLEDGE BASE:
{context}
"""

HUMAN_PROMPT = "{question}"


# ── Context Formatter ─────────────────────────────────────────────────────────

def format_docs(docs) -> str:
    """Format retrieved documents into a readable context block."""
    formatted = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source_type", "Unknown")
        url = doc.metadata.get("url", "")
        herb = doc.metadata.get("herb", "")

        header = f"[Source {i}: {source}"
        if herb:
            header += f" — {herb.replace('-', ' ').title()}"
        if url:
            header += f" | {url}"
        header += "]"

        formatted.append(f"{header}\n{doc.page_content}")

    return "\n\n---\n\n".join(formatted)


# ── Build Chain ───────────────────────────────────────────────────────────────

def build_rag_chain(
    backend: str = "chroma",
    k: int = 5,
    source_filter: Optional[str] = None,
    streaming: bool = False,
    model: str = "gpt-4o-mini",
):
    """
    Builds the full RAG chain.

    Args:
        backend: "chroma" or "pinecone"
        k: number of documents to retrieve
        source_filter: filter by source_type (e.g., "PubMed", "NCCIH")
        streaming: stream tokens to stdout
        model: OpenAI model name

    Returns:
        A LangChain LCEL chain
    """
    # Retriever
    retriever = get_retriever(backend=backend, k=k, source_filter=source_filter)

    # LLM
    callbacks = [StreamingStdOutCallbackHandler()] if streaming else []
    llm = ChatOpenAI(
        model=model,
        temperature=0.2,   # low temp = factual, consistent
        streaming=streaming,
        callbacks=callbacks,
        openai_api_key=os.environ["OPENAI_API_KEY"],
    )

    # Prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", HUMAN_PROMPT),
    ])

    # Chain (LCEL)
    rag_chain = (
        RunnableParallel({
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        })
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


# ── Chain with sources ────────────────────────────────────────────────────────

def build_rag_chain_with_sources(backend: str = "chroma", k: int = 5):
    """
    Returns both the answer AND the source documents used.
    Useful for building UIs that show citations.
    """
    retriever = get_retriever(backend=backend, k=k)

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.2,
        openai_api_key=os.environ["OPENAI_API_KEY"],
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", HUMAN_PROMPT),
    ])

    # Retrieve docs, keep them for source display
    retrieve_docs = RunnableParallel({
        "docs": retriever,
        "question": RunnablePassthrough(),
    })

    def format_with_sources(inputs):
        docs = inputs["docs"]
        question = inputs["question"]
        context = format_docs(docs)
        chain = prompt | llm | StrOutputParser()
        answer = chain.invoke({"context": context, "question": question})
        return {
            "answer": answer,
            "sources": [
                {
                    "source_type": d.metadata.get("source_type"),
                    "url": d.metadata.get("url", ""),
                    "snippet": d.page_content[:200] + "...",
                }
                for d in docs
            ]
        }

    return retrieve_docs | format_with_sources


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    chain = build_rag_chain(streaming=True)
    
    test_questions = [
        "What does the evidence say about rosemary oil for hair loss?",
        "Is saw palmetto effective for male pattern baldness?",
        "What are the safety concerns with taking biotin supplements?",
        "Compare traditional and modern uses of ashwagandha",
    ]

    for q in test_questions[:1]:  # run first question as smoke test
        print(f"\n{'='*60}")
        print(f"Q: {q}")
        print(f"{'='*60}")
        result = chain.invoke(q)
        print()
