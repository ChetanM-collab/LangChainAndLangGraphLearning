"""
Natural Health RAG - Embedding & Vector Store
Supports: ChromaDB (local/free) or Pinecone (cloud)
"""

import os
from pathlib import Path
from typing import List, Literal

from dotenv import load_dotenv

# Load natural-health-rag/.env when this module is imported (any cwd)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings  # free local option
from langchain_chroma import Chroma
from langchain_pinecone import PineconeVectorStore

# ── Config ────────────────────────────────────────────────────────────────────

CHROMA_DIR     = "./data/chroma_db"
COLLECTION_NAME = "natural_health"

# Choose embedding model: "openai" | "ollama" (free, local)
EMBEDDING_BACKEND: Literal["openai", "ollama"] = os.getenv("EMBEDDING_BACKEND", "openai")


# ── Embeddings ────────────────────────────────────────────────────────────────

def get_embeddings():
    """
    Returns an embedding model.
    - openai: best quality, requires OPENAI_API_KEY
    - ollama: free & local, run `ollama pull nomic-embed-text` first
    """
    if EMBEDDING_BACKEND == "ollama":
        print("🧠 Using Ollama embeddings (nomic-embed-text) — free & local")
        return OllamaEmbeddings(model="nomic-embed-text")
    else:
        print("🧠 Using OpenAI embeddings (text-embedding-3-small)")
        return OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=os.environ["OPENAI_API_KEY"],
        )


# ── ChromaDB (local, free) ────────────────────────────────────────────────────

def build_chroma(chunks: List[Document]) -> Chroma:
    """
    Build or update a local ChromaDB vector store.
    No API key needed. Data persisted to disk at CHROMA_DIR.
    """
    print(f"\n🗄️  Building ChromaDB at {CHROMA_DIR}...")
    embeddings = get_embeddings()

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
    )
    print(f"  ✅ ChromaDB built with {len(chunks)} chunks")
    return vectorstore


def load_chroma() -> Chroma:
    """Load an existing ChromaDB from disk."""
    embeddings = get_embeddings()
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )


# ── Pinecone (cloud, scales better) ──────────────────────────────────────────

def build_pinecone(chunks: List[Document], index_name: str = "natural-health") -> PineconeVectorStore:
    """
    Build a Pinecone vector store.
    Requires:
      - PINECONE_API_KEY env var
      - A Pinecone index named `index_name` with dimension=1536 (OpenAI)
        or dimension=768 (Ollama nomic-embed-text)
    """
    from pinecone import Pinecone, ServerlessSpec

    print(f"\n☁️  Building Pinecone index '{index_name}'...")
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

    # Create index if it doesn't exist
    if index_name not in [i.name for i in pc.list_indexes()]:
        dim = 1536 if EMBEDDING_BACKEND == "openai" else 768
        pc.create_index(
            name=index_name,
            dimension=dim,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        print(f"  📦 Created Pinecone index '{index_name}' (dim={dim})")

    embeddings = get_embeddings()
    vectorstore = PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=index_name,
    )
    print(f"  ✅ Pinecone index built with {len(chunks)} chunks")
    return vectorstore


# ── Retriever factory ─────────────────────────────────────────────────────────

def get_retriever(backend: Literal["chroma", "pinecone"] = "chroma", **kwargs):
    """
    Returns a LangChain retriever ready to use in a chain.

    Args:
        backend: "chroma" (local/free) or "pinecone" (cloud)
        k: number of chunks to retrieve (default 5)
        source_filter: optionally filter by source_type metadata
    """
    k = kwargs.get("k", 5)
    source_filter = kwargs.get("source_filter", None)

    if backend == "chroma":
        vs = load_chroma()
    else:
        index_name = kwargs.get("index_name", "natural-health")
        embeddings = get_embeddings()
        vs = PineconeVectorStore(
            index_name=index_name,
            embedding=embeddings,
        )

    search_kwargs = {"k": k}
    if source_filter:
        search_kwargs["filter"] = {"source_type": source_filter}

    return vs.as_retriever(search_type="similarity", search_kwargs=search_kwargs)
