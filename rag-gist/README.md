## RAG Gist

A minimal **Retrieval-Augmented Generation (RAG)** pipeline: load documents, chunk them, embed and store in Pinecone, then answer questions using retrieved context and an LLM.

### Files

| File | Purpose |
|------|---------|
| `ingestion.py` | Loads a text file, splits into chunks, embeds with OpenAI, and upserts into Pinecone |
| `main.py` | Runs RAG queries: retrieves top-k chunks, builds prompt with context, invokes the LLM |

### What I did

- **Ingestion**: Used `TextLoader` (UTF-8) on `mediumblog1.txt`, split with `CharacterTextSplitter` (chunk_size=1000), embedded with `OpenAIEmbeddings`, stored with `PineconeVectorStore.from_documents`.
- **RAG (Implementation 1 — without LCEL)**: Manually invoke retriever → `format_docs` → `prompt.format_messages` → `llm.invoke`.
- **RAG (Implementation 2 — with LCEL)**: Build a chain with `RunnablePassthrough.assign(context=itemgetter("question") \| retriever \| format_docs)` so the question flows into the retriever and `format_docs`; then pipe through `prompt \| llm \| StrOutputParser` to get a string answer.

### What I learned

- RAG = **retrieve** relevant chunks → **augment** the prompt with that context → **generate** with the LLM.
- LCEL (`RunnablePassthrough.assign`, `|`) composes retrieval and generation into a single runnable; the input `{"question": "..."}` drives both retrieval and the final prompt.
- Pinecone holds embeddings; the retriever returns the top-k closest chunks for the query embedding.
- Chunk overlap helps avoid splitting concepts across boundaries; `chunk_size` is a target, not a strict max.

### Setup

Create a `.env` with:

- `OPENAI_API_KEY` — for embeddings and chat
- `PINECONE_API_KEY` — for the vector store
- `INDEX_NAME` — Pinecone index (e.g. `rag-gist-index`)
- LangSmith (optional): `LANGSMITH_TRACING`, `LANGSMITH_API_KEY`, etc.

### Run

```bash
# 1. Ingest documents into Pinecone (do this first)
python ingestion.py

# 2. Run RAG queries
python main.py
```
