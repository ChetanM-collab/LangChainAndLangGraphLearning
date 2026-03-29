# 🌿 Natural Health RAG Assistant

A Retrieval-Augmented Generation system that bridges **ancient herbal wisdom** and **modern clinical evidence**. Ask plain-English questions and get evidence-backed answers with source citations.

## 📚 Knowledge Sources

| Source | Content | How loaded |
|--------|---------|------------|
| **NIH ODS** | Supplement fact sheets (biotin, zinc, iron, etc.) | WebBaseLoader |
| **NCCIH** | Herbs A–Z with evidence summaries | WebBaseLoader |
| **PubMed** | Clinical research abstracts via Entrez API | Custom fetcher |
| **ClinicalTrials.gov** | Ongoing & completed trial summaries | REST API v2 |
| **WHO / EMA PDFs** | Official herbal monographs (manual download) | PyPDFLoader |

---

## 🚀 Quick Start

### 1. Clone & install
```bash
git clone <your-repo>
cd natural-health-rag
pip install -r requirements.txt
```

### 2. Set environment variables
```bash
# .env file
OPENAI_API_KEY=sk-...

# Optional: use free local embeddings instead
EMBEDDING_BACKEND=ollama   # then run: ollama pull nomic-embed-text

# Optional: Pinecone cloud
PINECONE_API_KEY=...
```

### 3. Run ingestion (pulls all documents & builds vector DB)
```bash
python ingestion/ingest.py
```
This will:
- Scrape NIH ODS + NCCIH pages
- Fetch PubMed abstracts for 10 hair/herb queries
- Fetch ClinicalTrials.gov summaries
- Load any PDFs from `data/pdfs/`
- Chunk everything and store in ChromaDB at `data/chroma_db/`

### 4. Launch the chat UI
```bash
streamlit run app/ui.py
```

---

## 📁 Project Structure

```
natural-health-rag/
├── ingestion/
│   └── ingest.py          # Pulls & chunks all documents
├── embeddings/
│   └── vectorstore.py     # ChromaDB / Pinecone setup + retriever factory
├── retrieval/
│   └── chain.py           # RAG chain (retrieve → prompt → generate)
├── app/
│   └── ui.py              # Streamlit chat interface
├── data/
│   ├── pdfs/              # Put WHO/EMA PDFs here manually
│   ├── chroma_db/         # Auto-created after ingestion
│   └── ingestion_summary.json
└── requirements.txt
```

---

## ⚙️ Configuration Options

### Switch to free local embeddings (no OpenAI cost)
```bash
# Install Ollama: https://ollama.ai
ollama pull nomic-embed-text

# Set in .env:
EMBEDDING_BACKEND=ollama
```

### Switch from ChromaDB to Pinecone (for production scale)
```python
# In vectorstore.py, change default backend:
from embeddings.vectorstore import build_pinecone, get_retriever
get_retriever(backend="pinecone", index_name="natural-health")
```

### Add WHO Monograph PDFs
1. Download from https://www.who.int/publications/i/item/9241545178
2. Place PDFs in `data/pdfs/`
3. Re-run `python ingestion/ingest.py`

---

## 💬 Example Questions

- *"What does the evidence say about rosemary oil for hair loss?"*
- *"Is saw palmetto safe to combine with finasteride?"*
- *"Are there clinical trials on pumpkin seed oil for hair growth?"*
- *"What's the traditional vs modern use of ashwagandha?"*
- *"What are the NIH recommendations for biotin dosage?"*

---

## 🔧 Use as a Library

```python
from retrieval.chain import build_rag_chain, build_rag_chain_with_sources

# Simple chain
chain = build_rag_chain(k=5)
answer = chain.invoke("Does rosemary oil work for hair loss?")

# Chain with sources
chain = build_rag_chain_with_sources()
result = chain.invoke("What does PubMed say about saw palmetto?")
print(result["answer"])
print(result["sources"])  # list of {source_type, url, snippet}

# Filter to only PubMed sources
chain = build_rag_chain(source_filter="PubMed")
```

---

## ⚠️ Disclaimer

This tool is for **research and educational purposes only**. It is not medical advice. Always consult a qualified healthcare professional before making decisions about supplements or treatments.
