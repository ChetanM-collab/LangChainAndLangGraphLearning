"""
Natural Health RAG - Ingestion Pipeline
Pulls from: NIH ODS, NCCIH, WHO PDFs, PubMed API, ClinicalTrials.gov
"""

import os
import sys
import time
import json
import requests
from pathlib import Path
from typing import List

# Project root on sys.path so `import embeddings` works when run as:
#   python ingestion/ingest.py
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from bs4 import BeautifulSoup

from langchain_community.document_loaders import (
    WebBaseLoader,
    PyPDFLoader,
    DirectoryLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv(_ROOT / ".env")

from embeddings.vectorstore import build_chroma, build_pinecone

# ── Config ────────────────────────────────────────────────────────────────────

DATA_DIR = Path("data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE    = 800
CHUNK_OVERLAP = 150

# ── 1. NIH Office of Dietary Supplements ─────────────────────────────────────

NIH_ODS_URLS = [
    "https://ods.od.nih.gov/factsheets/Biotin-HealthProfessional/",
    "https://ods.od.nih.gov/factsheets/Zinc-HealthProfessional/",
    "https://ods.od.nih.gov/factsheets/Iron-HealthProfessional/",
    "https://ods.od.nih.gov/factsheets/VitaminD-HealthProfessional/",
    "https://ods.od.nih.gov/factsheets/VitaminE-HealthProfessional/",
    "https://ods.od.nih.gov/factsheets/Selenium-HealthProfessional/",
]

def load_nih_ods() -> List[Document]:
    """Scrape NIH ODS supplement fact sheets."""
    print("📥 Loading NIH ODS fact sheets...")
    docs = []
    for url in NIH_ODS_URLS:
        try:
            loader = WebBaseLoader(url)
            loaded = loader.load()
            for doc in loaded:
                doc.metadata["source_type"] = "NIH_ODS"
                doc.metadata["url"] = url
            docs.extend(loaded)
            time.sleep(0.5)  # be polite
        except Exception as e:
            print(f"  ⚠️  Failed {url}: {e}")
    print(f"  ✅ Loaded {len(docs)} NIH ODS pages")
    return docs


# ── 2. NCCIH Herbs A–Z ────────────────────────────────────────────────────────

NCCIH_HERBS = [
    "aloe-vera", "ashwagandha", "black-cohosh", "chamomile",
    "echinacea", "garlic", "ginger", "ginkgo", "ginseng",
    "green-tea", "lavender", "melatonin", "milk-thistle",
    "peppermint", "rosemary", "saw-palmetto", "st-johns-wort",
    "turmeric", "valerian",
]

def load_nccih() -> List[Document]:
    """Scrape NCCIH herb pages."""
    print("📥 Loading NCCIH herbs A–Z...")
    docs = []
    base = "https://www.nccih.nih.gov/health/"
    for herb in NCCIH_HERBS:
        url = f"{base}{herb}"
        try:
            loader = WebBaseLoader(url)
            loaded = loader.load()
            for doc in loaded:
                doc.metadata["source_type"] = "NCCIH"
                doc.metadata["herb"] = herb
                doc.metadata["url"] = url
            docs.extend(loaded)
            time.sleep(0.5)
        except Exception as e:
            print(f"  ⚠️  Failed {herb}: {e}")
    print(f"  ✅ Loaded {len(docs)} NCCIH herb pages")
    return docs


# ── 3. PubMed Abstracts (Entrez API) ─────────────────────────────────────────

PUBMED_QUERIES = [
    "rosemary oil hair loss clinical trial",
    "saw palmetto androgenic alopecia",
    "pumpkin seed oil hair growth",
    "minoxidil natural alternative",
    "scalp inflammation hair follicle",
    "caffeine hair follicle stimulation",
    "herbal medicine alopecia randomized",
    "natural DHT blocker hair loss",
    "vitamin D deficiency hair loss",
    "zinc supplementation alopecia",
]

ENTREZ_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

def fetch_pubmed_abstracts(query: str, max_results: int = 10) -> List[Document]:
    """Fetch PubMed abstracts for a query using the Entrez API."""
    docs = []

    # Step 1: search for IDs
    search_url = f"{ENTREZ_BASE}/esearch.fcgi"
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "json",
    }
    r = requests.get(search_url, params=params, timeout=10)
    ids = r.json().get("esearchresult", {}).get("idlist", [])

    if not ids:
        return docs

    # Step 2: fetch abstracts
    fetch_url = f"{ENTREZ_BASE}/efetch.fcgi"
    fetch_params = {
        "db": "pubmed",
        "id": ",".join(ids),
        "rettype": "abstract",
        "retmode": "text",
    }
    r2 = requests.get(fetch_url, params=fetch_params, timeout=15)
    raw_text = r2.text

    # Split on PMID boundaries (each abstract starts with a number)
    abstracts = [a.strip() for a in raw_text.split("\n\n\n") if a.strip()]
    for abstract in abstracts:
        docs.append(Document(
            page_content=abstract,
            metadata={
                "source_type": "PubMed",
                "query": query,
                "url": "https://pubmed.ncbi.nlm.nih.gov",
            }
        ))
    return docs

def load_pubmed() -> List[Document]:
    print("📥 Loading PubMed abstracts...")
    all_docs = []
    for query in PUBMED_QUERIES:
        try:
            docs = fetch_pubmed_abstracts(query, max_results=8)
            all_docs.extend(docs)
            time.sleep(0.4)  # NCBI rate limit: max 3 req/sec without API key
        except Exception as e:
            print(f"  ⚠️  PubMed query failed '{query}': {e}")
    print(f"  ✅ Loaded {len(all_docs)} PubMed abstracts")
    return all_docs


# ── 4. ClinicalTrials.gov ─────────────────────────────────────────────────────

CT_QUERIES = [
    "hair loss herbal",
    "alopecia natural treatment",
    "rosemary oil scalp",
    "saw palmetto hair",
]

def load_clinical_trials() -> List[Document]:
    """Fetch trial summaries from ClinicalTrials.gov API v2."""
    print("📥 Loading ClinicalTrials.gov data...")
    docs = []
    base = "https://clinicaltrials.gov/api/v2/studies"

    for query in CT_QUERIES:
        try:
            params = {
                "query.term": query,
                "pageSize": 10,
                "fields": "NCTId,BriefTitle,BriefSummary,Condition,InterventionName,Phase,OverallStatus",
            }
            r = requests.get(base, params=params, timeout=15)
            studies = r.json().get("studies", [])

            for study in studies:
                proto = study.get("protocolSection", {})
                id_mod = proto.get("identificationModule", {})
                desc_mod = proto.get("descriptionModule", {})
                
                nct_id = id_mod.get("nctId", "")
                title = id_mod.get("briefTitle", "")
                summary = desc_mod.get("briefSummary", "")

                if summary:
                    content = f"Clinical Trial: {title}\n\nSummary: {summary}"
                    docs.append(Document(
                        page_content=content,
                        metadata={
                            "source_type": "ClinicalTrials",
                            "nct_id": nct_id,
                            "url": f"https://clinicaltrials.gov/study/{nct_id}",
                        }
                    ))
            time.sleep(0.5)
        except Exception as e:
            print(f"  ⚠️  ClinicalTrials query failed: {e}")

    print(f"  ✅ Loaded {len(docs)} clinical trial summaries")
    return docs


# ── 5. WHO / EMA PDFs (local folder) ─────────────────────────────────────────

def load_local_pdfs(pdf_dir: str = "data/pdfs") -> List[Document]:
    """
    Load any WHO / EMA monograph PDFs you've manually downloaded.
    Place them in data/pdfs/ and they'll be auto-ingested.
    """
    pdf_path = Path(pdf_dir)
    if not pdf_path.exists() or not list(pdf_path.glob("*.pdf")):
        print("  ℹ️  No PDFs found in data/pdfs/ — skipping WHO/EMA step")
        print("     Download WHO monographs from: https://www.who.int/publications/i/item/9241545178")
        return []

    print(f"📥 Loading PDFs from {pdf_dir}...")
    loader = DirectoryLoader(pdf_dir, glob="**/*.pdf", loader_cls=PyPDFLoader)
    docs = loader.load()
    for doc in docs:
        doc.metadata["source_type"] = "WHO_EMA_PDF"
    print(f"  ✅ Loaded {len(docs)} PDF pages")
    return docs


# ── Chunking ──────────────────────────────────────────────────────────────────

def chunk_documents(docs: List[Document]) -> List[Document]:
    """Split documents into overlapping chunks for embedding."""
    print(f"\n✂️  Chunking {len(docs)} documents...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"  ✅ Created {len(chunks)} chunks")
    return chunks


# ── Main ──────────────────────────────────────────────────────────────────────

def run_ingestion() -> List[Document]:
    print("🌿 Natural Health RAG — Ingestion Pipeline")
    print("=" * 50)

    all_docs = []
    all_docs.extend(load_nih_ods())
    all_docs.extend(load_nccih())
    all_docs.extend(load_pubmed())
    all_docs.extend(load_clinical_trials())
    all_docs.extend(load_local_pdfs())

    print(f"\n📊 Total raw documents: {len(all_docs)}")
    chunks = chunk_documents(all_docs)

    # Save metadata summary
    summary = {
        "total_chunks": len(chunks),
        "sources": {
            "NIH_ODS": sum(1 for d in chunks if d.metadata.get("source_type") == "NIH_ODS"),
            "NCCIH": sum(1 for d in chunks if d.metadata.get("source_type") == "NCCIH"),
            "PubMed": sum(1 for d in chunks if d.metadata.get("source_type") == "PubMed"),
            "ClinicalTrials": sum(1 for d in chunks if d.metadata.get("source_type") == "ClinicalTrials"),
            "WHO_EMA_PDF": sum(1 for d in chunks if d.metadata.get("source_type") == "WHO_EMA_PDF"),
        }
    }
    with open("data/ingestion_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n📋 Ingestion Summary:")
    for src, count in summary["sources"].items():
        print(f"   {src}: {count} chunks")

    return chunks

if __name__ == "__main__":
    chunks = run_ingestion()      # pull + chunk
    build_pinecone(chunks)          # embed + store
    build_chroma(chunks)
    print("✅ Pipeline complete — ready to query!")
