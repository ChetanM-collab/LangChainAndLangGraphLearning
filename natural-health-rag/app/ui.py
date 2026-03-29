"""
Natural Health RAG - Streamlit Chat UI
Run with: streamlit run app/ui.py
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from retrieval.chain import build_rag_chain_with_sources

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="🌿 Natural Health Assistant",
    page_icon="🌿",
    layout="wide",
)

# ── Styling ───────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Lora:ital,wght@0,400;0,600;1,400&family=DM+Sans:wght@300;400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }
    h1, h2, h3 { font-family: 'Lora', serif; }

    .main { background: #f8f6f0; }

    .source-card {
        background: #fff;
        border-left: 3px solid #6b8f5e;
        padding: 0.6rem 0.9rem;
        border-radius: 0 6px 6px 0;
        margin: 0.4rem 0;
        font-size: 0.82rem;
        color: #555;
    }
    .source-badge {
        display: inline-block;
        background: #e8f0e4;
        color: #4a6741;
        font-size: 0.7rem;
        font-weight: 500;
        padding: 2px 7px;
        border-radius: 20px;
        margin-bottom: 4px;
    }
    .disclaimer {
        font-size: 0.75rem;
        color: #999;
        font-style: italic;
        margin-top: 0.3rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────

st.markdown("# 🌿 Natural Health Research Assistant")
st.markdown(
    "*Bridging ancient herbal wisdom and modern clinical evidence. "
    "Sources: NIH, NCCIH, PubMed, ClinicalTrials.gov, WHO monographs.*"
)
st.divider()

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Settings")

    backend = st.selectbox("Vector Store", ["chroma", "pinecone"], index=0)
    k = st.slider("Sources to retrieve", 3, 10, 5)
    source_filter = st.selectbox(
        "Filter by source",
        ["All", "PubMed", "NCCIH", "NIH_ODS", "ClinicalTrials", "WHO_EMA_PDF"],
        index=0,
    )

    st.divider()
    st.markdown("### 💡 Sample Questions")
    samples = [
        "Does rosemary oil really work for hair loss?",
        "What's the evidence for saw palmetto?",
        "Is biotin deficiency common?",
        "What does ashwagandha do for stress?",
        "Are there clinical trials on pumpkin seed oil?",
    ]
    for q in samples:
        if st.button(q, use_container_width=True):
            st.session_state.pending_question = q

    st.divider()
    st.markdown('<p class="disclaimer">⚠️ Not medical advice. Always consult a healthcare professional.</p>', unsafe_allow_html=True)

# ── Chat state ────────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Display history ───────────────────────────────────────────────────────────

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"📚 {len(msg['sources'])} sources used"):
                for src in msg["sources"]:
                    st.markdown(f"""
                    <div class="source-card">
                        <span class="source-badge">{src['source_type']}</span><br>
                        {src['snippet']}<br>
                        <a href="{src['url']}" target="_blank" style="color:#6b8f5e; font-size:0.75rem;">{src['url']}</a>
                    </div>
                    """, unsafe_allow_html=True)

# ── Input ─────────────────────────────────────────────────────────────────────

# Handle sidebar sample button
if "pending_question" in st.session_state:
    user_input = st.session_state.pop("pending_question")
else:
    user_input = st.chat_input("Ask about natural health, herbs, supplements...")

if user_input:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching knowledge base..."):
            try:
                sf = None if source_filter == "All" else source_filter
                chain = build_rag_chain_with_sources(backend=backend, k=k)
                result = chain.invoke(user_input)
                answer = result["answer"]
                sources = result["sources"]
            except Exception as e:
                answer = f"⚠️ Error: {e}\n\nMake sure you've run the ingestion pipeline first (`python ingestion/ingest.py`) and set your `OPENAI_API_KEY`."
                sources = []

        st.markdown(answer)
        if sources:
            with st.expander(f"📚 {len(sources)} sources used"):
                for src in sources:
                    st.markdown(f"""
                    <div class="source-card">
                        <span class="source-badge">{src['source_type']}</span><br>
                        {src['snippet']}<br>
                        <a href="{src['url']}" target="_blank" style="color:#6b8f5e; font-size:0.75rem;">{src['url']}</a>
                    </div>
                    """, unsafe_allow_html=True)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })
