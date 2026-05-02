# Corrective RAG (with Self-RAG + Adaptive-RAG flow)

This project implements a corrective Retrieval-Augmented Generation pipeline using LangGraph.
The graph combines:

- **Self-RAG checks**: grade retrieval quality, grade grounding (hallucination), and grade answer usefulness.
- **Adaptive-RAG routing**: route each question to either vector retrieval or web search first.

## What Changed

### Self-RAG updates

- Added a **hallucination grader** chain to verify that generation is supported by retrieved context.
- Added an **answer grader** chain to verify the generated response actually answers the question.
- Added retry/fallback decisions in the graph based on grader outputs (`useful`, `not useful`, `not supported` flow).

### Adaptive-RAG updates

- Added a **question router** (`graph/chains/router.py`) with structured output:
  - `vectorstore` for in-domain questions (agents, prompt engineering, adversarial attacks)
  - `web_search` for out-of-domain questions
- Updated graph entry to use `set_conditional_entry_point(...)` so routing happens before retrieval.
- Kept web search as a fallback when retrieval grading detects low relevance.

## Graph Flow

1. Route question: `vectorstore` or `web_search`
2. If vectorstore path:
   - retrieve documents
   - grade document relevance
   - optionally enrich with web search
3. Generate answer
4. Grade grounding (hallucination check)
5. Grade answer usefulness
6. End or fallback/retry based on grading result

## Key Files

- `graph/graph.py` - LangGraph wiring and routing/decision edges
- `graph/state.py` - shared graph state
- `graph/nodes/` - retrieve, grade documents, web search, generate
- `graph/chains/router.py` - Adaptive-RAG router
- `graph/chains/hallucination_grader.py` - Self-RAG grounding check
- `graph/chains/answer_grader.py` - Self-RAG answer relevance check
- `main.py` - simple invocation entrypoint

## Quick Start

### 1) Install dependencies

From `corrective-rag/`:

```bash
uv sync
```

or with pip:

```bash
pip install -e .
```

### 2) Configure environment

Create `.env` in `corrective-rag/` (or repo root) with:

```env
OPENAI_API_KEY=your_key
TAVILY_API_KEY=your_key
```

### 3) Run

```bash
python main.py
```

This runs:

```python
app.invoke({"question": "What is agent memory?"})
```

## Diagrams

- `graph_self_rag.png` - self-RAG style flow
- `graph_adaptive_rag.png` - adaptive + corrective flow (generated in `graph/graph.py`)

## Tests

```bash
pytest graph/chains/tests/test_chains.py
```

Note: tests call live LLM/tools, so valid API keys are required.

