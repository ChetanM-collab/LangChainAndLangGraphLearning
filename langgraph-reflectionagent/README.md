# LangGraph Reflection Agent

This example shows a simple reflection loop built with LangGraph.

The graph has two nodes:

- `generate`: produces an explanation for the user
- `reflect`: critiques the explanation like a strict reviewer

The workflow repeats until one of these conditions is met:

- the reviewer responds with `PASS`
- the graph reaches the built-in iteration limit

## How It Works

The project uses two LLM chains:

- `generation_chain` in [`chains.py`](/c:/workspace/LangChainAndLangGraph/langchain-course/langgraph-reflectionagent/chains.py)
- `reflection_chain` in [`chains.py`](/c:/workspace/LangChainAndLangGraph/langchain-course/langgraph-reflectionagent/chains.py)

The graph is defined in [`main.py`](/c:/workspace/LangChainAndLangGraph/langchain-course/langgraph-reflectionagent/main.py):

1. The user sends an initial message.
2. The `generate` node creates an explanation.
3. The `reflect` node critiques the answer.
4. If the critique starts with `PASS`, the graph stops.
5. Otherwise, the critique is fed back into the generator for another revision.

## Requirements

- Python `3.13+`
- An OpenAI API key

## Environment Setup

Create a `.env` file in `langgraph-reflectionagent/` with:

```env
OPENAI_API_KEY=your_openai_api_key
```

Optional:

- `LANGSMITH_TRACING=true`
- `LANGSMITH_API_KEY=...`
- `LANGSMITH_PROJECT=langgraph-reflectionagent`

If your machine uses proxy settings, make sure they are valid before running the script. Invalid `HTTP_PROXY` or `HTTPS_PROXY` values will cause OpenAI and Mermaid requests to fail.

## Install Dependencies

From the repo root:

```powershell
.\.venv\Scripts\python.exe -m pip install -e .\langgraph-reflectionagent
```

If you are using the workspace environment managed at the repo root, make sure your editor is using `.\.venv\Scripts\python.exe`.

## Run

From the repo root:

```powershell
.\.venv\Scripts\python.exe .\langgraph-reflectionagent\main.py
```

The script currently runs the graph with this prompt:

```text
What is crypto currency ?
```

## Files

- [`main.py`](/c:/workspace/LangChainAndLangGraph/langchain-course/langgraph-reflectionagent/main.py): defines the graph, nodes, and stop condition
- [`chains.py`](/c:/workspace/LangChainAndLangGraph/langchain-course/langgraph-reflectionagent/chains.py): defines the generator and reviewer prompts
- `flow1.png`: optional Mermaid graph render if network access is available

## Notes

- `draw_mermaid_png()` is wrapped in a `try/except`, so the app can still run if Mermaid image rendering is blocked.
- The graph stops after the message history reaches the current maximum iteration threshold in `main.py`.
- The reflection step converts the critic output into a `HumanMessage` so the generator can treat it as user feedback in the next round.
