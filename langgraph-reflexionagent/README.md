# LangGraph Reflexion Agent

This example implements a Reflexion-style agent with LangGraph.

The workflow follows a simple research loop:

- draft an initial answer
- critique that answer
- generate search queries
- run research tools
- revise the answer with citations

## What This Project Does

The graph starts with a user question and uses structured tool calls to keep the response format consistent.

It is built around three stages:

- `draft`: create an initial answer plus self-critique and search queries
- `execute_tools`: run the generated search queries
- `revise`: improve the answer using the research results

The loop continues until the graph reaches the current iteration limit in [`main.py`](/c:/workspace/LangChainAndLangGraph/langchain-course/langgraph-reflexionagent/main.py).

## Architecture

The main graph lives in [`main.py`](/c:/workspace/LangChainAndLangGraph/langchain-course/langgraph-reflexionagent/main.py).

It wires together:

- `first_responder` from [`chains.py`](/c:/workspace/LangChainAndLangGraph/langchain-course/langgraph-reflexionagent/chains.py)
- `revisor` from [`chains.py`](/c:/workspace/LangChainAndLangGraph/langchain-course/langgraph-reflexionagent/chains.py)
- `execute_tools` from [`tool_executor.py`](/c:/workspace/LangChainAndLangGraph/langchain-course/langgraph-reflexionagent/tool_executor.py)

The structured outputs are defined in [`schemas.py`](/c:/workspace/LangChainAndLangGraph/langchain-course/langgraph-reflexionagent/schemas.py):

- `AnswerQuestion`
- `ReviseAnswer`
- `Reflection`

## Graph Flow

1. A user question enters the `draft` node.
2. The model produces:
   - an answer
   - a reflection on what is missing or superfluous
   - a short list of search queries
3. The `execute_tools` node runs the search queries using Tavily.
4. The `revise` node rewrites the answer using the tool results.
5. The graph loops until the maximum iteration threshold is reached.

## Requirements

- Python `3.13+`
- OpenAI API key
- Tavily API key

## Environment Setup

Create a `.env` file in `langgraph-reflexionagent/` with:

```env
OPENAI_API_KEY=your_openai_api_key
TAVILY_API_KEY=your_tavily_api_key
```

Optional:

```env
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_PROJECT=langgraph-reflexionagent
```

If your shell has proxy variables such as `HTTP_PROXY`, `HTTPS_PROXY`, or `ALL_PROXY`, make sure they are valid. Incorrect proxy settings will cause OpenAI, Tavily, and LangSmith requests to fail.

## Install Dependencies

From the repo root:

```powershell
.\.venv\Scripts\python.exe -m pip install -e .\langgraph-reflexionagent
```

Important: the current code imports `langchain_tavily`, but `langchain-tavily` is not listed in [`pyproject.toml`](/c:/workspace/LangChainAndLangGraph/langchain-course/langgraph-reflexionagent/pyproject.toml). If it is not already installed in your environment, install it manually:

```powershell
.\.venv\Scripts\python.exe -m pip install langchain-tavily
```

Make sure your editor/interpreter is using:

```text
.\.venv\Scripts\python.exe
```

## Run

From the repo root:

```powershell
.\.venv\Scripts\python.exe .\langgraph-reflexionagent\main.py
```

The current sample prompt is:

```text
Write about AI-Powered SOC / autonomous soc problem domain, list startups that do that and raised capital.
```

## Files

- [`main.py`](/c:/workspace/LangChainAndLangGraph/langchain-course/langgraph-reflexionagent/main.py): defines the graph and iteration loop
- [`chains.py`](/c:/workspace/LangChainAndLangGraph/langchain-course/langgraph-reflexionagent/chains.py): defines the drafting and revision chains
- [`schemas.py`](/c:/workspace/LangChainAndLangGraph/langchain-course/langgraph-reflexionagent/schemas.py): structured schemas for answer, reflection, and citations
- [`tool_executor.py`](/c:/workspace/LangChainAndLangGraph/langchain-course/langgraph-reflexionagent/tool_executor.py): executes generated research queries with Tavily

## Notes

- This example uses tool calling to force structured intermediate outputs instead of relying on free-form text only.
- The revised answer is expected to include numerical citations and a `References` section.
- The current code prints the Mermaid graph text with `draw_mermaid()`, which is local text generation and does not require the external Mermaid image API.
- The iteration limit is currently controlled by `MAX_ITERATIONS` in [`main.py`](/c:/workspace/LangChainAndLangGraph/langchain-course/langgraph-reflexionagent/main.py).
