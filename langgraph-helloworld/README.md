# LangGraph Hello World

This example is a small ReAct-style LangGraph agent that can call tools.

It combines:

- an LLM reasoning step
- a tool execution step
- a loop that continues until the model stops requesting tools

## What This Example Does

The graph starts with a user message, lets the model decide whether to call a tool, executes any requested tool calls, and then returns control to the model for another reasoning pass.

In the current version, the project exposes these tools from [`react.py`](/c:/workspace/LangChainAndLangGraph/langchain-course/langgraph-helloworld/react.py):

- `triple(number)`: multiplies a number by 3
- `TavilySearch`: performs web search

## Graph Flow

The graph is defined in [`main.py`](/c:/workspace/LangChainAndLangGraph/langchain-course/langgraph-helloworld/main.py).

It has two nodes:

- `agent_reasoning`: the LLM decides what to do next
- `act`: executes tool calls

Flow:

1. The user sends a prompt.
2. The model responds.
3. If the response contains tool calls, the graph routes to the tool node.
4. Tool results are added to the message history.
5. The graph returns to the reasoning node.
6. The loop stops when the model returns an answer without any tool calls.

## Requirements

- Python `3.13+`
- OpenAI API access
- Tavily API access

## Environment Setup

Create a `.env` file in `langgraph-helloworld/` with:

```env
OPENAI_API_KEY=your_openai_api_key
TAVILY_API_KEY=your_tavily_api_key
```

Optional:

```env
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_PROJECT=langgraph-helloworld
```

If your machine uses proxy variables like `HTTP_PROXY` or `HTTPS_PROXY`, make sure they point to a real proxy. Invalid proxy values will break:

- OpenAI requests
- Tavily requests
- Mermaid graph rendering
- LangSmith tracing

## Install Dependencies

From the repo root:

```powershell
.\.venv\Scripts\python.exe -m pip install -e .\langgraph-helloworld
```

If you already use the repo-level virtual environment, make sure your IDE/interpreter is set to:

```text
.\.venv\Scripts\python.exe
```

## Run

From the repo root:

```powershell
.\.venv\Scripts\python.exe .\langgraph-helloworld\main.py
```

The script currently runs this prompt:

```text
What is the weather in Sydney? List it and then triple it
```

## Files

- [`main.py`](/c:/workspace/LangChainAndLangGraph/langchain-course/langgraph-helloworld/main.py): builds the graph and runs the example
- [`nodes.py`](/c:/workspace/LangChainAndLangGraph/langchain-course/langgraph-helloworld/nodes.py): contains the reasoning node and tool node
- [`react.py`](/c:/workspace/LangChainAndLangGraph/langchain-course/langgraph-helloworld/react.py): defines tools and configures the LLM
- `flow.png`: optional Mermaid diagram generated from the graph

## Notes

- Mermaid rendering is wrapped in `try/except`, so the script can still start when external graph rendering is unavailable.
- The agent relies on tool calling through `ChatOpenAI(...).bind_tools(tools)`.
- The example is intentionally small, so it is a good starting point for learning how LangGraph loops between reasoning and action.
