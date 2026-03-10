## Search agent using ReAct architecture

This folder contains my experiments building a simple search-style agent with the **ReAct (Reason + Act)** pattern.

### What I did
- **`main.py` – fake search tool**  
  - Defined a fake `search_tool` using `@tool` so the agent can “search” (currently just prints and returns a stub string).  
  - Wired the agent with `create_react_agent` + `OllamaLLM` and ran it with “What is the capital of France?” to see the ReAct loop (thought → tool call → observation → final answer).
- **`main-tavily-tool.py` – real Tavily search**  
  - Swapped the stub tool for a real **Tavily**-backed search tool.  
  - Used the same ReAct agent pattern, but now the tool actually hits Tavily’s API (using the `TAVILY_API_KEY` from `.env`) to return live search results.

### What I learned
- The **ReAct pattern** alternates between reasoning steps and tool calls, which is a clean mental model for building agents.
- In newer LangChain / LangGraph versions, agent helpers like `create_react_agent` may move between modules (e.g., into **LangGraph prebuilt agents**), so it’s important to check the current docs for the right import path and version.