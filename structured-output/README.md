## Structured output with LangChain

This folder contains my experiments using **structured outputs** with LangChain agents and Pydantic models.

### What I did
- **Defined Pydantic schemas**:
  - `Source` with a `url` field.
  - `AgentResponse` with an `answer` string and a list of `Source` objects.
- **Configured tools and model**:
  - Used `TavilySearch()` as a tool to pull web results.
  - Used `ChatOpenAI()` as the chat model.
- **Created an agent with `response_format`**:
  - Called `create_agent(model=llm, tools=tools, response_format=AgentResponse)` so the agent returns data shaped exactly like the `AgentResponse` schema (answer + sources).
  - Invoked the agent with a `HumanMessage` asking “What is the capital of France?” and printed the structured result.

### What I learned
- LangChain can **coerce model outputs into typed Pydantic models** using the `response_format` argument on `create_agent`.
- Using explicit schemas makes it much easier to **consume LLM outputs in code** (e.g., relying on `answer` and `sources` fields instead of parsing raw text).