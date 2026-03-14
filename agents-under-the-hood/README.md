## Agents under the hood

This folder explores **how LangChain agents work internally** by implementing the agent loop manually instead of using high-level helpers like `create_agent` or `create_react_agent`.

Use a local `.env` with LangSmith keys if you want tracing; the model is Ollama (`qwen3:1.7b`).

### What I did

- **Defined tools** with `@tool`:
  - `get_product_price(product)` — returns catalog prices (laptop, headphones, keyboard).
  - `apply_discount(price, discount_tier)` — applies bronze/silver/gold discount tiers.
- **Implemented the agent loop manually** in `layer1.py`:
  - Used `init_chat_model` and `llm.bind_tools(tools)` to give the model tool-calling capability.
  - Wrote a loop that: invokes the LLM → checks for `tool_calls` → if none, returns the final answer → otherwise runs the selected tool, appends a `ToolMessage`, and repeats.
- **Traced runs** with LangSmith (`@traceable`) to inspect the agent’s behavior.
- **Ran** a sample question: “What is the price of a laptop with a gold discount?” to see the multi-step reasoning (get price → apply discount → answer).

### What I learned

- The agent loop is essentially: **LLM invoke → tool calls? → execute tools → append `ToolMessage` → repeat until no tool calls**.
- `bind_tools` exposes the tool schemas to the model so it can request tool calls in its response.
- `ToolMessage` carries the tool result back into the conversation for the next LLM turn.
- Building this manually helps understand what higher-level APIs like `create_agent` are doing internally.
