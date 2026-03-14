## Agents under the hood

This folder explores **how LangChain agents work internally** by implementing the agent loop manually at multiple abstraction levels, instead of using helpers like `create_agent` or `create_react_agent`.

Use a local `.env` with LangSmith keys if you want tracing; the model is Ollama (`qwen3:1.7b`).

### Files

| File | Approach |
|------|----------|
| `layer1-agent-loop-langchain-tool-calling.py` | LangChain tool-calling: `@tool`, `init_chat_model`, `llm.bind_tools`, `ToolMessage` |
| `layer2-agent-loop-raw-function-calling.py` | Raw Ollama API: plain Python functions, JSON tool schemas, `ollama.chat(tools=...)` |
| `layer3-agent-loop-raw-react-prompt.py` | Pure ReAct prompt: no tool-calling API, single long prompt + regex parsing of `Action` / `Action Input` / `Final Answer` |

### What I did

- **Defined tools** (same semantics in both layers):
  - `get_product_price(product)` — returns catalog prices (laptop, headphones, keyboard).
  - `apply_discount(price, discount_tier)` — applies bronze/silver/gold discount tiers.
- **Layer 1 (LangChain)**:
  - Used `@tool`, `init_chat_model`, and `llm.bind_tools(tools)`.
  - Wrote a loop: LLM invoke → `tool_calls`? → execute tool → append `ToolMessage` → repeat.
- **Layer 2 (raw function calling)**:
  - Defined tool schemas by hand as JSON (OpenAI-style function format).
  - Called `ollama.chat(model=..., tools=tools_for_llm)` and processed `message.tool_calls` directly.
  - Used plain dict messages (`{"role": "tool", "content": str(observation)}`).
- **Layer 3 (raw ReAct prompt)**:
  - Wrote a classic ReAct-style prompt template with `Question`, `Thought`, `Action`, `Action Input`, `Observation`, `Final Answer`.
  - Fed the whole history as one growing string (prompt + scratchpad) and used `stop` tokens to let Python inject real `Observation` values.
  - Parsed `Action` / `Action Input` / `Final Answer` from the model’s text output using regex, then called the Python tools directly.
- **Traced runs** with LangSmith (`@traceable`) in both layers.
- **Ran** “What is the price of a laptop with a gold discount?” to see multi-step reasoning (get price → apply discount → answer).

### What I learned

- The agent loop is the same pattern at any layer: **LLM invoke → tool calls? → execute tools → append tool result → repeat until no tool calls**.
- LangChain’s `bind_tools` and `@tool` translate tools into the provider’s function-calling schema under the hood.
- The raw Ollama layer shows what’s actually sent: JSON tool definitions and `role: tool` messages.
- Building both layers clarifies what LangChain abstracts away (message types, tool schema, `tool_call_id` handling).
