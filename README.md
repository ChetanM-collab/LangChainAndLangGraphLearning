# LangChain & LangGraph learnings (Udemy)

My notes, experiments, and project work while taking the Udemy course **“LangChain – Develop AI Agents with LangChain and LangGraph”** by **Eden Marco**.

The first starting point in this repo is `hello-world/main.py`: a minimal “hello world” that creates a `PromptTemplate`, pipes it into an Ollama model via `OllamaLLM` (`prompt | llm`), and invokes it to generate a company name. Run it from the repo root with:

```bash
python hello-world/main.py
```

You’ll also need a `.env` file at the repo root with your own keys, for example:

- **OpenAI**: `OPENAI_API_KEY=...`
- **LangSmith**: `LANGSMITH_TRACING=true`, `LANGSMITH_ENDPOINT=...`, `LANGSMITH_API_KEY=...`, `LANGSMITH_PROJECT=...`

![LangChain Logo](/static/LangChain-logo.svg)
![LangGraph Logo](/static/LangGraph%20wordmark%20-%20dark.svg)

[![udemy](https://img.shields.io/badge/LangChain%20Udemy%20Course%20Coupon%20%2412.99-brightgreen)](https://www.udemy.com/course/langchain/?couponCode=FEB-2026)

