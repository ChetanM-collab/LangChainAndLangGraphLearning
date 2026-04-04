import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.graph import MessagesState, StateGraph, START, END
from nodes import run_agent_ressoning, tool_node
from react import llm

load_dotenv()

AGENT_REASONING_NODE = "agent_reasoning"
ACT = "act"
LAST = -1

def should_continue(state: MessagesState) -> bool:
    if not state["messages"][LAST].tool_calls:
        return END
    return ACT


graph = StateGraph(MessagesState)
graph.add_node(AGENT_REASONING_NODE, run_agent_ressoning)
graph.set_entry_point(AGENT_REASONING_NODE)

graph.add_node(ACT, tool_node)
graph.add_edge(START, AGENT_REASONING_NODE)
graph.add_conditional_edges(AGENT_REASONING_NODE, should_continue, {ACT: ACT, END: END})
graph.add_edge(ACT, AGENT_REASONING_NODE)
app = graph.compile()

try:
    app.get_graph().draw_mermaid_png(output_file_path="flow.png")
except Exception:
    # Rendering uses an external Mermaid API by default, so keep startup resilient.
    pass


if __name__ == "__main__":
    print("ReAct LangGraph with function calling")
    res = app.invoke({"messages" : [HumanMessage(content="What is the weather in Sydney? List it and then triple it")]})
    print(res["messages"][LAST].content)
