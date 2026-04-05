from typing import TypedDict, Annotated
from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import HumanMessage, BaseMessage
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from chains import generation_chain, reflection_chain

class MessageGraph(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

GENERATE = "generate"
REFLECT = "reflect"
LAST = -1

def generation_node(state: MessageGraph):
    return {"messages": generation_chain.invoke({"messages": state["messages"]})}

def reflection_node(state: MessageGraph):
    res = reflection_chain.invoke({"messages": state["messages"]})
    return {"messages": [HumanMessage(content=res.content)]}


def should_continue(state: MessageGraph) -> str:
    """
    Returns True if the loop should continue, False if it should stop.
    """
    if len(state["messages"]) >= 6:
        print(f"[Reflection] Max iterations reached 6 — stopping")
        return END

    last_message = state["messages"][LAST]

    if last_message.content.strip().upper().startswith("PASS"):
        print(f"[Reflection] PASS")
        return END

    print(f"[Reflection] FAIL — iterating")
    return REFLECT

builder = StateGraph(state_schema=MessageGraph)
builder.add_node(GENERATE, generation_node)
builder.add_node(REFLECT, reflection_node)

builder.set_entry_point(GENERATE)

builder.add_conditional_edges(GENERATE, should_continue, path_map={END:END, REFLECT: REFLECT})
builder.add_edge(REFLECT, GENERATE)

graph = builder.compile()

try:
    graph.get_graph().draw_mermaid_png(output_file_path="flow1.png")
except Exception:
    # Rendering uses an external Mermaid API by default, so keep startup resilient.
    pass


if __name__ == "__main__":
    print("Reflection Agent")
    res = graph.invoke({"messages" : [HumanMessage(content="What is crypto currency ?")]})

