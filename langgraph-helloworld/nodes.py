from dotenv import load_dotenv
from langchain_core.messages import SystemMessage
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode

from react import llm, tools

load_dotenv()

SYSTEM_PROMPT = """
You are a helpful assistant that can use the following tools to answer questions
"""

def run_agent_ressoning(state: MessagesState) -> MessagesState:
    """Run the agent reasoning node"""
    response = llm.invoke([SystemMessage(content=SYSTEM_PROMPT), *state["messages"]])
    return {"messages": [response]}

tool_node = ToolNode(tools)
