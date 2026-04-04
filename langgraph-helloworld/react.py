from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

load_dotenv()

@tool
def triple(number: float) -> float:
    """param num: The number to triple"""
    """
    Returns:
        The number tripled.
    """
    return number * 3


tools=[triple, TavilySearch(max_results=5)]

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0).bind_tools(tools)

