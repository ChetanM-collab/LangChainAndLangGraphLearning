from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
load_dotenv()

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from tavily import TavilyClient

tavily = TavilyClient()

@tool
def search_tool(query: str) -> str:
    """
    Tool to search the web for information.
    Args:
        query: The query to search for.

    Returns:
        A string with the search results.
    """  
    
    print(f"Searching for {query}")
    return tavily.search(query=query)

tools = [search_tool]
llm=ChatOpenAI()

agent = create_agent(
    model=llm,
    tools=tools
)

def main():
    print("Hello from search-agent-using-react-architecture!")
    #result = agent.invoke({"messages": "What is the capital of France?"})
    result = agent.invoke({"messages": HumanMessage(content="What is the capital of France?")})
    print(result)

if __name__ == "__main__":
    main()
