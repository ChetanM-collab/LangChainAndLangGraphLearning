from typing import List

from pydantic import BaseModel, Field

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
load_dotenv()

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch


class Source(BaseModel):
    """Schema for a source used by the agent"""
    url:str = Field(description="The URL of the source")

class AgentResponse(BaseModel):
    """Schema for the agent response with answers and sources"""

    answer:str = Field(description="The Agent's answer to the query")
    sources: List[Source] = Field(default_factory=list, description="List of sources used to generate the answer")

tools = [TavilySearch()]
llm=ChatOpenAI()

agent = create_agent(
    model=llm,
    tools=tools,
    response_format=AgentResponse
)

def main():
    print("Hello from structured-output!")
    #result = agent.invoke({"messages": "What is the capital of France?"})
    result = agent.invoke({"messages": HumanMessage(content="What is the capital of France?")})
    print(result)

if __name__ == "__main__":
    main()
