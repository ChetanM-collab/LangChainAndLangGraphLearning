from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable
from langchain_ollama import OllamaLLM

MAX_ITERATIONS = 10
MODEL = "ollama:qwen3:1.7b"

@tool
def get_product_price(product: str) -> float:
    """ Look up the price of a product in the catalog."""
    print(f"  >> Executing get_product_price(product = '{product}')")
    product_prices = {"laptop": 1299.99, "headphones": 149.95, "keyboard": 89.50}
    return product_prices.get(product, 0)

@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """ Apply a discount tier to a price and return the final price.
    Available tiers: bronze, silver, gold."""
    print(f"  >> Executing apply_discount(price={price}, discount_tier='{discount_tier}')")
    discount_percentages = {"bronze": 5, "silver":12, "gold": 23}
    discount = discount_percentages.get(discount_tier, 0)
    return round(price * (1 - discount/100), 2)

# ---- Agent Loop -----

@traceable(name="Langchain Agent Loop")
def run_agent(question : str):
    tools = [get_product_price, apply_discount]
    tools_dict = {t.name : t for t in tools}
    llm = init_chat_model(model=MODEL, temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    print(f"Question: {question}")

    messages = [
        SystemMessage(
        content=(
            "You are a shopping assistant that MUST use tools for all price and discount questions.\n\n"

            "Available tools:\n"
            "- get_product_price(product: str): returns the catalog price for a product\n"
            "- apply_discount(price: float, discount_tier: str): returns the final discounted price\n\n"

            "Rules:\n"
            "1. For any product price request, always call get_product_price first.\n"
            "2. Never invent, estimate, or assume a price.\n"
            "3. If the user asks for a discounted price, first call get_product_price.\n"
            "3a: Once the result is returned for get_product_price, do NOT call it again and "
            "4. CALL apply_discount using the exact price returned by get_product_price.\n"
            "5. Never calculate the discount yourself.\n"
            "6. If discount tier is missing, ask the user to specify bronze, silver, or gold.\n"
            "7. If the product is not found, say that the product is not available in the catalog.\n"
            "8. Do not answer with a final price until the required tool calls have been completed.\n"
            "9. When the question involves both product and discount, use tools in this order:\n"
            "   first get_product_price, then apply_discount.\n"
            )
        ),
        HumanMessage(content=question)
    ]

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n----- Iteration {iteration} -----")
        ai_message = llm_with_tools.invoke(messages)
        tool_calls = ai_message.tool_calls
        print(f"tool_calls: {tool_calls}")
        if not tool_calls:
            print(f"\nFinal Answer: {ai_message.content}")
            return ai_message.content

        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id")

        print(f"[Tool Selected] : {tool_name} with args {tool_args}")
        tools_to_use = tools_dict.get(tool_name)
        if tools_to_use is None:
            raise ValueError(f"Tool '{tool_name}' not found")
        
        observation = tools_to_use.invoke(tool_args)

        print(f" [Tool Result] {observation}")

        messages.append(ai_message)
        messages.append(
            ToolMessage(content=str(observation), tool_call_id=tool_call_id)
        )

    print("ERROR: Max iterations reached without a final answer.")
    return None
        
if __name__ == "__main__":
    print("Hello Langchain Agent (.bind_tools)")
    print()
    result = run_agent("What is the price of a laptop with a gold discount");
    print(f"Result = {result}")

