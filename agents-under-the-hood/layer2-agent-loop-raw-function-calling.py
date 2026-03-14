from dotenv import load_dotenv

load_dotenv()

import ollama
from ollama import chat
from langsmith import traceable

MAX_ITERATIONS = 10
MODEL = "qwen3:1.7b"

@traceable(run_type="tool")
def get_product_price(product: str) -> float:
    """ Look up the price of a product in the catalog."""
    print(f"  >> Executing get_product_price(product = '{product}')")
    product_prices = {"laptop": 1299.99, "headphones": 149.95, "keyboard": 89.50}
    return product_prices.get(product, 0)

@traceable(run_type="tool")
def apply_discount(price: float, discount_tier: str) -> float:
    """ Apply a discount tier to a price and return the final price.
    Available tiers: bronze, silver, gold."""
    print(f"  >> Executing apply_discount(price={price}, discount_tier='{discount_tier}')")
    discount_percentages = {"bronze": 5, "silver":12, "gold": 23}
    discount = discount_percentages.get(discount_tier, 0)
    return round(price * (1 - discount/100), 2)


tools_for_llm = [
    {
      "type": "function",
      "function": {
        "name": "get_product_price",
        "description": "Get the price of a product in the catalog",
        "parameters": {
          "type": "object",
          "required": ["product"],
          "properties": {
            "product": {"type": "string", "description": "The name of the product"}
          }
        }
      }
    },

    {
      "type": "function",
      "function": {
        "name": "apply_discount",
        "description": "Apply a discount tier to a price and return the final price",
        "parameters": {
          "type": "object",
          "required": ["price", "discount_tier"],
          "properties": {
            "price": {"type": "float", "description": "The original price of the product"},
            "discount_tier": {"type": "string", "description": "The name of the discount tier"}
          }
        }
      }
    }
  ]

@traceable(name="Ollama Chat", run_type="llm")
def ollama_chat_trace(messages):
    return chat(model=MODEL, tools=tools_for_llm, messages=messages)

# ---- Agent Loop -----

@traceable(name="Langchain Agent Loop")
def run_agent(question : str):
    tools_dict = {
        "get_product_price" : get_product_price,
        "apply_discount" : apply_discount
    } 

    print(f"Question: {question}")

    messages = [
        {"role" : "system",
         "content": (
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
        },
        {"role": "user", "content": question}
    ]

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n----- Iteration {iteration} -----")
        
        response = ollama_chat_trace(messages)
        ai_message = response.message
        
        tool_calls = ai_message.tool_calls
        
        print(f"tool_calls: {tool_calls}")
        if not tool_calls:
            print(f"\nFinal Answer: {ai_message.content}")
            return ai_message.content

        tool_call = tool_calls[0]

        tool_name = tool_call.function.name
        tool_args = tool_call.function.arguments
        
        print(f"[Tool Selected] : {tool_name} with args {tool_args}")
        tools_to_use = tools_dict.get(tool_name)
        
        if tools_to_use is None:
            raise ValueError(f"Tool '{tool_name}' not found")
        
        observation = tools_to_use(**tool_args)

        print(f" [Tool Result] {observation}")

        messages.append(ai_message)
        messages.append(
            {"role": "tool", "content": str(observation)}
        )

    print("ERROR: Max iterations reached without a final answer.")
    return None
        
if __name__ == "__main__":
    print("Hello Langchain Agent (.bind_tools)")
    print()
    result = run_agent("What is the price of a laptop with a gold discount");
    print(f"Result = {result}")

