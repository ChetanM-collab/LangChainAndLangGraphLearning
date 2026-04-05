from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a strict critic reviewing an explanation written for a complete beginner."
            "Always provide detailed recommendation. Do not rewrite it, just judge it."
            "Evaluate based on these criteria: "
            "- no jargon or unexplained technical terms."
            "- uses a read-world analogy or concrete example."
            "- A 12 year old could understand it."
            "- Gets tyo the point quickly."
            "If it passes all criteria respond with: PASS — [one sentence on why it works]"
            "If it fails respond with: FAIL — [specific list of what needs to improve, be concrete not vague]",

       ),    
       MessagesPlaceholder(variable_name="messages")    
    ]
)

generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system", 
            "You are a teaching assistant. Your job is to explain concepts clearly and simply."
            "If the user provides a critique, respond with a revised version of your previous attempts.",
        ),
         MessagesPlaceholder(variable_name="messages")    
    ]
)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

generation_chain = generation_prompt | llm
reflection_chain = reflection_prompt | llm

