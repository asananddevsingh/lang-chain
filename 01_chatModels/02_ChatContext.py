from langchain_core.messages import  SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

messages = [
    # SystemMessage("You are an expert in social media marketing"),
    # HumanMessage("Give a short description to create engaging posts on Instagram")
    SystemMessage("You are a calculator"),
    HumanMessage("What is 2 + 2?"),
    AIMessage("The result is 4"),
    HumanMessage("Add 2 more to the result")
]

llm = ChatOpenAI(model="gpt-4o")

result = llm.invoke(messages)

print(result.content)