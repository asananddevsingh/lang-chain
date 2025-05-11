from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()

messages = [
    SystemMessage("Solve the following math problem"),
    HumanMessage("What is 2 + 2 * 4 - 2?")
]

model = ChatOpenAI(model="gpt-4o")

result = model.invoke(messages)

print("OpenAI: ", result.content)


# model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# result = model.invoke(messages)

# print("Google: ", result.content)


model = ChatAnthropic(model="claude-3-opus-20240229")

result = model.invoke(messages)

print("Anthropic: ", result.content)

