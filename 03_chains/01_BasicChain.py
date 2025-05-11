from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

llm = ChatOpenAI(model="gpt-4o-mini")

messages = [
    ("system", "You are a comedian who tells jokes on {topic}"),
    ("user", "Tell me {count} jokes")
]

messages_prompt_template = ChatPromptTemplate.from_messages(messages)

# Simple Chain Exmaple
# chain = messages_prompt_template | llm

# response = chain.invoke({
#     "count": 2,
#     "topic": "Software Engineering"
# })

# print("# Response ::", response.content)

# Chain with Structured Output
chain = messages_prompt_template | llm | StrOutputParser()

response = chain.invoke({
    "count": 2,
    "topic": "Software Engineering"
})

print("# Response ::", response)

