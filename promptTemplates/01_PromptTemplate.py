from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")

# Example 1:
# template = "Write a {tone} email to {company} expressing interest in the {position} position, mentioning {skill} as a key strength. Keep it to 4 lines max"

# prompt_template = ChatPromptTemplate.from_template(template)

# print(prompt_template)

# prompt =  prompt_template.invoke({
#     "tone": "energetic", 
#     "company": "samsung", 
#     "position": "AI Engineer", 
#     "skill": "AI"
# })

# print("# Prompt Example 1 ::", prompt)

# Example 2:
messages = [
    ("system", "You are a comedian who tells jokes on {topic}"),
    ("user", "Tell me {count} jokes")
]

messages_prompt_template = ChatPromptTemplate.from_messages(messages)

print("# Prompt Example 2 ::", messages_prompt_template)

prompt = messages_prompt_template.invoke({
    "count": 2,
    "topic": "Software Engineering"
})

print("# Prompt Example 2 ::", prompt)

response = llm.invoke(prompt)

print("# Response ::", response.content)

