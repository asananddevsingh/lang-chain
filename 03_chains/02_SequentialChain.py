from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda
llm = ChatOpenAI(model="gpt-4o-mini")

# Animal Facts Template
animal_facts_prompt_tempalte = ChatPromptTemplate.from_messages([
    ("system", "You can tell facts about {animal}"),
    ("user", "Tell me {count} facts")
])

# Define prompt to convert fact into hindi language.
translation_prompt_tempalte = ChatPromptTemplate.from_messages([
    ("system", "Translate the following text into hindi: {text}"),
    ("user", "{text}")
])

# Prepare for translation.
prepare_for_translation = RunnableLambda(lambda output: {
    "text": output,
    "language": "Hindi"
})

# Create the chain
chain = animal_facts_prompt_tempalte | llm | prepare_for_translation | translation_prompt_tempalte | llm | StrOutputParser()

response = chain.invoke({
    "animal": "Dog",
    "count": 2
})

print("# Response ::", response)