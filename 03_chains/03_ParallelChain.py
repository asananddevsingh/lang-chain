from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableLambda

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")

summary_template = ChatPromptTemplate.from_messages([
    ("system", "You are a movie critic"),
    ("user", "Write a short summary of the movie: {movie} in 1 line.")
])

def analyze_plot(plot):
    plot_template = ChatPromptTemplate.from_messages([
        ("system", "You are a movie critic"),
        ("user", "Analyze the plot {plot} in 1 line.")
    ])

    return plot_template.format_prompt(plot=plot)

def analyze_characters(characters):
    character_template = ChatPromptTemplate.from_messages([
        ("system", "You are a movie critic"),
        ("user", "Analyze the characters {characters} in 1 line.")
    ])

    return character_template.format_prompt(characters=characters)

plot_branch_chain = (
    RunnableLambda(lambda x: analyze_plot(x)) | llm | StrOutputParser()
)

character_branch_chain = (
    RunnableLambda(lambda x: analyze_characters(x)) | llm | StrOutputParser()
)

# This is sequential chain.
# chain = summary_template | llm | StrOutputParser()

chain =  summary_template | llm | RunnableParallel(branches={"plot": plot_branch_chain, "characters": character_branch_chain}) | RunnableLambda(lambda x: x["branches"]["plot"] + "\n\n" + x["branches"]["characters"])

response = chain.invoke({"movie": "3 Idiots"})

print("Response: ", response)





