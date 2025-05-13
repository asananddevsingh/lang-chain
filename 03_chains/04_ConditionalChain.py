from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableBranch

# Load environment variables from .env
load_dotenv()

# Create a ChatOpenAI model
model = ChatOpenAI(model="gpt-4o-mini")

classification_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "Classify the following feedback as positive, neutral, negative: {feedback}."),
])

positive_feedback_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "Generate a thank you note for this positive feedback in 1 line: {feedback}."),
])

neutral_feedback_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "Generate a response addressing this neutral feedback in 1 line: {feedback}."),
])

negative_feedback_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "Generate a request for more details for this negative feedback: {feedback}."),
])

branches = RunnableBranch(
    (
        lambda x: print("Checking positive:", x) or "positive" in x,
        positive_feedback_template | model | StrOutputParser()
    ),
    (
        lambda x: print("Checking negative:", x) or "negative" in x,
        negative_feedback_template | model | StrOutputParser()
    ),
    neutral_feedback_template | model | StrOutputParser()
)

classification_chain = classification_template | model | StrOutputParser()

chain = classification_chain | branches

# response = chain.invoke({"feedback": "The product is Great."})
response = chain.invoke({"feedback": "The product is terrible. It broke after just one use and the quality is very poor."})

print(":: RESPONSE ::", response)