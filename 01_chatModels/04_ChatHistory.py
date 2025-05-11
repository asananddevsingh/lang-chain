from dotenv import load_dotenv
from google.cloud import firestore
from langchain_google_firestore import FirestoreChatMessageHistory
from langchain_openai import ChatOpenAI

load_dotenv()

# Setup Firebase Firestore
PROJECT_ID = "langchaintutorial-13e6e"
SESSION_ID = "1234"
COLLECTION_NAME = "LangChainCollection"

# Initialize Firestore client
client = firestore.Client(project=PROJECT_ID)

# Initialize FirestoreChatMessageHistory
message_history = FirestoreChatMessageHistory(
    session_id=SESSION_ID,
    client=client
)

# Initialize OpenAI model
model = ChatOpenAI(model="gpt-4o")

while True:
    human_message = input("User: ")
    if human_message.lower() == "exit":
        break

    # Add user message to history
    message_history.add_user_message(human_message)

    ai_response = model.invoke(message_history.messages)
    message_history.add_ai_message(ai_response.content)

    print(f"AI: {ai_response.content}")


    
    
    
