import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pinecone import Pinecone
from google import genai

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

client = genai.Client()

chat_history = []

def transformQuery(question):
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents = [
            "You are a query rewriting expert. Based on the provided chat history, rephrase the 'Follow Up user Question' into a complete, standalone question that can be understood without the chat history. Only output the rewritten question and nothing else.",
            f"chat_history: {chat_history}",
            f"Follow Up user Question: {question}"
        ]
    )
    #print(f"Chatbot: {response.text}")
    return response.text

def create_question_vector(question):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=GEMINI_API_KEY,
        )
    query_vector = embeddings.embed_query(question)
    return query_vector

def get_context(question):
    question = transformQuery(question)
    query_vector = create_question_vector(question)
    pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
    index = pc.Index(PINECONE_INDEX_NAME)
    results = index.query(
        vector=query_vector,
        top_k=3,  # number of results to return
        include_metadata=True  # include metadata if stored
    )
    context = "\n".join([match['metadata']['text'] for match in results['matches']])
    #print("Context retrieved:", context)
    return context

def chat():
    print("Welcome to the PDF Chatbot! Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Chatbot: Goodbye!")
            print("CHAT HISTORY:", chat_history)
            break
        chat_history.append({"role": "you", "content": user_input})
        context = get_context(user_input)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents = [
                "I have given you the context of a short story. Keep your answers clear, concise, and educational. Answer the question based on the context. If you don't find the answer in context, say 'I don't know'.",
                f"chat_history: {chat_history}",
                f"context: {context}",
                f"question: {user_input}"
            ]
        )
        print(f"Chatbot: {response.text}")
        chat_history.append({"role": "chatbot", "content": response.text})

if __name__ == "__main__":
    chat()
    
    