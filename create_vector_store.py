import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

def get_pdf_text(uploaded_files):
    text = ""
    for pdf in uploaded_files:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(raw_texts):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(raw_texts)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=GEMINI_API_KEY,
        )
    
    pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
    index = pc.Index(PINECONE_INDEX_NAME)
    
    documents = [Document(page_content=chunk) for chunk in text_chunks]
    vector_store = PineconeVectorStore.from_documents(
        documents=documents,
        embedding=embeddings,
        index_name=PINECONE_INDEX_NAME
    )
    
    print("test: vecotrstore", vector_store)
    return vector_store


def main():
    uploaded_files = ['docs/1_story.pdf']
    raw_texts = get_pdf_text(uploaded_files)
    text_chunks = get_text_chunks(raw_texts)
    vectorstore = get_vectorstore(text_chunks)
    print("Vector store created successfully with the provided PDF documents.", vectorstore)
main()