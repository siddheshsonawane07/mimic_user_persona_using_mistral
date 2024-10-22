import streamlit as st
import json
from langchain.vectorstores import FAISS
from langchain_mistralai.embeddings import MistralAIEmbeddings
from langchain.docstore.document import Document
from dotenv import load_dotenv

def process_messages(messages):
    return [msg for msg in messages if msg.get('message') and msg['message'] != "<Media omitted>"]

def create_vector_store(processed_messages):
    load_dotenv()

    # Convert the list of dictionaries into a list of Document objects, checking for the "message" key
    documents = [Document(page_content=msg['message']) for msg in processed_messages if msg.get('message')]

    embeddings = MistralAIEmbeddings(model="mistral-embed")
    vector_store = FAISS.from_documents(documents, embeddings)

    return vector_store

def setup_rag_chatbot(vector_store):
    return 1



st.title("WhatsApp RAG Chatbot")

uploaded_file = st.file_uploader("Choose a WhatsApp chat JSON file", type="json")

if uploaded_file is not None:
    json_data = json.load(uploaded_file)
    processed_messages = process_messages(json_data)
    
    st.write("Processing messages...")
    vector_store = create_vector_store(processed_messages)
    st.write("Vector store created.")
    
    convo_qa_chain = setup_rag_chatbot(vector_store)