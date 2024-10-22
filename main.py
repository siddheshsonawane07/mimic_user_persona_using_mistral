import streamlit as st
import json
import os
import faiss
import numpy as np
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from dotenv import load_dotenv

def process_messages(messages):
    """Process and filter WhatsApp messages"""
    return [msg for msg in messages if msg.get('message') and msg['message'] != "<Media omitted>"]

def create_vector_store(processed_messages):
    """Create and persist a FAISS vector store from processed messages."""
    load_dotenv()

    # Convert processed messages into Document objects
    documents = [Document(page_content=msg['message']) for msg in processed_messages if msg.get('message')]

    # Set the model for embeddings
    model_name = "intfloat/e5-large-v2"
    model_kwargs = {'device': 'cuda:0'}
    embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

    vector_store = FAISS.from_documents(documents=documents, embedding = embeddings)

    return vector_store

# Generate Response using Ollama's Llama 3.2
def generate_response(user_input, vector_store, conversation_history):
    """Generate contextual responses based on chat history"""
    # Get relevant context from vector store
    relevant_docs = vector_store.similarity_search(user_input, k=5)
    context = "\n".join([doc.page_content for doc in relevant_docs])
    
    # Create a prompt that uses the chat context to maintain personality
    prompt = f"""
    You are a chatbot that role-plays based on the style and personality shown in the following chat messages.
    Analyze these messages and respond in a similar style and tone:

    Previous messages for context:
    {context}

    Recent conversation:
    {conversation_history}

    User: {user_input}
    Assistant (responding in the style of the chat messages):"""
    
    # Generate response using Ollama's Llama 3.2
    local_llm = "llama3.2"
    llm = ChatOllama(model=local_llm, temperature=0)
    response = llm(prompt)
    
    return response.content


# Streamlit UI
st.title("💬 WhatsApp Context Bot")
st.write("Upload your WhatsApp chat history and I'll chat in the style of those messages!")

# Initialize session state for conversation history
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []



# File uploader
uploaded_file = st.file_uploader("Choose a WhatsApp chat JSON file", type="json")

if uploaded_file is not None:
    json_data = json.load(uploaded_file)
    processed_messages = process_messages(json_data)
    
    with st.spinner("Processing messages..."):
        vector_store = create_vector_store(processed_messages)
    st.success("Chat history processed successfully!")

    # Display conversation history
    for message in st.session_state.conversation_history:
        if message["role"] == "user":
            st.write(f"You: {message['content']}")
        else:
            st.write(f"Bot: {message['content']}")
    
    # Chat input
    user_input = st.text_input("Your message:", key="user_input")
    
    if user_input:
        # Add user message to history
        st.session_state.conversation_history.append({"role": "user", "content": user_input})
        
        # Generate response using context from recent messages
        response = generate_response(
            user_input,
            vector_store,
            "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.conversation_history[-5:]])
        )
        
        # Add response to history
        st.session_state.conversation_history.append({"role": "assistant", "content": response})
        
        # Rerun to update the display
        st.rerun()

else:
    st.info("Please upload a WhatsApp chat JSON file to start chatting!")

# Add a clear chat button
if st.button("Clear Chat History"):
    st.session_state.conversation_history = []
    st.rerun()