import streamlit as st
import json
import os
from langchain_community.vectorstores import Chroma
from langchain_mistralai.embeddings import MistralAIEmbeddings
from langchain.docstore.document import Document
from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv

def process_messages(messages):
    """Process and filter WhatsApp messages"""
    return [msg for msg in messages if msg.get('message') and msg['message'] != "<Media omitted>"]

def create_vector_store(processed_messages, persist_dir="./chroma_store"):
    """Create and persist a Chroma vector store from messages"""
    load_dotenv()
    
    documents = [Document(page_content=msg['message']) for msg in processed_messages if msg.get('message')]
    embeddings = MistralAIEmbeddings(model="mistral-embed")
    vector_store = Chroma.from_documents(documents=documents, embedding=embeddings, persist_directory=persist_dir)
    vector_store.persist()
    
    return vector_store

def load_vector_store(persist_dir="./chroma_store"):
    """Load existing vector store if it exists"""
    if os.path.exists(persist_dir):
        embeddings = MistralAIEmbeddings(model="mistral-embed")
        vector_store = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
        return vector_store
    return None

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
    
    # Generate response using Mistral
    mistral_llm = ChatMistralAI(model="mistral-large-latest")
    response = mistral_llm.invoke(prompt).content
    
    return response

# Streamlit UI
st.title("💬 WhatsApp Context Bot")
st.write("Upload your WhatsApp chat history and I'll chat in the style of those messages!")

# Initialize session state for conversation history
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

# Vector store handling
chroma_store_path = "./chroma_store"
vector_store = load_vector_store(chroma_store_path)

# File uploader
uploaded_file = st.file_uploader("Choose a WhatsApp chat JSON file", type="json")

if uploaded_file is not None:
    json_data = json.load(uploaded_file)
    processed_messages = process_messages(json_data)
    
    with st.spinner("Processing messages..."):
        vector_store = create_vector_store(processed_messages, persist_dir=chroma_store_path)
    st.success("Chat history processed successfully!")

if vector_store is not None:
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