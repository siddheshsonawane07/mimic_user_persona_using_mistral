import streamlit as st
import json
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
import random

def process_messages(messages):
    """Process and filter WhatsApp messages, removing duplicates"""
    processed_messages = []
    seen_messages = set()  # Track unique messages
    
    for msg in messages:
        if msg.get('message') and msg['message'] != "<Media omitted>":
            message_key = f"{msg['sender']}:{msg['message']}"
            if message_key not in seen_messages:
                sender = msg['sender']
                role = sender.split()[0]  # Extract the role from the sender's name
                processed_messages.append({
                    'sender': sender,
                    'role': role,
                    'message': msg['message']
                })
                seen_messages.add(message_key)
    
    return processed_messages

def create_vector_store(processed_messages):
    """Create and persist a FAISS vector store from processed messages."""
    load_dotenv()

    # Filter out duplicate messages
    unique_messages = list({msg['message']: msg for msg in processed_messages}.values())
    documents = [Document(page_content=msg['message']) for msg in unique_messages if msg.get('message')]

    model_name = "intfloat/e5-large-v2"
    model_kwargs = {'device': 'cuda:0'}
    embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

    vector_store = FAISS.from_documents(documents=documents, embedding=embeddings)
    return vector_store

def generate_response(user_input, vector_store, conversation_history, selected_role, tone):
    """Generate contextual responses based on chat history with improved variety"""
    # Get more relevant docs for variety
    relevant_docs = vector_store.similarity_search(user_input, k=8)
    
    selected_docs = random.sample(relevant_docs, min(5, len(relevant_docs)))
    context = "\n".join([doc.page_content for doc in selected_docs])

    tone_markers = {
        "Charming and fun": ["😊", "haha", "interesting", "wow"],
        "Sarcastic": ["well well well", "oh really", "fascinating", "*eye roll*"],
        "Professional": ["Indeed", "Certainly", "I understand", "Precisely"]
    }

    # Randomly select a tone marker
    selected_marker = random.choice(tone_markers.get(tone, [""]))

    system_prompt = f"""You are a chatbot that role-plays as {selected_role} based on the style and personality shown in the following chat messages.
    Respond in a {tone} style and tone, starting with "{selected_marker}" when appropriate.
    
    Important guidelines:
    1. Avoid repeating yourself or using the same phrases repeatedly
    2. Keep responses concise and varied
    3. Match the conversation style of the provided context
    4. Maintain consistent personality but vary expression

    Relevant context from chat history:
    {context}

    Previous conversation:
    {conversation_history[-3:] if conversation_history else "No previous conversation"}

    User's latest message:
    {user_input}"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_input)
    ]

    local_llm = "mistral"
    llm = ChatOllama(model=local_llm, temperature=0.2)  # Increased temperature for more variety
    response = llm(messages)

    # Format response with role
    response_text = f"{selected_role}: {response.content}"
    return response_text

def main():
    st.title("💬 WhatsApp Context Bot")
    st.write("Upload your WhatsApp chat history and I'll chat in the style of those messages!")

    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    
    if 'message_count' not in st.session_state:
        st.session_state.message_count = 0

    uploaded_file = st.file_uploader("Choose a WhatsApp chat JSON file", type="json")

    if uploaded_file is not None:
        json_data = json.load(uploaded_file)
        processed_messages = process_messages(json_data)

        roles = sorted(set([msg['role'] for msg in processed_messages]))

        selected_role = st.selectbox("Select your role", roles)

        filtered_messages = [msg for msg in processed_messages if msg['role'] == selected_role]

        if 'vector_store' not in st.session_state:
            with st.spinner("Processing messages..."):
                st.session_state.vector_store = create_vector_store(filtered_messages)
            st.success("Chat history processed successfully!")

        for msg in st.session_state.conversation_history:
            if msg["role"] == "user":
                st.write(f"You: {msg['content']}")
            else:
                st.write(msg['content'])  

        tone_descriptions = {
            "Charming and fun": "Light-hearted and engaging responses",
            "Sarcastic": "Witty and playfully critical responses",
            "Professional": "Formal and business-like responses"
        }
        tone = st.selectbox(
            "Select a tone or style",
            list(tone_descriptions.keys()),
            help="\n".join([f"{k}: {v}" for k, v in tone_descriptions.items()])
        )

        user_input = st.text_input("Your message:", key="user_input")

        if user_input:
            st.session_state.message_count += 1
            
            # Add user message to history
            st.session_state.conversation_history.append({"role": "user", "content": user_input})

            # Generate response
            response = generate_response(
                user_input,
                st.session_state.vector_store,
                [msg["content"] for msg in st.session_state.conversation_history[-6:]],
                selected_role,
                tone
            )

            st.session_state.conversation_history.append({"role": "assistant", "content": response})

            st.rerun()

    else:
        st.info("Please upload a WhatsApp chat JSON file to start chatting!")

    if st.button("Clear Chat History"):
        st.session_state.conversation_history = []
        st.session_state.message_count = 0
        st.rerun()

if __name__ == "__main__":
    main()