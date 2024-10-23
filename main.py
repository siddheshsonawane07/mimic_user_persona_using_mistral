import streamlit as st
import json
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

def process_messages(messages):
    """Process and filter WhatsApp messages"""
    processed_messages = []
    for msg in messages:
        if msg.get('message') and msg['message'] != "<Media omitted>":
            sender = msg['sender']
            role = sender.split()[0]  # Extract the role from the sender's name
            processed_messages.append({
                'sender': sender,
                'role': role,
                'message': msg['message']
            })
    return processed_messages

def create_vector_store(processed_messages):
    """Create and persist a FAISS vector store from processed messages."""
    load_dotenv()

    documents = [Document(page_content=msg['message']) for msg in processed_messages if msg.get('message')]

    model_name = "intfloat/e5-large-v2"
    model_kwargs = {'device': 'cuda:0'}
    embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

    vector_store = FAISS.from_documents(documents=documents, embedding=embeddings)

    return vector_store

def generate_response(user_input, vector_store, conversation_history, selected_role):
    """Generate contextual responses based on chat history"""
    relevant_docs = vector_store.similarity_search(user_input, k=5)
    context = "\n".join([doc.page_content for doc in relevant_docs])

    system_prompt = f"""You are a chatbot that role-plays as {selected_role} based on the style and personality shown in the following chat messages.
    Analyze these messages and respond in a similar style and tone.

    Relevant context from chat history:s
    {context}

    Recent conversation:
    {conversation_history}"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_input)
    ]

    local_llm = "mistral"
    llm = ChatOllama(model=local_llm, temperature=0)
    response = llm(messages)

    # Attribute the response to the selected role
    response_text = f"{selected_role}: {response.content}"

    return response_text


# Streamlit UI
def main():
    st.title("💬 WhatsApp Context Bot")
    st.write("Upload your WhatsApp chat history and I'll chat in the style of those messages!")

    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []

    uploaded_file = st.file_uploader("Choose a WhatsApp chat JSON file", type="json")

    if uploaded_file is not None:
        json_data = json.load(uploaded_file)
        processed_messages = process_messages(json_data)

        roles = set([msg['role'] for msg in processed_messages])

        selected_role = st.selectbox("Select your role", roles)

        filtered_messages = [msg for msg in processed_messages if msg['role'] == selected_role]

        with st.spinner("Processing messages..."):
            vector_store = create_vector_store(filtered_messages)
        st.success("Chat history processed successfully!")

        for message in st.session_state.conversation_history:
            if message["role"] == "user":
                st.write(f"You: {message['content']}")
            else:
                st.write(f"Bot: {message['content']}")

        user_input = st.text_input("Your message:", key="user_input")

        if user_input:
            st.session_state.conversation_history.append({"role": "user", "content": user_input})

            response = generate_response(
                user_input,
                vector_store,
                "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.conversation_history[-5:]]),
                selected_role
            )

            st.session_state.conversation_history.append({"role": "assistant", "content": response})

            st.rerun()

    else:
        st.info("Please upload a WhatsApp chat JSON file to start chatting!")

    if st.button("Clear Chat History"):
        st.session_state.conversation_history = []
        st.rerun()

if __name__ == "__main__":
    main()
