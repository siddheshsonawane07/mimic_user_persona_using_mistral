import json
import streamlit as st
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document

def process_messages(messages):
    return [msg['message'] for msg in messages if msg['message'] != "<Media omitted>"]

def create_vector_store(messages):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_text('\n'.join(messages))
    documents = [Document(page_content=text) for text in texts]
    
    embeddings_model = OllamaEmbeddings(model="llama3.2")
    
    # Create embeddings from the text documents
    texts = [doc.page_content for doc in documents]
    text_embeddings = embeddings_model.embed_documents(texts)
    
    # Use FAISS to create the vector store with both texts and embeddings
    vector_store = FAISS.from_embeddings(text_embeddings, texts)
    
    return vector_store

def setup_rag_chatbot(vector_store):
    llm = ChatOllama(model="llama3.2", temperature=0.0)

    condense_question_system_template = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )

    condense_question_prompt = ChatPromptTemplate.from_messages([
        ("system", condense_question_system_template),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm, 
        vector_store.as_retriever(), 
        condense_question_prompt
    )

    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
    ])

    qa_chain = create_stuff_documents_chain(llm, qa_prompt)
    return create_retrieval_chain(history_aware_retriever, qa_chain)

# Streamlit UI
st.title("WhatsApp RAG Chatbot")

uploaded_file = st.file_uploader("Choose a WhatsApp chat JSON file", type="json")

if uploaded_file is not None:
    json_data = json.load(uploaded_file)
    processed_messages = process_messages(json_data)
    
    st.write("Processing messages...")
    vector_store = create_vector_store(processed_messages)
    st.write("Vector store created.")
    
    convo_qa_chain = setup_rag_chatbot(vector_store)

    st.subheader("Chat with the WhatsApp persona")
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.text_input("Your message:")
    
    if user_input:
        response = convo_qa_chain.invoke({
            "input": user_input,
            "chat_history": st.session_state.chat_history
        })
        st.write(f"Chatbot: {response['answer']}")
        st.session_state.chat_history.append((user_input, response['answer']))

    for user_msg, bot_msg in st.session_state.chat_history:
        st.write(f"You: {user_msg}")
        st.write(f"Chatbot: {bot_msg}")
