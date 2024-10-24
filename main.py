import streamlit as st
import json
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain_mistralai import ChatMistralAI
from langchain_mistralai.embeddings import MistralAIEmbeddings
from dotenv import load_dotenv
import random
import tiktoken
from langchain_core.messages import HumanMessage, SystemMessage
import os

# Load environment variables
load_dotenv()

if not os.getenv("MISTRAL_API_KEY"):
    raise EnvironmentError("MISTRAL_API_KEY environment variable is not set")

def estimate_token_count(text, model_name="mistral-large-latest"):
    """Estimate token count using tiktoken."""
    # Use cl100k_base for Mistral models as it's closest to their tokenizer
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    return len(tokens)

def limit_context_size(contexts, max_tokens, model_name="mistral-large-latest"):
    """Trim the context to stay under the token limit."""
    total_tokens = 0
    limited_context = []
    
    for context in reversed(contexts):
        token_count = estimate_token_count(context, model_name)
        
        if total_tokens + token_count > max_tokens:
            break
            
        limited_context.append(context)
        total_tokens += token_count
    
    return list(reversed(limited_context))

def process_messages(messages):
    """Process and filter WhatsApp messages, removing duplicates and system messages."""
    processed_messages = []
    seen_messages = set()
    
    for msg in messages:
        # Skip empty or media messages
        if not msg.get('message') or msg['message'] == "<Media omitted>":
            continue
            
        # Create unique message identifier
        message_key = f"{msg['sender']}:{msg['message']}"
        
        if message_key not in seen_messages:
            # Extract the first word from sender's name as role
            sender = msg['sender'].strip()
            role = sender.split()[0]
            
            processed_messages.append({
                'sender': sender,
                'role': role,
                'message': msg['message'].strip()
            })
            seen_messages.add(message_key)
    
    return processed_messages

@st.cache_resource
def create_vector_store(processed_messages):
    """Create and persist a FAISS vector store from processed messages."""
    # Filter out duplicate messages while preserving order
    unique_messages = []
    seen = set()
    
    for msg in processed_messages:
        if msg['message'] not in seen:
            unique_messages.append(msg)
            seen.add(msg['message'])
    
    # Create documents for vectorization
    documents = [
        Document(
            page_content=msg['message'],
            metadata={'sender': msg['sender'], 'role': msg['role']}
        ) 
        for msg in unique_messages
    ]
    
    # Initialize embeddings and create vector store
    embeddings = MistralAIEmbeddings(
        model="mistral-embed",
        mistral_api_key=os.getenv("MISTRAL_API_KEY")
    )
    
    vector_store = FAISS.from_documents(documents=documents, embedding=embeddings)
    return vector_store

def generate_response(
    user_input, 
    vector_store, 
    conversation_history, 
    selected_role, 
    tone, 
    max_context_tokens=2000
):
    """Generate contextual responses based on chat history."""
    # Get relevant documents
    relevant_docs = vector_store.similarity_search(
        user_input,
        k=8,
        fetch_k=10  # Fetch more docs for better sampling
    )
    
    # Sample random subset of relevant docs
    selected_docs = random.sample(relevant_docs, min(5, len(relevant_docs)))
    
    # Combine context from vector store and recent conversation
    full_context = [
        f"{doc.metadata['sender']}: {doc.page_content}" 
        for doc in selected_docs
    ]
    full_context.extend(conversation_history[-6:])
    
    # Limit context size
    limited_context = limit_context_size(full_context, max_context_tokens)
    
    # Define tone markers and styles
    tone_styles = {
        "Charming and fun": {
            "markers": ["😊", "haha", "interesting", "wow"],
            "style": "friendly and engaging"
        },
        "Sarcastic": {
            "markers": ["well well well", "oh really", "fascinating", "*eye roll*"],
            "style": "witty and slightly sarcastic"
        },
        "Professional": {
            "markers": ["Indeed", "Certainly", "I understand", "Precisely"],
            "style": "formal and professional"
        }
    }
    
    selected_style = tone_styles.get(tone, {"markers": [""], "style": "neutral"})
    selected_marker = random.choice(selected_style["markers"])
    
    # Construct system prompt
    system_prompt = f"""You are role-playing as "{selected_role}" based on WhatsApp chat history.
    Your responses should:
    1. Closely mirror {selected_role}'s communication style from the provided context
    2. Maintain a {selected_style['style']} tone
    3. Be natural and varied in expression
    4. Stay concise and relevant
    5. Avoid repetitive phrases or patterns

    Context from chat history:
    {"\n".join(limited_context)}

    Recent conversation:
    {"\n".join(conversation_history[-3:]) if conversation_history else "No previous conversation"}"""

    # Create message objects for LLM
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_input)
    ]
    
    # Generate response
    llm = ChatMistralAI(
        model="mistral-large-latest",
        mistral_api_key=os.getenv("MISTRAL_API_KEY"),
        temperature=0.7
    )
    
    response = llm(messages)
    
    # Format response with selected marker
    response_text = f"{selected_role}: {selected_marker} {response.content.strip()}"
    return response_text

def main():
    st.set_page_config(
        page_title="WhatsApp Mimic Bot",
        page_icon="💬",
        layout="wide"
    )
    
    st.title("💬 WhatsApp Mimic Bot")
    st.write("Upload your WhatsApp chat JSON file and I'll chat in the style of those messages!")
    
    # Initialize session state
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    
    if 'message_count' not in st.session_state:
        st.session_state.message_count = 0
        
    if 'processed_messages' not in st.session_state:
        st.session_state.processed_messages = None
        
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a WhatsApp chat JSON file",
        type="json",
        help="Upload a JSON file exported from WhatsApp"
    )
    
    if uploaded_file is not None:
        # Process uploaded file
        try:
            json_data = json.load(uploaded_file)
            
            if st.session_state.processed_messages is None:
                st.session_state.processed_messages = process_messages(json_data)
                
            # Get unique roles
            roles = sorted(set(
                msg['role'] for msg in st.session_state.processed_messages
            ))
            
            # Role selection
            col1, col2 = st.columns(2)
            
            with col1:
                selected_role = st.selectbox(
                    "Select role to imitate",
                    roles,
                    help="Choose whose style to mimic"
                )
                
            with col2:
                tone = st.selectbox(
                    "Select conversation tone",
                    [
                        "Charming and fun",
                        "Sarcastic",
                        "Professional"
                    ],
                    help="Choose the tone of responses"
                )
            
            # Filter messages for selected role
            filtered_messages = [
                msg for msg in st.session_state.processed_messages
                if msg['role'] == selected_role
            ]
            
            # Create or get vector store
            if 'vector_store' not in st.session_state:
                with st.spinner("Processing messages..."):
                    st.session_state.vector_store = create_vector_store(filtered_messages)
                st.success("Chat history processed successfully!")
            
            # Display conversation history
            st.container()
            for msg in st.session_state.conversation_history:
                if msg["role"] == "user":
                    st.write(f"You: {msg['content']}")
                else:
                    st.write(msg['content'])
            
            # Chat input
            user_input = st.text_input(
                "Your message:",
                key="user_input",
                placeholder="Type your message here..."
            )
            
            if user_input:
                st.session_state.message_count += 1
                
                # Add user message to history
                st.session_state.conversation_history.append({
                    "role": "user",
                    "content": user_input
                })
                
                # Generate and add response
                response = generate_response(
                    user_input,
                    st.session_state.vector_store,
                    [msg["content"] for msg in st.session_state.conversation_history],
                    selected_role,
                    tone
                )
                
                st.session_state.conversation_history.append({
                    "role": "assistant",
                    "content": response
                })
                
                st.rerun()
            
            # Clear chat button
            if st.button("Clear Chat History"):
                st.session_state.conversation_history = []
                st.session_state.message_count = 0
                st.rerun()
                
        except json.JSONDecodeError:
            st.error("Error: Invalid JSON file. Please upload a valid WhatsApp chat export.")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    
    else:
        st.info("Please upload a WhatsApp chat JSON file to start chatting!")

if __name__ == "__main__":
    main()