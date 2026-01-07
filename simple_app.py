import streamlit as st
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Page config
st.set_page_config(page_title="BRC Chatbot", page_icon="🤖")
st.title("RAG enabled Chatbot for BRC")

# Initialize session state for messages if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize RAG components
@st.cache_resource
def initialize_rag():
    """Initialize the RAG pipeline components"""
    # Set up LLM
    llm = Ollama(model="llama3.1", request_timeout=120.0)
    
    # Configure settings
    Settings.llm = llm
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5"
    )
    
    # Load and index documents
    documents = SimpleDirectoryReader("data").load_data()
    index = VectorStoreIndex.from_documents(documents)
    
    return index.as_query_engine()

# Initialize the query engine
query_engine = initialize_rag()

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
if prompt := st.chat_input("Ask a question about BRC"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate and display assistant response
    with st.chat_message("assistant"):
        response = query_engine.query(prompt)
        st.markdown(str(response))
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": str(response)})