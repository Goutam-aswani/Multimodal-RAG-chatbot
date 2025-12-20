import streamlit as st
from components.sidebar import render_sidebar
from components.chat_interface import render_chat
from services.rag_pipeline import RAGPipeline

st.set_page_config(
    page_title="Multimodal RAG Chat",
    page_icon="🔍",
    layout="wide"
)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag_pipeline" not in st.session_state:
    st.session_state.rag_pipeline = RAGPipeline()
if "documents_loaded" not in st.session_state:
    st.session_state.documents_loaded = False

render_sidebar()
render_chat()
