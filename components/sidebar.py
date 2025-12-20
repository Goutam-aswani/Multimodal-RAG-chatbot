"""
Sidebar Component
Streamlit sidebar for document upload and settings.
"""

import streamlit as st
import os
from config import settings

def render_sidebar():
    """Render the file upload sidebar."""
    
    with st.sidebar:
        st.header("📁 Document Upload")
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Upload PDFs, Images, or Text files",
            type=["pdf", "png", "jpg", "jpeg", "txt"],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            st.write(f"**{len(uploaded_files)} file(s) selected**")
            
            if st.button("🚀 Process Documents", type="primary"):
                with st.spinner("Processing documents..."):
                    # Save files
                    file_paths = []
                    for file in uploaded_files:
                        path = os.path.join(settings.upload_dir, file.name)
                        os.makedirs(settings.upload_dir, exist_ok=True)
                        with open(path, "wb") as f:
                            f.write(file.getbuffer())
                        file_paths.append(path)
                    
                    # Process documents
                    num_chunks = st.session_state.rag_pipeline.process_documents(file_paths)
                    st.session_state.documents_loaded = True
                    
                st.success(f"✅ Processed {num_chunks} chunks from {len(uploaded_files)} files")
        
        # Status indicator
        st.divider()
        if st.session_state.documents_loaded:
            st.success("🟢 Ready to chat!")
        else:
            st.warning("🟡 Upload documents to start")
        
        # Settings
        st.divider()
        st.subheader("⚙️ Settings")
        
        model = st.selectbox(
            "Model",
            ["gemini-2.5-flash", "gemini-2.5-pro", "llama-3.1-8b"],
            index=0
        )
        
        # Memory info
        memory_len = len(st.session_state.rag_pipeline.memory) if hasattr(st.session_state.rag_pipeline, 'memory') else 0
        if memory_len > 0:
            st.caption(f"💭 Memory: {memory_len // 2} conversation turns")
        
        # Clear options
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 New Chat"):
                st.session_state.messages = []
                st.session_state.rag_pipeline.clear_memory()
                st.rerun()
        
        with col2:
            if st.button("🗑️ Clear All"):
                st.session_state.messages = []
                st.session_state.documents_loaded = False
                st.session_state.rag_pipeline.clear()
                st.rerun()
