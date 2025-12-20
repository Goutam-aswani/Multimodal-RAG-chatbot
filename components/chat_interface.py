"""
Chat Interface Component
Main chat UI with streaming support and memory-aware querying.
"""

import streamlit as st

def render_chat():
    """Render the main chat interface."""
    
    st.title("🔍 Multimodal RAG Chat")
    st.caption("💭 Conversational memory enabled - I remember our chat context!")
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
                with st.expander("📚 Sources"):
                    st.markdown(message["sources"])
            # Show reformulated query if different from original
            if "reformulated_query" in message and message.get("reformulated_query"):
                original = message.get("original_query", "")
                reformulated = message["reformulated_query"]
                if original and reformulated and original.lower().strip() != reformulated.lower().strip():
                    with st.expander("🔄 Query Reformulation"):
                        st.caption(f"**Original:** {original}")
                        st.caption(f"**Reformulated:** {reformulated}")
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
            
            # Stream response
            for token in st.session_state.rag_pipeline.query(prompt):
                full_response += token
                response_placeholder.markdown(full_response + "▌")
            
            response_placeholder.markdown(full_response)
            
            # Show sources
            sources = st.session_state.rag_pipeline.get_citations_markdown()
            if sources:
                with st.expander("📚 Sources"):
                    st.markdown(sources)
            
            # Show reformulated query if different
            reformulated = st.session_state.rag_pipeline.get_last_reformulated_query()
            if reformulated and prompt.lower().strip() != reformulated.lower().strip():
                with st.expander("🔄 Query Reformulation"):
                    st.caption(f"**Original:** {prompt}")
                    st.caption(f"**Reformulated:** {reformulated}")
        
        # Save to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response,
            "sources": sources,
            "original_query": prompt,
            "reformulated_query": reformulated
        })
