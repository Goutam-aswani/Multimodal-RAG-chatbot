"""
Document Viewer Component
Display sources and citations with document preview.
"""

import streamlit as st
from typing import List
from core.document_processor import Chunk

def render_sources(chunks: List[Chunk]):
    """Render source chunks in an expandable view."""
    if not chunks:
        st.info("No sources to display.")
        return
    
    for i, chunk in enumerate(chunks, 1):
        with st.expander(f"📄 Source {i}: {chunk.source_file} - Page {chunk.page_number}"):
            # Display chunk type badge
            chunk_type_colors = {
                "text": "blue",
                "table": "green",
                "image": "orange"
            }
            color = chunk_type_colors.get(chunk.chunk_type, "gray")
            st.markdown(f"**Type:** :{color}[{chunk.chunk_type.upper()}]")
            
            # Display content
            st.markdown("**Content:**")
            st.text_area(
                label=f"Content {i}",
                value=chunk.content,
                height=200,
                disabled=True,
                label_visibility="collapsed"
            )
            
            # Display metadata if any
            if chunk.metadata:
                st.markdown("**Metadata:**")
                st.json(chunk.metadata)

def render_citation_list(chunks: List[Chunk]) -> str:
    """Render a formatted citation list."""
    if not chunks:
        return ""
    
    citations = []
    for i, chunk in enumerate(chunks, 1):
        citations.append(
            f"**[{i}]** {chunk.source_file} - Page {chunk.page_number} ({chunk.chunk_type})"
        )
    return "\n".join(citations)

def render_document_preview(file_path: str, page_number: int = 1):
    """
    Render a preview of a document page.
    Note: This is a placeholder - can be enhanced with PDF rendering.
    """
    st.info(f"Preview for: {file_path}, Page {page_number}")
    st.markdown("*Document preview functionality can be added here.*")
