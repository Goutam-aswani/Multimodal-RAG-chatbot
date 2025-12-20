"""
LLM Service Module
Functions:
- build_prompt(query, context) → str
- generate_response(prompt) → Generator[str]
- format_citations(chunks) → str
- reformulate_query(query, history) → str
Classes:
- ConversationMemory: Manages conversation history
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from config import settings
from typing import List, Generator, Tuple, Optional
from dataclasses import dataclass, field
from core.document_processor import Chunk


@dataclass
class Message:
    """A single message in the conversation."""
    role: str
    content: str


class ConversationMemory:
    """Manages conversation history for context-aware querying."""
    
    def __init__(self, max_turns: int = 5):
        """
        Initialize conversation memory.
        
        Args:
            max_turns: Maximum number of conversation turns to keep (1 turn = 1 user + 1 assistant message)
        """
        self.max_turns = max_turns
        self.messages: List[Message] = []
    
    def add_user_message(self, content: str):
        """Add a user message to history."""
        self.messages.append(Message(role="user", content=content))
        self._trim_history()
    
    def add_assistant_message(self, content: str):
        """Add an assistant message to history."""
        self.messages.append(Message(role="assistant", content=content))
        self._trim_history()
    
    def _trim_history(self):
        """Keep only the last N turns of conversation."""
        max_messages = self.max_turns * 2  # Each turn has 2 messages
        if len(self.messages) > max_messages:
            self.messages = self.messages[-max_messages:]
    
    def get_history_for_reformulation(self) -> str:
        """Get formatted history for query reformulation."""
        if not self.messages:
            return ""
        
        history_parts = []
        for msg in self.messages[-6:]:  # Last 3 turns max for reformulation
            role = "Human" if msg.role == "user" else "Assistant"
            # Truncate long messages
            content = msg.content[:500] + "..." if len(msg.content) > 500 else msg.content
            history_parts.append(f"{role}: {content}")
        
        return "\n".join(history_parts)
    
    def get_history_for_context(self) -> List[Tuple[str, str]]:
        """Get history as list of (role, content) tuples for prompt building."""
        return [(msg.role, msg.content) for msg in self.messages[-4:]]  # Last 2 turns
    
    def clear(self):
        """Clear all conversation history."""
        self.messages = []
    
    def __len__(self):
        return len(self.messages)


def get_llm(model_name: str = None, streaming: bool = True):
    """Get LLM instance."""
    model = model_name or settings.default_model
    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=settings.google_api_key,
        streaming=streaming
    )


def reformulate_query(query: str, memory: ConversationMemory) -> str:
    """
    Reformulate the user's query using conversation history to make it standalone.
    
    This handles cases like:
    - "Tell me more about that" → "Tell me more about [topic from history]"
    - "What about the second point?" → "What about [second point from previous answer]?"
    """
    history = memory.get_history_for_reformulation()
    if not history or len(memory) < 2:
        return query
    reference_indicators = [
        "it", "this", "that", "these", "those", "they", "them",
        "more", "else", "other", "another", "same", "previous",
        "above", "mentioned", "said", "explain", "elaborate"
    ]
    query_lower = query.lower()
    needs_reformulation = any(indicator in query_lower.split() for indicator in reference_indicators)
    if len(query.split()) <= 4:
        needs_reformulation = True
    
    if not needs_reformulation:
        return query
    try:
        llm = get_llm(streaming=False)
        
        reformulation_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a query reformulation assistant. Given a conversation history and a follow-up question, reformulate the question to be standalone and self-contained.

RULES:
1. If the question references something from the conversation, include that context
2. Keep the reformulated question concise
3. If the question is already standalone, return it unchanged
4. Only output the reformulated question, nothing else

CONVERSATION HISTORY:
{history}
"""),
            ("human", "Follow-up question: {query}\n\nReformulated standalone question:")
        ])
        
        chain = reformulation_prompt | llm
        result = chain.invoke({"history": history, "query": query})
        
        reformulated = result.content.strip()
        if len(reformulated) > len(query) * 3 or len(reformulated) > 500:
            return query
        
        return reformulated if reformulated else query
        
    except Exception as e:
        print(f"Query reformulation failed: {e}")
        return query

def build_prompt(query: str, chunks: List[Chunk], memory: Optional[ConversationMemory] = None) -> Tuple[str, str]:
    """Build RAG prompt with context, instructions, and optional conversation history."""
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        source_info = f"[Source {i}: {chunk.source_file}, Page {chunk.page_number}]"
        context_parts.append(f"{source_info}\n{chunk.content}")
    
    context = "\n\n---\n\n".join(context_parts)
    history_section = ""
    if memory and len(memory) > 0:
        history_for_context = memory.get_history_for_context()
        if history_for_context:
            history_parts = []
            for role, content in history_for_context:
                prefix = "User" if role == "user" else "Assistant"
                content_truncated = content[:300] + "..." if len(content) > 300 else content
                history_parts.append(f"{prefix}: {content_truncated}")
            history_section = "\n\nRECENT CONVERSATION:\n" + "\n".join(history_parts)
    
    system_prompt = """You are a document analysis assistant. Answer questions based ONLY on the provided context.

RULES:
1. Only use information from the context
2. Cite sources using [Source X] format
3. If information is not in context, say "I cannot find this information in the documents"
4. Be concise but complete
5. If there's conversation history, use it to understand follow-up questions    

CONTEXT:
{context}{history}
"""
    
    return system_prompt.format(context=context, history=history_section), query


def generate_response(query: str, chunks: List[Chunk], memory: Optional[ConversationMemory] = None) -> Generator[str, None, None]:
    """Generate streaming response from LLM."""
    llm = get_llm()
    system_prompt, user_query = build_prompt(query, chunks, memory)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{query}")
    ])
    
    chain = prompt | llm
    
    for chunk in chain.stream({"query": user_query}):
        yield chunk.content


def format_citations(chunks: List[Chunk]) -> str:
    """Format citation list for display."""
    if not chunks:
        return ""
    
    citations = []
    for i, chunk in enumerate(chunks, 1):
        citations.append(
            f"**[{i}]** {chunk.source_file} - Page {chunk.page_number}"
        )
    return "\n".join(citations)

