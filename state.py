"""
State definition for the blog generation workflow.

This module defines the BlogState TypedDict that represents the shared state
flowing through all agents in the LangGraph workflow.
"""

from typing import TypedDict, List, Dict, Any, Optional


class BlogState(TypedDict, total=False):
    """
    Shared state object that flows through the entire blog generation graph.
    Each agent reads from and writes to this state.
    
    Attributes:
        topic: The blog topic provided by the user (required)
        plan: Structured blog plan including audience, length, sections, keywords
        research_docs: List of summarized research documents
        outline: Ordered list of section titles
        sections: Dictionary mapping section titles to their content
        draft: Combined sections before editing
        edited: Final polished blog content
        seo_meta: SEO metadata including title, description, keywords, etc.
        _vector_store: Internal reference to vector store (not serialized)
    """
    
    # Core input (required)
    topic: str
    
    # Planning phase
    plan: Dict[str, Any]  # {target_audience, blog_length, section_titles, keywords, tone}
    
    # Research phase
    research_docs: List[str]  # List of summarized research documents
    
    # Outline phase
    outline: List[str]  # Ordered list of section titles
    
    # Writing phase
    sections: Dict[str, str]  # {section_title: section_content}
    
    # Editing phase
    draft: str  # Combined sections before editing
    edited: str  # Final polished blog content
    
    # SEO phase
    seo_meta: Dict[str, Any]  # {meta_title, meta_description, slug, keywords, faq, keyword_density}
    
    # Internal state (not part of final output)
    _vector_store: Optional[Any]  # Reference to vector store instance