"""
Writer Agent - Writes individual blog sections using RAG.

This agent generates content for individual sections by retrieving relevant
context from the vector store and using it to inform the LLM generation.
"""

import os
import logging
from typing import Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

logger = logging.getLogger(__name__)


def load_prompt(prompt_file: str) -> Optional[str]:
    """
    Load prompt template from file.
    
    Args:
        prompt_file: Name of the prompt file in prompts/ directory
        
    Returns:
        Prompt template content or None if file not found
    """
    prompt_path = os.path.join("prompts", prompt_file)
    try:
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        logger.warning(f"Prompt file not found: {prompt_path}")
        return None


def writer_node(state: Dict[str, Any], section_title: str) -> Dict[str, Any]:
    """
    Writer agent: Writes a single section using RAG.
    
    This agent retrieves relevant context from the vector store and generates
    section content informed by the research.
    
    Input:
        state["topic"]: Blog topic
        state["plan"]: Blog plan
        state["_vector_store"]: Vector store for RAG
        section_title: Title of section to write
        
    Output:
        state["sections"][section_title]: Generated section content
        
    Args:
        state: Current blog state
        section_title: The section to write
        
    Returns:
        Updated state with new section
    """
    topic = state["topic"]
    plan = state.get("plan", {})
    vector_store = state.get("_vector_store")
    
    logger.info(f"Writer: Writing section - '{section_title}'")
    
    # Retrieve relevant context via RAG
    context_docs = []
    if vector_store:
        try:
            query = f"{topic} {section_title}"
            context_docs = vector_store.similarity_search(query, k=3)
            logger.info(f"Writer: Retrieved {len(context_docs)} context documents")
        except Exception as e:
            logger.warning(f"Writer: Failed to retrieve context: {e}")
    
    # Initialize LLM
    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4"),
        temperature=0.7
    )
    
    # Load writer prompt template
    prompt_template_str = load_prompt("writer.txt")
    if not prompt_template_str:
        prompt_template_str = _get_default_writer_prompt()
    
    # Format context
    context_text = _format_context(context_docs)
    
    # Create prompt
    prompt = PromptTemplate(
        template=prompt_template_str,
        input_variables=["topic", "section_title", "target_audience", "tone", "context"]
    )
    
    # Create chain
    chain = prompt | llm
    
    # Generate section
    try:
        response = chain.invoke({
            "topic": topic,
            "section_title": section_title,
            "target_audience": plan.get("target_audience", "general audience"),
            "tone": plan.get("tone", "professional"),
            "context": context_text
        })
        
        section_content = response.content if isinstance(response.content, str) else str(response.content)
        
        # Validate section length
        word_count = len(section_content.split())
        logger.info(f"Writer: Generated section with {word_count} words")
        
        if word_count < 100:
            logger.warning(f"Writer: Section too short ({word_count} words), regenerating...")
            section_content = _generate_fallback_section(section_title, topic, context_text)
        
    except Exception as e:
        logger.error(f"Writer: Failed to generate section: {e}")
        section_content = _generate_fallback_section(section_title, topic, context_text)
    
    # Initialize sections dict if needed
    if "sections" not in state:
        state["sections"] = {}
    
    # Store section
    state["sections"][section_title] = section_content
    
    logger.info(f"Writer: Completed section - '{section_title}'")
    return state


def write_all_sections(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Write all sections from the outline.
    
    This function iterates through all sections in the outline and
    generates content for each one using the writer_node.
    
    Args:
        state: Current blog state
        
    Returns:
        Updated state with all sections written
    """
    outline = state.get("outline", [])
    
    logger.info(f"Writer: Writing {len(outline)} sections")
    
    for section_title in outline:
        state = writer_node(state, section_title)
    
    logger.info(f"Writer: Completed all {len(outline)} sections")
    return state


def _format_context(context_docs: list) -> str:
    """
    Format context documents for the prompt.
    
    Args:
        context_docs: List of document strings
        
    Returns:
        Formatted context string
    """
    if not context_docs:
        return "No additional context available. Use your general knowledge."
    
    formatted = []
    for idx, doc in enumerate(context_docs, 1):
        # Truncate long documents
        doc_text = doc[:500] if len(doc) > 500 else doc
        formatted.append(f"Source {idx}:\n{doc_text}")
    
    return "\n\n".join(formatted)


def _get_default_writer_prompt() -> str:
    """
    Get default writer prompt template.
    
    Returns:
        Default prompt template string
    """
    return """You are an expert blog writer. Write a compelling section for a blog post.

Topic: {topic}
Section Title: {section_title}
Target Audience: {target_audience}
Tone: {tone}

Research Context:
{context}

Instructions:
1. Write 300-500 words
2. Use the research context to ensure accuracy and depth
3. Match the specified tone ({tone})
4. Make it engaging and informative for the target audience
5. Use clear subheadings (###) if needed to organize content
6. Include specific examples or data points from the context when relevant
7. Write in markdown format
8. Start directly with the content (no need to repeat the section title)

Write ONLY the section content, no meta-commentary or explanations."""


def _generate_fallback_section(
    section_title: str,
    topic: str,
    context: str
) -> str:
    """
    Generate a basic fallback section when generation fails.
    
    Args:
        section_title: Section title
        topic: Blog topic
        context: Available context
        
    Returns:
        Fallback section content
    """
    logger.warning(f"Writer: Using fallback content for '{section_title}'")
    
    return f"""### {section_title}

This section covers important aspects of {topic} related to {section_title.lower()}.

The key points to understand about this topic include:

- **Fundamental concepts**: Understanding the basic principles is essential for grasping the broader implications
- **Practical applications**: These concepts have real-world applications across various domains
- **Best practices**: Following established guidelines ensures optimal results

{context[:200] if context else ""}

As we continue to explore {topic}, it's important to consider how these elements work together to form a comprehensive understanding of the subject matter.
"""