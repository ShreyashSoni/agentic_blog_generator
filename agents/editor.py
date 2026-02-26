"""
Editor Agent - Combines and refines blog sections.

This agent takes all generated sections, combines them into a draft,
and then polishes the content for clarity, flow, and consistency.
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


def editor_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Editor agent: Combines and refines all sections.
    
    This agent combines all individual sections into a cohesive draft,
    then edits for clarity, flow, consistency, and overall quality.
    
    Input:
        state["topic"]: Blog topic
        state["sections"]: Dictionary of section content
        state["outline"]: List of section titles
        state["plan"]: Blog plan
        
    Output:
        state["draft"]: Combined sections before editing
        state["edited"]: Final polished blog content
        
    Args:
        state: Current blog state
        
    Returns:
        Updated state with draft and edited content
    """
    topic = state["topic"]
    sections = state.get("sections", {})
    outline = state.get("outline", [])
    plan = state.get("plan", {})
    
    logger.info(f"Editor: Combining {len(sections)} sections")
    
    # Combine sections into draft
    draft = _combine_sections(sections, outline)
    state["draft"] = draft
    
    logger.info(f"Editor: Draft created with {len(draft.split())} words")
    
    # Initialize LLM
    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4"),
        temperature=0.3  # Lower temperature for more consistent editing
    )
    
    # Load editor prompt template
    prompt_template_str = load_prompt("editor.txt")
    if not prompt_template_str:
        prompt_template_str = _get_default_editor_prompt()
    
    # Create prompt
    prompt = PromptTemplate(
        template=prompt_template_str,
        input_variables=["topic", "draft", "target_audience", "tone"]
    )
    
    # Create chain
    chain = prompt | llm
    
    # Edit content
    try:
        logger.info("Editor: Polishing content...")
        response = chain.invoke({
            "topic": topic,
            "draft": draft,
            "target_audience": plan.get("target_audience", "general audience"),
            "tone": plan.get("tone", "professional")
        })
        
        edited_content = response.content if isinstance(response.content, str) else str(response.content)
        
        # Validate edited content
        word_count = len(edited_content.split())
        logger.info(f"Editor: Final content has {word_count} words")
        
        if word_count < len(draft.split()) * 0.7:
            logger.warning("Editor: Edited content significantly shorter than draft, using draft")
            edited_content = draft
        
    except Exception as e:
        logger.error(f"Editor: Editing failed: {e}")
        edited_content = draft
    
    # Update state
    state["edited"] = edited_content
    
    logger.info("Editor: Content editing completed")
    return state


def _combine_sections(
    sections: Dict[str, str],
    outline: list
) -> str:
    """
    Combine individual sections into a single draft.
    
    Args:
        sections: Dictionary mapping section titles to content
        outline: Ordered list of section titles
        
    Returns:
        Combined draft content
    """
    draft_parts = []
    
    for section_title in outline:
        if section_title in sections:
            # Add section with title as H2 header
            draft_parts.append(f"## {section_title}\n\n{sections[section_title]}")
        else:
            logger.warning(f"Editor: Missing section '{section_title}'")
    
    # Join with double newline for spacing
    return "\n\n".join(draft_parts)


def _get_default_editor_prompt() -> str:
    """
    Get default editor prompt template.
    
    Returns:
        Default prompt template string
    """
    return """You are an expert blog editor. Refine and polish the following blog draft.

Topic: {topic}
Target Audience: {target_audience}
Tone: {tone}

Draft:
{draft}

Your editing tasks:
1. Improve clarity and readability
2. Remove redundancy and repetition
3. Add smooth transitions between sections
4. Ensure consistent tone throughout
5. Fix any grammatical errors
6. Enhance engagement and flow
7. Keep all factual information intact
8. Maintain the markdown formatting
9. Ensure sections connect logically
10. Polish the introduction and conclusion

Guidelines:
- Preserve all section headings (##)
- Keep the overall structure and length similar
- Make minimal changes to well-written parts
- Focus on improving weak areas
- Ensure the blog reads as a cohesive whole

Output the final polished blog in markdown format."""


def add_blog_title(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Add a title to the edited blog content.
    
    This is a helper function that can be called after editing
    to prepend a title to the blog.
    
    Args:
        state: Current blog state
        
    Returns:
        Updated state with titled content
    """
    topic = state["topic"]
    edited = state.get("edited", "")
    
    # Add title as H1 if not already present
    if not edited.startswith("#"):
        titled_content = f"# {topic}\n\n{edited}"
        state["edited"] = titled_content
        logger.info("Editor: Added title to blog")
    
    return state