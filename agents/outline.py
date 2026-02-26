"""
Outline Agent - Generates structured blog outline.

This agent uses the blog plan and research summaries to create a detailed
outline with section titles that will guide the content generation.
"""

import os
import logging
from typing import Dict, Any, List
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)


def outline_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Outline agent: Generates structured blog outline.
    
    This agent creates a detailed outline based on the plan and research,
    ensuring logical flow from introduction to conclusion.
    
    Input:
        state["topic"]: Blog topic
        state["plan"]: Blog plan with suggested sections
        state["research_docs"]: Research summaries
        
    Output:
        state["outline"]: List of section titles
        
    Args:
        state: Current blog state
        
    Returns:
        Updated state with outline
    """
    topic = state["topic"]
    plan = state.get("plan", {})
    research_docs = state.get("research_docs", [])
    
    logger.info(f"Outline: Generating outline for - '{topic}'")
    
    # Initialize LLM
    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4"),
        temperature=0.5
    )
    
    # Build prompt
    prompt = _build_outline_prompt(topic, plan, research_docs)
    
    try:
        # Generate outline
        response = llm.invoke(prompt)
        content = response.content if isinstance(response.content, str) else str(response.content)
        
        # Parse outline from response
        outline = _parse_outline(content)
        
        # Validate outline
        if not outline or len(outline) < 3:
            logger.warning("Outline: Generated outline too short, using fallback")
            outline = _create_fallback_outline(plan)
        
        # Ensure starts with Introduction and ends with Conclusion
        outline = _ensure_intro_conclusion(outline)
        
        logger.info(f"Outline: Generated {len(outline)} sections")
        for idx, section in enumerate(outline, 1):
            logger.info(f"  {idx}. {section}")
        
    except Exception as e:
        logger.error(f"Outline generation failed: {e}")
        outline = _create_fallback_outline(plan)
    
    # Update state
    state["outline"] = outline
    return state


def _build_outline_prompt(
    topic: str,
    plan: Dict[str, Any],
    research_docs: List[str]
) -> str:
    """
    Build the prompt for outline generation.
    
    Args:
        topic: Blog topic
        plan: Blog plan
        research_docs: Research summaries
        
    Returns:
        Prompt string
    """
    target_audience = plan.get("target_audience", "general audience")
    suggested_sections = plan.get("section_titles", [])
    
    # Format research docs
    research_text = "\n".join([f"- {doc}" for doc in research_docs[:5]])
    
    # Format suggested sections
    suggestions_text = "\n".join([f"- {section}" for section in suggested_sections])
    
    prompt = f"""Create a detailed blog outline based on the topic, plan, and research.

Topic: {topic}
Target Audience: {target_audience}

Suggested Sections:
{suggestions_text}

Research Summary:
{research_text}

Generate an ordered list of 5-7 section titles that:
1. Start with "Introduction"
2. Flow logically from basic to advanced concepts
3. End with "Conclusion"
4. Cover all key aspects of the topic informed by the research
5. Are specific and informative (not generic)
6. Match the target audience level

Output format (one section title per line, numbered):
1. Introduction
2. [Section Title]
3. [Section Title]
...
N. Conclusion

Provide ONLY the numbered list, no additional text."""

    return prompt


def _parse_outline(content: str) -> List[str]:
    """
    Parse outline from LLM response.
    
    Args:
        content: LLM response content
        
    Returns:
        List of section titles
    """
    outline = []
    lines = content.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Remove numbering (handles various formats: "1.", "1)", "1 -", etc.)
        if line[0].isdigit() or line.startswith('-') or line.startswith('•'):
            # Find the first letter or after the number/bullet
            parts = line.split('.', 1)
            if len(parts) > 1:
                title = parts[1].strip()
            else:
                parts = line.split(')', 1)
                if len(parts) > 1:
                    title = parts[1].strip()
                else:
                    # Remove leading special chars
                    title = line.lstrip('0123456789-•. ').strip()
            
            if title:
                outline.append(title)
    
    return outline


def _ensure_intro_conclusion(outline: List[str]) -> List[str]:
    """
    Ensure outline starts with Introduction and ends with Conclusion.
    
    Args:
        outline: List of section titles
        
    Returns:
        Modified outline with proper structure
    """
    if not outline:
        return ["Introduction", "Conclusion"]
    
    # Check if first section is Introduction
    if not outline[0].lower().startswith('intro'):
        outline.insert(0, "Introduction")
    
    # Check if last section is Conclusion
    if not outline[-1].lower().startswith('concl'):
        outline.append("Conclusion")
    
    return outline


def _create_fallback_outline(plan: Dict[str, Any]) -> List[str]:
    """
    Create fallback outline from plan or use default.
    
    Args:
        plan: Blog plan
        
    Returns:
        Fallback outline
    """
    suggested_sections = plan.get("section_titles", [])
    
    if suggested_sections and len(suggested_sections) >= 3:
        outline = suggested_sections.copy()
    else:
        outline = [
            "Introduction",
            "Background and Context",
            "Key Concepts",
            "Practical Applications",
            "Best Practices",
            "Conclusion"
        ]
    
    # Ensure proper structure
    outline = _ensure_intro_conclusion(outline)
    
    logger.info("Outline: Using fallback outline")
    return outline