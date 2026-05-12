"""
Outline Agent - Generates structured blog outline.

This agent uses the blog plan and research summaries to create a detailed
outline with section titles that will guide the content generation.
"""

import logging
from typing import Dict, Any, List
from utils.llm_factory import get_llm

logger = logging.getLogger(__name__)


def outline_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Outline agent: Generates structured blog outline.
    
    This agent creates a detailed outline based on the plan and research,
    ensuring logical flow from introduction to conclusion. Enhanced to handle
    user feedback and regeneration.
    
    Input:
        state["topic"]: Blog topic
        state["plan"]: Blog plan with suggested sections
        state["research_docs"]: Research summaries
        state["user_feedback"]["outline"]: Optional user feedback for revision
        state["approval_attempt_count"]["outline"]: Current attempt number
        
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
    
    # Check for user feedback from previous rejection
    user_feedback = state.get("user_feedback", {}).get("outline", "")
    attempts = state.get("approval_attempt_count", {}).get("outline", 0)
    
    if user_feedback and attempts > 0:
        logger.info(f"Outline: Incorporating user feedback (attempt {attempts})")
        logger.info(f"Feedback: {user_feedback}")
    else:
        logger.info(f"Outline: Generating outline for - '{topic}'")

    llm = get_llm(provider=state.get("llm_provider"),
                  model_name=state.get("model_name"),
                  temperature=0.5)
    
    # Build prompt with optional feedback
    prompt = _build_outline_prompt(topic, plan, research_docs, user_feedback, attempts)
    
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
        
        # Validate that outline addresses user feedback (if any)
        if user_feedback:
            addresses_feedback = _validate_outline_against_feedback(outline, user_feedback)
            if addresses_feedback:
                logger.info(f"Outline: Outline appears to address user feedback")
            else:
                logger.warning(f"Outline: Outline may not fully address user feedback")
            
            # Log feedback incorporation context
            context = _get_outline_revision_context(attempts, user_feedback)
            logger.info(f"Outline: {context}")
        
        logger.info(f"Outline: Generated {len(outline)} sections")
        for idx, section in enumerate(outline, 1):
            logger.info(f"  {idx}. {section}")
        
        # Add metadata about revision process
        if attempts > 0:
            outline.append(f"_revision_metadata_{attempts}_{len(user_feedback)}")  # Hidden metadata
        
    except Exception as e:
        logger.error(f"Outline generation failed: {e}")
        logger.error(f"Raw response content: {str(content)[:200]}..." if 'content' in locals() else "No content received")
        outline = _create_fallback_outline(plan)
        if user_feedback:
            logger.warning(f"Outline generation failed with feedback: {user_feedback[:50]}...")
    
    # Update state
    state["outline"] = outline
    return state


def _build_outline_prompt(
    topic: str,
    plan: Dict[str, Any],
    research_docs: List[str],
    user_feedback: str = "",
    attempts: int = 0
) -> str:
    """
    Build the prompt for outline generation with optional user feedback.
    
    Args:
        topic: Blog topic
        plan: Blog plan
        research_docs: Research summaries
        user_feedback: Optional user feedback for revision
        attempts: Current attempt number
        
    Returns:
        Enhanced prompt string
    """
    target_audience = plan.get("target_audience", "general audience")
    suggested_sections = plan.get("section_titles", [])
    
    # Format research docs
    research_text = "\n".join([f"- {doc}" for doc in research_docs[:7]])
    
    # Format suggested sections
    suggestions_text = "\n".join([f"- {section}" for section in suggested_sections])
    
    # Base prompt
    prompt = f"""Create a detailed blog outline based on the topic, plan, and research.

Topic: {topic}
Target Audience: {target_audience}

Suggested Sections:
{suggestions_text}

Research Summary:
{research_text}"""

    # Add feedback instructions if this is a revision
    if user_feedback and attempts > 0:
        prompt += f"""

IMPORTANT REVISION INSTRUCTIONS:
This is revision attempt {attempts}. The user has provided the following feedback on the previous outline:

User Feedback: "{user_feedback}"

Please carefully revise the outline to address this feedback. Make sure to:
1. Incorporate the specific changes requested in the feedback
2. Maintain the overall structure and flow requirements
3. Keep the outline focused on the target audience
4. Ensure all sections are relevant and well-organized
5. Address each point in the feedback specifically

Previous feedback to incorporate: {user_feedback}
"""

    # Add standard generation instructions
    prompt += """

Generate an ordered list of 5-7 section titles that:
1. Start with "Introduction"
2. Flow logically from basic to advanced concepts
3. End with "Conclusion"
4. Cover all key aspects of the topic informed by the research
5. Are specific and informative (not generic)
6. Match the target audience level"""

    # Add feedback-specific requirements if revision
    if user_feedback:
        prompt += "\n7. Address all points mentioned in the user feedback above"

    prompt += """

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


def _validate_outline_against_feedback(outline: List[str], user_feedback: str) -> bool:
    """
    Basic validation that outline addresses user feedback.
    
    Args:
        outline: Generated outline list
        user_feedback: User feedback to validate against
        
    Returns:
        True if outline seems to address feedback, False otherwise
    """
    if not user_feedback:
        return True
    
    feedback_lower = user_feedback.lower()
    outline_str = " ".join(outline).lower()
    
    # Basic keyword matching to check if feedback concepts appear in outline
    feedback_keywords = [
        word.strip('.,!?') for word in feedback_lower.split()
        if len(word) > 3 and word.isalpha()
    ]
    
    matches = sum(1 for keyword in feedback_keywords if keyword in outline_str)
    match_ratio = matches / len(feedback_keywords) if feedback_keywords else 1.0
    
    # Consider addressing feedback if at least 25% of keywords appear
    return match_ratio >= 0.25


def _get_outline_revision_context(attempts: int, user_feedback: str) -> str:
    """
    Get context string for outline revision attempts.
    
    Args:
        attempts: Current attempt number
        user_feedback: User feedback
        
    Returns:
        Context string for logging and prompts
    """
    if attempts <= 1:
        return "Initial outline generation"
    else:
        return f"Revision attempt {attempts} based on feedback: {user_feedback[:100]}..."


def _enhance_outline_with_feedback(outline: List[str], user_feedback: str) -> List[str]:
    """
    Enhance outline based on user feedback patterns.
    
    Args:
        outline: Original outline
        user_feedback: User feedback
        
    Returns:
        Enhanced outline list
    """
    if not user_feedback:
        return outline
    
    enhanced_outline = outline.copy()
    feedback_lower = user_feedback.lower()
    
    # Common feedback patterns and responses
    if "more detail" in feedback_lower or "more specific" in feedback_lower:
        # Try to make section titles more specific
        for i, section in enumerate(enhanced_outline):
            if not section.lower().startswith(('intro', 'concl')) and len(section.split()) < 4:
                enhanced_outline[i] = f"Detailed {section}"
    
    elif "remove" in feedback_lower or "delete" in feedback_lower:
        # Filter out sections that might match removal requests
        words_to_remove = []
        for word in feedback_lower.split():
            if len(word) > 4 and word.isalpha():
                words_to_remove.append(word)
        
        if words_to_remove:
            enhanced_outline = [
                section for section in enhanced_outline
                if not any(remove_word in section.lower() for remove_word in words_to_remove[:3])
            ]
    
    elif "add" in feedback_lower or "include" in feedback_lower:
        # Try to identify what to add
        add_keywords = []
        words = feedback_lower.split()
        for i, word in enumerate(words):
            if word in ['add', 'include'] and i + 1 < len(words):
                add_keywords.append(words[i + 1])
        
        # Insert new sections before conclusion
        if add_keywords and len(enhanced_outline) > 1:
            conclusion_idx = len(enhanced_outline) - 1
            for keyword in add_keywords[:2]:  # Limit to 2 additions
                new_section = f"Understanding {keyword.capitalize()}"
                enhanced_outline.insert(conclusion_idx, new_section)
                conclusion_idx += 1
    
    # Ensure structure is maintained
    enhanced_outline = _ensure_intro_conclusion(enhanced_outline)
    
    return enhanced_outline


def _clean_outline_metadata(outline: List[str]) -> List[str]:
    """
    Remove metadata entries from outline for final output.
    
    Args:
        outline: Outline with potential metadata
        
    Returns:
        Clean outline without metadata
    """
    return [section for section in outline if not section.startswith("_revision_metadata_")]