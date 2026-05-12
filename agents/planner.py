"""
Planner Agent - Analyzes topic and creates structured blog plan.

This agent takes a blog topic and generates a comprehensive plan including
target audience, blog length, section titles, and keywords.
"""

import json
import logging
import os
from typing import Dict, Any
from langchain_core.prompts import PromptTemplate
from utils.data_utils import strip_markdown_wrapper
from utils.llm_factory import get_llm

logger = logging.getLogger(__name__)


def load_prompt(prompt_file: str) -> str:
    """
    Load prompt template from file.
    
    Args:
        prompt_file: Name of the prompt file in prompts/ directory
        
    Returns:
        Prompt template content
    """
    prompt_path = os.path.join("prompts", prompt_file)
    with open(prompt_path, 'r', encoding='utf-8') as f:
        return f.read()


def planner_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Planner agent: Analyzes topic and creates structured plan.
    
    This is a LangGraph node that reads the topic from state and generates
    a comprehensive blog plan. Enhanced to handle user feedback and regeneration.
    
    Input:
        state["topic"]: The blog topic
        state["length"]: Blog complexity ('simple' or 'complex')
        state["user_feedback"]["plan"]: Optional user feedback for revision
        state["approval_attempt_count"]["plan"]: Current attempt number
        
    Output:
        state["plan"]: Structured plan dictionary
        
    Args:
        state: Current blog state
        
    Returns:
        Updated state with plan
    """
    topic = state["topic"]
    length = state.get("length", "complex")
    
    # Check for user feedback from previous rejection
    user_feedback = state.get("user_feedback", {}).get("plan", "")
    attempts = state.get("approval_attempt_count", {}).get("plan", 0)
    
    if user_feedback and attempts > 0:
        logger.info(f"Planner: Incorporating user feedback (attempt {attempts})")
        logger.info(f"Feedback: {user_feedback}")
    else:
        logger.info(f"Planner: Analyzing topic - '{topic}' (length: {length})")

    llm = get_llm(provider=state.get("llm_provider"),
                  model_name=state.get("model_name"),
                  temperature=0.7)
    
    # Load and modify prompt template based on feedback
    try:
        base_prompt_template = load_prompt("planner.txt")
        prompt_template = _enhance_prompt_with_feedback(base_prompt_template, user_feedback, attempts)
    except FileNotFoundError:
        logger.warning("Planner prompt template not found, using default")
        base_prompt_template = _get_default_prompt_template()
        prompt_template = _enhance_prompt_with_feedback(base_prompt_template, user_feedback, attempts)
    
    # Determine input variables based on whether we have feedback
    if user_feedback:
        input_variables = ["topic", "length", "user_feedback"]
        prompt_values = {"topic": topic, "length": length, "user_feedback": user_feedback}
    else:
        input_variables = ["topic", "length"]
        prompt_values = {"topic": topic, "length": length}
    
    # Create prompt
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=input_variables
    )
    
    # Create chain
    chain = prompt | llm
    
    # Generate plan
    try:
        response = chain.invoke(prompt_values)
        
        # Parse JSON response
        content = response.content if isinstance(response.content, str) else str(response.content)
        # Strip markdown wrappers if present
        cleaned_content = strip_markdown_wrapper(content)
        plan = json.loads(cleaned_content)
        
        # Validate plan structure
        required_keys = ["target_audience", "blog_length", "section_titles", "keywords"]
        for key in required_keys:
            if key not in plan:
                raise ValueError(f"Plan missing required key: {key}")
        
        # Add default tone if not present
        if "tone" not in plan:
            plan["tone"] = "technical"
        
        # Validate that plan addresses user feedback (if any)
        if user_feedback:
            addresses_feedback = _validate_plan_against_feedback(plan, user_feedback)
            if addresses_feedback:
                logger.info(f"Planner: Plan appears to address user feedback")
            else:
                logger.warning(f"Planner: Plan may not fully address user feedback")
            
            # Log feedback incorporation context
            context = _get_revision_context(attempts, user_feedback)
            logger.info(f"Planner: {context}")
        
        logger.info(f"Planner: Generated plan with {len(plan['section_titles'])} sections")
        logger.info(f"Planner: Target audience - {plan['target_audience']}")
        
        # Add metadata about revision process
        if attempts > 0:
            plan["_revision_metadata"] = {
                "attempt_number": attempts,
                "had_feedback": bool(user_feedback),
                "feedback_summary": user_feedback[:100] + "..." if len(user_feedback) > 100 else user_feedback
            }
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse plan JSON: {e}")
        logger.error(f"Raw response content: {content[:200]}...")
        # Fallback to default plan with feedback context
        plan = _create_default_plan(topic)
        if user_feedback:
            plan["_fallback_reason"] = f"JSON parse error with feedback: {user_feedback[:50]}..."
    except Exception as e:
        logger.error(f"Planner error: {e}")
        # Fallback to default plan with feedback context
        plan = _create_default_plan(topic)
        if user_feedback:
            plan["_fallback_reason"] = f"Generation error with feedback: {user_feedback[:50]}..."
    
    # Update state
    state["plan"] = plan
    return state


def _get_default_prompt_template() -> str:
    """
    Get default planner prompt template.
    
    Returns:
        Default prompt template string
    """
    return """You are an expert blog planning assistant. Analyze the given topic and create a comprehensive blog plan.

Topic: {topic}

Generate a detailed plan in JSON format with the following structure:
{{
    "target_audience": "Who is this blog for?",
    "blog_length": 1500,
    "section_titles": ["Section 1", "Section 2", ...],
    "keywords": ["keyword1", "keyword2", ...],
    "tone": "professional/casual/technical"
}}

Requirements:
- Target audience should be specific (e.g., "Software developers", "Marketing professionals")
- Blog length in words (800-2000 range)
- 4-7 section titles that flow logically
- 5-10 relevant keywords
- Appropriate tone for the topic

Output ONLY valid JSON, no additional text."""


def _create_default_plan(topic: str) -> Dict[str, Any]:
    """
    Create a default fallback plan.
    
    Args:
        topic: Blog topic
        
    Returns:
        Default plan dictionary
    """
    logger.warning("Using default fallback plan")
    return {
        "target_audience": "General audience",
        "blog_length": 1200,
        "section_titles": [
            "Introduction",
            "Overview",
            "Key Concepts",
            "Practical Applications",
            "Conclusion"
        ],
        "keywords": [word.lower() for word in topic.split()[:5]],
        "tone": "technical"
    }


def _enhance_prompt_with_feedback(base_template: str, user_feedback: str, attempts: int) -> str:
    """
    Enhance prompt template with user feedback for plan revision.
    
    Args:
        base_template: Base prompt template
        user_feedback: User feedback from previous rejection
        attempts: Current attempt number
        
    Returns:
        Enhanced prompt template with feedback instructions
    """
    if not user_feedback or attempts <= 0:
        return base_template
    
    # Add feedback section to the prompt
    feedback_instruction = f"""

IMPORTANT REVISION INSTRUCTIONS:
This is revision attempt {attempts}. The user has provided the following feedback on the previous plan:

User Feedback: "{user_feedback}"

Please carefully incorporate this feedback and generate a revised plan that addresses the user's concerns and requirements. Make sure to:

1. Address each point in the feedback specifically
2. Maintain all the original requirements (JSON format, required fields, etc.)
3. Improve upon the previous plan based on the feedback
4. Ensure the revision is substantial and meaningful

Previous feedback must be incorporated: {user_feedback}
"""
    
    # Check if template has user_feedback variable, if not add it
    if "{user_feedback}" not in base_template:
        # Insert feedback instruction before the final instruction line
        if "Output ONLY valid JSON" in base_template:
            enhanced_template = base_template.replace(
                "Output ONLY valid JSON, no additional text.",
                f"{feedback_instruction}\n\nOutput ONLY valid JSON, no additional text."
            )
        else:
            enhanced_template = base_template + feedback_instruction
    else:
        enhanced_template = base_template
    
    return enhanced_template


def _get_revision_context(attempts: int, user_feedback: str) -> str:
    """
    Get context string for revision attempts.
    
    Args:
        attempts: Current attempt number
        user_feedback: User feedback
        
    Returns:
        Context string for logging and prompts
    """
    if attempts <= 1:
        return "Initial plan generation"
    else:
        return f"Revision attempt {attempts} based on feedback: {user_feedback[:100]}..."


def _validate_plan_against_feedback(plan: Dict[str, Any], user_feedback: str) -> bool:
    """
    Basic validation that plan addresses user feedback.
    
    Args:
        plan: Generated plan dictionary
        user_feedback: User feedback to validate against
        
    Returns:
        True if plan seems to address feedback, False otherwise
    """
    if not user_feedback:
        return True
    
    feedback_lower = user_feedback.lower()
    plan_str = json.dumps(plan, default=str).lower()
    
    # Basic keyword matching to check if feedback concepts appear in plan
    feedback_keywords = [
        word.strip('.,!?') for word in feedback_lower.split()
        if len(word) > 3 and word.isalpha()
    ]
    
    matches = sum(1 for keyword in feedback_keywords if keyword in plan_str)
    match_ratio = matches / len(feedback_keywords) if feedback_keywords else 1.0
    
    # Consider addressing feedback if at least 30% of keywords appear
    return match_ratio >= 0.3