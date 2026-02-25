"""
Planner Agent - Analyzes topic and creates structured blog plan.

This agent takes a blog topic and generates a comprehensive plan including
target audience, blog length, section titles, and keywords.
"""

import json
import logging
import os
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

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
    a comprehensive blog plan.
    
    Input:
        state["topic"]: The blog topic
        
    Output:
        state["plan"]: Structured plan dictionary
        
    Args:
        state: Current blog state
        
    Returns:
        Updated state with plan
    """
    topic = state["topic"]
    logger.info(f"Planner: Analyzing topic - '{topic}'")
    
    # Initialize LLM (API key is read from OPENAI_API_KEY env var automatically)
    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4"),
        temperature=0.7
    )
    
    # Load prompt template
    try:
        prompt_template = load_prompt("planner.txt")
    except FileNotFoundError:
        logger.warning("Planner prompt template not found, using default")
        prompt_template = _get_default_prompt_template()
    
    # Create prompt
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["topic"]
    )
    
    # Create chain
    chain = prompt | llm
    
    # Generate plan
    try:
        response = chain.invoke({"topic": topic})
        
        # Parse JSON response
        content = response.content if isinstance(response.content, str) else str(response.content)
        plan = json.loads(content)
        
        # Validate plan structure
        required_keys = ["target_audience", "blog_length", "section_titles", "keywords"]
        for key in required_keys:
            if key not in plan:
                raise ValueError(f"Plan missing required key: {key}")
        
        # Add default tone if not present
        if "tone" not in plan:
            plan["tone"] = "professional"
        
        logger.info(f"Planner: Generated plan with {len(plan['section_titles'])} sections")
        logger.info(f"Planner: Target audience - {plan['target_audience']}")
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse plan JSON: {e}")
        # Fallback to default plan
        plan = _create_default_plan(topic)
    except Exception as e:
        logger.error(f"Planner error: {e}")
        # Fallback to default plan
        plan = _create_default_plan(topic)
    
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
        "tone": "professional"
    }