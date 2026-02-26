"""
SEO Agent - Generates SEO metadata and analysis.

This agent creates comprehensive SEO metadata including title, description,
slug, keywords, FAQ, and keyword density analysis.
"""

import os
import re
import json
import logging
from typing import Dict, Any, List
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)


def seo_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    SEO agent: Generates SEO metadata and analysis.
    
    This agent analyzes the final blog content and generates comprehensive
    SEO metadata to optimize search engine visibility.
    
    Input:
        state["topic"]: Blog topic
        state["edited"]: Final edited content
        state["plan"]: Blog plan with keywords
        
    Output:
        state["seo_meta"]: SEO metadata dictionary
        
    Args:
        state: Current blog state
        
    Returns:
        Updated state with SEO metadata
    """
    topic = state["topic"]
    edited_content = state.get("edited", "")
    plan = state.get("plan", {})
    keywords = plan.get("keywords", [])
    
    logger.info(f"SEO: Generating metadata for - '{topic}'")
    
    # Initialize LLM
    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4"),
        temperature=0.5
    )
    
    # Build prompt for SEO metadata
    prompt = _build_seo_prompt(topic, edited_content, keywords)
    
    try:
        # Generate SEO metadata
        response = llm.invoke(prompt)
        content = response.content if isinstance(response.content, str) else str(response.content)
        
        # Parse JSON response
        seo_meta = json.loads(content)
        
        # Validate SEO metadata
        seo_meta = _validate_seo_metadata(seo_meta, topic, keywords)
        
    except (json.JSONDecodeError, Exception) as e:
        logger.error(f"SEO: Failed to generate metadata: {e}")
        seo_meta = _create_fallback_seo_meta(topic, keywords)
    
    # Add keyword density analysis
    seo_meta["keyword_density"] = calculate_keyword_density(edited_content, keywords)
    
    # Log SEO metadata
    logger.info(f"SEO: Meta title - '{seo_meta.get('meta_title', 'N/A')}'")
    logger.info(f"SEO: Slug - '{seo_meta.get('slug', 'N/A')}'")
    logger.info(f"SEO: FAQ items - {len(seo_meta.get('faq', []))}")
    
    # Update state
    state["seo_meta"] = seo_meta
    return state


def _build_seo_prompt(topic: str, content: str, keywords: List[str]) -> str:
    """
    Build the prompt for SEO metadata generation.
    
    Args:
        topic: Blog topic
        content: Blog content
        keywords: Target keywords
        
    Returns:
        Prompt string
    """
    # Get first 1000 characters for context
    content_preview = content[:1000] + "..." if len(content) > 1000 else content
    
    prompt = f"""Generate comprehensive SEO metadata for this blog post.

Topic: {topic}
Target Keywords: {', '.join(keywords)}

Blog Content (preview):
{content_preview}

Generate JSON with the following structure:
{{
    "meta_title": "50-60 character SEO title",
    "meta_description": "150-160 character description",
    "slug": "url-friendly-slug",
    "keywords": ["primary", "keywords", "list"],
    "faq": [
        {{"question": "Q1", "answer": "A1"}},
        {{"question": "Q2", "answer": "A2"}},
        {{"question": "Q3", "answer": "A3"}}
    ]
}}

Requirements:

1. META TITLE (50-60 characters):
   - Include main keyword naturally
   - Be compelling and click-worthy
   - Accurately represent content
   - Stay within character limit

2. META DESCRIPTION (150-160 characters):
   - Include primary keyword
   - Summarize key value proposition
   - Include call-to-action if appropriate
   - Stay within character limit

3. SLUG:
   - Use lowercase letters
   - Replace spaces with hyphens
   - Remove special characters
   - Keep it concise (3-6 words max)
   - Include main keyword

4. KEYWORDS (5-10 items):
   - Mix of target keywords and variations
   - Include long-tail keywords
   - Prioritize most relevant terms

5. FAQ (3-5 items):
   - Common questions related to the topic
   - Provide clear, concise answers (2-3 sentences)
   - Use question format for questions
   - Make answers informative

Output ONLY valid JSON, no additional text."""

    return prompt


def _validate_seo_metadata(
    seo_meta: Dict[str, Any],
    topic: str,
    keywords: List[str]
) -> Dict[str, Any]:
    """
    Validate and fix SEO metadata.
    
    Args:
        seo_meta: Generated SEO metadata
        topic: Blog topic
        keywords: Target keywords
        
    Returns:
        Validated SEO metadata
    """
    # Validate meta_title
    if "meta_title" not in seo_meta or len(seo_meta["meta_title"]) > 70:
        seo_meta["meta_title"] = topic[:60]
        logger.warning("SEO: meta_title missing or too long, using topic")
    
    # Validate meta_description
    if "meta_description" not in seo_meta or len(seo_meta["meta_description"]) > 170:
        seo_meta["meta_description"] = f"Learn about {topic} in this comprehensive guide."[:160]
        logger.warning("SEO: meta_description missing or too long, using default")
    
    # Validate slug
    if "slug" not in seo_meta:
        seo_meta["slug"] = _slugify(topic)
        logger.warning("SEO: slug missing, generating from topic")
    
    # Validate keywords
    if "keywords" not in seo_meta or not isinstance(seo_meta["keywords"], list):
        seo_meta["keywords"] = keywords[:5]
        logger.warning("SEO: keywords missing or invalid, using plan keywords")
    
    # Validate FAQ
    if "faq" not in seo_meta or not isinstance(seo_meta["faq"], list):
        seo_meta["faq"] = []
        logger.warning("SEO: FAQ missing or invalid, using empty list")
    
    return seo_meta


def _create_fallback_seo_meta(topic: str, keywords: List[str]) -> Dict[str, Any]:
    """
    Create fallback SEO metadata.
    
    Args:
        topic: Blog topic
        keywords: Target keywords
        
    Returns:
        Fallback SEO metadata
    """
    logger.warning("SEO: Using fallback metadata")
    
    return {
        "meta_title": topic[:60],
        "meta_description": f"Comprehensive guide to {topic}. Learn key concepts, best practices, and practical applications."[:160],
        "slug": _slugify(topic),
        "keywords": keywords[:5] if keywords else [topic.lower()],
        "faq": []
    }


def _slugify(text: str) -> str:
    """
    Convert text to URL-friendly slug.
    
    Args:
        text: Text to slugify
        
    Returns:
        URL-friendly slug
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters
    text = re.sub(r'[^\w\s-]', '', text)
    
    # Replace spaces and multiple hyphens with single hyphen
    text = re.sub(r'[-\s]+', '-', text)
    
    # Limit length
    text = text[:60]
    
    # Remove leading/trailing hyphens
    text = text.strip('-')
    
    return text if text else "blog-post"


def calculate_keyword_density(content: str, keywords: List[str]) -> Dict[str, float]:
    """
    Calculate keyword density for primary keywords.
    
    Args:
        content: Blog content
        keywords: List of keywords to analyze
        
    Returns:
        Dictionary mapping keywords to density percentages
    """
    # Convert content to lowercase for case-insensitive matching
    content_lower = content.lower()
    
    # Count total words
    words = re.findall(r'\b\w+\b', content_lower)
    total_words = len(words)
    
    if total_words == 0:
        return {}
    
    # Calculate density for each keyword
    density = {}
    for keyword in keywords:
        keyword_lower = keyword.lower()
        
        # Count occurrences (handle multi-word keywords)
        if ' ' in keyword_lower:
            # For phrases, count exact matches
            count = len(re.findall(r'\b' + re.escape(keyword_lower) + r'\b', content_lower))
        else:
            # For single words, count word occurrences
            count = content_lower.split().count(keyword_lower)
        
        # Calculate percentage
        percentage = round((count / total_words) * 100, 2)
        density[keyword] = percentage
    
    logger.info(f"SEO: Calculated keyword density for {len(keywords)} keywords")
    return density