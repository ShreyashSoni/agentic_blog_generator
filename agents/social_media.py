"""
Social Media Agent - Generates platform-specific content from blog posts.

This agent creates Twitter threads, LinkedIn post variations, image suggestions,
and hashtags for social media distribution of blog content.
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional
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
    try:
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        logger.error(f"Prompt file not found: {prompt_path}")
        raise


def generate_content_summary(content: str, max_length: int = 1500) -> str:
    """
    Generate a summary of blog content for prompt context.
    
    Args:
        content: Full blog content
        max_length: Maximum length of summary (default: 1500 chars)
        
    Returns:
        Content summary
    """
    # Get first N characters as summary
    if len(content) > max_length:
        # Try to break at a paragraph
        summary = content[:max_length]
        last_para = summary.rfind('\n\n')
        if last_para > max_length // 2:
            summary = summary[:last_para]
        summary += "\n\n[... content continues ...]"
        return summary
    return content


def generate_twitter_thread(
    title: str,
    description: str,
    keywords: List[str],
    content: str,
    target_audience: str = "general audience",
    llm_provider: str = "anthropic",
    model_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate Twitter thread from blog content.
    
    Args:
        title: Blog title
        description: Blog description
        keywords: Blog keywords
        content: Blog content
        target_audience: Target audience
        llm_provider: LLM provider
        model_name: Model name
        
    Returns:
        Dictionary containing thread data
    """
    logger.info("Generating Twitter thread...")
    
    llm = get_llm(provider=llm_provider, model_name=model_name, temperature=0.7)
    
    # Load prompt template
    prompt_template_str = load_prompt("social_media_twitter.txt")
    
    # Create prompt
    prompt = PromptTemplate(
        template=prompt_template_str,
        input_variables=["title", "description", "keywords", "target_audience", "content_summary"]
    )
    
    # Create chain
    chain = prompt | llm
    
    try:
        # Generate thread using streaming to prevent timeouts
        logger.info("Streaming Twitter thread generation...")
        response_text = ""
        for chunk in chain.stream({
            "title": title,
            "description": description,
            "keywords": ", ".join(keywords) if isinstance(keywords, list) else keywords,
            "target_audience": target_audience,
            "content_summary": generate_content_summary(content)
        }):
            if hasattr(chunk, 'content'):
                chunk_content = chunk.content
                if isinstance(chunk_content, str):
                    response_text += chunk_content
                elif isinstance(chunk_content, list):
                    response_text += ''.join(str(item) for item in chunk_content)
            else:
                response_text += str(chunk)
        
        # Parse JSON response
        cleaned_content = strip_markdown_wrapper(response_text)
        thread_data = json.loads(cleaned_content)
        
        logger.info(f"Generated Twitter thread with {len(thread_data.get('thread', []))} tweets")
        return thread_data
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse Twitter thread JSON: {e}")
        return _create_fallback_twitter_thread(title, description)
    except Exception as e:
        logger.error(f"Error generating Twitter thread: {e}")
        return _create_fallback_twitter_thread(title, description)


def generate_linkedin_posts(
    title: str,
    description: str,
    keywords: List[str],
    content: str,
    target_audience: str = "general audience",
    llm_provider: str = "anthropic",
    model_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate LinkedIn post variations from blog content.
    
    Args:
        title: Blog title
        description: Blog description
        keywords: Blog keywords
        content: Blog content
        target_audience: Target audience
        llm_provider: LLM provider
        model_name: Model name
        
    Returns:
        Dictionary containing post variations
    """
    logger.info("Generating LinkedIn post variations...")
    
    llm = get_llm(provider=llm_provider, model_name=model_name, temperature=0.7)
    
    # Load prompt template
    prompt_template_str = load_prompt("social_media_linkedin.txt")
    
    # Create prompt
    prompt = PromptTemplate(
        template=prompt_template_str,
        input_variables=["title", "description", "keywords", "target_audience", "content_summary"]
    )
    
    # Create chain
    chain = prompt | llm
    
    try:
        # Generate posts using streaming to prevent timeouts
        logger.info("Streaming LinkedIn posts generation...")
        response_text = ""
        for chunk in chain.stream({
            "title": title,
            "description": description,
            "keywords": ", ".join(keywords) if isinstance(keywords, list) else keywords,
            "target_audience": target_audience,
            "content_summary": generate_content_summary(content)
        }):
            if hasattr(chunk, 'content'):
                chunk_content = chunk.content
                if isinstance(chunk_content, str):
                    response_text += chunk_content
                elif isinstance(chunk_content, list):
                    response_text += ''.join(str(item) for item in chunk_content)
            else:
                response_text += str(chunk)
        
        # Parse JSON response
        cleaned_content = strip_markdown_wrapper(response_text)
        posts_data = json.loads(cleaned_content)
        
        logger.info(f"Generated {len(posts_data.get('variations', []))} LinkedIn post variations")
        return posts_data
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LinkedIn posts JSON: {e}")
        return _create_fallback_linkedin_posts(title, description)
    except Exception as e:
        logger.error(f"Error generating LinkedIn posts: {e}")
        return _create_fallback_linkedin_posts(title, description)


def generate_image_suggestions(
    title: str,
    description: str,
    keywords: List[str],
    content: str,
    llm_provider: str = "anthropic",
    model_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate image suggestions for social media posts.
    
    Args:
        title: Blog title
        description: Blog description
        keywords: Blog keywords
        content: Blog content
        llm_provider: LLM provider
        model_name: Model name
        
    Returns:
        Dictionary containing image suggestions
    """
    logger.info("Generating image suggestions...")
    
    llm = get_llm(provider=llm_provider, model_name=model_name, temperature=0.7)
    
    # Load prompt template
    prompt_template_str = load_prompt("social_media_images.txt")
    
    # Create prompt
    prompt = PromptTemplate(
        template=prompt_template_str,
        input_variables=["title", "description", "keywords", "content_summary"]
    )
    
    # Create chain
    chain = prompt | llm
    
    try:
        # Generate suggestions using streaming to prevent timeouts
        logger.info("Streaming image suggestions generation...")
        response_text = ""
        for chunk in chain.stream({
            "title": title,
            "description": description,
            "keywords": ", ".join(keywords) if isinstance(keywords, list) else keywords,
            "content_summary": generate_content_summary(content, 1500)
        }):
            if hasattr(chunk, 'content'):
                chunk_content = chunk.content
                if isinstance(chunk_content, str):
                    response_text += chunk_content
                elif isinstance(chunk_content, list):
                    response_text += ''.join(str(item) for item in chunk_content)
            else:
                response_text += str(chunk)
        
        # Parse JSON response
        cleaned_content = strip_markdown_wrapper(response_text)
        image_data = json.loads(cleaned_content)
        
        twitter_count = len(image_data.get('twitter_images', []))
        linkedin_count = len(image_data.get('linkedin_images', []))
        logger.info(f"Generated {twitter_count} Twitter and {linkedin_count} LinkedIn image suggestions")
        return image_data
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse image suggestions JSON: {e}")
        return _create_fallback_image_suggestions()
    except Exception as e:
        logger.error(f"Error generating image suggestions: {e}")
        return _create_fallback_image_suggestions()


def generate_hashtags(
    title: str,
    description: str,
    keywords: List[str],
    target_audience: str = "general audience",
    llm_provider: str = "anthropic",
    model_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate platform-specific hashtags.
    
    Args:
        title: Blog title
        description: Blog description
        keywords: Blog keywords
        target_audience: Target audience
        llm_provider: LLM provider
        model_name: Model name
        
    Returns:
        Dictionary containing hashtag data
    """
    logger.info("Generating hashtags...")
    
    llm = get_llm(provider=llm_provider, model_name=model_name, temperature=0.5)
    
    # Load prompt template
    prompt_template_str = load_prompt("social_media_hashtags.txt")
    
    # Create prompt
    prompt = PromptTemplate(
        template=prompt_template_str,
        input_variables=["title", "description", "keywords", "target_audience"]
    )
    
    # Create chain
    chain = prompt | llm
    
    try:
        # Generate hashtags using streaming to prevent timeouts
        logger.info("Streaming hashtags generation...")
        response_text = ""
        for chunk in chain.stream({
            "title": title,
            "description": description,
            "keywords": ", ".join(keywords) if isinstance(keywords, list) else keywords,
            "target_audience": target_audience
        }):
            if hasattr(chunk, 'content'):
                chunk_content = chunk.content
                if isinstance(chunk_content, str):
                    response_text += chunk_content
                elif isinstance(chunk_content, list):
                    response_text += ''.join(str(item) for item in chunk_content)
            else:
                response_text += str(chunk)
        
        # Parse JSON response
        cleaned_content = strip_markdown_wrapper(response_text)
        hashtag_data = json.loads(cleaned_content)
        
        total_hashtags = len(hashtag_data.get('combined_hashtag_list', []))
        logger.info(f"Generated {total_hashtags} unique hashtags")
        return hashtag_data
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse hashtags JSON: {e}")
        return _create_fallback_hashtags(keywords)
    except Exception as e:
        logger.error(f"Error generating hashtags: {e}")
        return _create_fallback_hashtags(keywords)


def generate_social_media_content(
    blog_data: Dict[str, Any],
    llm_provider: str = "anthropic",
    model_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate all social media content for a blog post.
    
    Args:
        blog_data: Dictionary containing blog metadata and content
        llm_provider: LLM provider
        model_name: Model name
        
    Returns:
        Complete social media content package
    """
    metadata = blog_data.get('metadata', {})
    content = blog_data.get('content', '')
    
    title = metadata.get('title', 'Blog Post')
    description = metadata.get('description', '')
    keywords = metadata.get('keywords', [])
    target_audience = metadata.get('target_audience', 'general audience')
    
    logger.info(f"Generating social media content for: {title}")
    
    # Generate all content types
    twitter_data = generate_twitter_thread(
        title, description, keywords, content, target_audience, llm_provider, model_name
    )
    
    linkedin_data = generate_linkedin_posts(
        title, description, keywords, content, target_audience, llm_provider, model_name
    )
    
    image_data = generate_image_suggestions(
        title, description, keywords, content, llm_provider, model_name
    )
    
    hashtag_data = generate_hashtags(
        title, description, keywords, target_audience, llm_provider, model_name
    )
    
    # Combine all data
    social_media_content = {
        'twitter': twitter_data,
        'linkedin': linkedin_data,
        'images': image_data,
        'hashtags': hashtag_data,
        'metadata': {
            'blog_title': title,
            'blog_slug': metadata.get('slug', 'blog-post'),
            'generated_at': _get_current_datetime()
        }
    }
    
    logger.info("Social media content generation complete")
    return social_media_content


# Fallback functions

def _create_fallback_twitter_thread(title: str, description: str) -> Dict[str, Any]:
    """Create fallback Twitter thread if generation fails."""
    logger.warning("Using fallback Twitter thread")
    return {
        "thread": [
            {
                "tweet_number": 1,
                "content": f"📝 New blog post: {title}\n\n{description[:150]}...\n\n#Blog #ContentCreation",
                "character_count": len(f"📝 New blog post: {title}\n\n{description[:150]}...\n\n#Blog #ContentCreation"),
                "image_suggestion": None
            }
        ],
        "metadata": {
            "total_tweets": 1,
            "avg_character_count": 200,
            "hashtags_used": ["#Blog", "#ContentCreation"],
            "engagement_hooks": ["New post announcement"]
        }
    }


def _create_fallback_linkedin_posts(title: str, description: str) -> Dict[str, Any]:
    """Create fallback LinkedIn posts if generation fails."""
    logger.warning("Using fallback LinkedIn posts")
    return {
        "variations": [
            {
                "variation_type": "General",
                "title": title,
                "content": f"**{title}**\n\n{description}\n\nRead the full article to learn more.\n\n#Blog #ContentMarketing",
                "character_count": len(f"**{title}**\n\n{description}\n\nRead the full article to learn more.\n\n#Blog #ContentMarketing"),
                "word_count": 50,
                "hashtags": ["#Blog", "#ContentMarketing"],
                "image_suggestion": "Blog header image",
                "key_points": ["Main topic covered"],
                "cta": "Read the full article"
            }
        ],
        "metadata": {
            "blog_link_placement": "Link in comments",
            "avg_character_count": 300,
            "engagement_strategies": ["Direct CTA"]
        }
    }


def _create_fallback_image_suggestions() -> Dict[str, Any]:
    """Create fallback image suggestions if generation fails."""
    logger.warning("Using fallback image suggestions")
    return {
        "twitter_images": [
            {
                "tweet_number": 1,
                "image_type": "Header Image",
                "description": "Blog header or featured image",
                "elements": ["Title text", "Key visual"],
                "color_scheme": "Brand colors",
                "text_overlay": None,
                "dimensions": "1200x675px",
                "suggested_tools": ["Canva"],
                "alt_text": "Blog post header image",
                "purpose": "Visual appeal"
            }
        ],
        "linkedin_images": [
            {
                "variation": "General",
                "image_type": "Header Image",
                "description": "Professional blog header",
                "elements": ["Title", "Subtitle"],
                "color_scheme": "Professional palette",
                "text_overlay": None,
                "dimensions": "1200x627px",
                "suggested_tools": ["Canva"],
                "alt_text": "Blog post header",
                "purpose": "Brand awareness"
            }
        ],
        "general_suggestions": {
            "brand_colors": "Use consistent brand colors",
            "style_guide": "Professional and clean",
            "stock_photo_keywords": ["technology", "business"],
            "icon_libraries": ["Font Awesome"],
            "design_tips": ["Keep it simple", "Use high contrast"]
        }
    }


def _create_fallback_hashtags(keywords: List[str]) -> Dict[str, Any]:
    """Create fallback hashtags if generation fails."""
    logger.warning("Using fallback hashtags")
    
    # Convert keywords to hashtags
    hashtags = [f"#{keyword.replace(' ', '').replace('-', '').capitalize()}" 
                for keyword in keywords[:5]] if keywords else ["#Blog", "#Content"]
    
    return {
        "twitter_hashtags": {
            "primary": hashtags[:2],
            "secondary": hashtags[2:4],
            "trending": ["#ContentCreation"],
            "recommended_per_tweet": 2,
            "distribution_strategy": "1 primary + 1 secondary per tweet"
        },
        "linkedin_hashtags": {
            "primary": hashtags[:2],
            "technical": hashtags[2:],
            "industry": ["#Technology", "#Innovation"],
            "audience": ["#Professionals"],
            "recommended_per_post": 5,
            "distribution_strategy": "Mix of primary and industry tags"
        },
        "combined_hashtag_list": [
            {
                "hashtag": tag,
                "platforms": ["twitter", "linkedin"],
                "category": "primary",
                "popularity": "medium",
                "relevance_score": 70,
                "usage_note": "Based on blog keywords"
            } for tag in hashtags
        ],
        "metadata": {
            "total_unique_hashtags": len(hashtags),
            "trend_analysis": "Based on blog keywords",
            "seasonal_tags": [],
            "avoid_tags": []
        }
    }


def _get_current_datetime() -> str:
    """Get current datetime in ISO format."""
    from datetime import datetime
    return datetime.now().isoformat()