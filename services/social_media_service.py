"""
Social Media Service - Service layer for generating and formatting social media content.

This module provides reusable functions for generating social media content
from blog posts and formatting the output.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

from services.markdown_parser import parse_blog_file
from agents.social_media import generate_social_media_content

logger = logging.getLogger(__name__)


def generate_from_blog_file(
    blog_path: Optional[str] = None,
    output_dir: str = "outputs/social_media",
    llm_provider: str = "anthropic",
    model_name: Optional[str] = None,
    blog_data: Optional[Dict[str, Any]] = None
) -> str:
    """
    Generate social media content from a blog markdown file or pre-parsed blog data.
    
    Args:
        blog_path: Path to blog markdown file (optional if blog_data provided)
        output_dir: Directory to save social media content
        llm_provider: LLM provider to use
        model_name: Specific model name
        blog_data: Pre-parsed blog data (optional, will parse blog_path if not provided)
        
    Returns:
        Path to generated social media file
        
    Raises:
        FileNotFoundError: If blog file doesn't exist
        ValueError: If blog file is invalid or neither blog_path nor blog_data provided
    """
    # Validate input
    if blog_data is None and blog_path is None:
        raise ValueError("Either blog_path or blog_data must be provided")
    
    # Parse blog file if blog_data not provided
    if blog_data is None:
        if blog_path is None:
            raise ValueError("blog_path cannot be None when blog_data is not provided")
        logger.info(f"Generating social media content from: {blog_path}")
        try:
            blog_data = parse_blog_file(blog_path)
        except Exception as e:
            logger.error(f"Failed to parse blog file: {e}")
            raise
    else:
        logger.info("Generating social media content from pre-parsed blog data")
    
    # Generate social media content
    social_content = generate_social_media_content(
        blog_data=blog_data,
        llm_provider=llm_provider,
        model_name=model_name
    )
    
    # Format and save output
    output_path = save_social_media_file(
        social_content=social_content,
        blog_data=blog_data,
        output_dir=output_dir
    )
    
    logger.info(f"Social media content saved to: {output_path}")
    return output_path


def save_social_media_file(
    social_content: Dict[str, Any],
    blog_data: Dict[str, Any],
    output_dir: str = "outputs/social_media"
) -> str:
    """
    Save social media content to markdown file.
    
    Args:
        social_content: Generated social media content
        blog_data: Original blog data
        output_dir: Output directory
        
    Returns:
        Path to saved file
    """
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Get blog slug for filename
    metadata = blog_data.get('metadata', {})
    slug = metadata.get('slug', 'blog-post')
    
    # Create filename
    filename = f"{slug}-social.md"
    output_path = os.path.join(output_dir, filename)
    
    # Format content as markdown
    markdown_content = format_social_media_markdown(social_content, metadata)
    
    # Save file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(markdown_content)
    
    return output_path


def format_social_media_markdown(
    social_content: Dict[str, Any],
    blog_metadata: Dict[str, Any]
) -> str:
    """
    Format social media content as markdown file.
    
    Args:
        social_content: Generated social media content
        blog_metadata: Blog metadata
        
    Returns:
        Formatted markdown string
    """
    # Extract data
    twitter_data = social_content.get('twitter', {})
    linkedin_data = social_content.get('linkedin', {})
    images_data = social_content.get('images', {})
    hashtags_data = social_content.get('hashtags', {})
    gen_metadata = social_content.get('metadata', {})
    
    # Build markdown
    parts = []
    
    # Frontmatter
    parts.append("---")
    parts.append(f"blog_title: \"{blog_metadata.get('title', 'Blog Post')}\"")
    parts.append(f"blog_slug: \"{blog_metadata.get('slug', 'blog-post')}\"")
    parts.append(f"generated_at: \"{gen_metadata.get('generated_at', datetime.now().isoformat())}\"")
    parts.append("platforms:")
    parts.append("  - twitter")
    parts.append("  - linkedin")
    parts.append("---")
    parts.append("")
    
    # Title
    parts.append(f"# Social Media Content - {blog_metadata.get('title', 'Blog Post')}")
    parts.append("")
    
    # Twitter Thread
    parts.append("## 📱 Twitter Thread")
    parts.append("")
    parts.extend(_format_twitter_thread(twitter_data))
    parts.append("")
    
    # LinkedIn Posts
    parts.append("## 💼 LinkedIn Posts (3 Variations)")
    parts.append("")
    parts.extend(_format_linkedin_posts(linkedin_data))
    parts.append("")
    
    # Image Suggestions
    parts.append("## 🖼️ Image Suggestions")
    parts.append("")
    parts.extend(_format_image_suggestions(images_data))
    parts.append("")
    
    # Hashtags Summary
    parts.append("## 🔖 Hashtags Summary")
    parts.append("")
    parts.extend(_format_hashtags_summary(hashtags_data))
    parts.append("")
    
    return "\n".join(parts)


def _format_twitter_thread(twitter_data: Dict[str, Any]) -> list:
    """Format Twitter thread section."""
    parts = []
    
    thread = twitter_data.get('thread', [])
    metadata = twitter_data.get('metadata', {})
    
    if not thread:
        parts.append("*No Twitter thread generated*")
        return parts
    
    total_tweets = len(thread)
    parts.append(f"**Thread Summary**: {total_tweets} tweets")
    parts.append("")
    
    for tweet in thread:
        tweet_num = tweet.get('tweet_number', 0)
        content = tweet.get('content', '')
        char_count = tweet.get('character_count', len(content))
        image_suggestion = tweet.get('image_suggestion')
        
        parts.append(f"### Tweet {tweet_num}/{total_tweets}")
        parts.append("")
        parts.append(content)
        parts.append("")
        parts.append(f"*[Character count: {char_count}/280]*")
        
        if image_suggestion:
            parts.append(f"*[Suggested image: {image_suggestion}]*")
        
        parts.append("")
        parts.append("---")
        parts.append("")
    
    # Thread metadata
    parts.append("### Thread Metadata")
    parts.append("")
    parts.append(f"- **Total tweets**: {metadata.get('total_tweets', total_tweets)}")
    parts.append(f"- **Average length**: {metadata.get('avg_character_count', 0)} characters")
    
    hashtags = metadata.get('hashtags_used', [])
    if hashtags:
        parts.append(f"- **Hashtags used**: {', '.join(hashtags)}")
    
    hooks = metadata.get('engagement_hooks', [])
    if hooks:
        parts.append(f"- **Engagement hooks**: {len(hooks)}")
    
    parts.append("")
    
    return parts


def _format_linkedin_posts(linkedin_data: Dict[str, Any]) -> list:
    """Format LinkedIn posts section."""
    parts = []
    
    variations = linkedin_data.get('variations', [])
    
    if not variations:
        parts.append("*No LinkedIn posts generated*")
        return parts
    
    for i, variation in enumerate(variations, 1):
        var_type = variation.get('variation_type', f'Variation {i}')
        title = variation.get('title', '')
        content = variation.get('content', '')
        char_count = variation.get('character_count', len(content))
        word_count = variation.get('word_count', 0)
        hashtags = variation.get('hashtags', [])
        image_suggestion = variation.get('image_suggestion', '')
        key_points = variation.get('key_points', [])
        cta = variation.get('cta', '')
        
        parts.append(f"### Variation {i}: {var_type}")
        parts.append("")
        
        if title:
            parts.append(f"**Title**: {title}")
            parts.append("")
        
        parts.append(content)
        parts.append("")
        
        parts.append(f"*[Word count: {word_count} | Character count: {char_count}/3000]*")
        
        if hashtags:
            parts.append(f"*[Hashtags: {', '.join(hashtags)}]*")
        
        if image_suggestion:
            parts.append(f"*[Suggested image: {image_suggestion}]*")
        
        if key_points:
            parts.append("")
            parts.append("**Key Points**:")
            for point in key_points:
                parts.append(f"- {point}")
        
        if cta:
            parts.append("")
            parts.append(f"**Call-to-Action**: {cta}")
        
        parts.append("")
        parts.append("---")
        parts.append("")
    
    return parts


def _format_image_suggestions(images_data: Dict[str, Any]) -> list:
    """Format image suggestions section."""
    parts = []
    
    twitter_images = images_data.get('twitter_images', [])
    linkedin_images = images_data.get('linkedin_images', [])
    general = images_data.get('general_suggestions', {})
    
    # Twitter images
    if twitter_images:
        parts.append("### Twitter Images")
        parts.append("")
        
        for img in twitter_images:
            tweet_num = img.get('tweet_number', '')
            img_type = img.get('image_type', '')
            desc = img.get('description', '')
            elements = img.get('elements', [])
            color_scheme = img.get('color_scheme', '')
            dimensions = img.get('dimensions', '1200x675px')
            tools = img.get('suggested_tools', [])
            alt_text = img.get('alt_text', '')
            purpose = img.get('purpose', '')
            
            parts.append(f"**Image for Tweet {tweet_num}**: {img_type}")
            parts.append("")
            parts.append(f"- **Description**: {desc}")
            parts.append(f"- **Dimensions**: {dimensions}")
            
            if elements:
                parts.append(f"- **Elements**: {', '.join(elements)}")
            
            if color_scheme:
                parts.append(f"- **Color scheme**: {color_scheme}")
            
            if tools:
                parts.append(f"- **Suggested tools**: {', '.join(tools)}")
            
            if alt_text:
                parts.append(f"- **Alt text**: {alt_text}")
            
            if purpose:
                parts.append(f"- **Purpose**: {purpose}")
            
            parts.append("")
    
    # LinkedIn images
    if linkedin_images:
        parts.append("### LinkedIn Images")
        parts.append("")
        
        for img in linkedin_images:
            variation = img.get('variation', '')
            img_type = img.get('image_type', '')
            desc = img.get('description', '')
            elements = img.get('elements', [])
            color_scheme = img.get('color_scheme', '')
            dimensions = img.get('dimensions', '1200x627px')
            tools = img.get('suggested_tools', [])
            alt_text = img.get('alt_text', '')
            purpose = img.get('purpose', '')
            
            parts.append(f"**Image for {variation}**: {img_type}")
            parts.append("")
            parts.append(f"- **Description**: {desc}")
            parts.append(f"- **Dimensions**: {dimensions}")
            
            if elements:
                parts.append(f"- **Elements**: {', '.join(elements)}")
            
            if color_scheme:
                parts.append(f"- **Color scheme**: {color_scheme}")
            
            if tools:
                parts.append(f"- **Suggested tools**: {', '.join(tools)}")
            
            if alt_text:
                parts.append(f"- **Alt text**: {alt_text}")
            
            if purpose:
                parts.append(f"- **Purpose**: {purpose}")
            
            parts.append("")
    
    # General suggestions
    if general:
        parts.append("### General Design Guidelines")
        parts.append("")
        
        if general.get('brand_colors'):
            parts.append(f"- **Brand colors**: {general['brand_colors']}")
        
        if general.get('style_guide'):
            parts.append(f"- **Style**: {general['style_guide']}")
        
        if general.get('stock_photo_keywords'):
            keywords = ', '.join(general['stock_photo_keywords'])
            parts.append(f"- **Stock photo keywords**: {keywords}")
        
        if general.get('icon_libraries'):
            libs = ', '.join(general['icon_libraries'])
            parts.append(f"- **Icon libraries**: {libs}")
        
        if general.get('design_tips'):
            parts.append("- **Design tips**:")
            for tip in general['design_tips']:
                parts.append(f"  - {tip}")
        
        parts.append("")
    
    return parts


def _format_hashtags_summary(hashtags_data: Dict[str, Any]) -> list:
    """Format hashtags summary section."""
    parts = []
    
    twitter_tags = hashtags_data.get('twitter_hashtags', {})
    linkedin_tags = hashtags_data.get('linkedin_hashtags', {})
    combined = hashtags_data.get('combined_hashtag_list', [])
    
    # Twitter hashtags
    if twitter_tags:
        parts.append("### Twitter Hashtags")
        parts.append("")
        parts.append(f"**Recommended per tweet**: {twitter_tags.get('recommended_per_tweet', 2)}")
        parts.append("")
        
        if twitter_tags.get('primary'):
            parts.append(f"**Primary**: {', '.join(twitter_tags['primary'])}")
        
        if twitter_tags.get('secondary'):
            parts.append(f"**Secondary**: {', '.join(twitter_tags['secondary'])}")
        
        if twitter_tags.get('trending'):
            parts.append(f"**Trending**: {', '.join(twitter_tags['trending'])}")
        
        if twitter_tags.get('distribution_strategy'):
            parts.append("")
            parts.append(f"*Strategy: {twitter_tags['distribution_strategy']}*")
        
        parts.append("")
    
    # LinkedIn hashtags
    if linkedin_tags:
        parts.append("### LinkedIn Hashtags")
        parts.append("")
        parts.append(f"**Recommended per post**: {linkedin_tags.get('recommended_per_post', 6)}")
        parts.append("")
        
        if linkedin_tags.get('primary'):
            parts.append(f"**Primary**: {', '.join(linkedin_tags['primary'])}")
        
        if linkedin_tags.get('technical'):
            parts.append(f"**Technical**: {', '.join(linkedin_tags['technical'])}")
        
        if linkedin_tags.get('industry'):
            parts.append(f"**Industry**: {', '.join(linkedin_tags['industry'])}")
        
        if linkedin_tags.get('audience'):
            parts.append(f"**Audience**: {', '.join(linkedin_tags['audience'])}")
        
        if linkedin_tags.get('distribution_strategy'):
            parts.append("")
            parts.append(f"*Strategy: {linkedin_tags['distribution_strategy']}*")
        
        parts.append("")
    
    # All hashtags with details
    if combined:
        parts.append("### All Hashtags (Detailed)")
        parts.append("")
        parts.append("| Hashtag | Platforms | Category | Popularity | Relevance |")
        parts.append("|---------|-----------|----------|------------|-----------|")
        
        for tag_info in combined:
            hashtag = tag_info.get('hashtag', '')
            platforms = ', '.join(tag_info.get('platforms', []))
            category = tag_info.get('category', '')
            popularity = tag_info.get('popularity', '')
            relevance = tag_info.get('relevance_score', 0)
            
            parts.append(f"| {hashtag} | {platforms} | {category} | {popularity} | {relevance}% |")
        
        parts.append("")
    
    return parts


def get_generation_stats(social_content: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract statistics from generated social media content.
    
    Args:
        social_content: Generated social media content
        
    Returns:
        Dictionary of statistics
    """
    twitter_data = social_content.get('twitter', {})
    linkedin_data = social_content.get('linkedin', {})
    images_data = social_content.get('images', {})
    
    stats = {
        'twitter': {
            'total_tweets': len(twitter_data.get('thread', [])),
            'avg_char_count': twitter_data.get('metadata', {}).get('avg_character_count', 0),
            'hashtags_count': len(twitter_data.get('metadata', {}).get('hashtags_used', []))
        },
        'linkedin': {
            'total_variations': len(linkedin_data.get('variations', [])),
            'avg_char_count': linkedin_data.get('metadata', {}).get('avg_character_count', 0)
        },
        'images': {
            'twitter_images': len(images_data.get('twitter_images', [])),
            'linkedin_images': len(images_data.get('linkedin_images', []))
        },
        'generated_at': social_content.get('metadata', {}).get('generated_at', '')
    }
    
    return stats