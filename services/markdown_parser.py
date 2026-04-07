"""
Markdown Parser for Blog Files

This module provides functionality to parse generated blog markdown files,
extracting YAML frontmatter and content for WordPress publishing.
"""

import re
import yaml
import logging
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class BlogParseError(Exception):
    """Custom exception for blog parsing errors."""
    pass


def parse_blog_file(filepath: str) -> Dict[str, Any]:
    """
    Parse a blog markdown file and extract frontmatter and content.
    
    Args:
        filepath: Path to the markdown file
        
    Returns:
        Dictionary containing:
            - metadata: Parsed frontmatter (title, description, keywords, etc.)
            - content: Blog content body
            - raw_content: Original file content
            
    Raises:
        BlogParseError: If file cannot be parsed or is invalid
        FileNotFoundError: If file does not exist
    """
    logger.info(f"Parsing blog file: {filepath}")
    
    # Validate file exists
    file_path = Path(filepath)
    if not file_path.exists():
        raise FileNotFoundError(f"Blog file not found: {filepath}")
    
    if not file_path.suffix == '.md':
        raise BlogParseError(f"File must be a markdown (.md) file: {filepath}")
    
    # Read file content
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_content = f.read()
    except Exception as e:
        raise BlogParseError(f"Error reading file: {e}")
    
    # Extract frontmatter and content
    try:
        metadata = extract_frontmatter(raw_content)
        content = extract_body(raw_content)
    except Exception as e:
        raise BlogParseError(f"Error parsing content: {e}")
    
    # Validate required fields
    if not validate_metadata(metadata):
        raise BlogParseError("Missing required metadata fields")
    
    logger.info(f"Successfully parsed blog: {metadata.get('title', 'Untitled')}")
    
    return {
        'metadata': metadata,
        'content': content,
        'raw_content': raw_content
    }


def extract_frontmatter(content: str) -> Dict[str, Any]:
    """
    Extract YAML frontmatter from markdown content.
    
    Args:
        content: Raw markdown content
        
    Returns:
        Dictionary of frontmatter metadata
        
    Raises:
        BlogParseError: If frontmatter is invalid or missing
    """
    # Match frontmatter between --- delimiters
    frontmatter_pattern = r'^---\s*\n(.*?)\n---\s*\n'
    match = re.match(frontmatter_pattern, content, re.DOTALL)
    
    if not match:
        raise BlogParseError("No valid frontmatter found. Expected format:\n---\ntitle: ...\n---")
    
    frontmatter_text = match.group(1)
    
    try:
        metadata = yaml.safe_load(frontmatter_text)
    except yaml.YAMLError as e:
        raise BlogParseError(f"Invalid YAML in frontmatter: {e}")
    
    if not isinstance(metadata, dict):
        raise BlogParseError("Frontmatter must be a valid YAML dictionary")
    
    return metadata


def extract_body(content: str) -> str:
    """
    Extract blog content body (everything after frontmatter).
    
    Args:
        content: Raw markdown content
        
    Returns:
        Blog content without frontmatter
    """
    # Remove frontmatter
    frontmatter_pattern = r'^---\s*\n.*?\n---\s*\n'
    body = re.sub(frontmatter_pattern, '', content, count=1, flags=re.DOTALL)
    
    # Strip leading/trailing whitespace
    body = body.strip()
    
    return body


def validate_metadata(metadata: Dict[str, Any]) -> bool:
    """
    Validate that required metadata fields are present.
    
    Args:
        metadata: Metadata dictionary to validate
        
    Returns:
        True if valid, False otherwise
    """
    required_fields = ['title', 'description', 'slug']
    
    for field in required_fields:
        if field not in metadata:
            logger.error(f"Missing required field in frontmatter: {field}")
            return False
        
        if not metadata[field] or not str(metadata[field]).strip():
            logger.error(f"Empty value for required field: {field}")
            return False
    
    # Validate title length
    if len(metadata['title']) > 200:
        logger.warning("Title is longer than 200 characters, may be truncated")
    
    # Validate description length
    if len(metadata['description']) > 500:
        logger.warning("Description is longer than 500 characters, may be truncated")
    
    # Validate keywords if present
    if 'keywords' in metadata:
        if not isinstance(metadata['keywords'], list):
            logger.warning("Keywords should be a list, converting to list")
            metadata['keywords'] = [str(metadata['keywords'])]
    
    return True


def get_metadata_field(metadata: Dict[str, Any], field: str, default: Any = None) -> Any:
    """
    Safely get a metadata field with a default value.
    
    Args:
        metadata: Metadata dictionary
        field: Field name to retrieve
        default: Default value if field not found
        
    Returns:
        Field value or default
    """
    return metadata.get(field, default)


def format_keywords_for_wordpress(keywords: list) -> str:
    """
    Format keywords list as comma-separated string for WordPress tags.
    
    Args:
        keywords: List of keywords
        
    Returns:
        Comma-separated string of keywords
    """
    if not keywords:
        return ""
    
    if isinstance(keywords, str):
        return keywords
    
    if isinstance(keywords, list):
        return ", ".join(str(k) for k in keywords)
    
    return str(keywords)


def extract_faq_section(content: str) -> Optional[str]:
    """
    Extract the FAQ section from blog content if present.
    
    Args:
        content: Blog content
        
    Returns:
        FAQ section content or None if not found
    """
    # Look for FAQ section header
    faq_pattern = r'##\s*Frequently Asked Questions\s*\n(.*?)(?=\n##|\Z)'
    match = re.search(faq_pattern, content, re.DOTALL | re.IGNORECASE)
    
    if match:
        return match.group(0).strip()
    
    return None


def get_content_preview(content: str, max_length: int = 200) -> str:
    """
    Get a preview of the blog content.
    
    Args:
        content: Full blog content
        max_length: Maximum preview length
        
    Returns:
        Content preview
    """
    # Remove markdown headers and formatting
    preview = re.sub(r'#+\s+', '', content)
    preview = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', preview)  # Remove links
    preview = re.sub(r'[*_`]', '', preview)  # Remove emphasis
    
    # Get first paragraph or max_length characters
    preview = preview.strip().split('\n\n')[0]
    
    if len(preview) > max_length:
        preview = preview[:max_length].rsplit(' ', 1)[0] + '...'
    
    return preview


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example: Parse a blog file
    try:
        result = parse_blog_file("my_blogs/introduction-to-generative-ai.md") #("outputs/example-blog.md")
        print(f"Title: {result['metadata']['title']}")
        print(f"Description: {result['metadata']['description']}")
        print(f"Content length: {len(result['content'])} characters")
    except (FileNotFoundError, BlogParseError) as e:
        print(f"Error: {e}")