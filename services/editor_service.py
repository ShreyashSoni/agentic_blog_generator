"""
On-Demand Blog Editing Service

This service provides standalone functionality to re-edit previously generated
blog posts using the same editor logic from the main workflow.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path

from langchain_core.prompts import PromptTemplate
from utils.llm_factory import get_llm
from agents.editor import load_prompt
from services.markdown_parser import (
    parse_blog_file,
    combine_frontmatter,
    BlogParseError
)

logger = logging.getLogger(__name__)


class EditorServiceError(Exception):
    """Custom exception for editor service errors."""
    pass


def edit_blog_file(
    input_path: str,
    output_dir: str = "outputs/edited_blogs",
    llm_provider: Optional[str] = None,
    model_name: Optional[str] = None,
    preserve_original: bool = True
) -> str:
    """
    Edit a markdown blog file using the editor agent.
    
    This function reads a markdown file, extracts its content and metadata,
    applies the same editing logic used in the main workflow, and saves the
    result with updated frontmatter.
    
    Args:
        input_path: Path to the markdown file to edit
        output_dir: Directory to save the edited blog (default: outputs/edited_blogs)
        llm_provider: LLM provider ('openai' or 'anthropic', default: from env)
        model_name: Specific model name (default: from env)
        preserve_original: Whether to keep original_date in frontmatter
        
    Returns:
        Path to the saved edited file
        
    Raises:
        EditorServiceError: If editing fails
        FileNotFoundError: If input file doesn't exist
        BlogParseError: If markdown parsing fails
    """
    logger.info(f"Starting editing service for: {input_path}")
    
    # Validate input file
    input_file = Path(input_path)
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    if input_file.suffix != '.md':
        raise EditorServiceError(f"Input must be a markdown (.md) file: {input_path}")
    
    try:
        # Parse the blog file
        logger.info("Parsing input file...")
        parsed = parse_blog_file(input_path)
        metadata = parsed['metadata']
        content = parsed['content']
        
        logger.info(f"Parsed blog: {metadata.get('title', 'Untitled')}")
        logger.info(f"Content length: {len(content.split())} words")
        
    except BlogParseError as e:
        raise EditorServiceError(f"Failed to parse blog file: {e}")
    except Exception as e:
        raise EditorServiceError(f"Unexpected error reading file: {e}")
    
    # Extract editing parameters from frontmatter
    target_audience = metadata.get('target_audience', 'general audience')
    tone = metadata.get('tone', 'technical')
    topic = metadata.get('title', 'Blog Post')
    
    logger.info(f"Editing parameters - Audience: {target_audience}, Tone: {tone}")
    
    try:
        # Perform editing using editor logic
        logger.info("Running editor agent...")
        edited_content = _run_editor(
            topic=topic,
            content=content,
            target_audience=target_audience,
            tone=tone,
            llm_provider=llm_provider,
            model_name=model_name
        )
        
        edited_word_count = len(edited_content.split())
        logger.info(f"Editing complete. Edited content: {edited_word_count} words")
        
        # Validate edited content
        if edited_word_count < len(content.split()) * 0.7:
            logger.warning("Edited content significantly shorter than original")
            raise EditorServiceError("Editing resulted in content too short, possible error")
        
    except Exception as e:
        raise EditorServiceError(f"Editor agent failed: {e}")
    
    # Update metadata with edit information
    metadata = _update_metadata(metadata, preserve_original)
    
    # Combine frontmatter and edited content
    full_content = combine_frontmatter(metadata, edited_content)
    
    # Save to output directory
    try:
        output_path = _save_edited_file(
            content=full_content,
            input_path=input_path,
            output_dir=output_dir
        )
        logger.info(f"✅ Edited blog saved to: {output_path}")
        return output_path
        
    except Exception as e:
        raise EditorServiceError(f"Failed to save edited file: {e}")


def _run_editor(
    topic: str,
    content: str,
    target_audience: str,
    tone: str,
    llm_provider: Optional[str] = None,
    model_name: Optional[str] = None
) -> str:
    """
    Run the editor agent on the provided content.
    
    This reuses the same editor logic and prompt from agents/editor.py.
    
    Args:
        topic: Blog topic/title
        content: Content to edit
        target_audience: Target audience
        tone: Desired tone
        llm_provider: LLM provider
        model_name: Model name
        
    Returns:
        Edited content
    """
    # Get LLM instance
    llm = get_llm(
        provider=llm_provider,
        model_name=model_name,
        temperature=0.3
    )
    
    # Load editor prompt template
    prompt_template_str = load_prompt("editor.txt")
    if not prompt_template_str:
        logger.warning("Editor prompt not found, using default")
        prompt_template_str = _get_fallback_prompt()
    
    # Create prompt
    prompt = PromptTemplate(
        template=prompt_template_str,
        input_variables=["topic", "draft", "target_audience", "tone"]
    )
    
    # Create chain
    chain = prompt | llm
    
    # Execute editing with streaming
    logger.info("Streaming edited content from LLM...")
    response = chain.stream({
        "topic": topic,
        "draft": content,
        "target_audience": target_audience,
        "tone": tone
    })
    
    # Collect streamed response
    edited_content = ""
    for chunk in response:
        if hasattr(chunk, 'content'):
            content_chunk = chunk.content
            # Handle both string and list responses
            if isinstance(content_chunk, str):
                edited_content += content_chunk
            elif isinstance(content_chunk, list):
                edited_content += ''.join(str(item) for item in content_chunk)
            else:
                edited_content += str(content_chunk)
        else:
            edited_content += str(chunk)
    
    return edited_content.strip()


def _update_metadata(metadata: Dict[str, Any], preserve_original: bool = True) -> Dict[str, Any]:
    """
    Update frontmatter metadata with editing information.
    
    Args:
        metadata: Original metadata
        preserve_original: Whether to preserve original date
        
    Returns:
        Updated metadata dictionary
    """
    # Add edited timestamp
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    metadata['edited_date'] = current_time
    
    # Preserve original date if requested
    if preserve_original and 'date' in metadata:
        if 'original_date' not in metadata:
            metadata['original_date'] = metadata['date']
    
    # Update date to edited date
    metadata['date'] = datetime.now().strftime("%Y-%m-%d")
    
    logger.info(f"Updated metadata with edited_date: {current_time}")
    
    return metadata


def _save_edited_file(content: str, input_path: str, output_dir: str) -> str:
    """
    Save edited content to output directory.
    
    Args:
        content: Full markdown content with frontmatter
        input_path: Original input file path
        output_dir: Output directory
        
    Returns:
        Path to saved file
    """
    # Create output directory if needed
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate output filename (same as input)
    input_filename = Path(input_path).name
    output_file = output_path / input_filename
    
    # Write file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    logger.info(f"Saved edited file: {output_file}")
    
    return str(output_file)


def _get_fallback_prompt() -> str:
    """
    Get fallback editor prompt if file not found.
    
    Returns:
        Default editor prompt template
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


def get_editing_stats(input_path: str, output_path: str) -> Dict[str, Any]:
    """
    Compare original and edited versions to generate statistics.
    
    Args:
        input_path: Path to original file
        output_path: Path to edited file
        
    Returns:
        Dictionary with comparison statistics
    """
    try:
        # Parse both files
        original = parse_blog_file(input_path)
        edited = parse_blog_file(output_path)
        
        # Calculate statistics
        original_words = len(original['content'].split())
        edited_words = len(edited['content'].split())
        word_diff = edited_words - original_words
        word_diff_pct = (word_diff / original_words * 100) if original_words > 0 else 0
        
        return {
            'original_words': original_words,
            'edited_words': edited_words,
            'word_difference': word_diff,
            'word_change_percent': round(word_diff_pct, 2),
            'original_title': original['metadata'].get('title', 'N/A'),
            'edited_date': edited['metadata'].get('edited_date', 'N/A')
        }
    except Exception as e:
        logger.error(f"Error calculating stats: {e}")
        return {}


# Example usage
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Example: Edit a blog file
    try:
        output = edit_blog_file(
            input_path="outputs/example-blog.md",
            output_dir="outputs/edited_blogs"
        )
        print(f"✅ Success! Edited blog saved to: {output}")
        
        # Get statistics
        stats = get_editing_stats("outputs/example-blog.md", output)
        print(f"\n📊 Statistics:")
        print(f"  Original: {stats['original_words']} words")
        print(f"  Edited: {stats['edited_words']} words")
        print(f"  Change: {stats['word_change_percent']}%")
        
    except (FileNotFoundError, EditorServiceError, BlogParseError) as e:
        print(f"❌ Error: {e}")