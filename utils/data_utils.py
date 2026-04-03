def strip_markdown_wrapper(content: str) -> str:
    """
    Strip markdown code block wrappers from JSON content.
    
    Handles cases where LLM wraps JSON in ```json or ``` markers.
    
    Args:
        content: Raw content from LLM response
        
    Returns:
        Cleaned JSON string
    """
    content = content.strip()
    
    # Remove ```json prefix
    if content.startswith("```json"):
        content = content[7:]  # Remove ```json
    elif content.startswith("```"):
        content = content[3:]  # Remove ```
    
    # Remove ``` suffix
    if content.endswith("```"):
        content = content[:-3]
    
    return content.strip()