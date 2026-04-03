"""
LLM Factory - Centralized model initialization
"""
import os
from typing import Any
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

def get_llm(provider: str | Any = "anthropic", 
            model_name: str | Any = "anthropic.claude-opus-4-6-v1", 
            temperature: float = 0.7):
    """
    Get LLM instance based on provider configuration.
    
    Args:
        provider: Model provider ('openai' or 'anthropic'). 
                  Defaults to LLM_PROVIDER env var or 'anthropic'
        model_name: Specific model name to use. 
                    Defaults to provider-specific env vars
        temperature: Temperature for generation (0.0-1.0)
    
    Returns:
        Configured LLM instance
    """
    # Determine provider
    if provider is None:
        provider = os.getenv("LLM_PROVIDER", "anthropic").lower()
    
    if provider == "openai":
        model = model_name or os.getenv("OPENAI_MODEL", "gpt-4")
        return ChatOpenAI(
            model=model,
            temperature=temperature
        )
    elif provider == "anthropic":
        model = model_name or os.getenv(
            "ANTHROPIC_MODEL", 
            "anthropic.claude-sonnet-4-5-20250929-v1:0"
        )
        return ChatAnthropic(
            model_name=model,
            base_url=os.getenv("ANTHROPIC_BASE_URL"),
            default_headers={"Authorization": f"Bearer {os.getenv("TOKEN")}"},
            temperature=temperature,
            timeout=30,
            stop=['exit']
        )
    else:
        raise ValueError(
            f"Unknown provider: {provider}. "
            f"Supported providers: 'openai', 'anthropic'"
        )
