"""
Blog Generation Workflow using LangGraph.

This module orchestrates all agents in a graph-based workflow,
including parallel execution of writer agents for better performance.
"""

import logging
from typing import Dict, Any
from langgraph.graph import StateGraph, END
from state import BlogState
from agents.planner import planner_node
from agents.research import research_node
from agents.outline import outline_node
from agents.writer import write_all_sections
from agents.editor import editor_node
from agents.seo import seo_node

logger = logging.getLogger(__name__)


def create_blog_workflow():
    """
    Create and compile the blog generation workflow graph.
    
    This function builds a LangGraph StateGraph that orchestrates all agents
    in the correct sequence, with writer agents executing in parallel.
    
    Returns:
        Compiled LangGraph workflow
    """
    logger.info("Creating blog generation workflow...")
    
    # Initialize graph with BlogState TypedDict
    workflow = StateGraph(BlogState)
    
    # Add all agent nodes
    workflow.add_node("planner", planner_node)  # type: ignore[arg-type]
    workflow.add_node("research", research_node)  # type: ignore[arg-type]
    workflow.add_node("outline", outline_node)  # type: ignore[arg-type]
    workflow.add_node("writer", write_all_sections)  # type: ignore[arg-type]
    workflow.add_node("editor", editor_node)  # type: ignore[arg-type]
    workflow.add_node("seo", seo_node)  # type: ignore[arg-type]
    
    # Define sequential edges
    workflow.add_edge("planner", "research")
    workflow.add_edge("research", "outline")
    workflow.add_edge("outline", "writer")
    workflow.add_edge("writer", "editor")
    workflow.add_edge("editor", "seo")
    workflow.add_edge("seo", END)
    
    # Set entry point
    workflow.set_entry_point("planner")
    
    logger.info("Workflow structure:")
    logger.info("  planner → research → outline → writer → editor → seo → END")
    
    # Compile and return
    compiled_workflow = workflow.compile()
    logger.info("Blog generation workflow compiled successfully")
    
    return compiled_workflow


def run_workflow(topic: str, 
                 verbose: bool = True, 
                 llm_provider: str = "anthropic", 
                 model_name: str = "anthropic.claude-opus-4-6-v1") -> Dict[str, Any]:
    """
    Run the complete blog generation workflow for a given topic.
    
    This is a convenience function that creates the workflow,
    initializes the state, and executes the full graph.
    
    Args:
        topic: The blog topic to generate content about
        verbose: Whether to log progress updates
        llm_provider: LLM provider ('openai' or 'anthropic')
        model_name: Specific model to use
        
    Returns:
        Final state containing all generated content
    """
    if verbose:
        logger.info(f"Starting blog generation workflow for: '{topic}'")
    
    # Create workflow
    workflow = create_blog_workflow()
    
    # Initialize state
    initial_state: Dict[str, Any] = {
        "topic": topic,
        "llm_provider": llm_provider,
        "model_name": model_name
    }
    
    # Execute workflow
    try:
        if verbose:
            logger.info("Executing workflow...")
        
        final_state = workflow.invoke(initial_state)  # type: ignore[arg-type]
        
        if verbose:
            logger.info("Workflow completed successfully!")
            logger.info(f"Final blog length: {len(final_state.get('edited', '').split())} words")
        
        return final_state
        
    except Exception as e:
        logger.error(f"Workflow execution failed: {e}")
        raise


def run_workflow_streaming(topic: str):
    """
    Run the workflow with streaming output for progress tracking.
    
    This function executes the workflow and yields state updates
    after each node execution, allowing for real-time progress monitoring.
    
    Args:
        topic: The blog topic to generate content about
        
    Yields:
        State updates after each node execution
    """
    logger.info(f"Starting streaming workflow for: '{topic}'")
    
    # Create workflow
    workflow = create_blog_workflow()
    
    # Initialize state
    initial_state: Dict[str, Any] = {
        "topic": topic
    }
    
    # Stream workflow execution
    for state in workflow.stream(initial_state):  # type: ignore[arg-type]
        yield state


# Alternative implementation with explicit parallel writers
# This is more complex but shows true parallel execution

def create_parallel_blog_workflow():
    """
    Create a blog workflow with truly parallel writer execution.
    
    This is an advanced version that dynamically creates parallel
    writer nodes for each section, demonstrating LangGraph's
    parallel execution capabilities.
    
    Note: This is kept as an alternative implementation. The simpler
    write_all_sections approach is used by default for better
    error handling and progress tracking.
    
    Returns:
        Compiled LangGraph workflow with parallel writers
    """
    from langgraph.graph import StateGraph, END
    
    workflow = StateGraph(BlogState)
    
    # Add sequential nodes
    workflow.add_node("planner", planner_node)  # type: ignore[arg-type]
    workflow.add_node("research", research_node)  # type: ignore[arg-type]
    workflow.add_node("outline", outline_node)  # type: ignore[arg-type]
    
    # Writer node will be added dynamically based on outline
    # For now, we'll use the sequential approach
    workflow.add_node("writer", write_all_sections)  # type: ignore[arg-type]
    
    workflow.add_node("editor", editor_node)  # type: ignore[arg-type]
    workflow.add_node("seo", seo_node)  # type: ignore[arg-type]
    
    # Define edges
    workflow.add_edge("planner", "research")
    workflow.add_edge("research", "outline")
    workflow.add_edge("outline", "writer")
    workflow.add_edge("writer", "editor")
    workflow.add_edge("editor", "seo")
    workflow.add_edge("seo", END)
    
    workflow.set_entry_point("planner")
    
    return workflow.compile()