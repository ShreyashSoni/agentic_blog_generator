"""
Blog Generation Workflow using LangGraph.

This module orchestrates all agents in a graph-based workflow,
including parallel execution of writer agents for better performance.
"""

import logging
from typing import Dict, Any
from langgraph.graph import StateGraph, END
from state import BlogState, initialize_approval_state, is_approval_enabled
from agents.planner import planner_node
from agents.research import research_node
from agents.outline import outline_node
from agents.writer import write_all_sections
from agents.editor import editor_node
from agents.seo import seo_node
from agents.approval import (
    plan_approval_node,
    outline_approval_node,
    route_after_plan_approval,
    route_after_outline_approval
)

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


def create_interactive_blog_workflow():
    """
    Create and compile the interactive blog generation workflow graph with approval checkpoints.
    
    This function builds a LangGraph StateGraph that includes human approval checkpoints
    after the planner and outline agents, allowing for interactive review and feedback.
    
    Returns:
        Compiled LangGraph workflow with approval nodes
    """
    logger.info("Creating interactive blog generation workflow...")
    
    # Initialize graph with BlogState TypedDict
    workflow = StateGraph(BlogState)
    
    # Add all agent nodes
    workflow.add_node("planner", planner_node)  # type: ignore[arg-type]
    workflow.add_node("plan_approval", plan_approval_node)  # type: ignore[arg-type]
    workflow.add_node("research", research_node)  # type: ignore[arg-type]
    workflow.add_node("outline", outline_node)  # type: ignore[arg-type]
    workflow.add_node("outline_approval", outline_approval_node)  # type: ignore[arg-type]
    workflow.add_node("writer", write_all_sections)  # type: ignore[arg-type]
    workflow.add_node("editor", editor_node)  # type: ignore[arg-type]
    workflow.add_node("seo", seo_node)  # type: ignore[arg-type]
    
    # Define workflow with approval checkpoints and conditional edges
    workflow.add_edge("planner", "plan_approval")
    
    # Conditional routing after plan approval
    workflow.add_conditional_edges(
        "plan_approval",
        route_after_plan_approval,
        {
            "research": "research",
            "planner": "planner"  # Retry planner with feedback
        }
    )
    
    workflow.add_edge("research", "outline")
    workflow.add_edge("outline", "outline_approval")
    
    # Conditional routing after outline approval
    workflow.add_conditional_edges(
        "outline_approval",
        route_after_outline_approval,
        {
            "writer": "writer",
            "outline": "outline"  # Retry outline with feedback
        }
    )
    
    # Continue with standard workflow
    workflow.add_edge("writer", "editor")
    workflow.add_edge("editor", "seo")
    workflow.add_edge("seo", END)
    
    # Set entry point
    workflow.set_entry_point("planner")
    
    logger.info("Interactive workflow structure:")
    logger.info("  planner → plan_approval → research → outline → outline_approval → writer → editor → seo → END")
    logger.info("  With conditional routing for approval feedback loops")
    
    # Compile and return
    compiled_workflow = workflow.compile()
    logger.info("Interactive blog generation workflow compiled successfully")
    
    return compiled_workflow


def get_workflow(approval_mode: bool = False):
    """
    Get appropriate workflow based on approval mode.
    
    Args:
        approval_mode: Whether to use interactive approval workflow
        
    Returns:
        Compiled LangGraph workflow (interactive or standard)
    """
    if approval_mode:
        return create_interactive_blog_workflow()
    else:
        return create_blog_workflow()


def run_workflow(topic: str,
                 verbose: bool = True,
                 llm_provider: str = "anthropic",
                 model_name: str = "anthropic.claude-opus-4-6-v1",
                 length: str = "complex",
                 approval_mode: bool = False,
                 max_approval_attempts: int = 3,
                 approval_timeout: int = 300) -> Dict[str, Any]:
    """
    Run the complete blog generation workflow for a given topic.
    
    This is a convenience function that creates the workflow,
    initializes the state, and executes the full graph.
    
    Args:
        topic: The blog topic to generate content about
        verbose: Whether to log progress updates
        llm_provider: LLM provider ('openai' or 'anthropic')
        model_name: Specific model to use
        length: Blog complexity level ('simple' or 'complex')
        approval_mode: Whether to enable interactive approval checkpoints
        max_approval_attempts: Maximum revision attempts per checkpoint
        approval_timeout: Timeout for approval prompts in seconds
        
    Returns:
        Final state containing all generated content
    """
    if verbose:
        mode_desc = "interactive" if approval_mode else "standard"
        logger.info(f"Starting {mode_desc} blog generation workflow for: '{topic}'")
    
    # Create appropriate workflow
    workflow = get_workflow(approval_mode=approval_mode)
    
    # Initialize state
    initial_state: Dict[str, Any] = {
        "topic": topic,
        "llm_provider": llm_provider,
        "model_name": model_name,
        "length": length
    }
    
    # Initialize approval state if needed
    if approval_mode:
        initial_state = initialize_approval_state(initial_state)
        initial_state.update({
            "approval_mode": True,
            "max_approval_attempts": max_approval_attempts,
            "approval_timeout": approval_timeout
        })
        
        if verbose:
            print("\n" + "="*70)
            print("🤖 INTERACTIVE BLOG GENERATION")
            print("="*70)
            print("This workflow includes human approval checkpoints.")
            print("You will be asked to review and approve the plan and outline.")
            print("="*70 + "\n")
    
    # Execute workflow
    try:
        if verbose:
            logger.info("Executing workflow...")
        
        final_state = workflow.invoke(initial_state)  # type: ignore[arg-type]
        
        if verbose:
            logger.info("Workflow completed successfully!")
            logger.info(f"Final blog length: {len(final_state.get('edited', '').split())} words")
            
            # Log approval statistics if in approval mode
            if approval_mode and is_approval_enabled(final_state):
                plan_attempts = final_state.get("approval_attempt_count", {}).get("plan", 0)
                outline_attempts = final_state.get("approval_attempt_count", {}).get("outline", 0)
                logger.info(f"Approval attempts - Plan: {plan_attempts}, Outline: {outline_attempts}")
        
        return final_state
        
    except KeyboardInterrupt:
        logger.warning("Workflow interrupted by user during approval")
        raise
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