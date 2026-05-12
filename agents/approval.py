"""
Approval Agent - Human-in-the-loop approval checkpoints.

This module contains LangGraph nodes for human approval of plans and outlines,
enabling interactive review and feedback collection during blog generation.
"""

import logging
from typing import Dict, Any
from state import initialize_approval_state, is_approval_enabled
from services.approval_service import HumanApprovalService

logger = logging.getLogger(__name__)


def plan_approval_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node for plan approval checkpoint.
    
    This node presents the generated plan to the user for review and approval.
    If rejected, it collects feedback that will be used by the planner for revision.
    
    Input:
        state["plan"]: Generated blog plan
        state["approval_mode"]: Whether approval is enabled
        
    Output:
        state["plan_approval_status"]: 'approved', 'rejected'
        state["user_feedback"]["plan"]: User feedback if rejected
        state["approval_attempt_count"]["plan"]: Incremented attempt counter
        
    Args:
        state: Current blog state
        
    Returns:
        Updated state with approval status
        
    Raises:
        ValueError: If maximum approval attempts exceeded
        KeyboardInterrupt: If user requests workflow termination
    """
    # Ensure approval state is initialized
    state = initialize_approval_state(state)
    
    # Skip approval if not in approval mode
    if not is_approval_enabled(state):
        logger.info("Plan approval: Skipping (approval mode disabled)")
        state["plan_approval_status"] = "approved"
        return state
    
    # Check if we've exceeded max attempts
    attempts = state.get("approval_attempt_count", {}).get("plan", 0)
    max_attempts = state.get("max_approval_attempts", 3)
    
    if attempts >= max_attempts:
        error_msg = f"Maximum plan approval attempts exceeded: {attempts}/{max_attempts}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Increment attempt counter
    if "approval_attempt_count" not in state:
        state["approval_attempt_count"] = {"plan": 0, "outline": 0}
    state["approval_attempt_count"]["plan"] += 1
    
    current_attempt = state["approval_attempt_count"]["plan"]
    logger.info(f"Plan approval checkpoint - Attempt {current_attempt}/{max_attempts}")
    
    # Initialize approval service
    timeout = state.get("approval_timeout", 300)
    approval_service = HumanApprovalService(timeout=timeout)
    
    try:
        # Display plan for review
        approval_service.display_plan_for_review(state)
        
        # Get user approval decision
        approval_status = approval_service.get_user_approval("plan")
        
        if approval_status == "approved":
            state["plan_approval_status"] = "approved"
            logger.info("Plan approved by user")
            
        elif approval_status == "rejected":
            # Collect detailed feedback for revision
            feedback = approval_service.collect_feedback_for_revision("plan")
            
            if feedback:
                # Store feedback for planner revision
                if "user_feedback" not in state:
                    state["user_feedback"] = {}
                state["user_feedback"]["plan"] = feedback
                state["plan_approval_status"] = "rejected"
                logger.info(f"Plan rejected with feedback: {feedback[:100]}...")
            else:
                # No feedback provided, treat as approval
                state["plan_approval_status"] = "approved"
                logger.info("No feedback provided, treating as approval")
                
        elif approval_status == "quit":
            logger.info("User requested to quit workflow")
            raise KeyboardInterrupt("User requested workflow termination")
            
        else:  # timeout or other
            logger.warning("Plan approval timed out, treating as approval")
            state["plan_approval_status"] = "approved"
            
        # Log approval result
        status = state["plan_approval_status"]
        logger.info(f"Plan approval result: {status} (attempt {current_attempt})")
    
    except KeyboardInterrupt:
        # Re-raise keyboard interrupts
        raise
    except Exception as e:
        logger.error(f"Error during plan approval: {e}")
        # Default to approval to prevent workflow breakage in production
        state["plan_approval_status"] = "approved"
        logger.warning("Plan approval failed, defaulting to approved status")
    
    return state


def outline_approval_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node for outline approval checkpoint.
    
    This node presents the generated outline to the user for review and approval.
    If rejected, it collects feedback that will be used by the outline agent for revision.
    
    Input:
        state["outline"]: Generated blog outline
        state["approval_mode"]: Whether approval is enabled
        
    Output:
        state["outline_approval_status"]: 'approved', 'rejected'
        state["user_feedback"]["outline"]: User feedback if rejected
        state["approval_attempt_count"]["outline"]: Incremented attempt counter
        
    Args:
        state: Current blog state
        
    Returns:
        Updated state with approval status
        
    Raises:
        ValueError: If maximum approval attempts exceeded
        KeyboardInterrupt: If user requests workflow termination
    """
    # Ensure approval state is initialized
    state = initialize_approval_state(state)
    
    # Skip approval if not in approval mode
    if not is_approval_enabled(state):
        logger.info("Outline approval: Skipping (approval mode disabled)")
        state["outline_approval_status"] = "approved"
        return state
    
    # Check if we've exceeded max attempts
    attempts = state.get("approval_attempt_count", {}).get("outline", 0)
    max_attempts = state.get("max_approval_attempts", 3)
    
    if attempts >= max_attempts:
        error_msg = f"Maximum outline approval attempts exceeded: {attempts}/{max_attempts}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Increment attempt counter
    if "approval_attempt_count" not in state:
        state["approval_attempt_count"] = {"plan": 0, "outline": 0}
    state["approval_attempt_count"]["outline"] += 1
    
    current_attempt = state["approval_attempt_count"]["outline"]
    logger.info(f"Outline approval checkpoint - Attempt {current_attempt}/{max_attempts}")
    
    # Initialize approval service
    timeout = state.get("approval_timeout", 300)
    approval_service = HumanApprovalService(timeout=timeout)
    
    try:
        # Display outline for review
        approval_service.display_outline_for_review(state)
        
        # Get user approval decision
        approval_status = approval_service.get_user_approval("outline")
        
        if approval_status == "approved":
            state["outline_approval_status"] = "approved"
            logger.info("Outline approved by user")
            
        elif approval_status == "rejected":
            # Collect detailed feedback for revision
            feedback = approval_service.collect_feedback_for_revision("outline")
            
            if feedback:
                # Store feedback for outline agent revision
                if "user_feedback" not in state:
                    state["user_feedback"] = {}
                state["user_feedback"]["outline"] = feedback
                state["outline_approval_status"] = "rejected"
                logger.info(f"Outline rejected with feedback: {feedback[:100]}...")
            else:
                # No feedback provided, treat as approval
                state["outline_approval_status"] = "approved"
                logger.info("No feedback provided, treating as approval")
                
        elif approval_status == "quit":
            logger.info("User requested to quit workflow")
            raise KeyboardInterrupt("User requested workflow termination")
            
        else:  # timeout or other
            logger.warning("Outline approval timed out, treating as approval")
            state["outline_approval_status"] = "approved"
            
        # Log approval result
        status = state["outline_approval_status"]
        logger.info(f"Outline approval result: {status} (attempt {current_attempt})")
    
    except KeyboardInterrupt:
        # Re-raise keyboard interrupts
        raise
    except Exception as e:
        logger.error(f"Error during outline approval: {e}")
        # Default to approval to prevent workflow breakage in production
        state["outline_approval_status"] = "approved"
        logger.warning("Outline approval failed, defaulting to approved status")
    
    return state


# Conditional routing functions for LangGraph

def route_after_plan_approval(state: Dict[str, Any]) -> str:
    """
    Determine next node after plan approval.
    
    This function is used by LangGraph conditional edges to route the workflow
    based on the plan approval status.
    
    Args:
        state: Current workflow state
        
    Returns:
        Next node name: 'research' if approved, 'planner' if rejected
        
    Raises:
        ValueError: If approval status is invalid or missing
    """
    status = state.get("plan_approval_status", "pending")
    
    if status == "approved":
        logger.info("Plan approved: routing to research")
        return "research"
    elif status == "rejected":
        logger.info("Plan rejected: routing back to planner for revision")
        return "planner"
    else:
        error_msg = f"Invalid plan approval status: {status}"
        logger.error(error_msg)
        raise ValueError(error_msg)


def route_after_outline_approval(state: Dict[str, Any]) -> str:
    """
    Determine next node after outline approval.
    
    This function is used by LangGraph conditional edges to route the workflow
    based on the outline approval status.
    
    Args:
        state: Current workflow state
        
    Returns:
        Next node name: 'writer' if approved, 'outline' if rejected
        
    Raises:
        ValueError: If approval status is invalid or missing
    """
    status = state.get("outline_approval_status", "pending")
    
    if status == "approved":
        logger.info("Outline approved: routing to writer")
        return "writer"
    elif status == "rejected":
        logger.info("Outline rejected: routing back to outline for revision")
        return "outline"
    else:
        error_msg = f"Invalid outline approval status: {status}"
        logger.error(error_msg)
        raise ValueError(error_msg)


# Utility functions for approval workflow

def reset_plan_approval(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Reset plan approval state for retry.
    
    Args:
        state: Current workflow state
        
    Returns:
        State with reset plan approval fields
    """
    state["plan_approval_status"] = "pending"
    if "user_feedback" in state and "plan" in state["user_feedback"]:
        # Keep feedback for logging but mark as processed
        state["user_feedback"]["plan_processed"] = state["user_feedback"]["plan"]
        del state["user_feedback"]["plan"]
    
    logger.info("Plan approval state reset for retry")
    return state


def reset_outline_approval(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Reset outline approval state for retry.
    
    Args:
        state: Current workflow state
        
    Returns:
        State with reset outline approval fields
    """
    state["outline_approval_status"] = "pending"
    if "user_feedback" in state and "outline" in state["user_feedback"]:
        # Keep feedback for logging but mark as processed
        state["user_feedback"]["outline_processed"] = state["user_feedback"]["outline"]
        del state["user_feedback"]["outline"]
    
    logger.info("Outline approval state reset for retry")
    return state


def get_approval_attempt_info(state: Dict[str, Any], approval_type: str) -> Dict[str, Any]:
    """
    Get approval attempt information for logging and display.
    
    Args:
        state: Current workflow state
        approval_type: 'plan' or 'outline'
        
    Returns:
        Dictionary with attempt information
    """
    attempts = state.get("approval_attempt_count", {}).get(approval_type, 0)
    max_attempts = state.get("max_approval_attempts", 3)
    remaining = max_attempts - attempts
    
    return {
        "current_attempt": attempts,
        "max_attempts": max_attempts,
        "remaining_attempts": remaining,
        "is_final_attempt": remaining <= 1,
        "has_exceeded_max": attempts >= max_attempts
    }


def should_continue_approval_workflow(state: Dict[str, Any]) -> bool:
    """
    Check if approval workflow should continue.
    
    Args:
        state: Current workflow state
        
    Returns:
        True if workflow should continue, False if should stop
    """
    # Check if approval mode is enabled
    if not is_approval_enabled(state):
        return True  # Non-approval mode always continues
    
    # Check if either approval process has exceeded limits
    plan_info = get_approval_attempt_info(state, "plan")
    outline_info = get_approval_attempt_info(state, "outline")
    
    if plan_info["has_exceeded_max"] or outline_info["has_exceeded_max"]:
        logger.warning("Approval workflow stopping: maximum attempts exceeded")
        return False
    
    return True