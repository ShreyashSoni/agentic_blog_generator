"""
State definition for the blog generation workflow.

This module defines the BlogState TypedDict that represents the shared state
flowing through all agents in the LangGraph workflow.
"""

from typing import TypedDict, List, Dict, Any, Optional, Literal


class BlogState(TypedDict, total=False):
    """
    Shared state object that flows through the entire blog generation graph.
    Each agent reads from and writes to this state.
    
    Attributes:
        topic: The blog topic provided by the user (required)
        llm_provider: LLM provider ('openai' or 'anthropic')
        model_name: Specific model to use
        length: 'simple' or 'complex' for short or long format
        plan: Structured blog plan including audience, length, sections, keywords
        research_docs: List of summarized research documents
        outline: Ordered list of section titles
        sections: Dictionary mapping section titles to their content
        draft: Combined sections before editing
        edited: Final polished blog content
        seo_meta: SEO metadata including title, description, keywords, etc.
        _vector_store: Internal reference to vector store (not serialized)
        
        # Human-in-the-Loop Approval Fields
        approval_mode: Enable/disable human approval checkpoints
        plan_approval_status: Status of plan approval ('pending', 'approved', 'rejected')
        outline_approval_status: Status of outline approval ('pending', 'approved', 'rejected')
        user_feedback: User feedback for revisions {'plan': 'feedback', 'outline': 'feedback'}
        approval_attempt_count: Retry attempt tracking {'plan': count, 'outline': count}
        max_approval_attempts: Maximum revision attempts per checkpoint
        approval_timeout: Timeout for approval prompts in seconds
        workflow_paused_at: Track where workflow paused for resumption
    """
    
    # Core input (required)
    topic: str
    llm_provider: str
    model_name: str
    length: str  # 'simple' or 'complex'
    
    # Planning phase
    plan: Dict[str, Any]  # {target_audience, blog_length, section_titles, keywords, tone}
    
    # Research phase
    research_docs: List[str]  # List of summarized research documents
    
    # Outline phase
    outline: List[str]  # Ordered list of section titles
    
    # Writing phase
    sections: Dict[str, str]  # {section_title: section_content}
    
    # Editing phase
    draft: str  # Combined sections before editing
    edited: str  # Final polished blog content
    
    # SEO phase
    seo_meta: Dict[str, Any]  # {meta_title, meta_description, slug, keywords, faq, keyword_density}
    
    # Internal state (not part of final output)
    _vector_store: Optional[Any]  # Reference to vector store instance
    
    # Human-in-the-Loop Approval Fields
    approval_mode: bool  # Enable/disable human approval checkpoints
    plan_approval_status: Literal['pending', 'approved', 'rejected']  # Plan approval status
    outline_approval_status: Literal['pending', 'approved', 'rejected']  # Outline approval status
    user_feedback: Dict[str, str]  # User feedback for revisions {'plan': 'feedback', 'outline': 'feedback'}
    approval_attempt_count: Dict[str, int]  # Retry attempt tracking {'plan': count, 'outline': count}
    max_approval_attempts: int  # Maximum revision attempts per checkpoint (default: 3)
    approval_timeout: int  # Timeout for approval prompts in seconds (default: 300)
    workflow_paused_at: Optional[Literal['plan_approval', 'outline_approval']]  # Track paused workflow state


# Type aliases for approval status
ApprovalStatus = Literal['pending', 'approved', 'rejected']
WorkflowPausePoint = Literal['plan_approval', 'outline_approval']


def initialize_approval_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Initialize approval-related fields in state with sensible defaults.
    
    Args:
        state: Existing state dictionary
        
    Returns:
        State with initialized approval fields
    """
    # Set defaults only if not already present (preserves existing values)
    state.setdefault("approval_mode", False)
    state.setdefault("plan_approval_status", "pending")
    state.setdefault("outline_approval_status", "pending")
    state.setdefault("user_feedback", {})
    state.setdefault("approval_attempt_count", {"plan": 0, "outline": 0})
    state.setdefault("max_approval_attempts", 3)
    state.setdefault("approval_timeout", 300)
    state.setdefault("workflow_paused_at", None)
    
    return state


def validate_approval_state(state: Dict[str, Any]) -> bool:
    """
    Validate that approval-related fields have valid values.
    
    Args:
        state: State dictionary to validate
        
    Returns:
        True if state is valid, False otherwise
        
    Raises:
        ValueError: If critical validation fails
    """
    if not isinstance(state, dict):
        raise ValueError("State must be a dictionary")
    
    # Validate approval_mode
    approval_mode = state.get("approval_mode", False)
    if not isinstance(approval_mode, bool):
        raise ValueError("approval_mode must be a boolean")
    
    # Validate approval statuses
    valid_statuses = {'pending', 'approved', 'rejected'}
    
    plan_status = state.get("plan_approval_status", "pending")
    if plan_status not in valid_statuses:
        raise ValueError(f"plan_approval_status must be one of {valid_statuses}")
    
    outline_status = state.get("outline_approval_status", "pending")
    if outline_status not in valid_statuses:
        raise ValueError(f"outline_approval_status must be one of {valid_statuses}")
    
    # Validate attempt counts
    attempt_count = state.get("approval_attempt_count", {})
    if not isinstance(attempt_count, dict):
        raise ValueError("approval_attempt_count must be a dictionary")
    
    for key, value in attempt_count.items():
        if not isinstance(value, int) or value < 0:
            raise ValueError(f"approval_attempt_count[{key}] must be a non-negative integer")
    
    # Validate max attempts
    max_attempts = state.get("max_approval_attempts", 3)
    if not isinstance(max_attempts, int) or max_attempts < 1:
        raise ValueError("max_approval_attempts must be a positive integer")
    
    # Validate timeout
    timeout = state.get("approval_timeout", 300)
    if not isinstance(timeout, int) or timeout < 1:
        raise ValueError("approval_timeout must be a positive integer")
    
    # Validate user feedback
    feedback = state.get("user_feedback", {})
    if not isinstance(feedback, dict):
        raise ValueError("user_feedback must be a dictionary")
    
    for key, value in feedback.items():
        if not isinstance(value, str):
            raise ValueError(f"user_feedback[{key}] must be a string")
    
    # Validate workflow pause point
    pause_point = state.get("workflow_paused_at")
    if pause_point is not None:
        valid_pause_points = {'plan_approval', 'outline_approval'}
        if pause_point not in valid_pause_points:
            raise ValueError(f"workflow_paused_at must be one of {valid_pause_points} or None")
    
    return True


def is_approval_enabled(state: Dict[str, Any]) -> bool:
    """
    Check if approval mode is enabled in the current state.
    
    Args:
        state: State dictionary
        
    Returns:
        True if approval mode is enabled, False otherwise
    """
    return state.get("approval_mode", False)


def reset_approval_state(state: Dict[str, Any], approval_type: str = "all") -> Dict[str, Any]:
    """
    Reset approval-related state fields to defaults.
    
    Args:
        state: State dictionary to reset
        approval_type: Type of approval to reset ('plan', 'outline', or 'all')
        
    Returns:
        State with reset approval fields
    """
    if approval_type in ("plan", "all"):
        state["plan_approval_status"] = "pending"
        if "approval_attempt_count" in state:
            state["approval_attempt_count"]["plan"] = 0
        if "user_feedback" in state and "plan" in state["user_feedback"]:
            del state["user_feedback"]["plan"]
    
    if approval_type in ("outline", "all"):
        state["outline_approval_status"] = "pending"
        if "approval_attempt_count" in state:
            state["approval_attempt_count"]["outline"] = 0
        if "user_feedback" in state and "outline" in state["user_feedback"]:
            del state["user_feedback"]["outline"]
    
    if approval_type == "all":
        state["workflow_paused_at"] = None
    
    return state


def get_approval_summary(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get a summary of the current approval state for logging/debugging.
    
    Args:
        state: State dictionary
        
    Returns:
        Summary dictionary with approval information
    """
    return {
        "approval_mode": state.get("approval_mode", False),
        "plan_approval_status": state.get("plan_approval_status", "pending"),
        "outline_approval_status": state.get("outline_approval_status", "pending"),
        "plan_attempts": state.get("approval_attempt_count", {}).get("plan", 0),
        "outline_attempts": state.get("approval_attempt_count", {}).get("outline", 0),
        "max_attempts": state.get("max_approval_attempts", 3),
        "has_plan_feedback": "plan" in state.get("user_feedback", {}),
        "has_outline_feedback": "outline" in state.get("user_feedback", {}),
        "workflow_paused_at": state.get("workflow_paused_at")
    }