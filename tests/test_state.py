"""
Unit tests for enhanced BlogState functionality.
"""

import pytest
from typing import Dict, Any
from state import (
    BlogState,
    initialize_approval_state,
    validate_approval_state,
    is_approval_enabled,
    reset_approval_state,
    get_approval_summary
)


class TestBlogStateEnhancement:
    """Test enhanced BlogState with approval fields."""
    
    def test_empty_state_initialization(self):
        """Test initializing empty state with approval fields."""
        state: Dict[str, Any] = {}
        initialized_state = initialize_approval_state(state)
        
        assert initialized_state["approval_mode"] is False
        assert initialized_state["plan_approval_status"] == "pending"
        assert initialized_state["outline_approval_status"] == "pending"
        assert initialized_state["user_feedback"] == {}
        assert initialized_state["approval_attempt_count"] == {"plan": 0, "outline": 0}
        assert initialized_state["max_approval_attempts"] == 3
        assert initialized_state["approval_timeout"] == 300
        assert initialized_state["workflow_paused_at"] is None
    
    def test_partial_state_initialization(self):
        """Test initializing state that already has some approval fields."""
        state: Dict[str, Any] = {
            "approval_mode": True,
            "plan_approval_status": "approved",
            "user_feedback": {"plan": "test feedback"}
        }
        
        initialized_state = initialize_approval_state(state)
        
        # Should preserve existing values
        assert initialized_state["approval_mode"] is True
        assert initialized_state["plan_approval_status"] == "approved"
        assert initialized_state["user_feedback"] == {"plan": "test feedback"}
        
        # Should add missing values
        assert initialized_state["outline_approval_status"] == "pending"
        assert initialized_state["max_approval_attempts"] == 3
    
    def test_state_validation_valid_state(self):
        """Test validation of valid state."""
        state = {
            "approval_mode": True,
            "plan_approval_status": "approved",
            "outline_approval_status": "pending",
            "user_feedback": {"plan": "test feedback"},
            "approval_attempt_count": {"plan": 1, "outline": 0},
            "max_approval_attempts": 3,
            "approval_timeout": 300,
            "workflow_paused_at": None
        }
        
        assert validate_approval_state(state) is True
    
    def test_state_validation_invalid_approval_mode(self):
        """Test validation fails for invalid approval_mode."""
        state = {"approval_mode": "invalid"}
        
        with pytest.raises(ValueError, match="approval_mode must be a boolean"):
            validate_approval_state(state)
    
    def test_state_validation_invalid_approval_status(self):
        """Test validation fails for invalid approval status."""
        state = {
            "approval_mode": True,
            "plan_approval_status": "invalid_status"
        }
        
        with pytest.raises(ValueError, match="plan_approval_status must be one of"):
            validate_approval_state(state)
    
    def test_state_validation_invalid_attempt_count(self):
        """Test validation fails for invalid attempt count."""
        state = {
            "approval_mode": True,
            "approval_attempt_count": {"plan": -1}
        }
        
        with pytest.raises(ValueError, match="must be a non-negative integer"):
            validate_approval_state(state)
    
    def test_state_validation_invalid_max_attempts(self):
        """Test validation fails for invalid max attempts."""
        state = {
            "approval_mode": True,
            "max_approval_attempts": 0
        }
        
        with pytest.raises(ValueError, match="max_approval_attempts must be a positive integer"):
            validate_approval_state(state)
    
    def test_state_validation_invalid_timeout(self):
        """Test validation fails for invalid timeout."""
        state = {
            "approval_mode": True,
            "approval_timeout": -1
        }
        
        with pytest.raises(ValueError, match="approval_timeout must be a positive integer"):
            validate_approval_state(state)
    
    def test_state_validation_invalid_workflow_pause_point(self):
        """Test validation fails for invalid workflow pause point."""
        state = {
            "approval_mode": True,
            "workflow_paused_at": "invalid_point"
        }
        
        with pytest.raises(ValueError, match="workflow_paused_at must be one of"):
            validate_approval_state(state)
    
    def test_is_approval_enabled(self):
        """Test approval mode detection."""
        # Test enabled
        state_enabled = {"approval_mode": True}
        assert is_approval_enabled(state_enabled) is True
        
        # Test disabled
        state_disabled = {"approval_mode": False}
        assert is_approval_enabled(state_disabled) is False
        
        # Test default (missing field)
        state_default = {}
        assert is_approval_enabled(state_default) is False
    
    def test_reset_approval_state_plan_only(self):
        """Test resetting only plan approval state."""
        state = {
            "plan_approval_status": "approved",
            "outline_approval_status": "rejected",
            "approval_attempt_count": {"plan": 2, "outline": 3},
            "user_feedback": {"plan": "test plan", "outline": "test outline"},
            "workflow_paused_at": "plan_approval"
        }
        
        reset_state = reset_approval_state(state, "plan")
        
        assert reset_state["plan_approval_status"] == "pending"
        assert reset_state["outline_approval_status"] == "rejected"  # Unchanged
        assert reset_state["approval_attempt_count"]["plan"] == 0
        assert reset_state["approval_attempt_count"]["outline"] == 3  # Unchanged
        assert "plan" not in reset_state["user_feedback"]
        assert "outline" in reset_state["user_feedback"]  # Unchanged
        assert reset_state["workflow_paused_at"] == "plan_approval"  # Unchanged for partial reset
    
    def test_reset_approval_state_all(self):
        """Test resetting all approval state."""
        state = {
            "plan_approval_status": "approved",
            "outline_approval_status": "rejected",
            "approval_attempt_count": {"plan": 2, "outline": 3},
            "user_feedback": {"plan": "test plan", "outline": "test outline"},
            "workflow_paused_at": "outline_approval"
        }
        
        reset_state = reset_approval_state(state, "all")
        
        assert reset_state["plan_approval_status"] == "pending"
        assert reset_state["outline_approval_status"] == "pending"
        assert reset_state["approval_attempt_count"]["plan"] == 0
        assert reset_state["approval_attempt_count"]["outline"] == 0
        assert reset_state["user_feedback"] == {}
        assert reset_state["workflow_paused_at"] is None
    
    def test_get_approval_summary(self):
        """Test approval summary generation."""
        state = {
            "approval_mode": True,
            "plan_approval_status": "approved",
            "outline_approval_status": "pending",
            "approval_attempt_count": {"plan": 2, "outline": 1},
            "max_approval_attempts": 3,
            "user_feedback": {"plan": "test feedback"},
            "workflow_paused_at": "outline_approval"
        }
        
        summary = get_approval_summary(state)
        
        expected_summary = {
            "approval_mode": True,
            "plan_approval_status": "approved",
            "outline_approval_status": "pending",
            "plan_attempts": 2,
            "outline_attempts": 1,
            "max_attempts": 3,
            "has_plan_feedback": True,
            "has_outline_feedback": False,
            "workflow_paused_at": "outline_approval"
        }
        
        assert summary == expected_summary
    
    def test_get_approval_summary_defaults(self):
        """Test approval summary with default values."""
        state = {}  # Empty state
        
        summary = get_approval_summary(state)
        
        expected_summary = {
            "approval_mode": False,
            "plan_approval_status": "pending",
            "outline_approval_status": "pending",
            "plan_attempts": 0,
            "outline_attempts": 0,
            "max_attempts": 3,
            "has_plan_feedback": False,
            "has_outline_feedback": False,
            "workflow_paused_at": None
        }
        
        assert summary == expected_summary


class TestBlogStateBackwardCompatibility:
    """Test backward compatibility of enhanced BlogState."""
    
    def test_existing_state_fields_preserved(self):
        """Test that existing state fields are preserved."""
        existing_state = {
            "topic": "Test Topic",
            "llm_provider": "anthropic",
            "model_name": "claude-3-opus",
            "length": "complex",
            "plan": {"target_audience": "developers"},
            "research_docs": ["doc1", "doc2"],
            "outline": ["intro", "body", "conclusion"],
            "sections": {"intro": "content"},
            "draft": "draft content",
            "edited": "edited content",
            "seo_meta": {"title": "Test Title"}
        }
        
        enhanced_state = initialize_approval_state(existing_state.copy())
        
        # All original fields should be preserved
        for key, value in existing_state.items():
            assert enhanced_state[key] == value
        
        # New fields should be added
        assert "approval_mode" in enhanced_state
        assert "plan_approval_status" in enhanced_state
    
    def test_state_can_be_used_without_approval_fields(self):
        """Test that state can function without approval fields."""
        minimal_state = {
            "topic": "Test Topic",
            "plan": {"target_audience": "developers"}
        }
        
        # Should not raise errors when accessing approval fields
        assert is_approval_enabled(minimal_state) is False
        
        # Validation should pass with defaults
        initialized = initialize_approval_state(minimal_state.copy())
        assert validate_approval_state(initialized) is True
    
    def test_state_type_compatibility(self):
        """Test that enhanced state maintains type compatibility."""
        state: BlogState = {
            "topic": "Test Topic",
            "llm_provider": "anthropic",
            "model_name": "claude-3-opus",
            "length": "complex"
        }
        
        # Should be able to add approval fields
        state["approval_mode"] = True
        state["plan_approval_status"] = "pending"
        state["user_feedback"] = {}
        
        # Type checker should not complain
        assert isinstance(state, dict)
        assert state["approval_mode"] is True


# Integration tests with mock data
class TestBlogStateIntegration:
    """Integration tests for state management."""
    
    def create_realistic_state(self) -> Dict[str, Any]:
        """Create a realistic blog state for testing."""
        return {
            "topic": "Introduction to Machine Learning",
            "llm_provider": "anthropic",
            "model_name": "claude-3-opus",
            "length": "complex",
            "plan": {
                "target_audience": "Software developers new to ML",
                "blog_length": 2500,
                "tone": "technical",
                "section_titles": [
                    "Introduction",
                    "Types of ML",
                    "Common Algorithms",
                    "Tools and Frameworks",
                    "Getting Started",
                    "Conclusion"
                ],
                "keywords": ["machine learning", "AI", "algorithms", "Python"]
            },
            "research_docs": [
                "ML is a subset of AI that enables computers to learn",
                "Supervised learning uses labeled training data",
                "Popular frameworks include TensorFlow and PyTorch"
            ],
            "outline": [
                "Introduction to Machine Learning",
                "Types of Machine Learning",
                "Popular ML Algorithms",
                "ML Tools and Frameworks",
                "Building Your First ML Model",
                "Conclusion and Next Steps"
            ]
        }
    
    def test_full_workflow_state_transitions(self):
        """Test state transitions through a full approval workflow."""
        state = self.create_realistic_state()
        
        # Initialize for approval workflow
        state = initialize_approval_state(state)
        state["approval_mode"] = True
        
        # Simulate plan approval process
        state["plan_approval_status"] = "rejected"
        state["approval_attempt_count"]["plan"] = 1
        state["user_feedback"]["plan"] = "Make it more beginner-friendly"
        
        # Validate state
        assert validate_approval_state(state) is True
        
        # Simulate plan revision and approval
        state["plan_approval_status"] = "approved"
        state["approval_attempt_count"]["plan"] = 2
        
        # Simulate outline approval
        state["outline_approval_status"] = "approved"
        state["approval_attempt_count"]["outline"] = 1
        
        # Final validation
        assert validate_approval_state(state) is True
        assert is_approval_enabled(state) is True
        
        summary = get_approval_summary(state)
        assert summary["plan_attempts"] == 2
        assert summary["outline_attempts"] == 1
        assert summary["has_plan_feedback"] is True
        assert summary["has_outline_feedback"] is False
    
    def test_error_recovery_scenarios(self):
        """Test state handling in error scenarios."""
        state = self.create_realistic_state()
        state = initialize_approval_state(state)
        
        # Simulate max attempts exceeded scenario
        state["approval_attempt_count"]["plan"] = 3
        state["max_approval_attempts"] = 3
        state["plan_approval_status"] = "rejected"
        
        # Should still validate
        assert validate_approval_state(state) is True
        
        # Reset after max attempts
        reset_state = reset_approval_state(state, "plan")
        assert reset_state["approval_attempt_count"]["plan"] == 0
        assert reset_state["plan_approval_status"] == "pending"