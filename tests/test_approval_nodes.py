"""
Unit tests for approval node functionality.
"""

import pytest
from unittest.mock import patch, MagicMock
from typing import Dict, Any

# Import the functions we need to test
from agents.approval import (
    plan_approval_node,
    outline_approval_node,
    route_after_plan_approval,
    route_after_outline_approval,
    get_approval_attempt_info,
    should_continue_approval_workflow
)
from state import initialize_approval_state


class TestApprovalNodes:
    """Test approval node functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.test_state = {
            "topic": "Test Topic",
            "plan": {
                "target_audience": "Software developers",
                "blog_length": 2500,
                "tone": "technical",
                "section_titles": ["Intro", "Main", "Conclusion"],
                "keywords": ["test", "python"]
            },
            "outline": ["Introduction", "Main Content", "Conclusion"],
            "approval_mode": True,
            "max_approval_attempts": 3,
            "approval_timeout": 10,  # Short for tests
            "approval_attempt_count": {"plan": 0, "outline": 0},
            "user_feedback": {}
        }
    
    def test_plan_approval_node_disabled_mode(self):
        """Test plan approval node when approval mode is disabled."""
        state = self.test_state.copy()
        state["approval_mode"] = False
        
        result = plan_approval_node(state)
        
        assert result["plan_approval_status"] == "approved"
        assert result["approval_attempt_count"]["plan"] == 0  # Should not increment
    
    @patch('agents.approval.HumanApprovalService')
    def test_plan_approval_node_approved(self, mock_approval_service):
        """Test plan approval node with user approval."""
        # Mock approval service
        mock_service = mock_approval_service.return_value
        mock_service.get_user_approval.return_value = "approved"
        
        state = self.test_state.copy()
        result = plan_approval_node(state)
        
        assert result["plan_approval_status"] == "approved"
        assert result["approval_attempt_count"]["plan"] == 1
        
        # Verify service was called correctly
        mock_service.display_plan_for_review.assert_called_once_with(result)
        mock_service.get_user_approval.assert_called_once_with("plan")
    
    @patch('agents.approval.HumanApprovalService')
    def test_plan_approval_node_rejected_with_feedback(self, mock_approval_service):
        """Test plan approval node with rejection and feedback."""
        # Mock approval service
        mock_service = mock_approval_service.return_value
        mock_service.get_user_approval.return_value = "rejected"
        mock_service.collect_feedback_for_revision.return_value = "Make it more beginner-friendly"
        
        state = self.test_state.copy()
        result = plan_approval_node(state)
        
        assert result["plan_approval_status"] == "rejected"
        assert result["user_feedback"]["plan"] == "Make it more beginner-friendly"
        assert result["approval_attempt_count"]["plan"] == 1
        
        # Verify feedback collection was called
        mock_service.collect_feedback_for_revision.assert_called_once_with("plan")
    
    @patch('agents.approval.HumanApprovalService')
    def test_plan_approval_node_quit(self, mock_approval_service):
        """Test plan approval node with user quit."""
        # Mock approval service
        mock_service = mock_approval_service.return_value
        mock_service.get_user_approval.return_value = "quit"
        
        state = self.test_state.copy()
        
        with pytest.raises(KeyboardInterrupt):
            plan_approval_node(state)
    
    def test_plan_approval_node_max_attempts_exceeded(self):
        """Test plan approval node when max attempts exceeded."""
        state = self.test_state.copy()
        state["approval_attempt_count"]["plan"] = 3  # At max
        
        with pytest.raises(ValueError, match="Maximum plan approval attempts exceeded"):
            plan_approval_node(state)
    
    @patch('agents.approval.HumanApprovalService')
    def test_plan_approval_node_timeout(self, mock_approval_service):
        """Test plan approval node with timeout."""
        # Mock approval service
        mock_service = mock_approval_service.return_value
        mock_service.get_user_approval.return_value = "timeout"
        
        state = self.test_state.copy()
        result = plan_approval_node(state)
        
        # Should default to approved on timeout
        assert result["plan_approval_status"] == "approved"
    
    @patch('agents.approval.HumanApprovalService')
    def test_outline_approval_node_approved(self, mock_approval_service):
        """Test outline approval node with user approval."""
        # Mock approval service
        mock_service = mock_approval_service.return_value
        mock_service.get_user_approval.return_value = "approved"
        
        state = self.test_state.copy()
        result = outline_approval_node(state)
        
        assert result["outline_approval_status"] == "approved"
        assert result["approval_attempt_count"]["outline"] == 1
        
        # Verify service was called correctly
        mock_service.display_outline_for_review.assert_called_once_with(result)
        mock_service.get_user_approval.assert_called_once_with("outline")
    
    @patch('agents.approval.HumanApprovalService')
    def test_outline_approval_node_rejected_with_feedback(self, mock_approval_service):
        """Test outline approval node with rejection and feedback."""
        # Mock approval service
        mock_service = mock_approval_service.return_value
        mock_service.get_user_approval.return_value = "rejected"
        mock_service.collect_feedback_for_revision.return_value = "Add more detail to section 2"
        
        state = self.test_state.copy()
        result = outline_approval_node(state)
        
        assert result["outline_approval_status"] == "rejected"
        assert result["user_feedback"]["outline"] == "Add more detail to section 2"
        assert result["approval_attempt_count"]["outline"] == 1
    
    def test_outline_approval_node_disabled_mode(self):
        """Test outline approval node when approval mode is disabled."""
        state = self.test_state.copy()
        state["approval_mode"] = False
        
        result = outline_approval_node(state)
        
        assert result["outline_approval_status"] == "approved"
        assert result["approval_attempt_count"]["outline"] == 0
    
    @patch('agents.approval.HumanApprovalService')
    def test_outline_approval_node_error_handling(self, mock_approval_service):
        """Test outline approval node error handling."""
        # Mock approval service to raise exception
        mock_service = mock_approval_service.return_value
        mock_service.get_user_approval.side_effect = Exception("Test error")
        
        state = self.test_state.copy()
        result = outline_approval_node(state)
        
        # Should default to approved on error
        assert result["outline_approval_status"] == "approved"
    
    @patch('agents.approval.HumanApprovalService')
    def test_approval_node_no_feedback_provided(self, mock_approval_service):
        """Test approval node when no feedback is provided after rejection."""
        # Mock approval service
        mock_service = mock_approval_service.return_value
        mock_service.get_user_approval.return_value = "rejected"
        mock_service.collect_feedback_for_revision.return_value = ""  # No feedback
        
        state = self.test_state.copy()
        result = plan_approval_node(state)
        
        # Should treat as approved when no feedback provided
        assert result["plan_approval_status"] == "approved"


class TestApprovalRouting:
    """Test routing functions for approval workflow."""
    
    def test_route_after_plan_approval_approved(self):
        """Test routing after plan approval when approved."""
        state = {"plan_approval_status": "approved"}
        result = route_after_plan_approval(state)
        assert result == "research"
    
    def test_route_after_plan_approval_rejected(self):
        """Test routing after plan approval when rejected."""
        state = {"plan_approval_status": "rejected"}
        result = route_after_plan_approval(state)
        assert result == "planner"
    
    def test_route_after_plan_approval_invalid_status(self):
        """Test routing after plan approval with invalid status."""
        state = {"plan_approval_status": "invalid"}
        with pytest.raises(ValueError):
            route_after_plan_approval(state)
    
    def test_route_after_plan_approval_missing_status(self):
        """Test routing after plan approval with missing status."""
        state = {}
        with pytest.raises(ValueError):
            route_after_plan_approval(state)
    
    def test_route_after_outline_approval_approved(self):
        """Test routing after outline approval when approved."""
        state = {"outline_approval_status": "approved"}
        result = route_after_outline_approval(state)
        assert result == "writer"
    
    def test_route_after_outline_approval_rejected(self):
        """Test routing after outline approval when rejected."""
        state = {"outline_approval_status": "rejected"}
        result = route_after_outline_approval(state)
        assert result == "outline"
    
    def test_route_after_outline_approval_invalid_status(self):
        """Test routing after outline approval with invalid status."""
        state = {"outline_approval_status": "invalid"}
        with pytest.raises(ValueError):
            route_after_outline_approval(state)


class TestApprovalUtilities:
    """Test utility functions for approval workflow."""
    
    def test_get_approval_attempt_info_normal(self):
        """Test getting approval attempt info for normal case."""
        state = {
            "approval_attempt_count": {"plan": 2, "outline": 1},
            "max_approval_attempts": 3
        }
        
        result = get_approval_attempt_info(state, "plan")
        expected = {
            "current_attempt": 2,
            "max_attempts": 3,
            "remaining_attempts": 1,
            "is_final_attempt": True,
            "has_exceeded_max": False
        }
        assert result == expected
    
    def test_get_approval_attempt_info_exceeded(self):
        """Test getting approval attempt info when exceeded."""
        state = {
            "approval_attempt_count": {"plan": 3, "outline": 1},
            "max_approval_attempts": 3
        }
        
        result = get_approval_attempt_info(state, "plan")
        assert result["has_exceeded_max"] is True
        assert result["remaining_attempts"] == 0
    
    def test_get_approval_attempt_info_defaults(self):
        """Test getting approval attempt info with default values."""
        state = {}
        
        result = get_approval_attempt_info(state, "plan")
        expected = {
            "current_attempt": 0,
            "max_attempts": 3,
            "remaining_attempts": 3,
            "is_final_attempt": False,
            "has_exceeded_max": False
        }
        assert result == expected
    
    def test_should_continue_approval_workflow_normal(self):
        """Test should continue approval workflow for normal case."""
        state = {
            "approval_mode": True,
            "approval_attempt_count": {"plan": 1, "outline": 1},
            "max_approval_attempts": 3
        }
        
        result = should_continue_approval_workflow(state)
        assert result is True
    
    def test_should_continue_approval_workflow_disabled(self):
        """Test should continue approval workflow when disabled."""
        state = {"approval_mode": False}
        
        result = should_continue_approval_workflow(state)
        assert result is True
    
    def test_should_continue_approval_workflow_exceeded(self):
        """Test should continue approval workflow when exceeded."""
        state = {
            "approval_mode": True,
            "approval_attempt_count": {"plan": 3, "outline": 1},
            "max_approval_attempts": 3
        }
        
        result = should_continue_approval_workflow(state)
        assert result is False
    
    def test_should_continue_approval_workflow_outline_exceeded(self):
        """Test should continue approval workflow when outline exceeded."""
        state = {
            "approval_mode": True,
            "approval_attempt_count": {"plan": 1, "outline": 3},
            "max_approval_attempts": 3
        }
        
        result = should_continue_approval_workflow(state)
        assert result is False


class TestApprovalIntegration:
    """Integration tests for approval workflow."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.complete_state = {
            "topic": "Advanced Machine Learning",
            "llm_provider": "anthropic",
            "model_name": "claude-3-opus",
            "length": "complex",
            "plan": {
                "target_audience": "Data scientists and ML engineers",
                "blog_length": 3500,
                "tone": "technical",
                "section_titles": [
                    "Introduction to Advanced ML",
                    "Deep Learning Fundamentals",
                    "Optimization Techniques",
                    "Model Deployment",
                    "Future Directions",
                    "Conclusion"
                ],
                "keywords": ["machine learning", "deep learning", "neural networks", "optimization"]
            },
            "outline": [
                "Introduction to Advanced ML",
                "Deep Learning Fundamentals",
                "Convolutional Neural Networks",
                "Recurrent Neural Networks",
                "Optimization and Regularization",
                "Production Deployment Strategies",
                "Conclusion and Future Trends"
            ]
        }
    
    def test_complete_approval_workflow_simulation(self):
        """Test complete approval workflow simulation."""
        state = initialize_approval_state(self.complete_state.copy())
        state["approval_mode"] = True
        
        # Test plan approval
        assert state["plan_approval_status"] == "pending"
        
        # Simulate plan rejection
        state["plan_approval_status"] = "rejected"
        state["user_feedback"]["plan"] = "Make it more accessible to beginners"
        state["approval_attempt_count"]["plan"] = 1
        
        # Test routing after rejection
        next_node = route_after_plan_approval(state)
        assert next_node == "planner"
        
        # Simulate plan approval on retry
        state["plan_approval_status"] = "approved"
        state["approval_attempt_count"]["plan"] = 2
        
        next_node = route_after_plan_approval(state)
        assert next_node == "research"
        
        # Test outline approval
        state["outline_approval_status"] = "approved"
        state["approval_attempt_count"]["outline"] = 1
        
        next_node = route_after_outline_approval(state)
        assert next_node == "writer"
        
        # Verify workflow should continue
        assert should_continue_approval_workflow(state) is True
    
    @patch('agents.approval.HumanApprovalService')
    def test_approval_workflow_with_multiple_rejections(self, mock_approval_service):
        """Test approval workflow with multiple rejections."""
        state = initialize_approval_state(self.complete_state.copy())
        state["approval_mode"] = True
        
        # Mock service to simulate multiple rejections
        mock_service = mock_approval_service.return_value
        mock_service.get_user_approval.return_value = "rejected"
        mock_service.collect_feedback_for_revision.side_effect = [
            "First feedback",
            "Second feedback",
            "Third feedback"
        ]
        
        # First rejection
        result1 = plan_approval_node(state)
        assert result1["plan_approval_status"] == "rejected"
        assert result1["approval_attempt_count"]["plan"] == 1
        assert result1["user_feedback"]["plan"] == "First feedback"
        
        # Second rejection
        result2 = plan_approval_node(result1)
        assert result2["plan_approval_status"] == "rejected"
        assert result2["approval_attempt_count"]["plan"] == 2
        assert result2["user_feedback"]["plan"] == "Second feedback"
        
        # Third rejection (should reach max)
        result3 = plan_approval_node(result2)
        assert result3["plan_approval_status"] == "rejected"
        assert result3["approval_attempt_count"]["plan"] == 3
        assert result3["user_feedback"]["plan"] == "Third feedback"
        
        # Fourth attempt should raise error
        with pytest.raises(ValueError, match="Maximum plan approval attempts exceeded"):
            plan_approval_node(result3)
    
    def test_state_persistence_through_approval(self):
        """Test that state is properly maintained through approval process."""
        original_state = initialize_approval_state(self.complete_state.copy())
        original_state["approval_mode"] = True
        
        # Simulate plan approval with feedback
        state = original_state.copy()
        state["plan_approval_status"] = "rejected"
        state["user_feedback"]["plan"] = "Test feedback"
        state["approval_attempt_count"]["plan"] = 1
        
        # Verify original fields are preserved
        assert state["topic"] == original_state["topic"]
        assert state["plan"] == original_state["plan"]
        assert state["outline"] == original_state["outline"]
        
        # Verify approval fields are correctly set
        assert state["user_feedback"]["plan"] == "Test feedback"
        assert state["approval_attempt_count"]["plan"] == 1
        
        # Verify can continue workflow
        assert should_continue_approval_workflow(state) is True
    
    def test_error_recovery_scenarios(self):
        """Test error recovery in approval scenarios."""
        state = initialize_approval_state(self.complete_state.copy())
        state["approval_mode"] = True
        
        # Test with malformed state
        broken_state = {"approval_mode": True}  # Missing required fields
        
        # Should handle gracefully and initialize
        fixed_state = initialize_approval_state(broken_state)
        assert "approval_attempt_count" in fixed_state
        assert "max_approval_attempts" in fixed_state
        
        # Should be able to get attempt info
        attempt_info = get_approval_attempt_info(fixed_state, "plan")
        assert attempt_info["current_attempt"] == 0