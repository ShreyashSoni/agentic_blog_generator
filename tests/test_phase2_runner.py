"""
Test runner for Phase 2: Agent Integration validation.
This runs comprehensive tests for approval nodes, enhanced agents, and routing.
"""

import sys
import os
import traceback
from typing import Dict, Any
from unittest.mock import MagicMock, patch

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from state import initialize_approval_state, validate_approval_state
from agents.approval import (
    plan_approval_node,
    outline_approval_node,
    route_after_plan_approval,
    route_after_outline_approval,
    get_approval_attempt_info,
    should_continue_approval_workflow
)
from agents.planner import planner_node, _enhance_prompt_with_feedback, _validate_plan_against_feedback
from agents.outline import (
    outline_node, 
    _build_outline_prompt, 
    _validate_outline_against_feedback,
    _get_outline_revision_context
)


class Phase2TestRunner:
    """Test runner for Phase 2 Agent Integration."""
    
    def __init__(self):
        self.tests_run = 0
        self.tests_passed = 0
        self.tests_failed = 0
        self.failures = []
    
    def assert_equal(self, actual, expected, message=""):
        """Assert that actual equals expected."""
        if actual != expected:
            raise AssertionError(f"Expected {expected}, got {actual}. {message}")
    
    def assert_true(self, condition, message=""):
        """Assert that condition is True."""
        if not condition:
            raise AssertionError(f"Expected True, got {condition}. {message}")
    
    def assert_false(self, condition, message=""):
        """Assert that condition is False."""
        if condition:
            raise AssertionError(f"Expected False, got {condition}. {message}")
    
    def assert_in(self, item, container, message=""):
        """Assert that item is in container."""
        if item not in container:
            raise AssertionError(f"Expected {item} to be in {container}. {message}")
    
    def assert_raises(self, exception_type, func, *args, **kwargs):
        """Assert that function raises specific exception."""
        try:
            func(*args, **kwargs)
            raise AssertionError(f"Expected {exception_type.__name__} to be raised")
        except exception_type:
            pass  # Expected exception
        except Exception as e:
            raise AssertionError(f"Expected {exception_type.__name__}, got {type(e).__name__}: {e}")
    
    def run_test(self, test_func):
        """Run a single test function."""
        test_name = test_func.__name__
        self.tests_run += 1
        
        try:
            test_func()
            print(f"✅ {test_name}")
            self.tests_passed += 1
        except Exception as e:
            print(f"❌ {test_name}: {e}")
            self.failures.append((test_name, str(e), traceback.format_exc()))
            self.tests_failed += 1
    
    def run_all_tests(self):
        """Run all Phase 2 tests."""
        print("🚀 Running Phase 2: Agent Integration Tests\n")
        
        # Approval node tests
        print("📋 Testing Approval Nodes...")
        self.test_approval_nodes()
        
        print("\n🔄 Testing Routing Functions...")
        self.test_routing_functions()
        
        print("\n🤖 Testing Enhanced Agent Integration...")
        self.test_enhanced_agents()
        
        print("\n🔧 Testing Utility Functions...")
        self.test_utility_functions()
        
        print("\n🎯 Testing Integration Scenarios...")
        self.test_integration_scenarios()
        
        print(f"\n📊 Phase 2 Test Results:")
        print(f"Tests run: {self.tests_run}")
        print(f"Passed: {self.tests_passed}")
        print(f"Failed: {self.tests_failed}")
        
        if self.failures:
            print(f"\n❌ Failed Tests:")
            for name, error, _ in self.failures:
                print(f"  - {name}: {error}")
        
        if self.tests_failed == 0:
            print(f"\n🎉 All Phase 2 tests passed! Agent integration is working correctly.")
            return True
        else:
            print(f"\n⚠️  Some Phase 2 tests failed. Please review the implementation.")
            return False
        
        return True
    
    # Approval node tests
    def test_approval_nodes(self):
        """Test approval node functionality."""
        self.run_test(self._test_plan_approval_disabled)
        self.run_test(self._test_plan_approval_max_attempts)
        self.run_test(self._test_outline_approval_disabled)
        self.run_test(self._test_outline_approval_max_attempts)
        self.run_test(self._test_approval_attempt_counting)
    
    def _test_plan_approval_disabled(self):
        """Test plan approval when disabled."""
        state = {
            "approval_mode": False,
            "plan": {"target_audience": "developers"},
            "approval_attempt_count": {"plan": 0, "outline": 0}
        }
        
        result = plan_approval_node(state)
        
        self.assert_equal(result["plan_approval_status"], "approved")
        self.assert_equal(result["approval_attempt_count"]["plan"], 0)
    
    def _test_plan_approval_max_attempts(self):
        """Test plan approval with max attempts exceeded."""
        state = {
            "approval_mode": True,
            "plan": {"target_audience": "developers"},
            "approval_attempt_count": {"plan": 3, "outline": 0},
            "max_approval_attempts": 3
        }
        
        self.assert_raises(ValueError, plan_approval_node, state)
    
    def _test_outline_approval_disabled(self):
        """Test outline approval when disabled."""
        state = {
            "approval_mode": False,
            "outline": ["intro", "body", "conclusion"],
            "approval_attempt_count": {"plan": 0, "outline": 0}
        }
        
        result = outline_approval_node(state)
        
        self.assert_equal(result["outline_approval_status"], "approved")
        self.assert_equal(result["approval_attempt_count"]["outline"], 0)
    
    def _test_outline_approval_max_attempts(self):
        """Test outline approval with max attempts exceeded."""
        state = {
            "approval_mode": True,
            "outline": ["intro", "body", "conclusion"],
            "approval_attempt_count": {"plan": 0, "outline": 3},
            "max_approval_attempts": 3
        }
        
        self.assert_raises(ValueError, outline_approval_node, state)
    
    def _test_approval_attempt_counting(self):
        """Test that approval attempts are counted correctly."""
        state = initialize_approval_state({})
        state["approval_mode"] = True
        state["plan"] = {"target_audience": "developers"}
        
        # Mock approval service to simulate rejection
        with patch('agents.approval.HumanApprovalService') as mock_service:
            mock_service.return_value.get_user_approval.return_value = "rejected"
            mock_service.return_value.collect_feedback_for_revision.return_value = "test feedback"
            
            result = plan_approval_node(state)
            
            self.assert_equal(result["approval_attempt_count"]["plan"], 1)
            self.assert_equal(result["plan_approval_status"], "rejected")
            self.assert_equal(result["user_feedback"]["plan"], "test feedback")
    
    # Routing function tests
    def test_routing_functions(self):
        """Test routing function functionality."""
        self.run_test(self._test_plan_routing_approved)
        self.run_test(self._test_plan_routing_rejected)
        self.run_test(self._test_plan_routing_invalid)
        self.run_test(self._test_outline_routing_approved)
        self.run_test(self._test_outline_routing_rejected)
        self.run_test(self._test_outline_routing_invalid)
    
    def _test_plan_routing_approved(self):
        """Test plan routing when approved."""
        state = {"plan_approval_status": "approved"}
        result = route_after_plan_approval(state)
        self.assert_equal(result, "research")
    
    def _test_plan_routing_rejected(self):
        """Test plan routing when rejected."""
        state = {"plan_approval_status": "rejected"}
        result = route_after_plan_approval(state)
        self.assert_equal(result, "planner")
    
    def _test_plan_routing_invalid(self):
        """Test plan routing with invalid status."""
        state = {"plan_approval_status": "invalid"}
        self.assert_raises(ValueError, route_after_plan_approval, state)
    
    def _test_outline_routing_approved(self):
        """Test outline routing when approved."""
        state = {"outline_approval_status": "approved"}
        result = route_after_outline_approval(state)
        self.assert_equal(result, "writer")
    
    def _test_outline_routing_rejected(self):
        """Test outline routing when rejected."""
        state = {"outline_approval_status": "rejected"}
        result = route_after_outline_approval(state)
        self.assert_equal(result, "outline")
    
    def _test_outline_routing_invalid(self):
        """Test outline routing with invalid status."""
        state = {"outline_approval_status": "invalid"}
        self.assert_raises(ValueError, route_after_outline_approval, state)
    
    # Enhanced agent tests
    def test_enhanced_agents(self):
        """Test enhanced agent functionality."""
        self.run_test(self._test_planner_feedback_integration)
        self.run_test(self._test_outline_feedback_integration)
        self.run_test(self._test_prompt_enhancement)
        self.run_test(self._test_feedback_validation)
    
    def _test_planner_feedback_integration(self):
        """Test planner agent feedback integration."""
        feedback = "Make it more beginner-friendly and add practical examples"
        base_prompt = "Create a plan for {topic} with {length} complexity."
        
        enhanced = _enhance_prompt_with_feedback(base_prompt, feedback, 2)
        
        self.assert_in("revision attempt 2", enhanced)
        self.assert_in(feedback, enhanced)
        self.assert_in("IMPORTANT REVISION INSTRUCTIONS", enhanced)
    
    def _test_outline_feedback_integration(self):
        """Test outline agent feedback integration."""
        topic = "Machine Learning"
        plan = {"target_audience": "developers", "section_titles": ["intro", "main", "conclusion"]}
        research_docs = ["ML is powerful", "Python is popular"]
        feedback = "Add more detail to the technical sections"
        
        enhanced_prompt = _build_outline_prompt(topic, plan, research_docs, feedback, 1)
        
        self.assert_in("revision attempt 1", enhanced_prompt)
        self.assert_in(feedback, enhanced_prompt)
        self.assert_in("IMPORTANT REVISION INSTRUCTIONS", enhanced_prompt)
    
    def _test_prompt_enhancement(self):
        """Test prompt enhancement with feedback."""
        base_template = "Generate content about {topic}. Output ONLY valid JSON, no additional text."
        feedback = "Make it more technical"
        
        enhanced = _enhance_prompt_with_feedback(base_template, feedback, 1)
        
        self.assert_in("IMPORTANT REVISION INSTRUCTIONS", enhanced)
        self.assert_in("revision attempt 1", enhanced)
        self.assert_in(feedback, enhanced)
    
    def _test_feedback_validation(self):
        """Test feedback validation functions."""
        # Test plan validation
        plan = {"target_audience": "beginners", "tone": "casual", "keywords": ["python", "tutorial"]}
        feedback = "make it more beginner friendly with python examples"
        
        result = _validate_plan_against_feedback(plan, feedback)
        self.assert_true(result)  # Should find matching keywords
        
        # Test outline validation
        outline = ["Introduction", "Python Basics", "Advanced Topics", "Conclusion"]
        outline_feedback = "add more python examples"
        
        outline_result = _validate_outline_against_feedback(outline, outline_feedback)
        self.assert_true(outline_result)  # Should find "python" keyword
    
    # Utility function tests
    def test_utility_functions(self):
        """Test utility function functionality."""
        self.run_test(self._test_attempt_info_normal)
        self.run_test(self._test_attempt_info_exceeded)
        self.run_test(self._test_workflow_continuation)
        self.run_test(self._test_revision_context)
    
    def _test_attempt_info_normal(self):
        """Test attempt info for normal case."""
        state = {
            "approval_attempt_count": {"plan": 2, "outline": 1},
            "max_approval_attempts": 3
        }
        
        result = get_approval_attempt_info(state, "plan")
        
        self.assert_equal(result["current_attempt"], 2)
        self.assert_equal(result["max_attempts"], 3)
        self.assert_equal(result["remaining_attempts"], 1)
        self.assert_true(result["is_final_attempt"])
        self.assert_false(result["has_exceeded_max"])
    
    def _test_attempt_info_exceeded(self):
        """Test attempt info when exceeded."""
        state = {
            "approval_attempt_count": {"plan": 3, "outline": 1},
            "max_approval_attempts": 3
        }
        
        result = get_approval_attempt_info(state, "plan")
        
        self.assert_true(result["has_exceeded_max"])
        self.assert_equal(result["remaining_attempts"], 0)
    
    def _test_workflow_continuation(self):
        """Test workflow continuation logic."""
        # Normal case
        state = {
            "approval_mode": True,
            "approval_attempt_count": {"plan": 1, "outline": 1},
            "max_approval_attempts": 3
        }
        
        self.assert_true(should_continue_approval_workflow(state))
        
        # Exceeded case
        state["approval_attempt_count"]["plan"] = 3
        self.assert_false(should_continue_approval_workflow(state))
        
        # Disabled case
        state["approval_mode"] = False
        self.assert_true(should_continue_approval_workflow(state))
    
    def _test_revision_context(self):
        """Test revision context generation."""
        feedback = "Make it more beginner-friendly"
        
        # Initial context
        context1 = _get_outline_revision_context(0, feedback)
        self.assert_in("Initial", context1)
        
        # Revision context
        context2 = _get_outline_revision_context(2, feedback)
        self.assert_in("Revision attempt 2", context2)
        self.assert_in("Make it more", context2)
    
    # Integration scenario tests
    def test_integration_scenarios(self):
        """Test integration scenarios."""
        self.run_test(self._test_complete_approval_flow)
        self.run_test(self._test_multiple_rejections)
        self.run_test(self._test_state_persistence)
    
    def _test_complete_approval_flow(self):
        """Test complete approval flow simulation."""
        # Initialize state
        state = initialize_approval_state({
            "topic": "Python Testing",
            "plan": {"target_audience": "developers", "section_titles": ["intro", "testing", "conclusion"]},
            "outline": ["Introduction", "Unit Testing", "Integration Testing", "Conclusion"]
        })
        state["approval_mode"] = True
        
        # Test plan approval flow
        state["plan_approval_status"] = "rejected"
        state["user_feedback"]["plan"] = "Make it more detailed"
        state["approval_attempt_count"]["plan"] = 1
        
        # Route should go back to planner
        next_node = route_after_plan_approval(state)
        self.assert_equal(next_node, "planner")
        
        # Simulate plan approval
        state["plan_approval_status"] = "approved"
        next_node = route_after_plan_approval(state)
        self.assert_equal(next_node, "research")
        
        # Test outline approval flow
        state["outline_approval_status"] = "approved"
        state["approval_attempt_count"]["outline"] = 1
        next_node = route_after_outline_approval(state)
        self.assert_equal(next_node, "writer")
        
        # Verify workflow can continue
        self.assert_true(should_continue_approval_workflow(state))
    
    def _test_multiple_rejections(self):
        """Test multiple rejections scenario."""
        state = initialize_approval_state({
            "plan": {"target_audience": "developers"},
            "approval_mode": True
        })
        
        # Simulate multiple plan rejections
        feedbacks = ["Add more examples", "Make it shorter", "Change audience"]
        
        for i, feedback in enumerate(feedbacks):
            state["plan_approval_status"] = "rejected"
            state["user_feedback"]["plan"] = feedback
            state["approval_attempt_count"]["plan"] = i + 1
            
            if i < 2:  # Within limits
                self.assert_true(should_continue_approval_workflow(state))
            else:  # At limit
                # Would exceed on next attempt
                state["approval_attempt_count"]["plan"] = 3
                self.assert_false(should_continue_approval_workflow(state))
    
    def _test_state_persistence(self):
        """Test state persistence through approval."""
        original = {
            "topic": "AI Ethics",
            "plan": {"target_audience": "ethicists", "tone": "academic"},
            "outline": ["Ethics 101", "AI Challenges", "Solutions"],
            "custom_field": "preserved"
        }
        
        state = initialize_approval_state(original.copy())
        
        # Simulate approval process
        state["plan_approval_status"] = "rejected"
        state["user_feedback"]["plan"] = "Add practical examples"
        state["approval_attempt_count"]["plan"] = 1
        
        # Verify original fields preserved
        self.assert_equal(state["topic"], original["topic"])
        self.assert_equal(state["plan"], original["plan"])
        self.assert_equal(state["outline"], original["outline"])
        self.assert_equal(state["custom_field"], original["custom_field"])
        
        # Verify approval fields added
        self.assert_equal(state["user_feedback"]["plan"], "Add practical examples")
        self.assert_equal(state["approval_attempt_count"]["plan"], 1)


def main():
    """Run all Phase 2 tests."""
    runner = Phase2TestRunner()
    success = runner.run_all_tests()
    
    if success:
        print("\n✅ Phase 2: Agent Integration validation completed successfully!")
        print("Ready to proceed to Phase 3: Workflow Integration")
    else:
        print("\n❌ Phase 2: Agent Integration has issues that need to be addressed.")
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)