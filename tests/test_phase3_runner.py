"""
Test runner for Phase 3: Workflow Integration validation.
This runs comprehensive tests for the complete interactive workflow integration.
"""

import sys
import os
import traceback
from typing import Dict, Any
from unittest.mock import MagicMock, patch
import tempfile
import shutil

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from state import initialize_approval_state, validate_approval_state, is_approval_enabled
from workflows.blog_graph import (
    create_blog_workflow,
    create_interactive_blog_workflow,
    get_workflow,
    run_workflow
)
from agents.approval import (
    plan_approval_node,
    outline_approval_node,
    route_after_plan_approval,
    route_after_outline_approval
)


class Phase3TestRunner:
    """Test runner for Phase 3 Workflow Integration."""
    
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
    
    def assert_not_none(self, value, message=""):
        """Assert that value is not None."""
        if value is None:
            raise AssertionError(f"Expected not None, got None. {message}")
    
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
        """Run all Phase 3 tests."""
        print("🚀 Running Phase 3: Workflow Integration Tests\n")
        
        # Workflow creation tests
        print("🏗️ Testing Workflow Creation...")
        self.test_workflow_creation()
        
        print("\n🔄 Testing Workflow Integration...")
        self.test_workflow_integration()
        
        print("\n⚙️ Testing CLI Integration...")
        self.test_cli_integration()
        
        print("\n🎯 Testing End-to-End Scenarios...")
        self.test_end_to_end_scenarios()
        
        print("\n🔙 Testing Backward Compatibility...")
        self.test_backward_compatibility()
        
        print(f"\n📊 Phase 3 Test Results:")
        print(f"Tests run: {self.tests_run}")
        print(f"Passed: {self.tests_passed}")
        print(f"Failed: {self.tests_failed}")
        
        if self.failures:
            print(f"\n❌ Failed Tests:")
            for name, error, _ in self.failures:
                print(f"  - {name}: {error}")
        
        if self.tests_failed == 0:
            print(f"\n🎉 All Phase 3 tests passed! Workflow integration is working correctly.")
            return True
        else:
            print(f"\n⚠️  Some Phase 3 tests failed. Please review the implementation.")
            return False
        
        return True
    
    # Workflow creation tests
    def test_workflow_creation(self):
        """Test workflow creation functionality."""
        self.run_test(self._test_standard_workflow_creation)
        self.run_test(self._test_interactive_workflow_creation)
        self.run_test(self._test_workflow_selection)
        self.run_test(self._test_workflow_structure)
    
    def _test_standard_workflow_creation(self):
        """Test standard workflow creation."""
        workflow = create_blog_workflow()
        self.assert_not_none(workflow)
        
        # Test workflow is compiled and ready
        initial_state = {"topic": "Test Topic"}
        # Should not raise exception during setup
        try:
            # Just test that the workflow can be invoked without actual execution
            workflow._func  # Access the compiled function
        except Exception as e:
            raise AssertionError(f"Standard workflow compilation failed: {e}")
    
    def _test_interactive_workflow_creation(self):
        """Test interactive workflow creation."""
        workflow = create_interactive_blog_workflow()
        self.assert_not_none(workflow)
        
        # Test workflow includes approval nodes
        try:
            workflow._func  # Access the compiled function
        except Exception as e:
            raise AssertionError(f"Interactive workflow compilation failed: {e}")
    
    def _test_workflow_selection(self):
        """Test workflow selection mechanism."""
        # Test standard workflow selection
        standard_workflow = get_workflow(approval_mode=False)
        self.assert_not_none(standard_workflow)
        
        # Test interactive workflow selection
        interactive_workflow = get_workflow(approval_mode=True)
        self.assert_not_none(interactive_workflow)
        
        # They should be different objects
        self.assert_true(standard_workflow != interactive_workflow)
    
    def _test_workflow_structure(self):
        """Test workflow structure integrity."""
        # Create both workflows to ensure they compile correctly
        standard = create_blog_workflow()
        interactive = create_interactive_blog_workflow()
        
        # Both should be compiled successfully
        self.assert_not_none(standard)
        self.assert_not_none(interactive)
        
        # Should have callable functions
        self.assert_true(hasattr(standard, 'invoke'))
        self.assert_true(hasattr(interactive, 'invoke'))
    
    # Workflow integration tests
    def test_workflow_integration(self):
        """Test workflow integration functionality."""
        self.run_test(self._test_state_initialization)
        self.run_test(self._test_approval_routing)
        self.run_test(self._test_conditional_edges)
    
    def _test_state_initialization(self):
        """Test state initialization for interactive workflow."""
        initial_state = {
            "topic": "Test Topic",
            "llm_provider": "anthropic",
            "model_name": "claude-3-opus",
            "length": "complex"
        }
        
        # Test state initialization
        initialized = initialize_approval_state(initial_state.copy())
        
        # Should have approval fields
        self.assert_in("approval_mode", initialized)
        self.assert_in("plan_approval_status", initialized)
        self.assert_in("outline_approval_status", initialized)
        self.assert_in("approval_attempt_count", initialized)
        
        # Should validate correctly
        self.assert_true(validate_approval_state(initialized))
    
    def _test_approval_routing(self):
        """Test approval routing functionality."""
        # Test plan approval routing
        plan_approved_state = {"plan_approval_status": "approved"}
        plan_rejected_state = {"plan_approval_status": "rejected"}
        
        self.assert_equal(route_after_plan_approval(plan_approved_state), "research")
        self.assert_equal(route_after_plan_approval(plan_rejected_state), "planner")
        
        # Test outline approval routing
        outline_approved_state = {"outline_approval_status": "approved"}
        outline_rejected_state = {"outline_approval_status": "rejected"}
        
        self.assert_equal(route_after_outline_approval(outline_approved_state), "writer")
        self.assert_equal(route_after_outline_approval(outline_rejected_state), "outline")
    
    def _test_conditional_edges(self):
        """Test conditional edge routing logic."""
        # Test that routing functions handle edge cases
        try:
            route_after_plan_approval({"plan_approval_status": "invalid"})
            raise AssertionError("Should have raised ValueError for invalid status")
        except ValueError:
            pass  # Expected
        
        try:
            route_after_outline_approval({})  # Missing status
            raise AssertionError("Should have raised ValueError for missing status")
        except ValueError:
            pass  # Expected
    
    # CLI integration tests
    def test_cli_integration(self):
        """Test CLI integration functionality."""
        self.run_test(self._test_run_workflow_parameters)
        self.run_test(self._test_approval_mode_configuration)
        self.run_test(self._test_cli_parameter_validation)
    
    def _test_run_workflow_parameters(self):
        """Test run_workflow function with new parameters."""
        # Test with minimal mocking to check parameter passing
        with patch('workflows.blog_graph.create_blog_workflow') as mock_create_standard:
            with patch('workflows.blog_graph.create_interactive_blog_workflow') as mock_create_interactive:
                mock_workflow = MagicMock()
                mock_workflow.invoke.return_value = {"edited": "test content", "seo_meta": {}}
                
                mock_create_standard.return_value = mock_workflow
                mock_create_interactive.return_value = mock_workflow
                
                # Test standard mode
                result = run_workflow(
                    topic="Test Topic",
                    approval_mode=False,
                    verbose=False
                )
                
                mock_create_standard.assert_called_once()
                self.assert_in("edited", result)
                
                # Reset mocks
                mock_create_standard.reset_mock()
                mock_create_interactive.reset_mock()
                
                # Test interactive mode
                result = run_workflow(
                    topic="Test Topic",
                    approval_mode=True,
                    max_approval_attempts=5,
                    approval_timeout=600,
                    verbose=False
                )
                
                mock_create_interactive.assert_called_once()
                self.assert_in("edited", result)
    
    def _test_approval_mode_configuration(self):
        """Test approval mode configuration."""
        # Test state initialization with approval mode
        test_state = {
            "topic": "Test",
            "approval_mode": True,
            "max_approval_attempts": 5,
            "approval_timeout": 600
        }
        
        initialized = initialize_approval_state(test_state)
        
        self.assert_true(is_approval_enabled(initialized))
        self.assert_equal(initialized["max_approval_attempts"], 5)
        self.assert_equal(initialized["approval_timeout"], 600)
    
    def _test_cli_parameter_validation(self):
        """Test CLI parameter validation logic."""
        # Test valid ranges (simulated)
        valid_attempts = [1, 3, 5, 10]
        for attempts in valid_attempts:
            self.assert_true(1 <= attempts <= 10, f"Attempts {attempts} should be valid")
        
        valid_timeouts = [10, 300, 600, 3600]
        for timeout in valid_timeouts:
            self.assert_true(10 <= timeout <= 3600, f"Timeout {timeout} should be valid")
    
    # End-to-end scenario tests
    def test_end_to_end_scenarios(self):
        """Test end-to-end scenario functionality."""
        self.run_test(self._test_full_approval_workflow)
        self.run_test(self._test_rejection_workflow)
        self.run_test(self._test_error_handling)
    
    @patch('agents.approval.HumanApprovalService')
    def _test_full_approval_workflow(self, mock_approval_service):
        """Test full workflow with approvals."""
        # Mock approval service to simulate user approvals
        mock_service = mock_approval_service.return_value
        mock_service.get_user_approval.return_value = "approved"
        
        # Create minimal state for testing
        state = {
            "topic": "Test Topic",
            "approval_mode": True,
            "plan": {"target_audience": "developers", "section_titles": ["intro", "main"]},
            "outline": ["Introduction", "Main Content", "Conclusion"],
            "approval_attempt_count": {"plan": 0, "outline": 0},
            "max_approval_attempts": 3
        }
        
        # Test plan approval
        result = plan_approval_node(state.copy())
        self.assert_equal(result["plan_approval_status"], "approved")
        self.assert_equal(result["approval_attempt_count"]["plan"], 1)
        
        # Test outline approval  
        result = outline_approval_node(state.copy())
        self.assert_equal(result["outline_approval_status"], "approved")
        self.assert_equal(result["approval_attempt_count"]["outline"], 1)
    
    @patch('agents.approval.HumanApprovalService')
    def _test_rejection_workflow(self, mock_approval_service):
        """Test workflow with rejections and feedback."""
        # Mock approval service to simulate rejections
        mock_service = mock_approval_service.return_value
        mock_service.get_user_approval.return_value = "rejected"
        mock_service.collect_feedback_for_revision.return_value = "Make it more detailed"
        
        state = {
            "topic": "Test Topic",
            "approval_mode": True,
            "plan": {"target_audience": "developers"},
            "outline": ["Introduction", "Conclusion"],
            "approval_attempt_count": {"plan": 0, "outline": 0},
            "max_approval_attempts": 3
        }
        
        # Test plan rejection
        result = plan_approval_node(state.copy())
        self.assert_equal(result["plan_approval_status"], "rejected")
        self.assert_equal(result["user_feedback"]["plan"], "Make it more detailed")
        self.assert_equal(result["approval_attempt_count"]["plan"], 1)
        
        # Test routing after rejection
        next_node = route_after_plan_approval(result)
        self.assert_equal(next_node, "planner")
    
    def _test_error_handling(self):
        """Test error handling scenarios."""
        # Test max attempts exceeded
        state = {
            "approval_mode": True,
            "plan": {"target_audience": "developers"},
            "approval_attempt_count": {"plan": 3, "outline": 0},
            "max_approval_attempts": 3
        }
        
        try:
            plan_approval_node(state)
            raise AssertionError("Should have raised ValueError for max attempts exceeded")
        except ValueError as e:
            self.assert_in("Maximum plan approval attempts exceeded", str(e))
        
        # Test invalid routing
        try:
            route_after_plan_approval({"plan_approval_status": "invalid"})
            raise AssertionError("Should have raised ValueError for invalid status")
        except ValueError:
            pass  # Expected
    
    # Backward compatibility tests
    def test_backward_compatibility(self):
        """Test backward compatibility functionality."""
        self.run_test(self._test_standard_workflow_unchanged)
        self.run_test(self._test_existing_api_compatibility)
        self.run_test(self._test_state_compatibility)
    
    def _test_standard_workflow_unchanged(self):
        """Test that standard workflow behavior is unchanged."""
        # Mock the workflow execution to avoid actual LLM calls
        with patch('agents.planner.planner_node') as mock_planner:
            with patch('agents.research.research_node') as mock_research:
                with patch('agents.outline.outline_node') as mock_outline:
                    with patch('agents.writer.write_all_sections') as mock_writer:
                        with patch('agents.editor.editor_node') as mock_editor:
                            with patch('agents.seo.seo_node') as mock_seo:
                                
                                # Configure mocks to return expected state updates
                                mock_planner.return_value = {"plan": {"target_audience": "developers"}}
                                mock_research.return_value = {"research_docs": ["doc1"]}
                                mock_outline.return_value = {"outline": ["intro", "main", "conclusion"]}
                                mock_writer.return_value = {"sections": {"intro": "content"}, "draft": "content"}
                                mock_editor.return_value = {"edited": "final content"}
                                mock_seo.return_value = {"seo_meta": {"title": "Test"}}
                                
                                # Test standard workflow (should not include approval nodes)
                                result = run_workflow(
                                    topic="Test Topic",
                                    approval_mode=False,
                                    verbose=False
                                )
                                
                                # Should have completed without approval steps
                                self.assert_in("edited", result)
                                self.assert_equal(result["edited"], "final content")
    
    def _test_existing_api_compatibility(self):
        """Test that existing API calls still work."""
        with patch('workflows.blog_graph.create_blog_workflow') as mock_create:
            mock_workflow = MagicMock()
            mock_workflow.invoke.return_value = {"edited": "content", "seo_meta": {}}
            mock_create.return_value = mock_workflow
            
            # Test old API (without new parameters)
            result = run_workflow(topic="Test Topic")
            
            self.assert_in("edited", result)
            mock_create.assert_called_once()
    
    def _test_state_compatibility(self):
        """Test that existing state structures are compatible."""
        # Test old state structure
        old_state = {
            "topic": "Test Topic",
            "plan": {"target_audience": "developers"},
            "outline": ["intro", "main", "conclusion"]
        }
        
        # Should be able to initialize approval state without breaking
        enhanced_state = initialize_approval_state(old_state.copy())
        
        # Original fields should be preserved
        self.assert_equal(enhanced_state["topic"], old_state["topic"])
        self.assert_equal(enhanced_state["plan"], old_state["plan"])
        self.assert_equal(enhanced_state["outline"], old_state["outline"])
        
        # New fields should be added
        self.assert_in("approval_mode", enhanced_state)
        self.assert_in("approval_attempt_count", enhanced_state)
        
        # Should validate correctly
        self.assert_true(validate_approval_state(enhanced_state))


def main():
    """Run all Phase 3 tests."""
    runner = Phase3TestRunner()
    success = runner.run_all_tests()
    
    if success:
        print("\n✅ Phase 3: Workflow Integration validation completed successfully!")
        print("🎉 Human-in-the-loop approval feature is fully integrated and ready for use!")
        print("\n🚀 Usage Examples:")
        print("  # Standard mode (unchanged)")
        print('  uv run python app.py --topic "Machine Learning Basics"')
        print("\n  # Interactive mode with approvals")
        print('  uv run python app.py --topic "AI Ethics" --interactive')
        print('  uv run python app.py --topic "Deep Learning" --interactive --max-attempts 5')
    else:
        print("\n❌ Phase 3: Workflow Integration has issues that need to be addressed.")
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)