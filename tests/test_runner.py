"""
Simple test runner for Phase 1 implementation validation.
This runs basic tests without requiring pytest installation.
"""

import sys
import os
import traceback
from typing import Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from state import (
    initialize_approval_state,
    validate_approval_state,
    is_approval_enabled,
    reset_approval_state,
    get_approval_summary
)

from services.approval_service import (
    validate_feedback,
    sanitize_feedback,
    format_attempt_info,
    get_approval_prompt_text
)


class SimpleTestRunner:
    """Basic test runner without external dependencies."""
    
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
        """Run all test methods."""
        print("🚀 Running Phase 1 Implementation Tests\n")
        
        # State tests
        print("📋 Testing Enhanced BlogState...")
        self.test_state_initialization()
        self.test_state_validation()
        self.test_approval_mode_detection()
        self.test_state_reset()
        self.test_approval_summary()
        self.test_backward_compatibility()
        
        print("\n🔧 Testing Approval Service Utilities...")
        self.test_feedback_validation()
        self.test_feedback_sanitization()
        self.test_utility_functions()
        
        print(f"\n📊 Test Results:")
        print(f"Tests run: {self.tests_run}")
        print(f"Passed: {self.tests_passed}")
        print(f"Failed: {self.tests_failed}")
        
        if self.failures:
            print(f"\n❌ Failed Tests:")
            for name, error, traceback_str in self.failures:
                print(f"  - {name}: {error}")
        
        if self.tests_failed == 0:
            print(f"\n🎉 All tests passed! Phase 1 implementation is working correctly.")
            return True
        else:
            print(f"\n⚠️  Some tests failed. Please review the implementation.")
            return False
    
    # State tests
    def test_state_initialization(self):
        """Test state initialization with approval fields."""
        self.run_test(self._test_empty_state_init)
        self.run_test(self._test_partial_state_init)
    
    def _test_empty_state_init(self):
        state = {}
        result = initialize_approval_state(state)
        
        self.assert_false(result["approval_mode"])
        self.assert_equal(result["plan_approval_status"], "pending")
        self.assert_equal(result["outline_approval_status"], "pending")
        self.assert_equal(result["user_feedback"], {})
        self.assert_equal(result["approval_attempt_count"], {"plan": 0, "outline": 0})
        self.assert_equal(result["max_approval_attempts"], 3)
        self.assert_equal(result["approval_timeout"], 300)
        self.assert_equal(result["workflow_paused_at"], None)
    
    def _test_partial_state_init(self):
        state = {
            "approval_mode": True,
            "plan_approval_status": "approved"
        }
        result = initialize_approval_state(state)
        
        # Should preserve existing
        self.assert_true(result["approval_mode"])
        self.assert_equal(result["plan_approval_status"], "approved")
        
        # Should add missing
        self.assert_equal(result["outline_approval_status"], "pending")
        self.assert_equal(result["max_approval_attempts"], 3)
    
    def test_state_validation(self):
        """Test state validation."""
        self.run_test(self._test_valid_state)
        self.run_test(self._test_invalid_approval_mode)
        self.run_test(self._test_invalid_status)
        self.run_test(self._test_invalid_attempt_count)
    
    def _test_valid_state(self):
        state = {
            "approval_mode": True,
            "plan_approval_status": "approved",
            "outline_approval_status": "pending",
            "user_feedback": {},
            "approval_attempt_count": {"plan": 1, "outline": 0},
            "max_approval_attempts": 3,
            "approval_timeout": 300,
            "workflow_paused_at": None
        }
        self.assert_true(validate_approval_state(state))
    
    def _test_invalid_approval_mode(self):
        state = {"approval_mode": "invalid"}
        self.assert_raises(ValueError, validate_approval_state, state)
    
    def _test_invalid_status(self):
        state = {"plan_approval_status": "invalid"}
        self.assert_raises(ValueError, validate_approval_state, state)
    
    def _test_invalid_attempt_count(self):
        state = {"approval_attempt_count": {"plan": -1}}
        self.assert_raises(ValueError, validate_approval_state, state)
    
    def test_approval_mode_detection(self):
        """Test approval mode detection."""
        self.run_test(self._test_approval_enabled)
        self.run_test(self._test_approval_disabled)
    
    def _test_approval_enabled(self):
        state = {"approval_mode": True}
        self.assert_true(is_approval_enabled(state))
    
    def _test_approval_disabled(self):
        state = {"approval_mode": False}
        self.assert_false(is_approval_enabled(state))
        
        # Test default (missing field)
        empty_state = {}
        self.assert_false(is_approval_enabled(empty_state))
    
    def test_state_reset(self):
        """Test state reset functionality."""
        self.run_test(self._test_reset_plan_only)
        self.run_test(self._test_reset_all)
    
    def _test_reset_plan_only(self):
        state = {
            "plan_approval_status": "approved",
            "outline_approval_status": "rejected",
            "approval_attempt_count": {"plan": 2, "outline": 3},
            "user_feedback": {"plan": "test", "outline": "test2"}
        }
        
        result = reset_approval_state(state, "plan")
        
        self.assert_equal(result["plan_approval_status"], "pending")
        self.assert_equal(result["outline_approval_status"], "rejected")  # Unchanged
        self.assert_equal(result["approval_attempt_count"]["plan"], 0)
        self.assert_equal(result["approval_attempt_count"]["outline"], 3)  # Unchanged
        self.assert_true("plan" not in result["user_feedback"])
        self.assert_true("outline" in result["user_feedback"])  # Unchanged
    
    def _test_reset_all(self):
        state = {
            "plan_approval_status": "approved",
            "outline_approval_status": "rejected",
            "approval_attempt_count": {"plan": 2, "outline": 3},
            "user_feedback": {"plan": "test", "outline": "test2"},
            "workflow_paused_at": "outline_approval"
        }
        
        result = reset_approval_state(state, "all")
        
        self.assert_equal(result["plan_approval_status"], "pending")
        self.assert_equal(result["outline_approval_status"], "pending")
        self.assert_equal(result["approval_attempt_count"]["plan"], 0)
        self.assert_equal(result["approval_attempt_count"]["outline"], 0)
        self.assert_equal(result["user_feedback"], {})
        self.assert_equal(result["workflow_paused_at"], None)
    
    def test_approval_summary(self):
        """Test approval summary generation."""
        self.run_test(self._test_summary_with_data)
        self.run_test(self._test_summary_defaults)
    
    def _test_summary_with_data(self):
        state = {
            "approval_mode": True,
            "plan_approval_status": "approved",
            "outline_approval_status": "pending",
            "approval_attempt_count": {"plan": 2, "outline": 1},
            "max_approval_attempts": 3,
            "user_feedback": {"plan": "test"},
            "workflow_paused_at": "outline_approval"
        }
        
        summary = get_approval_summary(state)
        expected = {
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
        
        self.assert_equal(summary, expected)
    
    def _test_summary_defaults(self):
        state = {}
        summary = get_approval_summary(state)
        
        self.assert_false(summary["approval_mode"])
        self.assert_equal(summary["plan_attempts"], 0)
        self.assert_equal(summary["outline_attempts"], 0)
        self.assert_false(summary["has_plan_feedback"])
        self.assert_false(summary["has_outline_feedback"])
    
    def test_backward_compatibility(self):
        """Test backward compatibility."""
        self.run_test(self._test_existing_fields_preserved)
        self.run_test(self._test_works_without_approval)
    
    def _test_existing_fields_preserved(self):
        existing = {
            "topic": "Test Topic",
            "plan": {"target_audience": "developers"},
            "outline": ["intro", "body", "conclusion"]
        }
        
        enhanced = initialize_approval_state(existing.copy())
        
        # All original fields preserved
        for key, value in existing.items():
            self.assert_equal(enhanced[key], value)
        
        # New fields added
        self.assert_in("approval_mode", enhanced)
        self.assert_in("plan_approval_status", enhanced)
    
    def _test_works_without_approval(self):
        minimal = {"topic": "Test"}
        
        # Should work fine without approval fields
        self.assert_false(is_approval_enabled(minimal))
        
        # Initialize and validate should work
        initialized = initialize_approval_state(minimal.copy())
        self.assert_true(validate_approval_state(initialized))
    
    # Approval service utility tests
    def test_feedback_validation(self):
        """Test feedback validation."""
        self.run_test(self._test_valid_feedback)
        self.run_test(self._test_invalid_feedback)
    
    def _test_valid_feedback(self):
        valid = "This is a good feedback message with sufficient length"
        self.assert_true(validate_feedback(valid))
    
    def _test_invalid_feedback(self):
        # Too short
        self.assert_false(validate_feedback("short"))
        
        # Too long
        long_feedback = "x" * 501
        self.assert_false(validate_feedback(long_feedback))
        
        # Not string
        self.assert_false(validate_feedback(123))
        self.assert_false(validate_feedback(None))
        self.assert_false(validate_feedback(["list"]))
        
        # No meaningful content
        self.assert_false(validate_feedback("!@#$%^&*()"))
    
    def test_feedback_sanitization(self):
        """Test feedback sanitization."""
        self.run_test(self._test_sanitize_basic)
        self.run_test(self._test_sanitize_special_chars)
        self.run_test(self._test_sanitize_edge_cases)
    
    def _test_sanitize_basic(self):
        feedback = "  This is test feedback  "
        result = sanitize_feedback(feedback)
        self.assert_equal(result, "This is test feedback")
    
    def _test_sanitize_special_chars(self):
        feedback = "This has <script>alert('xss')</script> content"
        result = sanitize_feedback(feedback)
        self.assert_true("<script>" not in result)
        self.assert_in("This has", result)
        self.assert_in("content", result)
    
    def _test_sanitize_edge_cases(self):
        # Empty
        self.assert_equal(sanitize_feedback(""), "")
        self.assert_equal(sanitize_feedback(None), "")
        
        # Too long
        long_feedback = "x" * 600
        result = sanitize_feedback(long_feedback)
        self.assert_true(len(result) <= 500)
        self.assert_true(result.endswith("..."))
    
    def test_utility_functions(self):
        """Test utility functions."""
        self.run_test(self._test_attempt_formatting)
        self.run_test(self._test_prompt_text)
    
    def _test_attempt_formatting(self):
        # Normal case
        result = format_attempt_info(2, 5)
        self.assert_in("2/5", result)
        
        # Final attempt
        result = format_attempt_info(5, 5)
        self.assert_in("Final attempt", result)
        self.assert_in("⚠️", result)
        
        # Last remaining
        result = format_attempt_info(4, 5)
        self.assert_in("Last attempt remaining", result)
        self.assert_in("⚡", result)
    
    def _test_prompt_text(self):
        # Initial
        result = get_approval_prompt_text("plan", 0, 3)
        self.assert_in("initial", result.lower())
        
        # Revision
        result = get_approval_prompt_text("plan", 2, 3)
        self.assert_in("revision 2", result.lower())
        self.assert_in("1 attempts remaining", result.lower())
        
        # Final
        result = get_approval_prompt_text("plan", 3, 3)
        self.assert_in("final", result.lower())


def main():
    """Run all tests."""
    runner = SimpleTestRunner()
    success = runner.run_all_tests()
    
    if success:
        print("\n✅ Phase 1 implementation validation completed successfully!")
        print("Ready to proceed to Phase 2: Agent Integration")
    else:
        print("\n❌ Phase 1 implementation has issues that need to be addressed.")
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)