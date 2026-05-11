"""
Unit tests for HumanApprovalService functionality.
"""

import pytest
import time
from unittest.mock import patch, MagicMock, call
from io import StringIO
from services.approval_service import (
    HumanApprovalService,
    validate_feedback,
    sanitize_feedback,
    format_attempt_info,
    get_approval_prompt_text
)


class TestHumanApprovalService:
    """Test HumanApprovalService class functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.service = HumanApprovalService(timeout=5)  # Short timeout for tests
        self.test_state = {
            "topic": "Test Machine Learning",
            "plan": {
                "target_audience": "Software developers",
                "blog_length": 2500,
                "tone": "technical",
                "section_titles": [
                    "Introduction",
                    "Core Concepts", 
                    "Implementation",
                    "Conclusion"
                ],
                "keywords": ["ML", "AI", "Python"]
            },
            "outline": [
                "Introduction to Machine Learning",
                "What is Machine Learning?",
                "Types of Machine Learning",
                "Popular Algorithms",
                "Tools and Frameworks", 
                "Building Your First Model",
                "Conclusion"
            ],
            "approval_attempt_count": {"plan": 0, "outline": 0},
            "max_approval_attempts": 3,
            "user_feedback": {}
        }
    
    @patch('services.approval_service.os.system')
    def test_clear_screen_posix(self, mock_system):
        """Test screen clearing on POSIX systems."""
        with patch('services.approval_service.os.name', 'posix'):
            self.service.clear_screen()
            mock_system.assert_called_once_with('clear')
    
    @patch('services.approval_service.os.system')
    def test_clear_screen_windows(self, mock_system):
        """Test screen clearing on Windows systems."""
        with patch('services.approval_service.os.name', 'nt'):
            self.service.clear_screen()
            mock_system.assert_called_once_with('cls')
    
    @patch('builtins.print')
    def test_print_separator_with_title(self, mock_print):
        """Test separator printing with title."""
        self.service.print_separator("Test Title")
        
        # Should print 3 lines: top border, title, bottom border
        assert mock_print.call_count == 3
        calls = mock_print.call_args_list
        
        # Check that separators contain "=" characters
        assert "=" in calls[0][0][0]
        assert "Test Title" in calls[1][0][0] 
        assert "=" in calls[2][0][0]
    
    @patch('builtins.print')
    def test_print_separator_without_title(self, mock_print):
        """Test separator printing without title."""
        self.service.print_separator()
        
        # Should print 1 line: border only
        mock_print.assert_called_once()
        assert "=" in mock_print.call_args[0][0]
    
    @patch('builtins.print')
    def test_print_box_with_content(self, mock_print):
        """Test box printing with content."""
        content = "Line 1\nLine 2\nLine 3"
        self.service.print_box(content, "Test Box")
        
        # Should print multiple lines for box structure
        assert mock_print.call_count >= 5  # At minimum: top, title, separator, content lines, bottom
        
        # Check that box characters are used
        calls = [call[0][0] for call in mock_print.call_args_list]
        box_chars_found = any("┌" in line or "└" in line or "│" in line for line in calls)
        assert box_chars_found
    
    @patch('services.approval_service.HumanApprovalService.clear_screen')
    @patch('services.approval_service.HumanApprovalService.print_separator')
    @patch('services.approval_service.HumanApprovalService.print_box')
    @patch('builtins.print')
    def test_display_plan_for_review_initial_attempt(self, mock_print, mock_print_box, mock_print_separator, mock_clear_screen):
        """Test displaying plan for initial review."""
        self.service.display_plan_for_review(self.test_state)
        
        mock_clear_screen.assert_called_once()
        mock_print_separator.assert_called_once_with("📋 PLAN REVIEW")
        mock_print_box.assert_called_once()
        
        # Should display topic
        topic_call = None
        for call in mock_print.call_args_list:
            if "Test Machine Learning" in str(call):
                topic_call = call
                break
        assert topic_call is not None
    
    @patch('services.approval_service.HumanApprovalService.clear_screen')
    @patch('services.approval_service.HumanApprovalService.print_separator')
    @patch('services.approval_service.HumanApprovalService.print_box')
    @patch('builtins.print')
    def test_display_plan_for_review_with_feedback(self, mock_print, mock_print_box, mock_print_separator, mock_clear_screen):
        """Test displaying plan with previous feedback."""
        # Set up state with previous attempt and feedback
        self.test_state["approval_attempt_count"]["plan"] = 1
        self.test_state["user_feedback"]["plan"] = "Make it more beginner-friendly"
        
        self.service.display_plan_for_review(self.test_state)
        
        # Should show revision information
        revision_call = None
        feedback_call = None
        for call in mock_print.call_args_list:
            call_str = str(call)
            if "Revision Attempt" in call_str:
                revision_call = call
            if "Previous Feedback" in call_str:
                feedback_call = call
        
        assert revision_call is not None
        assert feedback_call is not None
    
    @patch('services.approval_service.HumanApprovalService.clear_screen')
    @patch('services.approval_service.HumanApprovalService.print_separator') 
    @patch('services.approval_service.HumanApprovalService.print_box')
    @patch('builtins.print')
    def test_display_outline_for_review(self, mock_print, mock_print_box, mock_print_separator, mock_clear_screen):
        """Test displaying outline for review."""
        self.service.display_outline_for_review(self.test_state)
        
        mock_clear_screen.assert_called_once()
        mock_print_separator.assert_called_once_with("📋 OUTLINE REVIEW")
        mock_print_box.assert_called_once()
        
        # Check that print_box was called with outline content
        box_call_args = mock_print_box.call_args[0][0]
        assert "Introduction to Machine Learning" in box_call_args
    
    @patch('builtins.input', return_value='A')
    @patch('builtins.print')
    def test_get_user_approval_approve(self, mock_print, mock_input):
        """Test user approval with approve choice."""
        result = self.service.get_user_approval("plan")
        assert result == "approved"
        mock_input.assert_called_once()
    
    @patch('builtins.input', return_value='R') 
    @patch('builtins.print')
    def test_get_user_approval_reject(self, mock_print, mock_input):
        """Test user approval with reject choice."""
        result = self.service.get_user_approval("plan")
        assert result == "rejected"
        mock_input.assert_called_once()
    
    @patch('builtins.input', return_value='Q')
    @patch('builtins.print')
    def test_get_user_approval_quit(self, mock_print, mock_input):
        """Test user approval with quit choice."""
        result = self.service.get_user_approval("plan")
        assert result == "quit"
        mock_input.assert_called_once()
    
    @patch('builtins.input', side_effect=['X', 'Y', 'A'])  # Invalid, Invalid, Valid
    @patch('builtins.print')
    def test_get_user_approval_invalid_then_valid(self, mock_print, mock_input):
        """Test user approval with invalid input then valid."""
        result = self.service.get_user_approval("plan")
        assert result == "approved"
        assert mock_input.call_count == 3
    
    def test_get_user_approval_timeout(self):
        """Test user approval timeout behavior."""
        # Use very short timeout
        short_service = HumanApprovalService(timeout=0.1)
        
        with patch('builtins.input', side_effect=lambda _: time.sleep(1)):  # Simulate slow input
            result = short_service.get_user_approval("plan")
            assert result == "approved"  # Should auto-approve on timeout
    
    @patch('builtins.input', side_effect=KeyboardInterrupt())
    @patch('builtins.print')
    def test_get_user_approval_keyboard_interrupt(self, mock_print, mock_input):
        """Test user approval with keyboard interrupt."""
        result = self.service.get_user_approval("plan")
        assert result == "quit"
    
    @patch('builtins.input', side_effect=EOFError())
    @patch('builtins.print')
    def test_get_user_approval_eof_error(self, mock_print, mock_input):
        """Test user approval with EOF error."""
        result = self.service.get_user_approval("plan")
        assert result == "timeout"
    
    @patch('services.approval_service.HumanApprovalService.print_separator')
    @patch('builtins.input', return_value='This is valid feedback for testing purposes')
    @patch('builtins.print')
    def test_collect_feedback_valid(self, mock_print, mock_input, mock_separator):
        """Test collecting valid feedback."""
        with patch('builtins.input', side_effect=['This is valid feedback for testing purposes', 'y']):
            result = self.service.collect_feedback_for_revision("plan")
            assert result == "This is valid feedback for testing purposes"
    
    @patch('services.approval_service.HumanApprovalService.print_separator')
    @patch('builtins.print')
    def test_collect_feedback_too_short(self, mock_print, mock_separator):
        """Test collecting feedback that's too short."""
        with patch('builtins.input', side_effect=['short', 'This is now long enough feedback', 'y']):
            result = self.service.collect_feedback_for_revision("plan")
            assert result == "This is now long enough feedback"
    
    @patch('services.approval_service.HumanApprovalService.print_separator')
    @patch('builtins.input', side_effect=KeyboardInterrupt())
    @patch('builtins.print')
    def test_collect_feedback_keyboard_interrupt(self, mock_print, mock_input, mock_separator):
        """Test collecting feedback with keyboard interrupt."""
        result = self.service.collect_feedback_for_revision("plan")
        assert result == ""
    
    @patch('services.approval_service.HumanApprovalService.print_separator')
    @patch('builtins.print')
    def test_collect_feedback_confirmation_no(self, mock_print, mock_separator):
        """Test feedback collection with confirmation rejection."""
        with patch('builtins.input', side_effect=[
            'This is valid feedback',
            'n',  # Don't confirm
            'This is better feedback', 
            'y'   # Confirm
        ]):
            result = self.service.collect_feedback_for_revision("plan")
            assert result == "This is better feedback"
    
    @patch('builtins.print')
    def test_show_timeout_warning(self, mock_print):
        """Test timeout warning display."""
        self.service.show_timeout_warning(30)
        
        warning_call = None
        for call in mock_print.call_args_list:
            if "Auto-approval in 30 seconds" in str(call):
                warning_call = call
                break
        assert warning_call is not None
    
    @patch('builtins.print')
    def test_handle_approval_timeout(self, mock_print):
        """Test approval timeout handling."""
        result = self.service.handle_approval_timeout()
        assert result == "approved"
        
        # Should print timeout message
        timeout_call = None
        for call in mock_print.call_args_list:
            if "timeout" in str(call).lower():
                timeout_call = call
                break
        assert timeout_call is not None


class TestApprovalServiceUtilities:
    """Test utility functions for approval service."""
    
    def test_validate_feedback_valid(self):
        """Test feedback validation with valid input."""
        valid_feedback = "This is a good feedback message with sufficient length"
        assert validate_feedback(valid_feedback) is True
    
    def test_validate_feedback_too_short(self):
        """Test feedback validation with too short input."""
        short_feedback = "short"
        assert validate_feedback(short_feedback) is False
    
    def test_validate_feedback_too_long(self):
        """Test feedback validation with too long input."""
        long_feedback = "x" * 501  # Exceeds 500 character limit
        assert validate_feedback(long_feedback) is False
    
    def test_validate_feedback_not_string(self):
        """Test feedback validation with non-string input."""
        assert validate_feedback(123) is False
        assert validate_feedback(None) is False
        assert validate_feedback([]) is False
    
    def test_validate_feedback_no_meaningful_content(self):
        """Test feedback validation with no meaningful content."""
        meaningless_feedback = "!@#$%^&*()_+"
        assert validate_feedback(meaningless_feedback) is False
    
    def test_sanitize_feedback_basic(self):
        """Test basic feedback sanitization."""
        feedback = "  This is test feedback  "
        sanitized = sanitize_feedback(feedback)
        assert sanitized == "This is test feedback"
    
    def test_sanitize_feedback_special_characters(self):
        """Test feedback sanitization with special characters."""
        feedback = "This has <script>alert('xss')</script> dangerous content"
        sanitized = sanitize_feedback(feedback)
        assert "<script>" not in sanitized
        assert "This has  dangerous content" in sanitized
    
    def test_sanitize_feedback_empty(self):
        """Test sanitization of empty feedback."""
        assert sanitize_feedback("") == ""
        assert sanitize_feedback(None) == ""
    
    def test_sanitize_feedback_too_long(self):
        """Test sanitization of overly long feedback."""
        long_feedback = "x" * 600
        sanitized = sanitize_feedback(long_feedback)
        assert len(sanitized) <= 500
        assert sanitized.endswith("...")
    
    def test_format_attempt_info_normal(self):
        """Test attempt info formatting for normal case."""
        result = format_attempt_info(2, 5)
        assert "2/5" in result
    
    def test_format_attempt_info_final_attempt(self):
        """Test attempt info formatting for final attempt."""
        result = format_attempt_info(5, 5)
        assert "Final attempt" in result
        assert "⚠️" in result
    
    def test_format_attempt_info_last_remaining(self):
        """Test attempt info formatting for last remaining attempt."""
        result = format_attempt_info(4, 5)
        assert "Last attempt remaining" in result
        assert "⚡" in result
    
    def test_get_approval_prompt_text_initial(self):
        """Test approval prompt text for initial attempt."""
        result = get_approval_prompt_text("plan", 0, 3)
        assert "initial version" in result.lower()
    
    def test_get_approval_prompt_text_revision(self):
        """Test approval prompt text for revision attempt."""
        result = get_approval_prompt_text("plan", 2, 3)
        assert "revision 2" in result.lower()
        assert "1 attempts remaining" in result.lower()
    
    def test_get_approval_prompt_text_final(self):
        """Test approval prompt text for final attempt."""
        result = get_approval_prompt_text("plan", 3, 3)
        assert "final revision" in result.lower()


class TestApprovalServiceIntegration:
    """Integration tests for approval service with realistic scenarios."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.service = HumanApprovalService(timeout=1)  # Short timeout for tests
        self.realistic_plan_state = {
            "topic": "Understanding RAG in LLMs",
            "plan": {
                "target_audience": "AI engineers and researchers",
                "blog_length": 3500,
                "tone": "technical but accessible",
                "section_titles": [
                    "Introduction to RAG",
                    "How RAG Works",
                    "RAG vs Fine-tuning",
                    "Implementation Strategies",
                    "Performance Optimization",
                    "Future of RAG",
                    "Conclusion"
                ],
                "keywords": ["RAG", "retrieval", "LLM", "vector database", "embeddings"]
            },
            "approval_attempt_count": {"plan": 1, "outline": 0},
            "max_approval_attempts": 3,
            "user_feedback": {"plan": "Add more practical examples and reduce technical jargon"}
        }
    
    @patch('services.approval_service.HumanApprovalService.clear_screen')
    @patch('builtins.print')
    def test_realistic_plan_display(self, mock_print, mock_clear):
        """Test displaying a realistic plan with previous feedback."""
        self.service.display_plan_for_review(self.realistic_plan_state)
        
        # Verify key elements are displayed
        print_calls = [str(call) for call in mock_print.call_args_list]
        all_output = " ".join(print_calls)
        
        assert "Understanding RAG in LLMs" in all_output
        assert "AI engineers and researchers" in all_output
        assert "Revision Attempt: 1/3" in all_output
        assert "Add more practical examples" in all_output
    
    @patch('builtins.input')
    @patch('builtins.print') 
    def test_complete_approval_workflow(self, mock_print, mock_input):
        """Test complete approval workflow simulation."""
        # Simulate: reject -> provide feedback -> approve
        mock_input.side_effect = [
            'R',  # Reject
            'Add more code examples and simplify explanations',  # Feedback
            'y',  # Confirm feedback
        ]
        
        # First: get rejection
        choice = self.service.get_user_approval("plan")
        assert choice == "rejected"
        
        # Second: collect feedback
        feedback = self.service.collect_feedback_for_revision("plan")
        assert "Add more code examples" in feedback
        
        # Verify feedback was sanitized properly
        assert len(feedback) > 10
        assert len(feedback) <= 500


# Mock-based integration tests
class TestApprovalServiceMocking:
    """Test approval service with extensive mocking for edge cases."""
    
    @patch('signal.signal')
    def test_signal_handler_setup(self, mock_signal):
        """Test that signal handlers are set up correctly."""
        service = HumanApprovalService()
        # Should have called signal.signal at least twice (SIGINT and possibly SIGTERM)
        assert mock_signal.call_count >= 1
    
    @patch('time.time')
    @patch('builtins.input', return_value='A')
    @patch('builtins.print')
    def test_timeout_calculation(self, mock_print, mock_input, mock_time):
        """Test timeout calculation and timing."""
        # Mock time progression
        mock_time.side_effect = [0, 1, 2, 3]  # Simulate time progression
        
        service = HumanApprovalService(timeout=5)
        result = service.get_user_approval("plan")
        
        assert result == "approved"
        # Should have called time.time() multiple times for timeout checking
        assert mock_time.call_count >= 2
    
    @patch('services.approval_service.logger')
    def test_logging_integration(self, mock_logger):
        """Test that approval service logs appropriately."""
        service = HumanApprovalService()
        
        with patch('builtins.input', return_value='A'):
            with patch('builtins.print'):
                result = service.get_user_approval("plan")
        
        assert result == "approved"
        # Should have logged the approval
        mock_logger.info.assert_called()
        log_calls = [str(call) for call in mock_logger.info.call_args_list]
        assert any("approved" in call for call in log_calls)