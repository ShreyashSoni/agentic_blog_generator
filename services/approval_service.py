"""
Interactive CLI service for human approval of plans and outlines.

This service manages the human-in-the-loop approval checkpoints, providing
a clean terminal interface for reviewing and providing feedback on AI-generated
blog plans and outlines.
"""

import os
import sys
import time
import signal
import logging
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class HumanApprovalService:
    """Service for managing human approval interactions in CLI"""
    
    def __init__(self, timeout: float = 300):
        """
        Initialize the approval service.
        
        Args:
            timeout: Timeout for approval prompts in seconds (default: 300)
        """
        self.timeout = float(timeout)
        self.approval_start_time: Optional[float] = None
        self._setup_signal_handlers()
    
    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful interruption handling."""
        def signal_handler(signum, frame):
            print("\n\n⛔️ Approval interrupted by user")
            self._cleanup()
            sys.exit(1)
        
        signal.signal(signal.SIGINT, signal_handler)
        if hasattr(signal, 'SIGTERM'):
            signal.signal(signal.SIGTERM, signal_handler)
    
    def _cleanup(self) -> None:
        """Cleanup resources and restore terminal state."""
        # Reset terminal if needed
        pass
    
    def display_plan_for_review(self, state: Dict[str, Any]) -> None:
        """
        Display formatted plan for user review.
        
        Args:
            state: Current workflow state containing plan information
        """
        self.clear_screen()
        plan = state.get("plan", {})
        topic = state.get("topic", "Unknown Topic")
        attempts = state.get("approval_attempt_count", {}).get("plan", 0)
        max_attempts = state.get("max_approval_attempts", 3)
        
        self.print_separator("📋 PLAN REVIEW")
        print(f"\nTopic: \"{topic}\"")
        
        # Show revision information if this is a retry
        if attempts > 0:
            print(f"Revision Attempt: {attempts}/{max_attempts}")
            feedback = state.get("user_feedback", {}).get("plan", "")
            if feedback:
                print(f"Previous Feedback: {feedback}")
        
        # Format plan content
        content = []
        content.append(f"Target Audience: {plan.get('target_audience', 'Not specified')}")
        content.append(f"Blog Length: {plan.get('blog_length', 'Not specified')} words")
        content.append(f"Tone: {plan.get('tone', 'Not specified')}")
        content.append("")
        content.append("Sections:")
        
        sections = plan.get("section_titles", [])
        for i, section in enumerate(sections, 1):
            content.append(f"  {i}. {section}")
        
        content.append("")
        keywords = plan.get("keywords", [])
        if keywords:
            if isinstance(keywords, list):
                keywords_str = ", ".join(str(k) for k in keywords)
            else:
                keywords_str = str(keywords)
            content.append(f"Keywords: {keywords_str}")
        
        self.print_box("\n".join(content), "Generated Plan")
        print()  # Extra spacing
    
    def display_outline_for_review(self, state: Dict[str, Any]) -> None:
        """
        Display formatted outline for user review.
        
        Args:
            state: Current workflow state containing outline information
        """
        self.clear_screen()
        outline = state.get("outline", [])
        topic = state.get("topic", "Unknown Topic")
        attempts = state.get("approval_attempt_count", {}).get("outline", 0)
        max_attempts = state.get("max_approval_attempts", 3)
        
        self.print_separator("📋 OUTLINE REVIEW")
        print(f"\nTopic: \"{topic}\"")
        
        # Show revision information if this is a retry
        if attempts > 0:
            print(f"Revision Attempt: {attempts}/{max_attempts}")
            feedback = state.get("user_feedback", {}).get("outline", "")
            if feedback:
                print(f"Previous Feedback: {feedback}")
        
        # Format outline content
        content = []
        for i, section in enumerate(outline, 1):
            content.append(f"  {i}. {section}")
        
        if not content:
            content.append("  No sections generated")
        
        self.print_box("\n".join(content), "Generated Outline")
        print()  # Extra spacing
    
    def get_user_approval(self, item_type: str) -> str:
        """
        Get user approval choice with timeout handling.
        
        Args:
            item_type: Type of item being approved ('plan' or 'outline')
            
        Returns:
            User choice: 'approved', 'rejected', 'quit', or 'timeout'
        """
        self.approval_start_time = time.time()
        
        print("Options:")
        print("[A] Approve and continue")
        print("[R] Reject and provide feedback")
        print("[Q] Quit workflow")
        
        if self.timeout > 0:
            print(f"[T] Auto-approve in {self.timeout} seconds")
        
        while True:
            # Check timeout
            if self.timeout > 0 and self.approval_start_time:
                elapsed = time.time() - self.approval_start_time
                if elapsed > self.timeout:
                    return self.handle_approval_timeout()
                
                # Show warning at intervals
                remaining = self.timeout - elapsed
                if remaining <= 60 and remaining % 15 == 0:
                    self.show_timeout_warning(int(remaining))
            
            try:
                choice = input(f"\nYour choice: ").strip().upper()
                
                if choice in ['A', 'APPROVE', 'Y', 'YES']:
                    logger.info(f"User approved {item_type}")
                    return 'approved'
                elif choice in ['R', 'REJECT', 'N', 'NO']:
                    logger.info(f"User rejected {item_type}")
                    return 'rejected'
                elif choice in ['Q', 'QUIT', 'EXIT']:
                    logger.info(f"User quit during {item_type} approval")
                    return 'quit'
                else:
                    print("Invalid choice. Please enter A (approve), R (reject), or Q (quit).")
                    
            except KeyboardInterrupt:
                print("\n\nWorkflow interrupted by user.")
                logger.info(f"Keyboard interrupt during {item_type} approval")
                return 'quit'
            except EOFError:
                print("\n\nInput stream ended.")
                logger.warning(f"EOF during {item_type} approval")
                return 'timeout'
            except Exception as e:
                logger.error(f"Error during approval input: {e}")
                print(f"Error reading input: {e}. Please try again.")
    
    def collect_feedback_for_revision(self, item_type: str) -> str:
        """
        Collect detailed feedback for revision.
        
        Args:
            item_type: Type of item being revised ('plan' or 'outline')
            
        Returns:
            User feedback string (empty if cancelled)
        """
        self.print_separator("📝 FEEDBACK FOR REVISION")
        
        print(f"\nPlease provide specific feedback for the {item_type} revision:")
        print("\nExamples:")
        
        if item_type == "plan":
            examples = [
                '- "Change target audience to \'marketing professionals\'"',
                '- "Add a section about ethical considerations"',
                '- "Reduce blog length to 1500 words"',
                '- "Make tone more casual and beginner-friendly"'
            ]
        else:  # outline
            examples = [
                '- "Add more detail to section 3"',
                '- "Remove section about advanced topics"',
                '- "Reorder sections - put practical examples first"',
                '- "Add a section about common mistakes"'
            ]
        
        for example in examples:
            print(example)
        
        print(f"\nFeedback (minimum 10 characters, maximum 500):")
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                feedback = input("> ").strip()
                
                if not feedback:
                    print("Feedback cannot be empty. Please provide specific feedback.")
                    continue
                
                if not validate_feedback(feedback):
                    print("Feedback must be between 10 and 500 characters. Please try again.")
                    continue
                
                # Confirm feedback
                print(f"\nYour feedback: {feedback}")
                confirm = input("Is this correct? [Y/n]: ").strip().lower()
                
                if confirm in ['', 'y', 'yes']:
                    sanitized = sanitize_feedback(feedback)
                    logger.info(f"Collected feedback for {item_type}: {sanitized[:100]}...")
                    return sanitized
                else:
                    print("Please provide your feedback again:")
                    
            except KeyboardInterrupt:
                print("\n\nFeedback cancelled.")
                logger.info(f"Feedback collection cancelled for {item_type}")
                return ""
            except EOFError:
                print("\n\nInput stream ended.")
                logger.warning(f"EOF during feedback collection for {item_type}")
                return ""
            except Exception as e:
                logger.error(f"Error during feedback collection: {e}")
                print(f"Error reading feedback: {e}. Please try again.")
        
        print(f"Maximum retry attempts ({max_retries}) exceeded.")
        return ""
    
    def show_timeout_warning(self, remaining_time: int) -> None:
        """
        Show timeout countdown warning.
        
        Args:
            remaining_time: Seconds remaining before timeout
        """
        if remaining_time <= 60:
            print(f"\n⏰ Warning: Auto-approval in {remaining_time} seconds...")
    
    def handle_approval_timeout(self) -> str:
        """
        Handle approval timeout scenario.
        
        Returns:
            Action to take: 'approved' (default) or 'quit'
        """
        print("\n\n⏰ Approval timeout reached.")
        print("Automatically approving to continue workflow...")
        logger.warning("Approval timeout - automatically approving")
        return 'approved'
    
    def clear_screen(self) -> None:
        """Clear terminal screen for clean display."""
        try:
            os.system('clear' if os.name == 'posix' else 'cls')
        except Exception:
            # Fallback: print newlines
            print('\n' * 50)
    
    def print_separator(self, title: str = "") -> None:
        """
        Print visual separator with optional title.
        
        Args:
            title: Optional title to display in separator
        """
        width = 70
        if title:
            print("=" * width)
            # Center the title
            title_line = f" {title} "
            padding = (width - len(title_line)) // 2
            centered_title = "=" * padding + title_line + "=" * (width - padding - len(title_line))
            print(centered_title)
            print("=" * width)
        else:
            print("=" * width)
    
    def print_box(self, content: str, title: str = "") -> None:
        """
        Print content in a formatted box.
        
        Args:
            content: Content to display in box
            title: Optional title for the box
        """
        lines = content.split('\n')
        
        # Calculate box width
        max_content_width = max(len(line) for line in lines) if lines else 0
        title_width = len(title) + 4 if title else 0
        box_width = max(max_content_width + 4, title_width, 50)  # Minimum width of 50
        
        # Top border
        if title:
            print("┌" + "─" * (box_width - 2) + "┐")
            title_padding = (box_width - len(title) - 4) // 2
            title_line = "│ " + " " * title_padding + title + " " * (box_width - len(title) - title_padding - 4) + " │"
            print(title_line)
            print("├" + "─" * (box_width - 2) + "┤")
        else:
            print("┌" + "─" * (box_width - 2) + "┐")
        
        # Content lines
        for line in lines:
            padding = box_width - len(line) - 4
            print(f"│ {line}{' ' * padding} │")
        
        # Bottom border
        print("└" + "─" * (box_width - 2) + "┘")


# Utility functions

def validate_feedback(feedback: str) -> bool:
    """
    Validate user feedback meets requirements.
    
    Args:
        feedback: User feedback string
        
    Returns:
        True if feedback is valid, False otherwise
    """
    if not isinstance(feedback, str):
        return False
    
    # Check length requirements
    if len(feedback.strip()) < 10:
        return False
    
    if len(feedback.strip()) > 500:
        return False
    
    # Check for minimal content (not just whitespace/punctuation)
    meaningful_chars = sum(1 for c in feedback if c.isalnum())
    if meaningful_chars < 5:
        return False
    
    return True


def sanitize_feedback(feedback: str) -> str:
    """
    Sanitize feedback for security and processing.
    
    Args:
        feedback: Raw user feedback
        
    Returns:
        Sanitized feedback string
    """
    if not feedback:
        return ""
    
    # Basic sanitization
    sanitized = feedback.strip()
    
    # Remove potentially harmful characters but preserve meaningful punctuation
    safe_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?;:-_()[]{}/"\'')
    sanitized = ''.join(c for c in sanitized if c in safe_chars)
    
    # Collapse multiple spaces
    sanitized = ' '.join(sanitized.split())
    
    # Ensure reasonable length
    if len(sanitized) > 500:
        sanitized = sanitized[:497] + "..."
    
    return sanitized


def format_attempt_info(attempts: int, max_attempts: int) -> str:
    """
    Format attempt counter display.
    
    Args:
        attempts: Current attempt number
        max_attempts: Maximum allowed attempts
        
    Returns:
        Formatted attempt info string
    """
    remaining = max_attempts - attempts
    if remaining <= 0:
        return f"⚠️  Final attempt ({attempts}/{max_attempts})"
    elif remaining == 1:
        return f"⚡ Last attempt remaining ({attempts}/{max_attempts})"
    else:
        return f"Attempt {attempts}/{max_attempts}"


def get_approval_prompt_text(item_type: str, attempts: int = 0, max_attempts: int = 3) -> str:
    """
    Get contextual prompt text for approval.
    
    Args:
        item_type: Type of item ('plan' or 'outline')
        attempts: Current attempt number
        max_attempts: Maximum allowed attempts
        
    Returns:
        Formatted prompt text
    """
    base_text = f"Please review the generated {item_type} above."
    
    if attempts == 0:
        return f"{base_text} This is the initial version."
    elif attempts < max_attempts:
        remaining = max_attempts - attempts
        return f"{base_text} This is revision {attempts}. {remaining} attempts remaining."
    else:
        return f"{base_text} This is the final revision attempt."


# Quick test function for development
def test_approval_service():
    """Test function for development and debugging."""
    service = HumanApprovalService(timeout=10)
    
    # Test plan display
    test_state = {
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
        "approval_attempt_count": {"plan": 1},
        "max_approval_attempts": 3,
        "user_feedback": {"plan": "Make it more beginner-friendly"}
    }
    
    print("Testing plan display...")
    service.display_plan_for_review(test_state)
    
    print("\nTesting approval prompt...")
    choice = service.get_user_approval("plan")
    print(f"Choice: {choice}")
    
    if choice == "rejected":
        feedback = service.collect_feedback_for_revision("plan")
        print(f"Feedback: {feedback}")


if __name__ == "__main__":
    # Run test if called directly
    test_approval_service()