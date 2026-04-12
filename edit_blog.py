"""
On-Demand Blog Editing CLI

Command-line interface for editing previously generated blog posts
using the same editor logic from the main workflow.
"""

import argparse
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

from services.editor_service import (
    edit_blog_file,
    get_editing_stats,
    EditorServiceError
)
from services.markdown_parser import BlogParseError
from utils.env_utils import set_appleconnect_token, validate_environment

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def print_header():
    """Print CLI header."""
    print("\n" + "=" * 70)
    print("🤖 ON-DEMAND BLOG EDITING SERVICE")
    print("=" * 70 + "\n")


def print_stats(stats: dict):
    """
    Print editing statistics.
    
    Args:
        stats: Dictionary of statistics
    """
    if not stats:
        return
    
    print("\n" + "=" * 70)
    print("📊 EDITING STATISTICS")
    print("=" * 70)
    print(f"\n📝 Title: {stats.get('original_title', 'N/A')}")
    print(f"📅 Edited: {stats.get('edited_date', 'N/A')}")
    print(f"\n📈 Word Count:")
    print(f"  • Original: {stats.get('original_words', 0)} words")
    print(f"  • Edited: {stats.get('edited_words', 0)} words")
    print(f"  • Change: {stats.get('word_change_percent', 0):+.1f}%")
    print("\n" + "=" * 70 + "\n")


def validate_input_file(filepath: str) -> bool:
    """
    Validate input file exists and is a markdown file.
    
    Args:
        filepath: Path to input file
        
    Returns:
        True if valid, False otherwise
    """
    path = Path(filepath)
    
    if not path.exists():
        logger.error(f"❌ File not found: {filepath}")
        return False
    
    if not path.is_file():
        logger.error(f"❌ Path is not a file: {filepath}")
        return False
    
    if path.suffix != '.md':
        logger.error(f"❌ File must be a markdown (.md) file: {filepath}")
        return False
    
    return True


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="🤖 On-Demand Blog Editing Service - Re-edit generated blogs using AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Edit a blog from outputs directory
  python edit_blog.py --input outputs/my-blog.md

  # Edit with custom output directory
  python edit_blog.py -i blog.md -o custom_output/

  # Edit with specific LLM provider and model
  python edit_blog.py -i blog.md --provider openai --model gpt-4

  # Verbose mode with statistics
  python edit_blog.py -i blog.md --verbose --stats

For more information, see README.md
        """
    )
    
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to markdown file to edit (e.g., outputs/blog.md)"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="outputs/edited_blogs",
        help="Output directory for edited blog (default: outputs/edited_blogs)"
    )
    
    parser.add_argument(
        "--provider",
        type=str,
        choices=["openai", "anthropic"],
        help="LLM provider to use (default: from LLM_PROVIDER env var or 'anthropic')"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        help="Specific model name to use (overrides env var defaults)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--stats", "-s",
        action="store_true",
        help="Show detailed editing statistics"
    )
    
    parser.add_argument(
        "--no-preserve-date",
        action="store_true",
        help="Don't preserve original date in frontmatter"
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        # Reduce noise from non-critical loggers
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
    
    # Print header
    print_header()
    
    # Validate environment
    try:
        set_appleconnect_token()
        validate_environment()
    except Exception as e:
        logger.error(f"❌ Environment validation failed: {e}")
        sys.exit(1)
    
    # Validate input file
    if not validate_input_file(args.input):
        sys.exit(1)
    
    # Print task info
    print(f"📄 Input File: {args.input}")
    print(f"📁 Output Directory: {args.output_dir}")
    if args.provider:
        print(f"🤖 Provider: {args.provider}")
    if args.model:
        print(f"🧠 Model: {args.model}")
    print("\n" + "=" * 70)
    
    # Run editing service
    try:
        print("\n⏳ Editing blog... This may take a moment.\n")
        
        output_path = edit_blog_file(
            input_path=args.input,
            output_dir=args.output_dir,
            llm_provider=args.provider,
            model_name=args.model,
            preserve_original=not args.no_preserve_date
        )
        
        # Success message
        print(f"\n✅ Blog successfully edited!")
        print(f"📝 Saved to: {output_path}\n")
        
        # Show statistics if requested
        if args.stats:
            stats = get_editing_stats(args.input, output_path)
            print_stats(stats)
        
        print("✨ Editing completed successfully!\n")
        
    except FileNotFoundError as e:
        logger.error(f"\n❌ File not found: {e}")
        sys.exit(1)
    except BlogParseError as e:
        logger.error(f"\n❌ Failed to parse blog file: {e}")
        logger.error("   Make sure the file has valid YAML frontmatter")
        sys.exit(1)
    except EditorServiceError as e:
        logger.error(f"\n❌ Editing service error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.warning("\n\n⚠️  Editing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌ Unexpected error: {str(e)}")
        if args.verbose:
            logger.exception("Detailed error:")
        sys.exit(1)


if __name__ == "__main__":
    main()