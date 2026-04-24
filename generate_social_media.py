"""
Social Media Content Generator - CLI Tool

Generate social media content (Twitter threads, LinkedIn posts) from blog markdown files.
Similar to edit_blog.py but for social media content generation.
"""

import argparse
import sys
import logging
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

from services.social_media_service import (
    generate_from_blog_file,
    get_generation_stats
)
from services.markdown_parser import parse_blog_file
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
    print("🚀 SOCIAL MEDIA CONTENT GENERATOR")
    print("=" * 70)
    print()


def print_stats(stats: dict, blog_title: str):
    """
    Print generation statistics.
    
    Args:
        stats: Statistics dictionary
        blog_title: Blog title
    """
    print("\n" + "=" * 70)
    print("📊 GENERATION STATISTICS")
    print("=" * 70)
    print(f"\n📝 Blog: {blog_title}")
    print(f"🕒 Generated: {stats.get('generated_at', 'N/A')}")
    
    # Twitter stats
    twitter = stats.get('twitter', {})
    print(f"\n🐦 Twitter Thread:")
    print(f"   • Total tweets: {twitter.get('total_tweets', 0)}")
    print(f"   • Avg characters: {twitter.get('avg_char_count', 0)}")
    print(f"   • Hashtags used: {twitter.get('hashtags_count', 0)}")
    
    # LinkedIn stats
    linkedin = stats.get('linkedin', {})
    print(f"\n💼 LinkedIn Posts:")
    print(f"   • Variations: {linkedin.get('total_variations', 0)}")
    print(f"   • Avg characters: {linkedin.get('avg_char_count', 0)}")
    
    # Images
    images = stats.get('images', {})
    print(f"\n🖼️  Images:")
    print(f"   • Twitter images: {images.get('twitter_images', 0)}")
    print(f"   • LinkedIn images: {images.get('linkedin_images', 0)}")
    
    print("\n" + "=" * 70 + "\n")


def validate_and_parse_blog_file(filepath: str) -> Optional[dict]:
    """
    Validate and parse blog file in a single operation.
    
    Args:
        filepath: Path to blog file
        
    Returns:
        Parsed blog data if valid, None otherwise
    """
    path = Path(filepath)
    
    if not path.exists():
        logger.error(f"File not found: {filepath}")
        return None
    
    if not path.suffix == '.md':
        logger.error(f"File must be a markdown (.md) file: {filepath}")
        return None
    
    try:
        blog_data = parse_blog_file(filepath)
        return blog_data
    except Exception as e:
        logger.error(f"Invalid blog file: {e}")
        return None


def generate_single_file(
    input_file: str,
    output_dir: str,
    provider: Optional[str],
    model: Optional[str],
    show_stats: bool,
    verbose: bool,
    dry_run: bool,
    blog_data: Optional[dict] = None
) -> bool:
    """
    Generate social media content for a single blog file.
    
    Args:
        input_file: Path to input blog file
        output_dir: Output directory
        provider: LLM provider
        model: Model name
        show_stats: Whether to show statistics
        verbose: Verbose logging
        dry_run: Validate without generating
        blog_data: Pre-parsed blog data (optional, will parse if not provided)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Parse blog for metadata (only if not already provided)
        if blog_data is None:
            blog_data = parse_blog_file(input_file)
        
        metadata = blog_data.get('metadata', {})
        title = metadata.get('title', 'Unknown')
        
        print(f"📄 Processing: {input_file}")
        print(f"   Title: {title}")
        print(f"   Slug: {metadata.get('slug', 'N/A')}")
        
        if dry_run:
            print("   ✅ Validation successful (dry-run mode)")
            return True
        
        # Generate content (pass pre-parsed blog_data to avoid re-parsing)
        print("   ⏳ Generating social media content...")
        
        output_path = generate_from_blog_file(
            blog_path=input_file,
            output_dir=output_dir,
            llm_provider=provider or "anthropic",
            model_name=model,
            blog_data=blog_data
        )
        
        print(f"   ✅ Generated successfully!")
        print(f"   📁 Saved to: {output_path}")
        
        # Show statistics if requested
        if show_stats:
            # Parse the generated content to get stats
            # For now, just indicate stats would be shown
            print("   📊 Statistics available in output file")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to generate content: {e}")
        if verbose:
            logger.exception("Detailed error:")
        return False


def generate_batch(
    directory: str,
    output_dir: str,
    provider: Optional[str],
    model: Optional[str],
    verbose: bool
) -> None:
    """
    Generate social media content for all blog files in directory.
    
    Args:
        directory: Directory containing blog files
        output_dir: Output directory
        provider: LLM provider
        model: Model name
        verbose: Verbose logging
    """
    dir_path = Path(directory)
    
    if not dir_path.exists() or not dir_path.is_dir():
        logger.error(f"Directory not found: {directory}")
        return
    
    # Find all markdown files
    md_files = list(dir_path.glob("*.md"))
    
    if not md_files:
        logger.warning(f"No markdown files found in: {directory}")
        return
    
    print(f"\n🔄 Batch processing {len(md_files)} files from: {directory}")
    print()
    
    success_count = 0
    fail_count = 0
    
    for md_file in md_files:
        try:
            # Skip files that end with -social.md (already processed)
            if md_file.stem.endswith('-social'):
                logger.info(f"Skipping already processed file: {md_file.name}")
                continue
            
            success = generate_single_file(
                input_file=str(md_file),
                output_dir=output_dir,
                provider=provider,
                model=model,
                show_stats=False,
                verbose=verbose,
                dry_run=False
            )
            
            if success:
                success_count += 1
            else:
                fail_count += 1
            
            print()  # Spacing between files
            
        except Exception as e:
            logger.error(f"Error processing {md_file.name}: {e}")
            fail_count += 1
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 BATCH PROCESSING SUMMARY")
    print("=" * 70)
    print(f"\n✅ Successful: {success_count}")
    print(f"❌ Failed: {fail_count}")
    print(f"📁 Total processed: {success_count + fail_count}")
    print("\n" + "=" * 70 + "\n")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="🚀 Generate social media content from blog posts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate for single blog
  python generate_social_media.py --input outputs/my-blog.md
  
  # Specify output directory
  python generate_social_media.py -i outputs/blog.md -o custom_dir/
  
  # Use different LLM provider
  python generate_social_media.py -i blog.md --provider openai --model gpt-4
  
  # Batch process all blogs in directory
  python generate_social_media.py --directory outputs/ --batch
  
  # Dry run (validate without generating)
  python generate_social_media.py -i blog.md --dry-run
  
  # Show statistics
  python generate_social_media.py -i blog.md --stats

For more information, see README.md
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input", "-i",
        type=str,
        help="Path to blog markdown file"
    )
    input_group.add_argument(
        "--directory", "-d",
        type=str,
        help="Directory containing blog files (use with --batch)"
    )
    
    # Output options
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="outputs/social_media",
        help="Output directory for social media content (default: outputs/social_media)"
    )
    
    # LLM options
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
    
    # Mode options
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Enable batch processing mode (use with --directory)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate blog file without generating content"
    )
    
    # Display options
    parser.add_argument(
        "--stats", "-s",
        action="store_true",
        help="Show detailed generation statistics"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    set_appleconnect_token()
    # Validate environment
    logger.info("Validating environment...")
    validate_environment()
    
    # Print header
    print_header()
    
    # Validate batch mode usage
    if args.batch and not args.directory:
        logger.error("--batch requires --directory option")
        sys.exit(1)
    
    if args.directory and not args.batch:
        logger.error("--directory requires --batch option")
        sys.exit(1)
    
    # Show configuration
    print("📋 Configuration:")
    print(f"   • Output directory: {args.output_dir}")
    
    if args.provider:
        print(f"   • LLM provider: {args.provider}")
    if args.model:
        print(f"   • Model: {args.model}")
    if args.dry_run:
        print("   • Mode: DRY RUN (validation only)")
    
    print("\n" + "=" * 70 + "\n")
    
    try:
        if args.batch:
            # Batch mode
            generate_batch(
                directory=args.directory,
                output_dir=args.output_dir,
                provider=args.provider,
                model=args.model,
                verbose=args.verbose
            )
        else:
            # Single file mode - validate and parse in one operation
            blog_data = validate_and_parse_blog_file(args.input)
            if blog_data is None:
                sys.exit(1)
            
            # Generate with pre-parsed data
            success = generate_single_file(
                input_file=args.input,
                output_dir=args.output_dir,
                provider=args.provider,
                model=args.model,
                show_stats=args.stats,
                verbose=args.verbose,
                dry_run=args.dry_run,
                blog_data=blog_data
            )
            
            if not success:
                sys.exit(1)
        
        print("✨ Social media content generation completed successfully!\n")
        
    except KeyboardInterrupt:
        logger.warning("\n\n⚠️  Generation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌ Error: {str(e)}")
        if args.verbose:
            logger.exception("Detailed error:")
        sys.exit(1)


if __name__ == "__main__":
    main()