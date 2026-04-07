"""
WordPress Publishing CLI

Command-line interface for publishing generated blog posts to WordPress.
This script allows on-demand publishing of markdown blog files to WordPress as drafts.
"""

import argparse
import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv

from services.markdown_parser import parse_blog_file, BlogParseError
from services.wordpress_publisher import (
    WordPressPublisher,
    WordPressAuthError,
    WordPressPublishError
)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def validate_environment() -> Dict[str, str]:
    """
    Validate that required WordPress environment variables are set.
    
    Returns:
        Dictionary of WordPress configuration
        
    Raises:
        SystemExit: If required variables are missing
    """
    required_vars = {
        'WORDPRESS_SITE_URL': 'WordPress site URL',
        'WORDPRESS_USERNAME': 'WordPress username',
        'WORDPRESS_APP_PASSWORD': 'WordPress application password'
    }
    
    config = {}
    missing = []
    
    for var, description in required_vars.items():
        value = os.getenv(var)
        if not value or value.startswith('your_') or value.startswith('https://yourblog'):
            missing.append(f"  - {var}: {description}")
        else:
            config[var] = value
    
    if missing:
        logger.error("❌ Missing required WordPress configuration in .env file:")
        for item in missing:
            logger.error(item)
        logger.error("\nPlease set these variables in your .env file.")
        logger.error("See .env.example for reference.")
        sys.exit(1)
    
    return config


def publish_single_blog(
    filepath: str,
    publisher: WordPressPublisher,
    status: str = 'draft',
    categories: List[str] = None,
    tags: List[str] = None,
    dry_run: bool = False
) -> Dict[str, Any]:
    """
    Publish a single blog file to WordPress.
    
    Args:
        filepath: Path to markdown blog file
        publisher: WordPress publisher instance
        status: Post status (draft/publish/private)
        categories: List of category names
        tags: List of tag names
        dry_run: If True, validate without publishing
        
    Returns:
        Dictionary with publish result
        
    Raises:
        BlogParseError: If blog parsing fails
        WordPressPublishError: If publishing fails
    """
    logger.info(f"📄 Processing: {filepath}")
    
    # Parse blog file
    try:
        parsed = parse_blog_file(filepath)
        metadata = parsed['metadata']
        content = parsed['content']
    except BlogParseError as e:
        logger.error(f"❌ Parse error: {e}")
        raise
    except FileNotFoundError as e:
        logger.error(f"❌ File not found: {e}")
        raise
    
    # Display blog info
    logger.info(f"   Title: {metadata.get('title', 'Untitled')}")
    logger.info(f"   Slug: {metadata.get('slug', 'N/A')}")
    logger.info(f"   Content length: {len(content)} characters")
    
    if dry_run:
        logger.info("✅ Dry run - validation successful (not publishing)")
        return {
            'success': True,
            'dry_run': True,
            'title': metadata.get('title'),
            'filepath': filepath
        }
    
    # Publish to WordPress
    try:
        result = publisher.publish_blog(
            metadata=metadata,
            content=content,
            status=status,
            categories=categories,
            tags=tags
        )
        
        logger.info(f"✅ Published successfully!")
        logger.info(f"   Post ID: {result['id']}")
        logger.info(f"   Status: {result['status']}")
        logger.info(f"   URL: {result['url']}")
        logger.info(f"   Edit: {result['edit_url']}")
        
        return {
            'success': True,
            'filepath': filepath,
            **result
        }
        
    except WordPressPublishError as e:
        logger.error(f"❌ Publish error: {e}")
        raise


def publish_batch(
    directory: str,
    publisher: WordPressPublisher,
    status: str = 'draft',
    categories: List[str] = None,
    tags: List[str] = None,
    dry_run: bool = False
) -> Dict[str, Any]:
    """
    Publish all blog files in a directory.
    
    Args:
        directory: Directory containing markdown files
        publisher: WordPress publisher instance
        status: Post status for all posts
        categories: List of category names
        tags: List of tag names
        dry_run: If True, validate without publishing
        
    Returns:
        Dictionary with batch results
    """
    logger.info(f"📁 Batch processing directory: {directory}")
    
    # Find all markdown files
    dir_path = Path(directory)
    if not dir_path.exists():
        logger.error(f"❌ Directory not found: {directory}")
        sys.exit(1)
    
    md_files = list(dir_path.glob('*.md'))
    if not md_files:
        logger.warning(f"⚠️  No markdown files found in {directory}")
        return {'success': True, 'total': 0, 'published': 0, 'failed': 0}
    
    logger.info(f"Found {len(md_files)} markdown file(s)")
    
    # Process each file
    results = {
        'total': len(md_files),
        'published': 0,
        'failed': 0,
        'successes': [],
        'failures': []
    }
    
    for md_file in md_files:
        logger.info(f"\n{'='*70}")
        try:
            result = publish_single_blog(
                filepath=str(md_file),
                publisher=publisher,
                status=status,
                categories=categories,
                tags=tags,
                dry_run=dry_run
            )
            results['published'] += 1
            results['successes'].append({
                'file': md_file.name,
                'title': result.get('title'),
                'url': result.get('url')
            })
        except Exception as e:
            results['failed'] += 1
            results['failures'].append({
                'file': md_file.name,
                'error': str(e)
            })
            logger.error(f"Failed to publish {md_file.name}: {e}")
            # Continue with next file
    
    # Print summary
    logger.info(f"\n{'='*70}")
    logger.info("📊 BATCH PROCESSING SUMMARY")
    logger.info(f"{'='*70}")
    logger.info(f"Total files: {results['total']}")
    logger.info(f"✅ Published: {results['published']}")
    logger.info(f"❌ Failed: {results['failed']}")
    
    if results['successes']:
        logger.info("\n✅ Successfully published:")
        for item in results['successes']:
            logger.info(f"   • {item['file']}: {item.get('title', 'N/A')}")
    
    if results['failures']:
        logger.info("\n❌ Failed:")
        for item in results['failures']:
            logger.info(f"   • {item['file']}: {item['error']}")
    
    logger.info(f"{'='*70}\n")
    
    return results


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="📤 Publish generated blog posts to WordPress",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Publish single blog as draft
  python publish_to_wordpress.py --file outputs/my-blog.md
  
  # Publish as published post
  python publish_to_wordpress.py --file outputs/my-blog.md --status publish
  
  # Publish with categories and tags
  python publish_to_wordpress.py --file outputs/my-blog.md \\
      --categories "Tech,AI" --tags "python,machine-learning"
  
  # Batch publish all blogs
  python publish_to_wordpress.py --directory outputs/ --batch
  
  # Dry run (validate without publishing)
  python publish_to_wordpress.py --file outputs/my-blog.md --dry-run

For more information, see README.md
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--file',
        type=str,
        help='Path to markdown blog file to publish'
    )
    input_group.add_argument(
        '--directory',
        type=str,
        help='Directory containing blog files (use with --batch)'
    )
    
    # Publishing options
    parser.add_argument(
        '--status',
        type=str,
        choices=['draft', 'publish', 'private'],
        default=os.getenv('WORDPRESS_DEFAULT_STATUS', 'draft'),
        help='Post status (default: draft)'
    )
    
    parser.add_argument(
        '--categories',
        type=str,
        help='Comma-separated category names (e.g., "Tech,AI")'
    )
    
    parser.add_argument(
        '--tags',
        type=str,
        help='Comma-separated tag names (e.g., "python,ml"). Uses keywords from frontmatter if not provided.'
    )
    
    parser.add_argument(
        '--batch',
        action='store_true',
        help='Enable batch processing mode (with --directory)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Validate files without actually publishing'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    # WordPress config overrides
    parser.add_argument(
        '--site-url',
        type=str,
        help='WordPress site URL (overrides .env)'
    )
    
    parser.add_argument(
        '--username',
        type=str,
        help='WordPress username (overrides .env)'
    )
    
    parser.add_argument(
        '--app-password',
        type=str,
        help='WordPress application password (overrides .env)'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate batch mode usage
    if args.batch and not args.directory:
        logger.error("❌ --batch requires --directory")
        sys.exit(1)
    
    # Print header
    print("\n" + "="*70)
    print("📤 WORDPRESS BLOG PUBLISHER")
    print("="*70 + "\n")
    
    # Validate environment
    try:
        config = validate_environment()
    except SystemExit:
        sys.exit(1)
    
    # Override with CLI arguments if provided
    site_url = args.site_url or config['WORDPRESS_SITE_URL']
    username = args.username or config['WORDPRESS_USERNAME']
    app_password = args.app_password or config['WORDPRESS_APP_PASSWORD']
    
    # Initialize WordPress publisher
    try:
        logger.info("🔧 Initializing WordPress publisher...")
        publisher = WordPressPublisher(
            site_url=site_url,
            username=username,
            app_password=app_password,
            timeout=int(os.getenv('WORDPRESS_API_TIMEOUT', '30'))
        )
        
        # Authenticate
        if not args.dry_run:
            logger.info("🔐 Authenticating with WordPress...")
            publisher.authenticate()
            logger.info("✅ Authentication successful!\n")
        
    except WordPressAuthError as e:
        logger.error(f"\n❌ Authentication failed: {e}\n")
        logger.error("Please check your WordPress credentials in .env file.")
        logger.error("Generate an application password at:")
        logger.error(f"{site_url}/wp-admin/profile.php\n")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌ Initialization error: {e}\n")
        sys.exit(1)
    
    # Parse categories and tags
    categories = None
    if args.categories:
        categories = [c.strip() for c in args.categories.split(',') if c.strip()]
    
    tags = None
    if args.tags:
        tags = [t.strip() for t in args.tags.split(',') if t.strip()]
    
    # Publish
    try:
        if args.batch and args.directory:
            # Batch mode
            results = publish_batch(
                directory=args.directory,
                publisher=publisher,
                status=args.status,
                categories=categories,
                tags=tags,
                dry_run=args.dry_run
            )
            
            # Exit with error code if any failed
            if results['failed'] > 0:
                sys.exit(1)
        else:
            # Single file mode
            result = publish_single_blog(
                filepath=args.file,
                publisher=publisher,
                status=args.status,
                categories=categories,
                tags=tags,
                dry_run=args.dry_run
            )
        
        print("\n✨ Publishing completed successfully!\n")
        
    except KeyboardInterrupt:
        logger.warning("\n\n⚠️  Publishing interrupted by user\n")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌ Error: {e}\n")
        if args.verbose:
            logger.exception("Detailed error:")
        sys.exit(1)


if __name__ == "__main__":
    main()