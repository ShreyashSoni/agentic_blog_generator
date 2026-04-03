"""
Agentic Blog Generator - CLI Application

This is the main entry point for generating blog posts using the multi-agent system.
"""

import argparse
import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv
from workflows.blog_graph import run_workflow
from utils.env_utils import set_appleconnect_token, validate_environment

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def save_blog_to_file(content: str, seo_meta: Dict[str, Any], output_dir: str = "outputs") -> str:
    """
    Save blog content to markdown file with frontmatter.
    
    Args:
        content: Blog content in markdown
        seo_meta: SEO metadata dictionary
        output_dir: Directory to save the blog
        
    Returns:
        Path to the saved file
    """
    # Create output directory if needed
    Path(output_dir).mkdir(exist_ok=True)
    
    # Generate filename from slug
    slug = seo_meta.get("slug", "blog-post")
    filepath = os.path.join(output_dir, f"{slug}.md")
    
    # Create frontmatter
    keywords_str = ", ".join(seo_meta.get("keywords", []))
    frontmatter = f"""---
title: "{seo_meta.get('meta_title', 'Blog Post')}"
description: "{seo_meta.get('meta_description', '')}"
keywords: [{keywords_str}]
slug: "{slug}"
date: {_get_current_date()}
---

"""
    
    # Combine content
    full_content = frontmatter + content
    
    # Add FAQ section if available
    if "faq" in seo_meta and seo_meta["faq"]:
        faq_section = "\n\n---\n\n## Frequently Asked Questions\n\n"
        for faq_item in seo_meta["faq"]:
            faq_section += f"**Q: {faq_item['question']}**\n\n"
            faq_section += f"A: {faq_item['answer']}\n\n"
        full_content += faq_section
    
    # Save to file
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(full_content)
    
    return filepath


def print_seo_metadata(seo_meta: Dict[str, Any]):
    """
    Pretty print SEO metadata to console.
    
    Args:
        seo_meta: SEO metadata dictionary
    """
    print("\n" + "="*70)
    print("📊 SEO METADATA")
    print("="*70)
    
    print(f"\n📝 Title: {seo_meta.get('meta_title', 'N/A')}")
    print(f"📄 Description: {seo_meta.get('meta_description', 'N/A')}")
    print(f"🔗 Slug: {seo_meta.get('slug', 'N/A')}")
    
    if seo_meta.get('keywords'):
        print(f"\n🔑 Keywords: {', '.join(seo_meta['keywords'])}")
    
    if "keyword_density" in seo_meta and seo_meta["keyword_density"]:
        print("\n📈 Keyword Density:")
        for keyword, density in list(seo_meta["keyword_density"].items())[:5]:
            print(f"  • {keyword}: {density}%")
    
    if seo_meta.get("faq"):
        print(f"\n❓ FAQ: {len(seo_meta['faq'])} questions generated")
    
    print("\n" + "="*70 + "\n")


def print_statistics(final_state: Dict[str, Any]):
    """
    Print blog generation statistics.
    
    Args:
        final_state: Final state from workflow
    """
    edited = final_state.get("edited", "")
    sections = final_state.get("sections", {})
    research_docs = final_state.get("research_docs", [])
    
    word_count = len(edited.split())
    
    print("\n" + "="*70)
    print("📈 BLOG STATISTICS")
    print("="*70)
    print(f"\n📝 Total Words: {word_count}")
    print(f"📚 Sections: {len(sections)}")
    print(f"🔍 Research Documents: {len(research_docs)}")
    print(f"✅ Status: Complete")
    print("\n" + "="*70 + "\n")


def _get_current_date() -> str:
    """Get current date in YYYY-MM-DD format."""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="🤖 Agentic Blog Generator - Generate high-quality blog posts using AI agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python app.py --topic "Introduction to Machine Learning"
  python app.py --topic "RAG vs Fine-tuning" --output-dir my_blogs
  python app.py --topic "Python Best Practices" --verbose

For more information, see README.md
        """
    )
    
    parser.add_argument(
        "--topic",
        type=str,
        required=True,
        help="Blog topic to generate content about"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory to save generated blog (default: outputs)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save the blog to file (print to console only)"
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

    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    set_appleconnect_token()
    # Validate environment
    logger.info("Validating environment...")
    validate_environment()
    
    # Print header
    print("\n" + "="*70)
    print("🤖 AGENTIC BLOG GENERATOR")
    print("="*70)
    print(f"\n📝 Topic: {args.topic}")
    print(f"📁 Output Directory: {args.output_dir}")
    print("\n" + "="*70 + "\n")
    
    # Run workflow
    try:
        logger.info("🚀 Starting blog generation workflow...")
        print("⏳ Generating blog... This may take a few minutes.\n")
        
        final_state = run_workflow(args.topic, 
                                   verbose=True, 
                                   llm_provider=args.provider, 
                                   model_name=args.model)
        
        # Extract results
        edited_content = final_state.get("edited", "")
        seo_meta = final_state.get("seo_meta", {})
        
        if not edited_content:
            logger.error("❌ Blog generation failed: No content generated")
            sys.exit(1)
        
        # Print statistics
        print_statistics(final_state)
        
        # Save blog to file
        if not args.no_save:
            filepath = save_blog_to_file(
                content=edited_content,
                seo_meta=seo_meta,
                output_dir=args.output_dir
            )
            logger.info(f"\n✅ Blog saved to: {filepath}")
        else:
            logger.info("\n📄 Blog content (not saved):")
            print("\n" + "="*70)
            print(edited_content)
            print("="*70 + "\n")
        
        # Print SEO metadata
        print_seo_metadata(seo_meta)
        
        # Success message
        print("✨ Blog generation completed successfully!\n")
        
    except KeyboardInterrupt:
        logger.warning("\n\n⚠️  Generation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌ Error generating blog: {str(e)}")
        if args.verbose:
            logger.exception("Detailed error:")
        sys.exit(1)


if __name__ == "__main__":
    main()