"""
WordPress Publisher Service

This module provides functionality to publish blog posts to WordPress
using the WordPress REST API with Application Password authentication.
"""

import requests
import logging
import base64
from datetime import date, datetime
from typing import Dict, Any, List, Optional
from urllib.parse import urljoin
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger(__name__)


class WordPressPublishError(Exception):
    """Custom exception for WordPress publishing errors."""
    pass


class WordPressAuthError(Exception):
    """Custom exception for WordPress authentication errors."""
    pass


class WordPressPublisher:
    """
    WordPress Publisher service for creating and managing WordPress posts.
    
    This class handles authentication, post creation, and taxonomy management
    using the WordPress REST API.
    """
    
    def __init__(
        self,
        site_url: str,
        username: str,
        app_password: str,
        timeout: int = 30,
        verify_ssl: bool = True
    ):
        """
        Initialize WordPress Publisher.
        
        Args:
            site_url: WordPress site URL (e.g., https://yourblog.com)
            username: WordPress username
            app_password: WordPress Application Password
            timeout: Request timeout in seconds
            verify_ssl: Whether to verify SSL certificates
        """
        self.site_url = site_url.rstrip('/')
        self.username = username
        self.app_password = app_password.replace(' ', '')  # Remove spaces from app password
        self.timeout = timeout
        self.verify_ssl = verify_ssl
        
        # Build API base URL
        self.api_base = urljoin(self.site_url, '/wp-json/wp/v2/')
        
        # Create auth header
        credentials = f"{self.username}:{self.app_password}"
        token = base64.b64encode(credentials.encode()).decode()
        self.headers = {
            'Authorization': f'Basic {token}',
            'Content-Type': 'application/json',
            'User-Agent': 'Agentic Blog Generator/1.0'
        }
        
        logger.info(f"WordPress Publisher initialized for: {self.site_url}")
    
    def authenticate(self) -> bool:
        """
        Validate WordPress credentials by making a test API call.
        
        Returns:
            True if authentication successful
            
        Raises:
            WordPressAuthError: If authentication fails
        """
        logger.info("Validating WordPress credentials...")
        
        try:
            # Test with a simple GET request to users/me endpoint
            url = urljoin(self.api_base, 'users/me')
            response = requests.get(
                url,
                headers=self.headers,
                timeout=self.timeout,
                verify=self.verify_ssl
            )
            
            if response.status_code == 200:
                user_data = response.json()
                logger.info(f"Authenticated as: {user_data.get('name', 'Unknown')}")
                return True
            elif response.status_code == 401:
                raise WordPressAuthError(
                    "Authentication failed. Please check your username and application password.\n"
                    "Generate an application password at: WordPress Admin → Users → Profile → Application Passwords"
                )
            else:
                raise WordPressAuthError(f"Unexpected response: {response.status_code} - {response.text}")
                
        except requests.exceptions.SSLError:
            raise WordPressAuthError(
                "SSL certificate verification failed. "
                "Set verify_ssl=False to bypass (not recommended for production)"
            )
        except requests.exceptions.ConnectionError:
            raise WordPressAuthError(f"Cannot connect to WordPress site: {self.site_url}")
        except requests.exceptions.Timeout:
            raise WordPressAuthError("Connection timeout. Please check your site URL and network connection")
        except Exception as e:
            raise WordPressAuthError(f"Authentication error: {str(e)}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(requests.exceptions.RequestException),
        reraise=True
    )
    def create_post(
        self,
        title: str,
        content: str,
        status: str = 'draft',
        excerpt: Optional[str] = None,
        slug: Optional[str] = None,
        categories: Optional[List[int]] = None,
        tags: Optional[List[int]] = None,
        date: Optional[str] = None,
        author_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Create a new WordPress post.
        
        Args:
            title: Post title
            content: Post content (HTML or markdown)
            status: Post status ('draft', 'publish', 'private')
            excerpt: Post excerpt/description
            slug: Post slug (URL-friendly name)
            categories: List of category IDs
            tags: List of tag IDs
            date: Post date (ISO 8601 format)
            author_id: Author user ID
            
        Returns:
            Dictionary containing post data including ID and URL
            
        Raises:
            WordPressPublishError: If post creation fails
        """
        logger.info(f"Creating WordPress post: {title}")
        
        # Build post data
        post_data = {
            'title': title,
            'content': content,
            'status': status
        }
        
        # Add optional fields
        if excerpt:
            post_data['excerpt'] = excerpt
        if slug:
            post_data['slug'] = slug
        if categories:
            post_data['categories'] = categories
        if tags:
            post_data['tags'] = tags
        if date:
            post_data['date'] = date
        if author_id:
            post_data['author'] = author_id
        
        try:
            url = urljoin(self.api_base, 'posts')
            response = requests.post(
                url,
                json=post_data,
                headers=self.headers,
                timeout=self.timeout,
                verify=self.verify_ssl
            )
            
            if response.status_code in [200, 201]:
                post = response.json()
                logger.info(f"Post created successfully: ID {post['id']}")
                return post
            elif response.status_code == 400:
                error_msg = response.json().get('message', 'Bad request')
                raise WordPressPublishError(f"Invalid post data: {error_msg}")
            elif response.status_code == 401:
                raise WordPressAuthError("Authentication failed during post creation")
            elif response.status_code == 403:
                raise WordPressPublishError(
                    "Permission denied. User may not have rights to create posts."
                )
            else:
                raise WordPressPublishError(
                    f"Failed to create post: {response.status_code} - {response.text}"
                )
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error creating post: {e}")
            raise
        except Exception as e:
            raise WordPressPublishError(f"Error creating post: {str(e)}")
    
    def get_or_create_category(self, name: str) -> int:
        """
        Get category ID by name, or create if it doesn't exist.
        
        Args:
            name: Category name
            
        Returns:
            Category ID
            
        Raises:
            WordPressPublishError: If category operations fail
        """
        logger.info(f"Getting/creating category: {name}")
        
        try:
            # Search for existing category
            url = urljoin(self.api_base, 'categories')
            response = requests.get(
                url,
                params={'search': name},
                headers=self.headers,
                timeout=self.timeout,
                verify=self.verify_ssl
            )
            
            if response.status_code == 200:
                categories = response.json()
                # Check for exact match
                for cat in categories:
                    if cat['name'].lower() == name.lower():
                        logger.info(f"Found existing category: {name} (ID: {cat['id']})")
                        return cat['id']
            
            # Category doesn't exist, create it
            logger.info(f"Creating new category: {name}")
            response = requests.post(
                url,
                json={'name': name},
                headers=self.headers,
                timeout=self.timeout,
                verify=self.verify_ssl
            )
            
            if response.status_code in [200, 201]:
                category = response.json()
                logger.info(f"Category created: {name} (ID: {category['id']})")
                return category['id']
            else:
                raise WordPressPublishError(f"Failed to create category: {response.text}")
                
        except Exception as e:
            logger.warning(f"Error with category '{name}': {e}")
            raise WordPressPublishError(f"Category error: {str(e)}")
    
    def get_or_create_tag(self, name: str) -> int:
        """
        Get tag ID by name, or create if it doesn't exist.
        
        Args:
            name: Tag name
            
        Returns:
            Tag ID
            
        Raises:
            WordPressPublishError: If tag operations fail
        """
        logger.info(f"Getting/creating tag: {name}")
        
        try:
            # Search for existing tag
            url = urljoin(self.api_base, 'tags')
            response = requests.get(
                url,
                params={'search': name},
                headers=self.headers,
                timeout=self.timeout,
                verify=self.verify_ssl
            )
            
            if response.status_code == 200:
                tags = response.json()
                # Check for exact match
                for tag in tags:
                    if tag['name'].lower() == name.lower():
                        logger.info(f"Found existing tag: {name} (ID: {tag['id']})")
                        return tag['id']
            
            # Tag doesn't exist, create it
            logger.info(f"Creating new tag: {name}")
            response = requests.post(
                url,
                json={'name': name},
                headers=self.headers,
                timeout=self.timeout,
                verify=self.verify_ssl
            )
            
            if response.status_code in [200, 201]:
                tag = response.json()
                logger.info(f"Tag created: {name} (ID: {tag['id']})")
                return tag['id']
            else:
                raise WordPressPublishError(f"Failed to create tag: {response.text}")
                
        except Exception as e:
            logger.warning(f"Error with tag '{name}': {e}")
            raise WordPressPublishError(f"Tag error: {str(e)}")
    
    def process_categories(self, category_names: List[str]) -> List[int]:
        """
        Process a list of category names and return their IDs.
        
        Args:
            category_names: List of category names
            
        Returns:
            List of category IDs
        """
        category_ids = []
        for name in category_names:
            name = name.strip()
            if name:
                try:
                    cat_id = self.get_or_create_category(name)
                    category_ids.append(cat_id)
                except Exception as e:
                    logger.warning(f"Skipping category '{name}': {e}")
        
        return category_ids
    
    def process_tags(self, tag_names: List[str]) -> List[int]:
        """
        Process a list of tag names and return their IDs.
        
        Args:
            tag_names: List of tag names
            
        Returns:
            List of tag IDs
        """
        tag_ids = []
        for name in tag_names:
            name = name.strip()
            if name:
                try:
                    tag_id = self.get_or_create_tag(name)
                    tag_ids.append(tag_id)
                except Exception as e:
                    logger.warning(f"Skipping tag '{name}': {e}")
        
        return tag_ids
    
    def get_post_url(self, post_id: int) -> str:
        """
        Get the public URL for a post.
        
        Args:
            post_id: WordPress post ID
            
        Returns:
            Post URL
        """
        return f"{self.site_url}/?p={post_id}"
    
    def get_post_edit_url(self, post_id: int) -> str:
        """
        Get the WordPress admin edit URL for a post.
        
        Args:
            post_id: WordPress post ID
            
        Returns:
            Post edit URL
        """
        return f"{self.site_url}/wp-admin/post.php?post={post_id}&action=edit"
    
    def _format_date(self, date_value: Any) -> Optional[str]:
        """
        Convert date value to ISO 8601 string format for WordPress API.
        
        Args:
            date_value: Date value (string, date, or datetime object)
            
        Returns:
            ISO 8601 formatted date string or None
        """
        if date_value is None:
            return None
        
        # If already a string, return as-is
        if isinstance(date_value, str):
            return date_value
        
        # If datetime object, format it
        if isinstance(date_value, datetime):
            return date_value.isoformat()
        
        # If date object, convert to datetime at midnight and format
        if isinstance(date_value, date):
            return datetime.combine(date_value, datetime.min.time()).isoformat()
        
        # Try to convert to string
        return str(date_value)
    
    def publish_blog(
        self,
        metadata: Dict[str, Any],
        content: str,
        status: str = 'draft',
        categories: Optional[List[str]] = None,
        tags: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        High-level method to publish a complete blog post.
        
        Args:
            metadata: Blog metadata (title, description, slug, etc.)
            content: Blog content
            status: Post status ('draft', 'publish', 'private')
            categories: List of category names (optional)
            tags: List of tag names (optional, uses keywords from metadata if not provided)
            
        Returns:
            Dictionary with post information including ID, URL, and edit URL
            
        Raises:
            WordPressPublishError: If publishing fails
        """
        logger.info(f"Publishing blog: {metadata.get('title', 'Untitled')}")
        
        # Process categories
        category_ids = []
        if categories:
            category_ids = self.process_categories(categories)
        
        # Process tags (use keywords from metadata if not provided)
        tag_ids = []
        if tags:
            tag_ids = self.process_tags(tags)
        elif 'keywords' in metadata and metadata['keywords']:
            keywords = metadata['keywords']
            if isinstance(keywords, list):
                tag_ids = self.process_tags(keywords)
            elif isinstance(keywords, str):
                tag_ids = self.process_tags([k.strip() for k in keywords.split(',')])
        
        # Format date for WordPress API (convert Python date objects to ISO string)
        post_date = self._format_date(metadata.get('date'))
        
        # Create the post
        post = self.create_post(
            title=metadata.get('title', 'Untitled'),
            content=content,
            status=status,
            excerpt=metadata.get('description'),
            slug=metadata.get('slug'),
            categories=category_ids if category_ids else None,
            tags=tag_ids if tag_ids else None,
            date=post_date
        )
        
        # Build result
        result = {
            'id': post['id'],
            'title': post['title']['rendered'],
            'status': post['status'],
            'url': post['link'],
            'edit_url': self.get_post_edit_url(post['id']),
            'date': post['date'],
            'categories': len(category_ids),
            'tags': len(tag_ids)
        }
        
        logger.info(f"Blog published successfully: {result['title']}")
        return result


# Example usage
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    logging.basicConfig(level=logging.INFO)
    
    # Initialize publisher
    publisher = WordPressPublisher(
        site_url=os.getenv('WORDPRESS_SITE_URL', ''),
        username=os.getenv('WORDPRESS_USERNAME', ''),
        app_password=os.getenv('WORDPRESS_APP_PASSWORD', '')
    )
    
    # Test authentication
    try:
        publisher.authenticate()
        print("✅ Authentication successful!")
    except WordPressAuthError as e:
        print(f"❌ Authentication failed: {e}")