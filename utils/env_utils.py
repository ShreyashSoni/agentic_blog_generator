import os
import sys
import logging
import subprocess

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def set_appleconnect_token():
    command = [
        '/usr/local/bin/appleconnect',
        'getToken',
        '-C', 'hvys3fcwcteqrvw3qzkvtk86viuoqv',
        '--token-type=oauth',
        '--interactivity-type=none',
        '-E', 'prod',
        '-G', 'pkce',
        '-o', 'openid,dsid,accountname,profile,groups'
    ]

    result = subprocess.run(
        command,
        capture_output=True,  # Captures stdout and stderr
        text=True,            # Returns strings instead of bytes
        timeout=30            # Optional timeout
    )

    if result.returncode == 0:
        token = result.stdout.strip().split()[-1]
        os.environ['TOKEN'] = token
        logger.info("Successfully set AppleConnect Token for authentication")
    else:
        logger.error(f"AppleConnect Error: {result.stderr}")


def validate_environment():
    """
    Validate that required environment variables are set.
    
    Raises:
        SystemExit: If required variables are missing
    """
    required_vars = {
        "TAVILY_API_KEY": "Tavily API key for web research",
        "TOKEN": "AppleConnect token for authentication"
    }
    
    missing_vars = []
    for var, description in required_vars.items():
        if not os.getenv(var):
            missing_vars.append(f"  • {var}: {description}")
    
    if missing_vars:
        logger.error("Missing required environment variables:")
        for var in missing_vars:
            logger.error(var)
        logger.error("\nPlease set these variables in your .env file or environment")
        logger.error("See .env.example for reference")
        sys.exit(1)