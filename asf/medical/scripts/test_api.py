import sys
import json
import argparse
import logging
from typing import List, Optional
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
class APITester:
    """API tester class."""
    def __init__(self, base_url: str, timeout: int = 10):
        """
        Initialize the API tester.
        Args:
            base_url: Base URL of the API
            timeout: Request timeout in seconds
        Authenticate with the API.
        Args:
            username: Username
            password: Password
        Returns:
            True if authentication was successful, False otherwise
        Test an API endpoint.
        Args:
            endpoint: API endpoint (relative to base URL)
            method: HTTP method
            data: Request data
        Returns:
            Response data if successful, None otherwise
        Test the search endpoint.
        Args:
            query: Search query
            max_results: Maximum number of results
        Returns:
            Response data if successful, None otherwise
        Test the PICO search endpoint.
        Args:
            condition: Medical condition
            interventions: List of interventions
            outcomes: List of outcomes
        Returns:
            Response data if successful, None otherwise
        Test the contradiction endpoint.
        Args:
            text1: First text
            text2: Second text
        Returns:
            Response data if successful, None otherwise
        Test the knowledge base endpoint.
        Returns:
            Response data if successful, None otherwise
        Test the health endpoint.
        Returns:
            Response data if successful, None otherwise
        Test all endpoints.
        Args:
            username: Optional username for authentication
            password: Optional password for authentication
        Returns:
            Dictionary with test results
    parser = argparse.ArgumentParser(description="Test the Medical Research Synthesizer API")
    parser.add_argument("--url", default="http://localhost:8000", help="Base URL of the API")
    parser.add_argument("--timeout", type=int, default=10, help="Request timeout in seconds")
    parser.add_argument("--username", help="Username for authentication")
    parser.add_argument("--password", help="Password for authentication")
    parser.add_argument("--endpoint", help="Specific endpoint to test")
    parser.add_argument("--method", default="GET", help="HTTP method for specific endpoint")
    parser.add_argument("--data", help="JSON data for specific endpoint")
    args = parser.parse_args()
    tester = APITester(args.url, args.timeout)
    if args.endpoint:
        data = json.loads(args.data) if args.data else None
        result = tester.test_endpoint(args.endpoint, args.method, data)
        if result:
            print(json.dumps(result, indent=2))
            sys.exit(0)
        else:
            sys.exit(1)
    else:
        results = tester.test_all(args.username, args.password)
        print(json.dumps(results, indent=2))
        if all(results.values()):
            logger.info("All tests passed")
            sys.exit(0)
        else:
            logger.error("Some tests failed")
            sys.exit(1)
if __name__ == "__main__":
    main()