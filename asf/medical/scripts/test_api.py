#!/usr/bin/env python
"""
Script to test the Medical Research Synthesizer API endpoints.

This script sends requests to various API endpoints and reports the results.
It can be used as a simple testing tool or in CI/CD pipelines.
"""

import sys
import json
import argparse
import logging
import time
from pathlib import Path
import requests
from typing import Dict, Any, List, Optional

# Set up logging
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
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.token = None
        self.headers = {}
    
    def authenticate(self, username: str, password: str) -> bool:
        """
        Authenticate with the API.
        
        Args:
            username: Username
            password: Password
            
        Returns:
            True if authentication was successful, False otherwise
        """
        auth_url = f"{self.base_url}/v1/auth/token"
        logger.info(f"Authenticating at {auth_url}")
        
        try:
            response = requests.post(
                auth_url,
                data={"username": username, "password": password},
                timeout=self.timeout
            )
            response.raise_for_status()
            
            auth_data = response.json()
            self.token = auth_data.get("access_token")
            
            if self.token:
                logger.info("Authentication successful")
                self.headers = {"Authorization": f"Bearer {self.token}"}
                return True
            else:
                logger.error("Authentication failed: No token received")
                return False
        except requests.exceptions.RequestException as e:
            logger.error(f"Authentication error: {str(e)}")
            return False
    
    def test_endpoint(self, endpoint: str, method: str = "GET", data: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Test an API endpoint.
        
        Args:
            endpoint: API endpoint (relative to base URL)
            method: HTTP method
            data: Request data
            
        Returns:
            Response data if successful, None otherwise
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        logger.info(f"Testing endpoint: {method} {url}")
        
        try:
            if method.upper() == "GET":
                response = requests.get(url, headers=self.headers, timeout=self.timeout)
            elif method.upper() == "POST":
                response = requests.post(url, json=data, headers=self.headers, timeout=self.timeout)
            elif method.upper() == "PUT":
                response = requests.put(url, json=data, headers=self.headers, timeout=self.timeout)
            elif method.upper() == "DELETE":
                response = requests.delete(url, headers=self.headers, timeout=self.timeout)
            else:
                logger.error(f"Unsupported method: {method}")
                return None
            
            response.raise_for_status()
            
            response_data = response.json()
            logger.info(f"Response: {response.status_code} {response.reason}")
            return response_data
        except requests.exceptions.RequestException as e:
            logger.error(f"Error testing endpoint: {str(e)}")
            return None
    
    def test_search(self, query: str, max_results: int = 10) -> Optional[Dict[str, Any]]:
        """
        Test the search endpoint.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            
        Returns:
            Response data if successful, None otherwise
        """
        data = {
            "query": query,
            "max_results": max_results,
            "pagination": {
                "page": 1,
                "page_size": max_results
            }
        }
        return self.test_endpoint("v1/search", "POST", data)
    
    def test_pico_search(self, condition: str, interventions: List[str], outcomes: List[str]) -> Optional[Dict[str, Any]]:
        """
        Test the PICO search endpoint.
        
        Args:
            condition: Medical condition
            interventions: List of interventions
            outcomes: List of outcomes
            
        Returns:
            Response data if successful, None otherwise
        """
        data = {
            "condition": condition,
            "interventions": interventions,
            "outcomes": outcomes,
            "population": None,
            "study_design": None,
            "years": None,
            "max_results": 10,
            "pagination": {
                "page": 1,
                "page_size": 10
            }
        }
        return self.test_endpoint("v1/search/pico", "POST", data)
    
    def test_contradiction(self, text1: str, text2: str) -> Optional[Dict[str, Any]]:
        """
        Test the contradiction endpoint.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Response data if successful, None otherwise
        """
        data = {
            "text1": text1,
            "text2": text2,
            "context": None
        }
        return self.test_endpoint("v1/enhanced-contradiction/analyze", "POST", data)
    
    def test_knowledge_base(self) -> Optional[Dict[str, Any]]:
        """
        Test the knowledge base endpoint.
        
        Returns:
            Response data if successful, None otherwise
        """
        return self.test_endpoint("v1/knowledge-base")
    
    def test_health(self) -> Optional[Dict[str, Any]]:
        """
        Test the health endpoint.
        
        Returns:
            Response data if successful, None otherwise
        """
        return self.test_endpoint("health")
    
    def test_all(self, username: str = None, password: str = None) -> Dict[str, bool]:
        """
        Test all endpoints.
        
        Args:
            username: Optional username for authentication
            password: Optional password for authentication
            
        Returns:
            Dictionary with test results
        """
        results = {}
        
        # Test health endpoint
        results["health"] = self.test_health() is not None
        
        # Authenticate if credentials are provided
        if username and password:
            results["auth"] = self.authenticate(username, password)
            if not results["auth"]:
                logger.warning("Authentication failed, skipping authenticated endpoints")
                return results
            
            # Test authenticated endpoints
            results["search"] = self.test_search("pneumonia treatment") is not None
            results["pico_search"] = self.test_pico_search(
                "pneumonia",
                ["antibiotics", "corticosteroids"],
                ["mortality", "length of stay"]
            ) is not None
            results["contradiction"] = self.test_contradiction(
                "Antibiotics are effective for bacterial pneumonia",
                "Antibiotics are not recommended for viral pneumonia"
            ) is not None
            results["knowledge_base"] = self.test_knowledge_base() is not None
        
        return results

def main():
    """Main function."""
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
        # Test specific endpoint
        data = json.loads(args.data) if args.data else None
        result = tester.test_endpoint(args.endpoint, args.method, data)
        if result:
            print(json.dumps(result, indent=2))
            sys.exit(0)
        else:
            sys.exit(1)
    else:
        # Test all endpoints
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
