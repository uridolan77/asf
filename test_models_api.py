import os
import sys
import logging
import requests
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def login():
    """Login to get an access token."""
    try:
        # Get API base URL from environment variable or use default
        api_base_url = os.environ.get("API_BASE_URL", "http://localhost:8000")
        
        # Login credentials - using the format expected by OAuth2PasswordRequestForm
        login_data = {
            "username": "admin@example.com",
            "password": "password"
        }
        
        # Make login request
        url = f"{api_base_url}/api/login"
        logger.info(f"Making login request to {url}")
        
        response = requests.post(
            url, 
            data=login_data,
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )
        
        # Check response
        if response.status_code == 200:
            logger.info(f"Login successful: {response.status_code}")
            
            # Parse response
            data = response.json()
            logger.info(f"Got access token")
            
            return data.get("access_token")
        else:
            logger.error(f"Login failed: {response.status_code}")
            logger.error(f"Response: {response.text}")
            return None
    except Exception as e:
        logger.error(f"Error logging in: {e}")
        return None

def test_models_api():
    """Test the models API endpoint."""
    try:
        # Get API base URL from environment variable or use default
        api_base_url = os.environ.get("API_BASE_URL", "http://localhost:8000")
        
        # Get token from login or environment variable
        token = login() or os.environ.get("API_TOKEN", "")
        
        if not token:
            logger.error("No token available. Login failed and no API_TOKEN environment variable set.")
            return
        
        # Set up headers
        headers = {"Authorization": f"Bearer {token}"}
        
        # Make request to models endpoint
        url = f"{api_base_url}/api/llm/gateway/models"
        logger.info(f"Making request to {url}")
        
        response = requests.get(url, headers=headers)
        
        # Check response
        if response.status_code == 200:
            logger.info(f"Request successful: {response.status_code}")
            
            # Parse response
            data = response.json()
            logger.info(f"Response data: {json.dumps(data, indent=2)}")
            
            # Check if response is a list
            if isinstance(data, list):
                logger.info(f"Found {len(data)} models in the response")
                
                # Print models
                for model in data:
                    logger.info(f"Model: {model.get('model_id')}, Provider: {model.get('provider_id')}")
            else:
                logger.warning(f"Response is not a list: {data}")
        else:
            logger.error(f"Request failed: {response.status_code}")
            logger.error(f"Response: {response.text}")
    except Exception as e:
        logger.error(f"Error testing models API: {e}")
        raise

if __name__ == "__main__":
    test_models_api()
