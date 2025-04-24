"""
Standalone test for export utilities.

This script tests the export utilities without using pytest.
"""

import json
import sys
import os
import logging

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

from asf.medical.api.export_utils import export_to_json
from asf.medical.core.exceptions import ValidationError

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_export_to_json():
    """Test exporting data to JSON."""
    # Sample data
    sample_data = [
        {
            "pmid": "12345678",
            "title": "Test Article 1",
            "journal": "Test Journal",
            "authors": ["Smith, J."]
        },
        {
            "pmid": "87654321",
            "title": "Test Article 2",
            "journal": "Another Journal",
            "authors": ["Johnson, A."]
        }
    ]
    
    try:
        # Test with valid data
        logger.info("Testing export_to_json with valid data...")
        response = export_to_json(sample_data, query_text="test query")
        
        assert response.status_code == 200, f"Expected status code 200, got {response.status_code}"
        
        content = json.loads(response.body)
        assert content["count"] == 2, f"Expected count 2, got {content['count']}"
        assert len(content["results"]) == 2, f"Expected 2 results, got {len(content['results'])}"
        assert "query" in content, "Expected 'query' in content"
        assert content["query"] == "test query", f"Expected query 'test query', got {content['query']}"
        
        logger.info("Valid data test passed!")
        
        # Test with invalid data
        logger.info("Testing export_to_json with invalid data...")
        try:
            export_to_json("not a list")
            assert False, "Expected ValidationError but no exception was raised"
        except ValidationError:
            logger.info("Invalid data test passed!")
        except Exception as e:
            assert False, f"Expected ValidationError but got {type(e).__name__}: {str(e)}"
            
        logger.info("All tests passed!")
        return True
    except AssertionError as e:
        logger.error(f"Test failed: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error: {type(e).__name__}: {str(e)}")
        return False


if __name__ == "__main__":
    success = test_export_to_json()
    sys.exit(0 if success else 1)
