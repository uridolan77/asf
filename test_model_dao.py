import os
import sys
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the project root directory to the path so we can import the models
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)
    logger.info(f"Added {project_root} to Python path")

# Import the ModelDAO
from asf.bollm.backend.repositories.model_dao import ModelDAO

def test_model_dao():
    """Test the ModelDAO class."""
    try:
        # Create DAO
        logger.info("Creating ModelDAO instance")
        dao = ModelDAO()
        
        # Ensure table exists
        logger.info("Ensuring llm_models table exists")
        dao.ensure_table_exists()
        
        # Get all models
        logger.info("Getting all models")
        models = dao.get_all_models()
        logger.info(f"Found {len(models)} models in the database")
        
        # Print the models
        for model in models:
            logger.info(f"Model: {model['model_id']}, Provider: {model['provider_id']}")
        
        # Test getting a specific model
        if len(models) > 0:
            model_id = models[0]['model_id']
            provider_id = models[0]['provider_id']
            logger.info(f"Getting model {model_id} from provider {provider_id}")
            model = dao.get_model_by_id(model_id, provider_id)
            if model:
                logger.info(f"Found model: {model}")
            else:
                logger.warning(f"Model {model_id} from provider {provider_id} not found")
        
        # Test creating a model
        test_model_id = f"test-model-{datetime.utcnow().timestamp()}"
        test_provider_id = "test-provider"
        logger.info(f"Creating test model {test_model_id} for provider {test_provider_id}")
        
        model_data = {
            "model_id": test_model_id,
            "provider_id": test_provider_id,
            "display_name": "Test Model",
            "model_type": "chat",
            "context_window": 16000,
            "max_output_tokens": 2000,
            "capabilities": ["function_calling", "json_mode"],
            "parameters": {"temperature": 0.7, "top_p": 0.9}
        }
        
        created_model = dao.create_model(model_data)
        if created_model:
            logger.info(f"Created model: {created_model}")
            
            # Test updating the model
            logger.info(f"Updating model {test_model_id}")
            update_data = {
                "display_name": "Updated Test Model",
                "context_window": 32000,
                "capabilities": ["function_calling", "json_mode", "vision"]
            }
            
            updated_model = dao.update_model(test_model_id, test_provider_id, update_data)
            if updated_model:
                logger.info(f"Updated model: {updated_model}")
            else:
                logger.warning(f"Failed to update model {test_model_id}")
            
            # Test deleting the model
            logger.info(f"Deleting model {test_model_id}")
            deleted = dao.delete_model(test_model_id, test_provider_id)
            if deleted:
                logger.info(f"Deleted model {test_model_id}")
            else:
                logger.warning(f"Failed to delete model {test_model_id}")
        else:
            logger.warning(f"Failed to create test model {test_model_id}")
        
    except Exception as e:
        logger.error(f"Error testing ModelDAO: {e}")
        raise

if __name__ == "__main__":
    test_model_dao()
