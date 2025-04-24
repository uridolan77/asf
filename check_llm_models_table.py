import os
import sys
import logging
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the project root directory to the path so we can import the models
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)
    logger.info(f"Added {project_root} to Python path")

# Import the models
from asf.bollm.backend.models.base import Base
from asf.bollm.backend.models.llm_model import LLMModel
from asf.bollm.backend.config.config import SQLALCHEMY_DATABASE_URL

def check_database():
    """Check if we can connect to the database and if the llm_models table exists."""
    try:
        # Create engine
        logger.info(f"Connecting to database with URL: {SQLALCHEMY_DATABASE_URL}")
        engine = create_engine(SQLALCHEMY_DATABASE_URL)
        
        # Create session
        Session = sessionmaker(bind=engine)
        session = Session()
        
        # Check if we can connect to the database
        logger.info("Testing database connection...")
        result = session.execute(text("SELECT 1")).fetchone()
        logger.info(f"Database connection test result: {result}")
        
        # Check if the llm_models table exists
        logger.info("Checking if llm_models table exists...")
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        logger.info(f"Tables in database: {tables}")
        
        if "llm_models" in tables:
            logger.info("llm_models table exists")
            
            # Check if there are any models in the table
            logger.info("Checking if there are any models in the table...")
            models = session.query(LLMModel).all()
            logger.info(f"Found {len(models)} models in the database")
            
            # Print the models
            for model in models:
                logger.info(f"Model: {model.model_id}, Provider: {model.provider_id}")
        else:
            logger.warning("llm_models table does not exist")
            
            # Create the table
            logger.info("Creating llm_models table...")
            Base.metadata.create_all(engine)
            logger.info("llm_models table created")
            
            # Check if the table was created
            inspector = inspect(engine)
            tables = inspector.get_table_names()
            logger.info(f"Tables in database after creation: {tables}")
            
            if "llm_models" in tables:
                logger.info("llm_models table was created successfully")
            else:
                logger.error("Failed to create llm_models table")
        
        # Close the session
        session.close()
        
    except Exception as e:
        logger.error(f"Error checking database: {e}")
        raise

if __name__ == "__main__":
    check_database()
