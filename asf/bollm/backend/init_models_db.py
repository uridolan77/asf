import os
import sys
import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError

# Add the parent directory to the path so we can import the models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from asf.bollm.backend.models.base import Base
from asf.bollm.backend.models.llm_model import LLMModel
from asf.bollm.backend.config.config import SQLALCHEMY_DATABASE_URL

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def init_db():
    """Initialize the database."""
    try:
        # Create engine
        engine = create_engine(SQLALCHEMY_DATABASE_URL)
        
        # Create tables
        Base.metadata.create_all(engine)
        
        # Create session
        Session = sessionmaker(bind=engine)
        session = Session()
        
        logger.info("Database tables created successfully")
        
        return session
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        raise

def create_initial_models(session):
    """Check if models exist in the database."""
    try:
        # Check if models already exist
        existing_models = session.query(LLMModel).all()
        
        if existing_models:
            logger.info(f"Found {len(existing_models)} existing models in the database")
        else:
            logger.info("No models found in the database")
            
        # No hardcoded models - all models should be managed through the database
    except SQLAlchemyError as e:
        session.rollback()
        logger.error(f"Error creating initial models: {e}")
        raise

def main():
    """Main function."""
    try:
        # Initialize database
        session = init_db()
        
        # Create initial models
        create_initial_models(session)
        
        logger.info("Database initialization completed successfully")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        sys.exit(1)
    finally:
        if 'session' in locals():
            session.close()

if __name__ == "__main__":
    main()
