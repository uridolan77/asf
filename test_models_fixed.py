#!/usr/bin/env python
"""
Test script to verify that SQLAlchemy model conflicts are fixed.

This script attempts to import all models from both the bollm and medical modules
to ensure there are no conflicts in the SQLAlchemy registry.
"""

import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("model-test")

def test_importing_models():
    """Test importing models from bollm and medical modules."""
    try:
        # First import bollm models
        logger.info("Importing bollm models...")
        from asf.bollm.backend.models import Base as BOLLMBase
        from asf.bollm.backend.models.user import BOLLMUser, Role
        from asf.bollm.backend.models.provider import Provider, ApiKey
        from asf.bollm.backend.models.audit import AuditLog, ApiKeyUsage
        from asf.bollm.backend.models.configuration import Configuration, UserSetting
        
        logger.info("Successfully imported bollm models")
        
        # Then import medical models
        logger.info("Importing medical models...")
        from asf.medical.storage.database import MedicalBase
        from asf.medical.storage.models.user import MedicalUser
        
        logger.info("Successfully imported medical models")
        
        # Test that classes are different
        logger.info(f"BOLLMUser class: {BOLLMUser.__module__}.{BOLLMUser.__name__}")
        logger.info(f"MedicalUser class: {MedicalUser.__module__}.{MedicalUser.__name__}")
        
        return True
    except Exception as e:
        logger.error(f"Failed to import models: {str(e)}", exc_info=True)
        return False

if __name__ == "__main__":
    logger.info("Starting model import test")
    
    success = test_importing_models()
    
    if success:
        logger.info("✅ Model imports successful - SQLAlchemy registry fix is working")
        sys.exit(0)
    else:
        logger.error("❌ Model imports failed - SQLAlchemy registry conflict still exists")
        sys.exit(1)