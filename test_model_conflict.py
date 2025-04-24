#!/usr/bin/env python
"""
Test script to verify that we've resolved the model conflicts between
the bollm User and medical MedicalUser models.
"""

# Import the SQLAlchemy modules first
import sqlalchemy
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, registry

# Now try to import both model classes
try:
    print("Loading bollm User model...")
    from asf.bollm.backend.models import User
    print("Successfully loaded bollm User model")
    
    print("Loading medical MedicalUser model...")
    from asf.medical.storage.models import MedicalUser 
    print("Successfully loaded medical MedicalUser model")
    
    # Test creating instances of both models
    bollm_user = User(username="test_bollm", email="test_bollm@example.com")
    medical_user = MedicalUser(email="test_medical@example.com", hashed_password="password")
    
    print(f"\nSuccessfully instantiated bollm User: {bollm_user}")
    print(f"Successfully instantiated medical MedicalUser: {medical_user}")
    
    print("\nBoth models successfully loaded and instantiated without conflicts!")
except Exception as e:
    print(f"\nERROR: Failed to load or instantiate models: {str(e)}")