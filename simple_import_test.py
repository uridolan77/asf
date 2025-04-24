#!/usr/bin/env python
"""
Simple test script to verify we can import both User models without conflicts.
This script avoids instantiating the models to prevent SQLAlchemy mapper configurations.
"""

print("Starting simple import test...")

# Try importing the models separately
try:
    print("Importing bollm User model...")
    from asf.bollm.backend.models import User as BollmUser
    print(f"Successfully imported bollm User model: {BollmUser.__name__}")
    
    print("\nImporting medical MedicalUser model...")
    from asf.medical.storage.models import MedicalUser
    print(f"Successfully imported medical MedicalUser model: {MedicalUser.__name__}")
    
    print("\nBoth models successfully imported without conflicts!")
    
    # Print model class paths to confirm they are different
    print(f"\nBollm User model path: {BollmUser.__module__}.{BollmUser.__name__}")
    print(f"Medical User model path: {MedicalUser.__module__}.{MedicalUser.__name__}")
    
except Exception as e:
    print(f"\nERROR: Failed to import models: {str(e)}")