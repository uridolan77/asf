import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.abspath('.'))

# Try to import the BioMedLMService
try:
    from asf.medical.ml.models.biomedlm import BioMedLMService
    print("Successfully imported BioMedLMService")
except Exception as e:
    print(f"Error importing BioMedLMService: {e}")
