import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.abspath('.'))

# Try to import some of the modules
try:
    from asf.medical.ml.models.biomedlm import BioMedLMService
    print("Successfully imported BioMedLMService")
except Exception as e:
    print(f"Error importing BioMedLMService: {e}")

try:
    from asf.medical.ml.models.tsmixer import TSMixerService
    print("Successfully imported TSMixerService")
except Exception as e:
    print(f"Error importing TSMixerService: {e}")

try:
    from asf.medical.ml.services.contradiction_classifier_service import ContradictionClassifierService
    print("Successfully imported ContradictionClassifierService")
except Exception as e:
    print(f"Error importing ContradictionClassifierService: {e}")
