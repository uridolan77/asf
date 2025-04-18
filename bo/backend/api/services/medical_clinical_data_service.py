"""
Medical Clinical Data Service for BO backend.

This service provides a bridge between the BO frontend and the Medical Research
clinical data functionality, integrating terminology (SNOMED CT) with clinical trials data.
"""
import os
import logging
from typing import Dict, List, Optional, Union, Any

# Import from the medical module
from medical.services.clinical_data_service import ClinicalDataService
from medical.services.terminology_service import TerminologyService
from medical.clients.clinical_trials_gov.clinical_trials_client import ClinicalTrialsClient

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MedicalClinicalDataService:
    """
    Service for interacting with the medical module's ClinicalDataService.
    This provides a bridge between the BO frontend and the Medical Research clinical data functionality.
    """
    def __init__(self):
        """Initialize with direct access to the medical module's ClinicalDataService"""
        # Initialize the terminology service with default settings
        self.terminology_service = TerminologyService(
            snomed_access_mode=os.environ.get("SNOMED_ACCESS_MODE", "umls"),
            snomed_api_key=os.environ.get("UMLS_API_KEY"),
            snomed_cache_dir=os.environ.get("SNOMED_CACHE_DIR", "./terminology_cache"),
            snomed_edition=os.environ.get("SNOMED_EDITION", "US")
        )
        
        # Initialize the clinical trials client
        self.clinical_trials_client = ClinicalTrialsClient(
            cache_dir=os.environ.get("CLINICAL_TRIALS_CACHE_DIR", "./clinical_trials_cache")
        )
        
        # Initialize the clinical data service
        self.clinical_data_service = ClinicalDataService(
            terminology_service=self.terminology_service,
            clinical_trials_client=self.clinical_trials_client
        )
        
        logger.info("MedicalClinicalDataService initialized successfully")
    
    def search_concept_and_trials(self, term: str, max_trials: int = 10) -> Dict[str, Any]:
        """
        Search for a medical term and find related clinical trials.
        
        Args:
            term: The medical term to search for
            max_trials: Maximum number of trials to return
            
        Returns:
            Dictionary containing SNOMED CT concepts and related clinical trials
        """
        try:
            results = self.clinical_data_service.search_concept_and_trials(term, max_trials=max_trials)
            
            return {
                "success": True,
                "message": f"Found {len(results.get('trials', []))} trials for term: {term}",
                "data": results
            }
        except Exception as e:
            logger.error(f"Error searching for concept and trials: {str(e)}")
            return {
                "success": False,
                "message": f"Failed to search for concept and trials: {str(e)}",
                "data": None
            }
    
    def search_by_concept_id(
        self, 
        concept_id: str, 
        terminology: str = "SNOMEDCT",
        max_trials: int = 10
    ) -> Dict[str, Any]:
        """
        Search for clinical trials related to a specific medical concept.
        
        Args:
            concept_id: The concept identifier (e.g., SNOMED CT concept ID)
            terminology: The terminology system (currently only SNOMED CT supported)
            max_trials: Maximum number of trials to return
            
        Returns:
            Dictionary containing concept details and related clinical trials
        """
        try:
            results = self.clinical_data_service.search_by_concept_id(
                concept_id, 
                terminology=terminology, 
                max_trials=max_trials
            )
            
            if "error" in results:
                return {
                    "success": False,
                    "message": results["error"],
                    "data": None
                }
            
            return {
                "success": True,
                "message": f"Found {len(results.get('trials', []))} trials for concept ID: {concept_id}",
                "data": results
            }
        except Exception as e:
            logger.error(f"Error searching by concept ID: {str(e)}")
            return {
                "success": False,
                "message": f"Failed to search by concept ID: {str(e)}",
                "data": None
            }
    
    def map_trial_conditions(self, nct_id: str) -> Dict[str, Any]:
        """
        Map all conditions in a clinical trial to SNOMED CT concepts.
        
        Args:
            nct_id: The ClinicalTrials.gov identifier (NCT number)
            
        Returns:
            Dictionary with condition mappings
        """
        try:
            mappings = self.clinical_data_service.map_trial_conditions(nct_id)
            
            return {
                "success": True,
                "message": f"Mapped conditions for trial {nct_id} to SNOMED CT concepts",
                "data": mappings
            }
        except Exception as e:
            logger.error(f"Error mapping trial conditions: {str(e)}")
            return {
                "success": False,
                "message": f"Failed to map trial conditions: {str(e)}",
                "data": None
            }
    
    def find_trials_with_semantic_expansion(
        self,
        term: str,
        include_similar: bool = True,
        max_trials: int = 20
    ) -> Dict[str, Any]:
        """
        Find clinical trials with semantic expansion of the search term.
        
        Args:
            term: The medical term to search for
            include_similar: Whether to include similar concepts
            max_trials: Maximum number of trials to return
            
        Returns:
            Dictionary containing search results with semantic expansion
        """
        try:
            results = self.clinical_data_service.find_trials_with_semantic_expansion(
                term,
                include_similar=include_similar,
                max_trials=max_trials
            )
            
            return {
                "success": True,
                "message": f"Found {len(results.get('trials', []))} trials with semantic expansion for term: {term}",
                "data": results
            }
        except Exception as e:
            logger.error(f"Error finding trials with semantic expansion: {str(e)}")
            return {
                "success": False,
                "message": f"Failed to find trials with semantic expansion: {str(e)}",
                "data": None
            }
    
    def get_trial_semantic_context(self, nct_id: str) -> Dict[str, Any]:
        """
        Get semantic context for a clinical trial by mapping its conditions 
        and interventions to SNOMED CT.
        
        Args:
            nct_id: The ClinicalTrials.gov identifier (NCT number)
            
        Returns:
            Semantic context for the trial
        """
        try:
            context = self.clinical_data_service.get_trial_semantic_context(nct_id)
            
            return {
                "success": True,
                "message": f"Retrieved semantic context for trial {nct_id}",
                "data": context
            }
        except Exception as e:
            logger.error(f"Error getting trial semantic context: {str(e)}")
            return {
                "success": False,
                "message": f"Failed to get trial semantic context: {str(e)}",
                "data": None
            }
    
    def analyze_trial_phases_by_concept(
        self,
        concept_id: str,
        terminology: str = "SNOMEDCT",
        include_descendants: bool = True,
        max_results: int = 500
    ) -> Dict[str, Any]:
        """
        Analyze clinical trial phases for a medical concept.
        
        Args:
            concept_id: The concept identifier (e.g., SNOMED CT concept ID)
            terminology: The terminology system (currently only SNOMED CT supported)
            include_descendants: Whether to include descendant concepts
            max_results: Maximum number of trials to analyze
            
        Returns:
            Analysis of trial phases for the concept
        """
        try:
            analysis = self.clinical_data_service.analyze_trial_phases_by_concept(
                concept_id,
                terminology=terminology,
                include_descendants=include_descendants,
                max_results=max_results
            )
            
            if "error" in analysis:
                return {
                    "success": False,
                    "message": analysis["error"],
                    "data": None
                }
            
            return {
                "success": True,
                "message": f"Analyzed trial phases for concept ID: {concept_id}",
                "data": analysis
            }
        except Exception as e:
            logger.error(f"Error analyzing trial phases: {str(e)}")
            return {
                "success": False,
                "message": f"Failed to analyze trial phases: {str(e)}",
                "data": None
            }

# Dependency to get the medical clinical data service
def get_medical_clinical_data_service() -> MedicalClinicalDataService:
    """Factory function to create and provide a MedicalClinicalDataService instance."""
    return MedicalClinicalDataService()
