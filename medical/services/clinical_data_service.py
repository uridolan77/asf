#!/usr/bin/env python3
"""
Clinical Data Service

An integrated service that combines terminology (SNOMED CT) and clinical trials data
for the Medical Record Service Clinical API Platform (MRS CAP).

This service connects SNOMED CT terms with relevant clinical trials,
enabling powerful medical data integration capabilities.
"""
import os
import logging
from typing import Dict, List, Optional, Union, Any
import pandas as pd

from ..services.terminology_service import TerminologyService
from ..clients.clinical_trials_gov.clinical_trials_client import (
    ClinicalTrialsClient,
    StudyAnalyzer
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("clinical_data_service")


class ClinicalDataService:
    """
    Integrated clinical data service that combines terminology data with clinical trials.
    
    This service bridges the gap between medical terminology (SNOMED CT) and
    real-world clinical trials, allowing for semantic enrichment and contextual
    understanding of medical concepts.
    """
    
    def __init__(self, 
                 terminology_service: TerminologyService,
                 clinical_trials_client: Optional[ClinicalTrialsClient] = None,
                 clinical_trials_cache_dir: Optional[str] = None,
                 clinical_trials_api_key: Optional[str] = None):
        """
        Initialize the clinical data service.
        
        Args:
            terminology_service: An initialized TerminologyService instance
            clinical_trials_client: An optional initialized ClinicalTrialsClient instance
            clinical_trials_cache_dir: Directory for caching clinical trials data
            clinical_trials_api_key: API key for ClinicalTrials.gov (if required)
        """
        self.terminology_service = terminology_service
        
        # Initialize clinical trials client if not provided
        if clinical_trials_client:
            self.clinical_trials_client = clinical_trials_client
        else:
            self.clinical_trials_client = ClinicalTrialsClient(
                api_key=clinical_trials_api_key,
                cache_dir=clinical_trials_cache_dir
            )
        
        # Initialize the study analyzer
        self.study_analyzer = StudyAnalyzer(self.clinical_trials_client)
        
        logger.info("Clinical Data Service initialized")
    
    #
    # Integrated search functions
    #
    
    def search_concept_and_trials(self, term: str, max_trials: int = 10) -> Dict:
        """
        Search for a medical term and find related clinical trials.
        
        Args:
            term: The medical term to search for
            max_trials: Maximum number of trials to return
            
        Returns:
            Dictionary containing SNOMED CT concepts and related clinical trials
        """
        # Get SNOMED CT concepts
        concepts = self.terminology_service.map_to_snomed(term)
        
        # Get related clinical trials
        trials = self.clinical_trials_client.search_studies(
            query=term,
            fields=["NCTId", "BriefTitle", "OverallStatus", "Phase", "EnrollmentCount"],
            min_rank=1,
            max_rank=max_trials
        )
        
        return {
            "term": term,
            "concepts": concepts[:5],  # Limit to top 5 concepts
            "trials": trials.get("studies", [])
        }
    
    def search_by_concept_id(self, concept_id: str, 
                            terminology: str = "SNOMEDCT",
                            max_trials: int = 10) -> Dict:
        """
        Search for clinical trials related to a specific medical concept.
        
        Args:
            concept_id: The concept identifier (e.g., SNOMED CT concept ID)
            terminology: The terminology system (currently only SNOMED CT supported)
            max_trials: Maximum number of trials to return
            
        Returns:
            Dictionary containing concept details and related clinical trials
        """
        # Get the concept details
        concept = self.terminology_service.get_concept_details(concept_id, terminology)
        
        # Check if concept exists
        if not concept:
            return {
                "concept_id": concept_id,
                "error": f"Concept not found in {terminology}"
            }
        
        # Build a search query based on the concept's preferred term
        term = concept.get("preferredTerm", "")
        
        # Get related clinical trials
        if term:
            trials = self.clinical_trials_client.search_studies(
                query=term,
                fields=["NCTId", "BriefTitle", "OverallStatus", "Phase", "EnrollmentCount"],
                min_rank=1,
                max_rank=max_trials
            )
        else:
            trials = {"studies": []}
        
        return {
            "concept": concept,
            "trials": trials.get("studies", [])
        }
    
    #
    # Terminology mapping functions
    #
    
    def map_condition_to_snomed(self, condition_name: str) -> List[Dict]:
        """
        Map a clinical trial condition to SNOMED CT concepts.
        
        Args:
            condition_name: The condition name from ClinicalTrials.gov
            
        Returns:
            List of matching SNOMED CT concepts
        """
        # Search for matching SNOMED concepts
        return self.terminology_service.map_to_snomed(condition_name)
    
    def map_trial_conditions(self, nct_id: str) -> Dict:
        """
        Map all conditions in a clinical trial to SNOMED CT concepts.
        
        Args:
            nct_id: The ClinicalTrials.gov identifier (NCT number)
            
        Returns:
            Dictionary with condition mappings
        """
        # Get the study details
        study = self.clinical_trials_client.get_study(nct_id)
        
        # Extract the study data and conditions
        study_data = study.get("data", {}).get("study", {})
        conditions = study_data.get("condition", [])
        
        # Map each condition to SNOMED CT
        mappings = {}
        for condition in conditions:
            snomed_concepts = self.map_condition_to_snomed(condition)
            
            # Keep only the top 3 matches for each condition
            mappings[condition] = snomed_concepts[:3] if snomed_concepts else []
        
        return {
            "nct_id": nct_id,
            "study_title": study_data.get("briefTitle", ""),
            "condition_mappings": mappings
        }
    
    #
    # Enhanced analytical functions
    #
    
    def analyze_trial_phases_by_concept(self, concept_id: str, 
                                       terminology: str = "SNOMEDCT",
                                       include_descendants: bool = True,
                                       max_results: int = 500) -> Dict:
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
        # Get the concept and its descendants if requested
        concept = self.terminology_service.get_concept_details(concept_id, terminology)
        
        if not concept:
            return {
                "concept_id": concept_id,
                "error": f"Concept not found in {terminology}"
            }
        
        search_terms = [concept.get("preferredTerm", "")]
        
        # Include descendants if requested
        if include_descendants:
            descendants = self.terminology_service.get_descendants(concept_id, terminology)
            search_terms.extend([d.get("preferredTerm", "") for d in descendants[:10]])  # Limit to top 10
        
        # Build a search query
        query = " OR ".join([f'"{term}"' for term in search_terms if term])
        
        # Get the phase distribution for this query
        phase_distribution = self.study_analyzer.get_phase_distribution(
            query=query,
            max_results=max_results
        )
        
        return {
            "concept": concept,
            "included_terms": search_terms,
            "analysis": phase_distribution
        }
    
    def find_trials_with_semantic_expansion(self, term: str, 
                                          include_similar: bool = True,
                                          max_trials: int = 20) -> Dict:
        """
        Find clinical trials with semantic expansion of the search term.
        
        Args:
            term: The medical term to search for
            include_similar: Whether to include similar concepts
            max_trials: Maximum number of trials to return
            
        Returns:
            Clinical trials related to the term and semantically similar terms
        """
        # First normalize the term using SNOMED CT
        normalized = self.terminology_service.normalize_clinical_term(term)
        
        # Get the primary concept and its preferred term
        primary_concept = normalized["concepts"][0] if normalized["concepts"] else None
        
        search_terms = [term]  # Always include the original term
        
        # Add the normalized term if available and different from original
        if normalized["normalized_term"] != term:
            search_terms.append(normalized["normalized_term"])
        
        # Get similar concepts if requested
        if include_similar and primary_concept:
            similar_concepts = self.terminology_service.find_similar_concepts(
                primary_concept["conceptId"],
                "SNOMEDCT"
            )
            
            # Add terms from similar concepts
            for concept in similar_concepts[:5]:  # Limit to top 5 similar concepts
                term = concept.get("preferredTerm", "")
                if term and term not in search_terms:
                    search_terms.append(term)
        
        # Build a search query
        query = " OR ".join([f'"{t}"' for t in search_terms])
        
        # Search for clinical trials
        trials = self.clinical_trials_client.search_studies(
            query=query,
            fields=["NCTId", "BriefTitle", "OverallStatus", "Phase", 
                   "EnrollmentCount", "Condition", "LeadSponsorName"],
            min_rank=1,
            max_rank=max_trials
        )
        
        return {
            "original_term": term,
            "normalized_term": normalized["normalized_term"],
            "search_terms_used": search_terms,
            "confidence": normalized.get("confidence", 0),
            "trials": trials.get("studies", [])
        }
    
    def get_trial_semantic_context(self, nct_id: str) -> Dict:
        """
        Get semantic context for a clinical trial by mapping its conditions 
        and interventions to SNOMED CT.
        
        Args:
            nct_id: The ClinicalTrials.gov identifier (NCT number)
            
        Returns:
            Semantic context for the trial
        """
        # Get the study details
        study = self.clinical_trials_client.get_study(nct_id)
        
        # Extract the study data
        study_data = study.get("data", {}).get("study", {})
        
        # Map conditions to SNOMED CT
        conditions = study_data.get("condition", [])
        condition_mappings = {}
        
        for condition in conditions:
            concepts = self.map_condition_to_snomed(condition)
            condition_mappings[condition] = concepts[:3] if concepts else []  # Top 3 matches
        
        # Map interventions to SNOMED CT where applicable
        interventions = study_data.get("intervention", [])
        intervention_mappings = {}
        
        for intervention in interventions:
            name = intervention.get("name", "")
            if name:
                concepts = self.map_condition_to_snomed(name)  # Use same function for simplicity
                intervention_mappings[name] = concepts[:2] if concepts else []  # Top 2 matches
        
        # Create hierarchical context for the primary condition
        hierarchical_context = []
        if conditions and condition_mappings.get(conditions[0]):
            primary_concept_id = condition_mappings[conditions[0]][0].get("conceptId")
            if primary_concept_id:
                # Get the ancestors to build the hierarchy
                ancestors = self.terminology_service.get_ancestors(primary_concept_id)
                hierarchical_context = ancestors
        
        return {
            "nct_id": nct_id,
            "study_title": study_data.get("briefTitle", ""),
            "condition_mappings": condition_mappings,
            "intervention_mappings": intervention_mappings,
            "hierarchical_context": hierarchical_context
        }
    
    #
    # Data conversion and utility functions
    #
    
    def trials_to_dataframe(self, trials_data: Dict) -> pd.DataFrame:
        """
        Convert clinical trials search results to a pandas DataFrame.
        
        Args:
            trials_data: The trials data returned from search functions
            
        Returns:
            DataFrame containing the trial data
        """
        # Check if we have a list of studies or need to extract it
        if isinstance(trials_data, dict) and "studies" in trials_data:
            trials = trials_data["studies"]
        elif isinstance(trials_data, list):
            trials = trials_data
        else:
            # If no valid data, return empty DataFrame
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(trials)
        
        # Rename columns for better readability if needed
        column_map = {
            "NCTId": "NCT ID",
            "BriefTitle": "Title",
            "OverallStatus": "Status",
            "Phase": "Phase",
            "EnrollmentCount": "Enrollment",
            "StartDate": "Start Date",
            "CompletionDate": "Completion Date",
            "LeadSponsorName": "Sponsor"
        }
        
        # Apply column renaming for columns that exist in the DataFrame
        for old_name, new_name in column_map.items():
            if old_name in df.columns:
                df.rename(columns={old_name: new_name}, inplace=True)
        
        return df


# Example usage
def example_usage():
    """Demonstrate usage of the Clinical Data Service."""
    from asf.medical.services.terminology_service import TerminologyService
    
    # Initialize the terminology service
    terminology_service = TerminologyService(
        snomed_access_mode="umls",
        snomed_api_key=os.environ.get("UMLS_API_KEY", "your-umls-api-key-here"),
        snomed_cache_dir="./terminology_cache"
    )
    
    # Initialize the clinical trials client
    clinical_trials_client = ClinicalTrialsClient(
        cache_dir="./ctgov_cache"
    )
    
    # Initialize the clinical data service
    service = ClinicalDataService(
        terminology_service=terminology_service,
        clinical_trials_client=clinical_trials_client
    )
    
    # Search for a term and find related trials
    results = service.search_concept_and_trials("type 2 diabetes")
    
    print(f"Search results for 'type 2 diabetes':")
    print(f"Found {len(results['concepts'])} SNOMED CT concepts")
    print(f"Found {len(results['trials'])} related clinical trials")
    
    # Print the top concept and trial
    if results['concepts']:
        concept = results['concepts'][0]
        print(f"\nTop concept: {concept['conceptId']} - {concept['preferredTerm']}")
    
    if results['trials']:
        trial = results['trials'][0]
        print(f"\nTop trial: {trial['NCTId']} - {trial['BriefTitle']}")
    
    # Try semantic expansion
    expanded_results = service.find_trials_with_semantic_expansion("heart attack")
    
    print(f"\nExpanded search for 'heart attack':")
    print(f"Normalized term: {expanded_results['normalized_term']}")
    print(f"Search terms used: {expanded_results['search_terms_used']}")
    print(f"Found {len(expanded_results['trials'])} related trials")
    
    # Get semantic context for a trial
    context = service.get_trial_semantic_context("NCT03980509")  # Example NCT ID
    
    print(f"\nSemantic context for trial {context['nct_id']}:")
    print(f"Title: {context['study_title']}")
    print(f"Mapped {len(context['condition_mappings'])} conditions to SNOMED CT")
    print(f"Mapped {len(context['intervention_mappings'])} interventions to SNOMED CT")
    
    # Convert to DataFrame
    df = service.trials_to_dataframe(expanded_results['trials'])
    print(f"\nDataFrame shape: {df.shape}")


if __name__ == "__main__":
    example_usage()