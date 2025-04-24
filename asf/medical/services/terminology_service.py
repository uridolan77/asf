#!/usr/bin/env python3
"""
Terminology Service

A service layer that wraps the SNOMED CT client and other terminology clients
to provide high-level terminology features for the Medical Record Service
Clinical API Platform (MRS CAP).

Features:
- Term normalization
- Code lookup
- Semantic search
- Hierarchical navigation
- Concept relationships
- Expression Constraint Language (ECL) support
- Reference set access
"""
import os
import logging
from typing import Dict, List, Optional, Union, Set, Tuple
from functools import lru_cache
from pathlib import Path

from ..clients.snomed.snomed_client import (
    SnomedClient,
    SnomedSubsumptionTester,
    SnomedExpressionConverter,
    SnomedReferenceSetManager,
    IS_A_RELATIONSHIP,
    FINDING_SITE,
    ASSOCIATED_WITH,
    CAUSATIVE_AGENT,
    PATHOLOGICAL_PROCESS,
    METHOD,
    PROCEDURE_SITE,
    CLINICAL_FINDING,
    PROCEDURE,
    BODY_STRUCTURE,
    ORGANISM,
    SUBSTANCE,
    PHARMACEUTICAL_PRODUCT,
    SITUATION,
    EVENT
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("terminology_service")


class TerminologyService:
    """
    Service layer for terminology operations in the MRS CAP.
    
    This service wraps terminology clients (SNOMED CT, ICD-10, etc.) and provides
    high-level functionality for working with medical terminologies.
    """
    
    def __init__(self, 
                 snomed_access_mode: str = "umls",
                 snomed_api_key: Optional[str] = None,
                 snomed_api_url: Optional[str] = None,
                 snomed_cache_dir: Optional[str] = None,
                 snomed_edition: str = "US",
                 snomed_version: Optional[str] = None,
                 snomed_local_data_path: Optional[str] = None):
        """
        Initialize the terminology service.
        
        Args:
            snomed_access_mode: How to access SNOMED CT. Options: "umls", "api", "local"
            snomed_api_key: API key for authentication (required for "umls" and may be required for "api")
            snomed_api_url: Base URL for API access (required for "api" mode)
            snomed_cache_dir: Directory to store cache data
            snomed_edition: SNOMED CT edition to use (e.g., "US", "INT")
            snomed_version: Specific version of SNOMED CT to use
            snomed_local_data_path: Path to local SNOMED CT files (required for "local" mode)
        """
        # Initialize SNOMED CT client
        self.snomed_client = SnomedClient(
            access_mode=snomed_access_mode,
            api_key=snomed_api_key or os.environ.get("UMLS_API_KEY"),
            api_url=snomed_api_url,
            cache_dir=snomed_cache_dir,
            edition=snomed_edition,
            version=snomed_version,
            local_data_path=snomed_local_data_path
        )
        
        # Initialize helper classes
        self.subsumption_tester = SnomedSubsumptionTester(self.snomed_client)
        self.expression_converter = SnomedExpressionConverter(self.snomed_client)
        self.refset_manager = SnomedReferenceSetManager(self.snomed_client)
        
        logger.info("Terminology service initialized")
    
    #
    # Term normalization functions
    #
    
    def normalize_clinical_term(self, term: str) -> Dict:
        """
        Normalize a clinical term to its standard form.
        
        Args:
            term: The clinical term to normalize
            
        Returns:
            Dictionary containing normalized term and matching concepts
        """
        # Search for matching concepts
        matches = self.snomed_client.search(term)
        
        if not matches:
            return {
                "original_term": term,
                "normalized_term": term,
                "confidence": 0.0,
                "concepts": []
            }
        
        # Use the highest-scored match as the normalized form
        best_match = matches[0]
        
        return {
            "original_term": term,
            "normalized_term": best_match["preferredTerm"],
            "confidence": min(best_match["score"] / 100, 1.0),
            "concepts": matches[:5]  # Return top 5 matches
        }
    
    def map_to_snomed(self, term: str) -> List[Dict]:
        """
        Map a term to SNOMED CT concepts.
        
        Args:
            term: The term to map
            
        Returns:
            List of matching SNOMED CT concepts
        """
        return self.snomed_client.search(term)
    
    #
    # Code lookup functions
    #
    
    def get_concept_details(self, code: str, terminology: str = "SNOMEDCT") -> Dict:
        """
        Get detailed information about a concept.
        
        Args:
            code: The concept code
            terminology: The terminology system (currently only SNOMED CT supported)
            
        Returns:
            Dictionary containing concept details
        """
        if terminology.upper() == "SNOMEDCT":
            return self.snomed_client.get_concept(code)
        else:
            raise ValueError(f"Unsupported terminology: {terminology}")
    
    def get_concept_name(self, code: str, terminology: str = "SNOMEDCT") -> str:
        """
        Get the preferred name of a concept.
        
        Args:
            code: The concept code
            terminology: The terminology system (currently only SNOMED CT supported)
            
        Returns:
            Preferred name as a string
        """
        try:
            concept = self.get_concept_details(code, terminology)
            return concept.get("preferredTerm", "")
        except Exception as e:
            logger.error(f"Error getting concept name for {code}: {e}")
            return ""
    
    #
    # Semantic search functions
    #
    
    def semantic_search(self, query: str, max_results: int = 20) -> List[Dict]:
        """
        Perform a semantic search for clinical concepts.
        
        Args:
            query: The search query
            max_results: Maximum number of results to return
            
        Returns:
            List of matching concepts with relevance scores
        """
        return self.snomed_client.search(query, max_results=max_results)
    
    def find_similar_concepts(self, concept_id: str, terminology: str = "SNOMEDCT") -> List[Dict]:
        """
        Find concepts similar to a given concept.
        
        Args:
            concept_id: The concept identifier
            terminology: The terminology system
            
        Returns:
            List of similar concepts
        """
        if terminology.upper() == "SNOMEDCT":
            # Get the concept details
            concept = self.snomed_client.get_concept(concept_id)
            
            # Use the preferred term to search for similar concepts
            similar = self.snomed_client.search(concept.get("preferredTerm", ""))
            
            # Filter out the original concept
            return [c for c in similar if c["conceptId"] != concept_id]
        else:
            raise ValueError(f"Unsupported terminology: {terminology}")
    
    #
    # Hierarchical navigation functions
    #
    
    def get_parents(self, concept_id: str, terminology: str = "SNOMEDCT") -> List[Dict]:
        """
        Get direct parent concepts.
        
        Args:
            concept_id: The concept identifier
            terminology: The terminology system
            
        Returns:
            List of parent concepts
        """
        if terminology.upper() == "SNOMEDCT":
            return self.snomed_client.get_parents(concept_id, direct_only=True)
        else:
            raise ValueError(f"Unsupported terminology: {terminology}")
    
    def get_children(self, concept_id: str, terminology: str = "SNOMEDCT") -> List[Dict]:
        """
        Get direct child concepts.
        
        Args:
            concept_id: The concept identifier
            terminology: The terminology system
            
        Returns:
            List of child concepts
        """
        if terminology.upper() == "SNOMEDCT":
            return self.snomed_client.get_children(concept_id, direct_only=True)
        else:
            raise ValueError(f"Unsupported terminology: {terminology}")
    
    def get_ancestors(self, concept_id: str, terminology: str = "SNOMEDCT") -> List[Dict]:
        """
        Get all ancestor concepts.
        
        Args:
            concept_id: The concept identifier
            terminology: The terminology system
            
        Returns:
            List of ancestor concepts
        """
        if terminology.upper() == "SNOMEDCT":
            return self.snomed_client.get_parents(concept_id, direct_only=False)
        else:
            raise ValueError(f"Unsupported terminology: {terminology}")
    
    def get_descendants(self, concept_id: str, terminology: str = "SNOMEDCT") -> List[Dict]:
        """
        Get all descendant concepts.
        
        Args:
            concept_id: The concept identifier
            terminology: The terminology system
            
        Returns:
            List of descendant concepts
        """
        if terminology.upper() == "SNOMEDCT":
            return self.snomed_client.get_children(concept_id, direct_only=False)
        else:
            raise ValueError(f"Unsupported terminology: {terminology}")
    
    def is_a(self, concept_id: str, potential_parent_id: str, terminology: str = "SNOMEDCT") -> bool:
        """
        Check if a concept is a subtype of another concept.
        
        Args:
            concept_id: The concept to check
            potential_parent_id: The potential parent concept
            terminology: The terminology system
            
        Returns:
            True if the concept is a subtype of the potential parent, False otherwise
        """
        if terminology.upper() == "SNOMEDCT":
            return self.subsumption_tester.is_subsumed_by(concept_id, potential_parent_id)
        else:
            raise ValueError(f"Unsupported terminology: {terminology}")
    
    #
    # Concept relationship functions
    #
    
    def get_relationships(self, concept_id: str, relationship_type: Optional[str] = None, 
                        terminology: str = "SNOMEDCT") -> List[Dict]:
        """
        Get relationships for a given concept.
        
        Args:
            concept_id: The concept identifier
            relationship_type: Optional relationship type ID to filter by
            terminology: The terminology system
            
        Returns:
            List of relationship dictionaries
        """
        if terminology.upper() == "SNOMEDCT":
            return self.snomed_client.get_relationships(concept_id, relationship_type)
        else:
            raise ValueError(f"Unsupported terminology: {terminology}")
    
    def get_finding_sites(self, diagnosis_concept_id: str) -> List[Dict]:
        """
        Get anatomical sites associated with a diagnosis.
        
        Args:
            diagnosis_concept_id: The SNOMED CT concept ID for the diagnosis
            
        Returns:
            List of anatomical sites
        """
        attributes = self.snomed_client.get_concept_attributes(
            diagnosis_concept_id, 
            attribute_type=FINDING_SITE
        )
        
        sites = []
        for attribute in attributes:
            site_id = attribute.get("destinationId")
            if site_id:
                try:
                    site = self.snomed_client.get_concept(site_id)
                    sites.append(site)
                except Exception as e:
                    logger.warning(f"Error getting site concept {site_id}: {e}")
        
        return sites
    
    def get_causative_agents(self, diagnosis_concept_id: str) -> List[Dict]:
        """
        Get causative agents associated with a diagnosis.
        
        Args:
            diagnosis_concept_id: The SNOMED CT concept ID for the diagnosis
            
        Returns:
            List of causative agents
        """
        attributes = self.snomed_client.get_concept_attributes(
            diagnosis_concept_id, 
            attribute_type=CAUSATIVE_AGENT
        )
        
        agents = []
        for attribute in attributes:
            agent_id = attribute.get("destinationId")
            if agent_id:
                try:
                    agent = self.snomed_client.get_concept(agent_id)
                    agents.append(agent)
                except Exception as e:
                    logger.warning(f"Error getting agent concept {agent_id}: {e}")
        
        return agents
    
    #
    # Advanced features
    #
    
    def evaluate_ecl(self, expression: str, max_results: int = 200) -> List[Dict]:
        """
        Evaluate an Expression Constraint Language (ECL) expression.
        
        Args:
            expression: The ECL expression
            max_results: Maximum number of results to return
            
        Returns:
            List of matching concepts
        """
        return self.snomed_client.evaluate_ecl(expression, max_results=max_results)
    
    def find_all_diabetes_types(self) -> List[Dict]:
        """
        Find all types of diabetes using ECL.
        
        Returns:
            List of diabetes concept types
        """
        # 73211009 is the concept ID for "Diabetes mellitus"
        return self.evaluate_ecl("<<73211009")
    
    def get_reference_set_members(self, refset_id: str, max_results: int = 100) -> List[Dict]:
        """
        Get members of a reference set.
        
        Args:
            refset_id: The reference set identifier
            max_results: Maximum number of results to return
            
        Returns:
            List of reference set member dictionaries
        """
        return self.refset_manager.get_reference_set_members(refset_id, max_results=max_results)
    
    def convert_expression_to_human_readable(self, expression: str) -> str:
        """
        Convert a SNOMED CT expression to human-readable form.
        
        Args:
            expression: SNOMED CT expression in standard form
            
        Returns:
            Human-readable version of the expression
        """
        return self.expression_converter.to_human_readable(expression)


# Example usage
def example_usage():
    """Demonstrate the usage of the Terminology Service."""
    # Initialize the service with UMLS access
    service = TerminologyService(
        snomed_access_mode="umls",
        snomed_api_key=os.environ.get("UMLS_API_KEY", "your-umls-api-key-here"),
        snomed_cache_dir="./terminology_cache"
    )
    
    # Normalize a clinical term
    normalization = service.normalize_clinical_term("heart attack")
    print(f"Normalized 'heart attack' to: {normalization['normalized_term']}")
    print(f"Confidence: {normalization['confidence']:.2f}")
    
    # Get details for a concept
    concept = service.get_concept_details("22298006")  # Myocardial infarction
    print(f"\nDetails for concept {concept['conceptId']}:")
    print(f"  FSN: {concept['fsn']}")
    print(f"  Preferred Term: {concept['preferredTerm']}")
    
    # Find anatomical sites for a diagnosis
    sites = service.get_finding_sites("22298006")  # Myocardial infarction
    print(f"\nAnatomical sites for Myocardial infarction:")
    for site in sites:
        print(f"  {site['conceptId']}: {site['preferredTerm']}")
    
    # Find all types of diabetes
    diabetes_types = service.find_all_diabetes_types()
    print(f"\nFound {len(diabetes_types)} types of diabetes")
    for i, diabetes in enumerate(diabetes_types[:5]):  # Print first 5
        print(f"  {i+1}. {diabetes['preferredTerm']}")
    
    # Get members of a reference set
    members = service.get_reference_set_members("723264001", max_results=5)  # Lateralizable body structure reference set
    print(f"\nRetrieved {len(members)} members from reference set 723264001")


if __name__ == "__main__":
    example_usage()