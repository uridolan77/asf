"""
UMLS API Module for Medical Research Synthesizer
This module provides connectivity to the UMLS (Unified Medical Language System) API,
facilitating access to standardized medical terminology and relationships.
Prerequisites:
- UMLS account (https://uts.nlm.nih.gov/uts/signup-login)
- API key from UMLS
- Python 3.6+
- Required packages: requests, dotenv
Usage:
1. Store your UMLS API key and credentials in a .env file
2. Use the UMLSClient class to interact with the UMLS API
"""
import time
import json
import logging
import requests
from typing import Dict, List, Optional, Any
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('umls_api')
class UMLSClient:
    """Client for interacting with the UMLS API."""
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the UMLS API client.
        Args:
            api_key: UMLS API key. If not provided, will attempt to load from environment variables.
        current_time = time.time()
        return (not self.tgt or 
                current_time - self.token_timestamp > self.token_lifetime)
    def authenticate(self) -> bool:
        """
        Authenticate with the UMLS API and obtain a ticket granting ticket (TGT).
        Returns:
            bool: True if authentication was successful, False otherwise.
        """
        if not self._need_new_token():
            logger.debug("Using existing authentication token")
            return True
        if not self.api_key:
            logger.error("Cannot authenticate: No API key provided")
            return False
        try:
            auth_params = {'apikey': self.api_key}
            response = requests.post(
                self.auth_endpoint,
                data=auth_params
            )
            response.raise_for_status()
            self.tgt = response.text
            self.token_timestamp = time.time()
            logger.info("Successfully authenticated with UMLS API")
            return True
        except requests.exceptions.RequestException as e:
            logger.error(f"Authentication failed: {str(e)}")
            return False
    def get_service_ticket(self) -> Optional[str]:
        """
        Get a service ticket for API access.
        Returns:
            Optional[str]: Service ticket if successful, None otherwise.
        """
        if not self.authenticate():
            return None
        try:
            service_params = {'service': self.base_url}
            response = requests.post(
                self.tgt,
                data=service_params
            )
            response.raise_for_status()
            service_ticket = response.text
            return service_ticket
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get service ticket: {str(e)}")
            return None
    def search_term(self, term: str, search_type: str = "exact") -> Optional[Dict]:
        """
        Search for a medical term in UMLS.
        Args:
            term: The medical term to search for
            search_type: The type of search to perform (exact, words, approximate, etc.)
        Returns:
            Optional[Dict]: Search results if successful, None otherwise
        """
        service_ticket = self.get_service_ticket()
        if not service_ticket:
            return None
        try:
            search_url = f"{self.base_url}/search/current"
            params = {
                'string': term,
                'searchType': search_type,
                'ticket': service_ticket
            }
            response = requests.get(search_url, params=params)
            response.raise_for_status()
            results = response.json()
            logger.info(f"Successfully searched for term: '{term}'")
            return results
        except requests.exceptions.RequestException as e:
            logger.error(f"Search failed: {str(e)}")
            return None
    def get_concept(self, cui: str) -> Optional[Dict]:
        """
        Retrieve information about a specific concept by CUI.
        Args:
            cui: The Concept Unique Identifier (CUI)
        Returns:
            Optional[Dict]: Concept information if successful, None otherwise
        """
        service_ticket = self.get_service_ticket()
        if not service_ticket:
            return None
        try:
            concept_url = f"{self.base_url}/content/current/CUI/{cui}"
            params = {'ticket': service_ticket}
            response = requests.get(concept_url, params=params)
            response.raise_for_status()
            concept_info = response.json()
            logger.info(f"Successfully retrieved information for CUI: {cui}")
            return concept_info
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to retrieve concept {cui}: {str(e)}")
            return None
    def get_semantic_types(self, cui: str) -> Optional[List[Dict]]:
        """
        Retrieve semantic types for a specific concept.
        Args:
            cui: The Concept Unique Identifier (CUI)
        Returns:
            Optional[List[Dict]]: List of semantic types if successful, None otherwise
        """
        service_ticket = self.get_service_ticket()
        if not service_ticket:
            return None
        try:
            semantic_url = f"{self.base_url}/content/current/CUI/{cui}/SemanticTypes"
            params = {'ticket': service_ticket}
            response = requests.get(semantic_url, params=params)
            response.raise_for_status()
            semantic_info = response.json()
            logger.info(f"Successfully retrieved semantic types for CUI: {cui}")
            return semantic_info.get('result', [])
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to retrieve semantic types for {cui}: {str(e)}")
            return None
    def get_relations(self, cui: str, relation_type: Optional[str] = None) -> Optional[List[Dict]]:
        """
        Retrieve relationships for a specific concept.
        Args:
            cui: The Concept Unique Identifier (CUI)
            relation_type: Optional filter for specific relation types
        Returns:
            Optional[List[Dict]]: List of relations if successful, None otherwise
        """
        service_ticket = self.get_service_ticket()
        if not service_ticket:
            return None
        try:
            relations_url = f"{self.base_url}/content/current/CUI/{cui}/relations"
            params = {'ticket': service_ticket}
            if relation_type:
                params['relationTypes'] = relation_type
            response = requests.get(relations_url, params=params)
            response.raise_for_status()
            relations_info = response.json()
            logger.info(f"Successfully retrieved relations for CUI: {cui}")
            return relations_info.get('result', [])
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to retrieve relations for {cui}: {str(e)}")
            return None
    def get_source_concepts(self, cui: str, source: str) -> Optional[List[Dict]]:
        """
        Retrieve source-specific concept information.
        Args:
            cui: The Concept Unique Identifier (CUI)
            source: The vocabulary source (e.g., 'SNOMEDCT_US', 'ICD10CM')
        Returns:
            Optional[List[Dict]]: List of source concepts if successful, None otherwise
        """
        service_ticket = self.get_service_ticket()
        if not service_ticket:
            return None
        try:
            source_url = f"{self.base_url}/content/current/CUI/{cui}/source/{source}"
            params = {'ticket': service_ticket}
            response = requests.get(source_url, params=params)
            response.raise_for_status()
            source_info = response.json()
            logger.info(f"Successfully retrieved {source} concepts for CUI: {cui}")
            return source_info.get('result', [])
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to retrieve {source} concepts for {cui}: {str(e)}")
            return None
    def get_definition(self, cui: str, source: Optional[str] = None) -> Optional[List[Dict]]:
        """
        Retrieve definitions for a specific concept.
        Args:
            cui: The Concept Unique Identifier (CUI)
            source: Optional source vocabulary to filter definitions
        Returns:
            Optional[List[Dict]]: List of definitions if successful, None otherwise
        """
        service_ticket = self.get_service_ticket()
        if not service_ticket:
            return None
        try:
            definitions_url = f"{self.base_url}/content/current/CUI/{cui}/definitions"
            params = {'ticket': service_ticket}
            if source:
                params['source'] = source
            response = requests.get(definitions_url, params=params)
            response.raise_for_status()
            definitions_info = response.json()
            logger.info(f"Successfully retrieved definitions for CUI: {cui}")
            return definitions_info.get('result', [])
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to retrieve definitions for {cui}: {str(e)}")
            return None
    def find_contradictions(self, term1: str, term2: str) -> Dict[str, Any]:
        """
        Find potential contradictions between two medical terms by examining their
        relationships and semantic types.
        This is a simplified example and would need to be expanded for a real system.
        Args:
            term1: First medical term
            term2: Second medical term
        Returns:
            Dict[str, Any]: Structure containing potential contradiction information
        """
        result = {
            'term1': term1,
            'term2': term2,
            'contradiction_found': False,
            'contradiction_type': None,
            'explanation': None,
            'confidence': 0.0,
            'term1_info': None,
            'term2_info': None
        }
        term1_search = self.search_term(term1)
        term2_search = self.search_term(term2)
        if not term1_search or not term2_search:
            result['explanation'] = "Could not find one or both terms in UMLS"
            return result
        try:
            cui1 = term1_search['result']['results'][0]['ui']
            cui2 = term2_search['result']['results'][0]['ui']
            concept1 = self.get_concept(cui1)
            concept2 = self.get_concept(cui2)
            result['term1_info'] = concept1
            result['term2_info'] = concept2
            sem_types1 = self.get_semantic_types(cui1)
            sem_types2 = self.get_semantic_types(cui2)
            relations1 = self.get_relations(cui1)
            relations2 = self.get_relations(cui2)
            if sem_types1 and sem_types2:
                st1_names = [st['name'] for st in sem_types1]
                st2_names = [st['name'] for st in sem_types2]
                common_types = set(st1_names).intersection(set(st2_names))
                if common_types:
                    result['contradiction_type'] = "Semantic type conflict"
                    result['contradiction_found'] = True
                    result['confidence'] = 0.7
                    result['explanation'] = f"Terms share semantic types {common_types} but have potential contradictory relationships"
            return result
        except (KeyError, IndexError) as e:
            logger.error(f"Error in contradiction detection: {str(e)}")
            result['explanation'] = f"Error processing terms: {str(e)}"
            return result
def extract_pneumonia_concepts():
    """
    Extract pneumonia-related concepts from UMLS.
    This is a simplified example for CAP-related research.
    """
    client = UMLSClient()
    pneumonia_results = client.search_term("pneumonia")
    if not pneumonia_results:
        logger.error("Failed to find pneumonia concepts")
        return None
    try:
        pneumonia_cui = None
        for result in pneumonia_results['result']['results']:
            if "community-acquired pneumonia" in result['name'].lower():
                pneumonia_cui = result['ui']
                break
        if not pneumonia_cui:
            pneumonia_cui = pneumonia_results['result']['results'][0]['ui']
        pneumonia_info = client.get_concept(pneumonia_cui)
        pneumonia_semantics = client.get_semantic_types(pneumonia_cui)
        pneumonia_relations = client.get_relations(pneumonia_cui)
        treatments = []
        pathogens = []
        for relation in pneumonia_relations:
            related_cui = relation.get('relatedId')
            if not related_cui:
                continue
            related_concept = client.get_concept(related_cui)
            related_semantics = client.get_semantic_types(related_cui)
            for sem in related_semantics:
                sem_type = sem.get('name', '').lower()
                if 'bacterium' in sem_type or 'virus' in sem_type:
                    pathogens.append(related_concept)
                elif 'pharmacologic' in sem_type or 'antibiotic' in sem_type:
                    treatments.append(related_concept)
        return {
            'concept': pneumonia_info,
            'semantics': pneumonia_semantics,
            'treatments': treatments,
            'pathogens': pathogens
        }
    except (KeyError, IndexError) as e:
        logger.error(f"Error extracting pneumonia concepts: {str(e)}")
        return None
if __name__ == "__main__":
    client = UMLSClient()
    print("Searching for 'community-acquired pneumonia'...")
    results = client.search_term("community-acquired pneumonia")
    if results and 'result' in results:
        print(f"Found {results['result']['results'][0]['name']} with CUI: {results['result']['results'][0]['ui']}")
        print("\nChecking for contradictions in treatments...")
        contradiction = client.find_contradictions(
            "macrolide antibiotics pneumonia", 
            "fluoroquinolones pneumonia"
        )
        if contradiction['contradiction_found']:
            print(f"Potential contradiction detected: {contradiction['explanation']}")
        else:
            print("No direct contradiction found between these treatments")
    else:
        print("No results found or error occurred")