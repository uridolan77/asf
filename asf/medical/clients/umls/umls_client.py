"""
UMLS client for the Medical Research Synthesizer.
This module provides a client for interacting with the UMLS API.
"""
import logging
import time
import json
import asyncio
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import httpx

logger = logging.getLogger(__name__)

class UMLSClient:
    """
    Client for interacting with the UMLS API.
    This client provides methods for searching UMLS concepts and retrieving concept details.
    """
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://uts-ws.nlm.nih.gov/rest",
        version: str = "current"
    ):
        """
        Initialize a new UMLS API client.
        
        Args:
            api_key: The UMLS API key
            base_url: Base URL for the API (default: https://uts-ws.nlm.nih.gov/rest)
            version: API version (default: "current")
        """
        self.api_key = api_key
        self.base_url = base_url
        self.version = version
        self.client = httpx.AsyncClient(timeout=30.0)
        self.tgt = None
        self.tgt_expires = 0
        self.requests_per_second = 5  # UMLS recommends no more than 5 requests per second
        self.last_request_time = 0
    
    async def close(self):
        """
        Close the HTTP client.
        
        Should be called when the client is no longer needed.
        """
        await self.client.aclose()
    
    async def _rate_limit(self):
        """
        Implement rate limiting for UMLS API.
        UMLS recommends no more than 5 requests per second.
        """
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        # If we've made a request less than 1/requests_per_second seconds ago, wait
        if time_since_last_request < (1.0 / self.requests_per_second):
            await asyncio.sleep((1.0 / self.requests_per_second) - time_since_last_request)
        
        self.last_request_time = time.time()
    
    async def _get_tgt(self) -> Optional[str]:
        """
        Get a Ticket Granting Ticket (TGT) from the UMLS API.
        
        Returns:
            TGT URL if successful, None otherwise
            
        Raises:
            httpx.HTTPError: If the request fails
        """
        # Check if current TGT is still valid
        current_time = datetime.now().timestamp()
        if self.tgt and current_time < self.tgt_expires:
            return self.tgt
        
        try:
            auth_endpoint = f"{self.base_url}/auth/authenticateUser"
            auth_params = {
                'apiKey': self.api_key
            }
            
            await self._rate_limit()
            response = await self.client.post(auth_endpoint, data=auth_params)
            response.raise_for_status()
            
            # Extract TGT URL
            tgt_url = response.headers.get('location')
            if not tgt_url:
                logger.error("Could not find TGT URL in response headers")
                return None
            
            # TGT is valid for 8 hours according to UMLS
            self.tgt = tgt_url
            self.tgt_expires = current_time + (8 * 60 * 60)
            
            return tgt_url
        except httpx.HTTPError as e:
            logger.error(f"Failed to get TGT: {str(e)}")
            return None
    
    async def _get_service_ticket(self) -> Optional[str]:
        """
        Get a Service Ticket (ST) from the UMLS API.
        
        Returns:
            Service Ticket if successful, None otherwise
            
        Raises:
            httpx.HTTPError: If the request fails
        """
        tgt_url = await self._get_tgt()
        if not tgt_url:
            return None
        
        try:
            service_params = {'service': f"{self.base_url}"}
            
            await self._rate_limit()
            response = await self.client.post(tgt_url, data=service_params)
            response.raise_for_status()
            
            service_ticket = response.text
            return service_ticket
        except httpx.HTTPError as e:
            logger.error(f"Failed to get service ticket: {str(e)}")
            return None
    
    async def _make_request(
        self, 
        endpoint: str, 
        params: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Make a request to the UMLS API.
        
        Args:
            endpoint: API endpoint
            params: Request parameters
            
        Returns:
            Response data if successful, None otherwise
            
        Raises:
            httpx.HTTPError: If the request fails
        """
        service_ticket = await self._get_service_ticket()
        if not service_ticket:
            return None
        
        if not params:
            params = {}
        
        # Add ticket to params
        params['ticket'] = service_ticket
        
        try:
            url = f"{self.base_url}/{endpoint}"
            
            await self._rate_limit()
            response = await self.client.get(url, params=params)
            response.raise_for_status()
            
            return response.json()
        except httpx.HTTPError as e:
            logger.error(f"API request failed: {str(e)}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode API response: {str(e)}")
            return None
    
    async def search(
        self, 
        query: str, 
        search_type: str = "words", 
        max_results: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Search for UMLS concepts.
        
        Args:
            query: Search query
            search_type: Search type (default: "words")
                Options: "exact", "words", "approximate", "normalizedString"
            max_results: Maximum number of results to return (default: 20)
            
        Returns:
            List of concept summaries
        """
        endpoint = f"search/{self.version}"
        params = {
            'string': query,
            'searchType': search_type,
            'pageSize': max_results,
            'returnIdType': 'concept'
        }
        
        response = await self._make_request(endpoint, params)
        
        if not response or 'result' not in response:
            return []
        
        return response.get('result', {}).get('results', [])
    
    async def get_concept(self, concept_id: str) -> Optional[Dict[str, Any]]:
        """
        Get details for a specific UMLS concept.
        
        Args:
            concept_id: Concept ID (e.g., "C0012634")
            
        Returns:
            Concept details if successful, None otherwise
        """
        endpoint = f"content/{self.version}/CUI/{concept_id}"
        response = await self._make_request(endpoint)
        
        if not response or 'result' not in response:
            return None
        
        return response.get('result', {})
    
    async def get_concept_atoms(self, concept_id: str) -> List[Dict[str, Any]]:
        """
        Get atoms for a specific UMLS concept.
        
        Args:
            concept_id: Concept ID (e.g., "C0012634")
            
        Returns:
            List of atoms
        """
        endpoint = f"content/{self.version}/CUI/{concept_id}/atoms"
        response = await self._make_request(endpoint)
        
        if not response or 'result' not in response:
            return []
        
        return response.get('result', [])
    
    async def get_concept_definitions(self, concept_id: str) -> List[Dict[str, Any]]:
        """
        Get definitions for a specific UMLS concept.
        
        Args:
            concept_id: Concept ID (e.g., "C0012634")
            
        Returns:
            List of definitions
        """
        endpoint = f"content/{self.version}/CUI/{concept_id}/definitions"
        response = await self._make_request(endpoint)
        
        if not response or 'result' not in response:
            return []
        
        return response.get('result', [])
    
    async def get_concept_relations(self, concept_id: str) -> List[Dict[str, Any]]:
        """
        Get relations for a specific UMLS concept.
        
        Args:
            concept_id: Concept ID (e.g., "C0012634")
            
        Returns:
            List of relations
        """
        endpoint = f"content/{self.version}/CUI/{concept_id}/relations"
        response = await self._make_request(endpoint)
        
        if not response or 'result' not in response:
            return []
        
        return response.get('result', [])
    
    async def get_semantic_types(self, concept_id: str) -> List[Dict[str, Any]]:
        """
        Get semantic types for a specific UMLS concept.
        
        Args:
            concept_id: Concept ID (e.g., "C0012634")
            
        Returns:
            List of semantic types
        """
        endpoint = f"content/{self.version}/CUI/{concept_id}/semanticTypes"
        response = await self._make_request(endpoint)
        
        if not response or 'result' not in response:
            return []
        
        return response.get('result', [])
    
    async def crosswalk(
        self, 
        source_vocab: str, 
        source_code: str, 
        target_vocab: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Crosswalk a code from one vocabulary to another.
        
        Args:
            source_vocab: Source vocabulary (e.g., "ICD10CM")
            source_code: Source code
            target_vocab: Target vocabulary (optional, e.g., "SNOMEDCT_US")
            
        Returns:
            List of mappings
        """
        # First, find the CUI for the source code
        endpoint = f"crosswalk/{self.version}/source/{source_vocab}/{source_code}"
        response = await self._make_request(endpoint)
        
        if not response or 'result' not in response:
            return []
        
        results = response.get('result', [])
        
        # If target vocabulary specified, filter results
        if target_vocab and results:
            return [r for r in results if r.get('rootSource') == target_vocab]
        
        return results
    
    async def find_contradictions(
        self, 
        term1: str, 
        term2: str
    ) -> Dict[str, Any]:
        """
        Find potential contradictions between two medical terms.
        
        Args:
            term1: First medical term
            term2: Second medical term
            
        Returns:
            Dictionary with contradiction information
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
        
        # Search for both terms
        term1_results = await self.search(term1)
        term2_results = await self.search(term2)
        
        if not term1_results or not term2_results:
            result['explanation'] = "Could not find one or both terms in UMLS"
            return result
        
        try:
            # Get concept IDs
            cui1 = term1_results[0]['ui']
            cui2 = term2_results[0]['ui']
            
            # Get concept details
            concept1 = await self.get_concept(cui1)
            concept2 = await self.get_concept(cui2)
            
            result['term1_info'] = concept1
            result['term2_info'] = concept2
            
            # Get semantic types
            sem_types1 = await self.get_semantic_types(cui1)
            sem_types2 = await self.get_semantic_types(cui2)
            
            # Check for direct contradictions
            # This is a simplified check - in a real system, you'd want more sophisticated logic
            sem_type1_names = [sem['name'].lower() for sem in sem_types1]
            sem_type2_names = [sem['name'].lower() for sem in sem_types2]
            
            # Check for semantic type contradictions
            contradictory_pairs = [
                ('pharmacologic substance', 'disease or syndrome'),
                ('antibiotic', 'bacterium'),
                ('therapeutic procedure', 'contraindicated procedure')
            ]
            
            for type1 in sem_type1_names:
                for type2 in sem_type2_names:
                    if (type1, type2) in contradictory_pairs or (type2, type1) in contradictory_pairs:
                        result['contradiction_found'] = True
                        result['contradiction_type'] = 'semantic_type'
                        result['explanation'] = f"Semantic types '{type1}' and '{type2}' may be contradictory"
                        result['confidence'] = 0.7
            
            # Check relations for potential contradictions
            if not result['contradiction_found']:
                relations1 = await self.get_concept_relations(cui1)
                
                # Look for opposites or contradictions in relations
                for relation in relations1:
                    if relation.get('relatedId') == cui2 and relation.get('relationLabel') == 'contradicts':
                        result['contradiction_found'] = True
                        result['contradiction_type'] = 'direct_relation'
                        result['explanation'] = "Direct contradiction relationship found"
                        result['confidence'] = 0.9
            
            return result
            
        except (IndexError, KeyError) as e:
            logger.error(f"Error analyzing contradictions: {str(e)}")
            result['explanation'] = f"Error analyzing contradictions: {str(e)}"
            return result
    
    async def get_vocabulary_metadata(self, vocabulary: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a specific vocabulary.
        
        Args:
            vocabulary: Vocabulary abbreviation (e.g., "SNOMEDCT_US")
            
        Returns:
            Vocabulary metadata if successful, None otherwise
        """
        endpoint = f"metadata/{self.version}/sources/{vocabulary}"
        return await self._make_request(endpoint)
    
    async def extract_medical_concepts(self, text: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Extract medical concepts from free text.
        
        This is a simplified implementation. For production use, consider
        using the MetaMap API or a similar service.
        
        Args:
            text: Free text to analyze
            max_results: Maximum number of concepts to return
            
        Returns:
            List of extracted concepts
        """
        # Naive approach: split text and search for each word/phrase
        # This is just an example - real concept extraction is much more complex
        words = text.split()
        phrases = []
        
        # Generate phrases of different lengths
        for i in range(len(words)):
            for j in range(i + 1, min(i + 4, len(words) + 1)):
                phrase = ' '.join(words[i:j])
                if len(phrase) > 3:  # Ignore very short phrases
                    phrases.append(phrase)
        
        results = []
        for phrase in phrases:
            search_results = await self.search(phrase, search_type="exact")
            if search_results:
                # Add the first result for this phrase
                concept = search_results[0]
                concept['matched_text'] = phrase
                results.append(concept)
            
            if len(results) >= max_results:
                break
        
        return results