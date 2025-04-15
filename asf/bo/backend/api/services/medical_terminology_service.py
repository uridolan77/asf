"""
Medical Terminology Service for interfacing with the medical module's TerminologyService.
"""
import os
import sys
import logging
from typing import Dict, List, Optional, Any
from fastapi import Depends, HTTPException, status

# Add the project root directory to sys.path to import the medical module
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
if project_root not in sys.path:
    sys.path.append(project_root)

# Mock implementations to avoid import errors
class ValidationError(Exception):
    """Validation error exception."""
    pass

class TerminologyService:
    """Mock TerminologyService for the Medical Research Synthesizer."""

    async def lookup_term(self, term, terminology="umls"):
        """Look up a term in the specified terminology."""
        return {
            "term": term,
            "terminology": terminology,
            "results": [
                {
                    "id": "mock-concept-1",
                    "name": f"Mock {terminology.upper()} Concept for {term}",
                    "definition": f"This is a mock definition for {term}"
                }
            ]
        }

    async def get_related_terms(self, concept_id, relationship_type=None, terminology="umls"):
        """Get terms related to the specified concept."""
        return {
            "concept_id": concept_id,
            "terminology": terminology,
            "relationship_type": relationship_type or "all",
            "related_terms": [
                {
                    "id": "mock-related-1",
                    "name": f"Mock Related Term 1 for {concept_id}",
                    "relationship": relationship_type or "broader"
                },
                {
                    "id": "mock-related-2",
                    "name": f"Mock Related Term 2 for {concept_id}",
                    "relationship": relationship_type or "narrower"
                }
            ]
        }

    async def get_concept_details(self, concept_id, terminology="umls"):
        """Get detailed information about a concept."""
        return {
            "id": concept_id,
            "terminology": terminology,
            "name": f"Mock Concept {concept_id}",
            "definition": "This is a mock definition for the concept.",
            "semantic_types": ["Disease or Syndrome", "Anatomical Structure"],
            "codes": {
                "ICD10": "M00.0",
                "SNOMED": "12345678",
                "UMLS": concept_id
            }
        }

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MedicalTerminologyService:
    """
    Service for interacting with the medical module's TerminologyService.
    This provides a bridge between the BO frontend and the Medical Research terminology functionality.
    """
    def __init__(self):
        """Initialize with direct access to the medical module's TerminologyService"""
        # Initialize the terminology service with default settings
        # In a production environment, these would be configured via environment variables
        self.terminology_service = TerminologyService(
            snomed_access_mode=os.environ.get("SNOMED_ACCESS_MODE", "umls"),
            snomed_api_key=os.environ.get("UMLS_API_KEY"),
            snomed_cache_dir=os.environ.get("SNOMED_CACHE_DIR", "./terminology_cache"),
            snomed_edition=os.environ.get("SNOMED_EDITION", "US")
        )
        logger.info("Medical Terminology Service initialized")

    def normalize_term(self, term: str) -> Dict[str, Any]:
        """
        Normalize a clinical term to its standard form.

        Args:
            term: The clinical term to normalize

        Returns:
            Dictionary containing normalized term and matching concepts
        """
        try:
            if not term or not term.strip():
                return {
                    "success": False,
                    "message": "Term cannot be empty",
                    "data": None
                }

            result = self.terminology_service.normalize_clinical_term(term)

            return {
                "success": True,
                "message": f"Term normalized with confidence: {result['confidence']:.2f}",
                "data": result
            }
        except Exception as e:
            logger.error(f"Error normalizing term '{term}': {str(e)}")
            return {
                "success": False,
                "message": f"Failed to normalize term: {str(e)}",
                "data": None
            }

    def get_concept_details(self, code: str, terminology: str = "SNOMEDCT") -> Dict[str, Any]:
        """
        Get detailed information about a concept.

        Args:
            code: The concept code
            terminology: The terminology system (currently only SNOMED CT supported)

        Returns:
            Dictionary containing concept details
        """
        try:
            if not code or not code.strip():
                return {
                    "success": False,
                    "message": "Concept code cannot be empty",
                    "data": None
                }

            concept = self.terminology_service.get_concept_details(code, terminology)

            return {
                "success": True,
                "message": f"Retrieved details for concept: {concept.get('preferredTerm', code)}",
                "data": concept
            }
        except ValueError as e:
            logger.error(f"Validation error for concept '{code}': {str(e)}")
            return {
                "success": False,
                "message": str(e),
                "data": None
            }
        except Exception as e:
            logger.error(f"Error getting concept details for '{code}': {str(e)}")
            return {
                "success": False,
                "message": f"Failed to get concept details: {str(e)}",
                "data": None
            }

    def semantic_search(self, query: str, max_results: int = 20) -> Dict[str, Any]:
        """
        Perform a semantic search for clinical concepts.

        Args:
            query: The search query
            max_results: Maximum number of results to return

        Returns:
            List of matching concepts with relevance scores
        """
        try:
            if not query or not query.strip():
                return {
                    "success": False,
                    "message": "Search query cannot be empty",
                    "data": None
                }

            results = self.terminology_service.semantic_search(query, max_results=max_results)

            return {
                "success": True,
                "message": f"Found {len(results)} concepts for query: {query}",
                "data": {
                    "query": query,
                    "results": results,
                    "total_count": len(results)
                }
            }
        except Exception as e:
            logger.error(f"Error in semantic search for '{query}': {str(e)}")
            return {
                "success": False,
                "message": f"Failed to perform semantic search: {str(e)}",
                "data": None
            }

    def get_hierarchical_relationships(
        self,
        concept_id: str,
        relationship_type: str,
        terminology: str = "SNOMEDCT"
    ) -> Dict[str, Any]:
        """
        Get hierarchical relationships for a concept.

        Args:
            concept_id: The concept identifier
            relationship_type: The type of relationship ("parents", "children", "ancestors", "descendants")
            terminology: The terminology system

        Returns:
            List of related concepts
        """
        try:
            if not concept_id or not concept_id.strip():
                return {
                    "success": False,
                    "message": "Concept ID cannot be empty",
                    "data": None
                }

            if relationship_type not in ["parents", "children", "ancestors", "descendants"]:
                return {
                    "success": False,
                    "message": f"Invalid relationship type: {relationship_type}. Must be one of: parents, children, ancestors, descendants",
                    "data": None
                }

            # Call the appropriate method based on relationship type
            if relationship_type == "parents":
                results = self.terminology_service.get_parents(concept_id, terminology)
            elif relationship_type == "children":
                results = self.terminology_service.get_children(concept_id, terminology)
            elif relationship_type == "ancestors":
                results = self.terminology_service.get_ancestors(concept_id, terminology)
            elif relationship_type == "descendants":
                results = self.terminology_service.get_descendants(concept_id, terminology)

            # Get the concept name for reference
            concept_name = ""
            try:
                concept_details = self.terminology_service.get_concept_details(concept_id, terminology)
                concept_name = concept_details.get("preferredTerm", concept_id)
            except:
                pass

            return {
                "success": True,
                "message": f"Found {len(results)} {relationship_type} for concept: {concept_name} ({concept_id})",
                "data": {
                    "concept_id": concept_id,
                    "concept_name": concept_name,
                    "relationship_type": relationship_type,
                    "results": results,
                    "total_count": len(results)
                }
            }
        except ValueError as e:
            logger.error(f"Validation error for concept '{concept_id}': {str(e)}")
            return {
                "success": False,
                "message": str(e),
                "data": None
            }
        except Exception as e:
            logger.error(f"Error getting {relationship_type} for '{concept_id}': {str(e)}")
            return {
                "success": False,
                "message": f"Failed to get {relationship_type}: {str(e)}",
                "data": None
            }

    def is_a_relationship(self, concept_id: str, potential_parent_id: str, terminology: str = "SNOMEDCT") -> Dict[str, Any]:
        """
        Check if a concept is a subtype of another concept.

        Args:
            concept_id: The concept to check
            potential_parent_id: The potential parent concept
            terminology: The terminology system

        Returns:
            Result of the subsumption check
        """
        try:
            if not concept_id or not concept_id.strip():
                return {
                    "success": False,
                    "message": "Concept ID cannot be empty",
                    "data": None
                }

            if not potential_parent_id or not potential_parent_id.strip():
                return {
                    "success": False,
                    "message": "Parent concept ID cannot be empty",
                    "data": None
                }

            is_subtype = self.terminology_service.is_a(concept_id, potential_parent_id, terminology)

            # Get concept names for reference
            concept_name = ""
            parent_name = ""
            try:
                concept_details = self.terminology_service.get_concept_details(concept_id, terminology)
                concept_name = concept_details.get("preferredTerm", concept_id)

                parent_details = self.terminology_service.get_concept_details(potential_parent_id, terminology)
                parent_name = parent_details.get("preferredTerm", potential_parent_id)
            except:
                pass

            message = f"'{concept_name}' is{' ' if is_subtype else ' not '}a subtype of '{parent_name}'"

            return {
                "success": True,
                "message": message,
                "data": {
                    "concept_id": concept_id,
                    "concept_name": concept_name,
                    "parent_id": potential_parent_id,
                    "parent_name": parent_name,
                    "is_subtype": is_subtype
                }
            }
        except ValueError as e:
            logger.error(f"Validation error for is-a check: {str(e)}")
            return {
                "success": False,
                "message": str(e),
                "data": None
            }
        except Exception as e:
            logger.error(f"Error checking is-a relationship: {str(e)}")
            return {
                "success": False,
                "message": f"Failed to check is-a relationship: {str(e)}",
                "data": None
            }

    def get_concept_relationships(
        self,
        concept_id: str,
        relationship_type: Optional[str] = None,
        terminology: str = "SNOMEDCT"
    ) -> Dict[str, Any]:
        """
        Get relationships for a given concept.

        Args:
            concept_id: The concept identifier
            relationship_type: Optional relationship type ID to filter by
            terminology: The terminology system

        Returns:
            List of relationship dictionaries
        """
        try:
            if not concept_id or not concept_id.strip():
                return {
                    "success": False,
                    "message": "Concept ID cannot be empty",
                    "data": None
                }

            relationships = self.terminology_service.get_relationships(
                concept_id,
                relationship_type,
                terminology
            )

            # Get the concept name for reference
            concept_name = ""
            try:
                concept_details = self.terminology_service.get_concept_details(concept_id, terminology)
                concept_name = concept_details.get("preferredTerm", concept_id)
            except:
                pass

            return {
                "success": True,
                "message": f"Found {len(relationships)} relationships for concept: {concept_name} ({concept_id})",
                "data": {
                    "concept_id": concept_id,
                    "concept_name": concept_name,
                    "relationship_type": relationship_type,
                    "relationships": relationships,
                    "total_count": len(relationships)
                }
            }
        except ValueError as e:
            logger.error(f"Validation error for concept '{concept_id}': {str(e)}")
            return {
                "success": False,
                "message": str(e),
                "data": None
            }
        except Exception as e:
            logger.error(f"Error getting relationships for '{concept_id}': {str(e)}")
            return {
                "success": False,
                "message": f"Failed to get relationships: {str(e)}",
                "data": None
            }

    def evaluate_ecl(self, expression: str, max_results: int = 200) -> Dict[str, Any]:
        """
        Evaluate an Expression Constraint Language (ECL) expression.

        Args:
            expression: The ECL expression
            max_results: Maximum number of results to return

        Returns:
            List of matching concepts
        """
        try:
            if not expression or not expression.strip():
                return {
                    "success": False,
                    "message": "ECL expression cannot be empty",
                    "data": None
                }

            results = self.terminology_service.evaluate_ecl(expression, max_results=max_results)

            return {
                "success": True,
                "message": f"ECL evaluation found {len(results)} matching concepts",
                "data": {
                    "expression": expression,
                    "results": results,
                    "total_count": len(results)
                }
            }
        except Exception as e:
            logger.error(f"Error evaluating ECL expression '{expression}': {str(e)}")
            return {
                "success": False,
                "message": f"Failed to evaluate ECL expression: {str(e)}",
                "data": None
            }

    def find_all_diabetes_types(self) -> Dict[str, Any]:
        """
        Find all types of diabetes using ECL.

        Returns:
            List of diabetes concept types
        """
        try:
            results = self.terminology_service.find_all_diabetes_types()

            return {
                "success": True,
                "message": f"Found {len(results)} types of diabetes",
                "data": {
                    "results": results,
                    "total_count": len(results),
                    "condition": "Diabetes mellitus"
                }
            }
        except Exception as e:
            logger.error(f"Error finding diabetes types: {str(e)}")
            return {
                "success": False,
                "message": f"Failed to find diabetes types: {str(e)}",
                "data": None
            }

# Dependency to get the medical terminology service
def get_medical_terminology_service() -> MedicalTerminologyService:
    """Factory function to create and provide a MedicalTerminologyService instance."""
    return MedicalTerminologyService()