"""
Ontology Integration Service for Medical Knowledge Graph

This module provides services for integrating medical ontologies (SNOMED CT,
MeSH, etc.) into the knowledge graph, enabling enhanced search, retrieval,
and cross-ontology concept mapping.

Features:
- Map free text to standardized medical concepts
- Link concepts across different ontologies (e.g., SNOMED CT to MeSH)
- Enrich knowledge graph with ontology relationships
- Enhance graph-based retrieval with ontology awareness
- Provide explainable concept relationships
- Advanced NLP with biomedical Named Entity Recognition
- Relation extraction for medical knowledge
- Semantic similarity using domain-specific models

Dependencies:
- UMLS API access for ontology concept resolution
- Graph database connection (Neo4j or Memgraph)
- Transformers library for biomedical language models
- BioBERT or PubMedBERT models for domain-specific NLP
"""

import os
import re
import json
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
import logging
from datetime import datetime

# Import required modules
from medical.core.logging_config import get_logger
from medical.clients.umls import UMLSClient
from medical.clients.snomed.snomed_client import SnomedClient
from medical.graph.graph_service import GraphService
from medical.core.cache import CacheInterface, LocalCache, RedisCache, CacheManager, get_cache_manager as get_cache

# Advanced NLP imports
try:
    import torch
    from transformers import (
        AutoTokenizer, 
        AutoModelForTokenClassification,
        AutoModelForSequenceClassification,
        AutoModel,
        pipeline
    )
    NLP_ADVANCED_AVAILABLE = True
except ImportError:
    NLP_ADVANCED_AVAILABLE = False
    get_logger(__name__).warning("Advanced NLP features unavailable. Install with: pip install torch transformers")

# Configure logger
logger = get_logger(__name__)

# Constants
DEFAULT_CONFIDENCE_THRESHOLD = 0.7
ONTOLOGY_NODE_LABEL = "OntologyConcept"
ONTOLOGY_RELATIONSHIP = "MENTIONS_CONCEPT"
CROSS_ONTOLOGY_RELATIONSHIP = "MAPS_TO"

# Biomedical NER constants
BIO_NER_ENTITIES = [
    "DISEASE", "DRUG", "GENE", "PROCEDURE", "ANATOMY", 
    "DOSAGE", "FREQUENCY", "DEMOGRAPHIC", "SYMPTOM", "OUTCOME"
]

# Biomedical Relation constants
BIO_RELATIONS = [
    "TREATS", "CAUSES", "ASSOCIATED_WITH", "CONTRAINDICATES",
    "IMPROVES", "WORSENS", "PREVENTS", "DIAGNOSES", "AFFECTS"
]

@dataclass
class BiomedicalEntity:
    """Represents a biomedical entity extracted from text using NER."""
    text: str
    entity_type: str
    start_char: int
    end_char: int
    confidence: float
    concept_id: Optional[str] = None
    ontology_source: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert entity to dictionary representation."""
        return {
            "text": self.text,
            "entity_type": self.entity_type,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "confidence": self.confidence,
            "concept_id": self.concept_id,
            "ontology_source": self.ontology_source
        }

@dataclass
class BiomedicalRelation:
    """Represents a semantic relationship between biomedical entities."""
    relation_type: str
    source_entity: BiomedicalEntity
    target_entity: BiomedicalEntity
    confidence: float
    evidence_text: str
    source_doc_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert relation to dictionary representation."""
        return {
            "relation_type": self.relation_type,
            "source_entity": self.source_entity.to_dict(),
            "target_entity": self.target_entity.to_dict(),
            "confidence": self.confidence,
            "evidence_text": self.evidence_text,
            "source_doc_id": self.source_doc_id
        }

class OntologyIntegrationService:
    """
    Service for integrating medical ontologies with knowledge graph using advanced NLP.
    
    This service provides methods for:
    1. Biomedical Named Entity Recognition (NER)
    2. Relation Extraction between biomedical entities
    3. Semantic similarity computation with domain-specific models
    4. Cross-ontology concept mapping
    5. Knowledge graph enrichment with ontological concepts
    """
    
    def __init__(
        self,
        graph_service: Optional[GraphService] = None,
        umls_client: Optional[UMLSClient] = None,
        snomed_client: Optional[SnomedClient] = None,
        cache: Optional[CacheInterface] = None,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        use_advanced_nlp: bool = True,
        model_path: Optional[str] = None
    ):
        """
        Initialize the ontology integration service.
        
        Args:
            graph_service: Service for interacting with graph database
            umls_client: Client for UMLS API interactions
            snomed_client: Client for SNOMED CT API interactions
            cache: Cache instance for storing mapped concepts
            confidence_threshold: Minimum confidence score for entity extraction
            use_advanced_nlp: Whether to use advanced NLP features (BioBERT/PubMedBERT)
            model_path: Path to pre-downloaded model (if not using HF hub)
        """
        self.graph_service = graph_service
        self.umls_client = umls_client or UMLSClient()
        self.snomed_client = snomed_client or SnomedClient()
        self.cache = cache or get_cache()
        self.confidence_threshold = confidence_threshold
        self.use_advanced_nlp = use_advanced_nlp and NLP_ADVANCED_AVAILABLE
        
        # Advanced NLP components (initialized lazily)
        self._ner_model = None
        self._ner_tokenizer = None
        self._relation_model = None
        self._relation_tokenizer = None
        self._embeddings_model = None
        self._embeddings_tokenizer = None
        
        # Set default model paths if not provided
        self.model_path = model_path
        self.default_ner_model = "dmis-lab/biobert-base-cased-v1.1-ner"
        self.default_relation_model = "allenai/biomed_roberta_base"
        self.default_embeddings_model = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
        
        # Initialize NLP components if advanced NLP is enabled
        if self.use_advanced_nlp:
            self._initialize_nlp_components()
    
    def _initialize_nlp_components(self) -> None:
        """Initialize the NLP components if not already loaded."""
        try:
            if not NLP_ADVANCED_AVAILABLE:
                logger.warning("Advanced NLP features unavailable. Please install required packages.")
                return
                
            logger.info("Initializing advanced biomedical NLP components...")
            
            # Initialize NER components
            if self._ner_model is None:
                logger.info(f"Loading biomedical NER model: {self.default_ner_model}")
                self._ner_tokenizer = AutoTokenizer.from_pretrained(self.default_ner_model)
                self._ner_model = AutoModelForTokenClassification.from_pretrained(self.default_ner_model)
                self._ner_pipeline = pipeline("ner", model=self._ner_model, tokenizer=self._ner_tokenizer)
            
            # Initialize embedding model for semantic similarity
            if self._embeddings_model is None:
                logger.info(f"Loading biomedical embeddings model: {self.default_embeddings_model}")
                self._embeddings_tokenizer = AutoTokenizer.from_pretrained(self.default_embeddings_model)
                self._embeddings_model = AutoModel.from_pretrained(self.default_embeddings_model)
            
            # For relation extraction, we'll use a fine-tuned model
            # This could be lazily loaded when needed to conserve resources
            
            logger.info("Advanced biomedical NLP components initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize NLP components: {str(e)}")
            self.use_advanced_nlp = False

class GraphRAGOntologyEnhancer:
    """
    Enhances GraphRAG retrieval with medical ontology integration.
    
    This class provides methods to enhance search queries and retrieval results
    using medical ontologies like SNOMED CT and MeSH. It expands queries with
    related ontology concepts and improves retrieval by incorporating
    ontological relationships.
    """
    
    def __init__(
        self,
        ontology_service: OntologyIntegrationService,
        expansion_depth: int = 1,
        include_narrower_concepts: bool = True,
        include_broader_concepts: bool = False,
        ontology_weight: float = 0.3
    ):
        """
        Initialize the GraphRAG ontology enhancer.
        
        Args:
            ontology_service: Service for ontology integration
            expansion_depth: How many levels to expand ontology concepts
            include_narrower_concepts: Whether to include more specific concepts
            include_broader_concepts: Whether to include more general concepts
            ontology_weight: Weight of ontology concepts in the enhanced query
        """
        self.ontology_service = ontology_service
        self.expansion_depth = expansion_depth
        self.include_narrower_concepts = include_narrower_concepts
        self.include_broader_concepts = include_broader_concepts
        self.ontology_weight = ontology_weight
        self.logger = get_logger(__name__)
    
    async def enhance_search_query(self, query: str) -> str:
        """
        Enhance a search query with related ontology concepts.
        
        This method extracts medical concepts from the query and expands them
        with related concepts from the ontology, creating a more comprehensive
        search query that can capture relevant results that wouldn't match
        the original query directly.
        
        Args:
            query: Original search query
            
        Returns:
            Enhanced search query with ontology expansion
        """
        if not query or not self.ontology_service:
            return query
            
        try:
            # Extract biomedical entities from the query
            entities = await self.ontology_service.extract_biomedical_entities(query)
            
            if not entities:
                return query
                
            # Only use entities that were mapped to ontology concepts
            mapped_entities = [e for e in entities if e.concept_id]
            
            if not mapped_entities:
                return query
                
            # Build enhanced query
            enhanced_terms = []
            
            for entity in mapped_entities:
                # Add the original term
                enhanced_terms.append(entity.text)
                
                # Add related terms based on ontology relationships
                if entity.concept_id and entity.ontology_source:
                    related_concepts = []
                    
                    # Get narrower (more specific) concepts if requested
                    if self.include_narrower_concepts:
                        try:
                            if entity.ontology_source == "SNOMED":
                                narrower = await self.ontology_service.snomed_client.get_children(
                                    entity.concept_id, direct_only=(self.expansion_depth == 1)
                                )
                                related_concepts.extend(
                                    {"term": c.get("preferredTerm", ""), "weight": 0.8} 
                                    for c in narrower
                                )
                            elif entity.ontology_source == "UMLS":
                                narrower = await self.ontology_service.umls_client.get_narrower_concepts(
                                    entity.concept_id
                                )
                                related_concepts.extend(
                                    {"term": c.get("name", ""), "weight": 0.8} 
                                    for c in narrower
                                )
                        except Exception as e:
                            self.logger.warning(f"Error getting narrower concepts: {str(e)}")
                    
                    # Get broader (more general) concepts if requested
                    if self.include_broader_concepts:
                        try:
                            if entity.ontology_source == "SNOMED":
                                broader = await self.ontology_service.snomed_client.get_parents(
                                    entity.concept_id, direct_only=(self.expansion_depth == 1)
                                )
                                related_concepts.extend(
                                    {"term": c.get("preferredTerm", ""), "weight": 0.6} 
                                    for c in broader
                                )
                            elif entity.ontology_source == "UMLS":
                                broader = await self.ontology_service.umls_client.get_broader_concepts(
                                    entity.concept_id
                                )
                                related_concepts.extend(
                                    {"term": c.get("name", ""), "weight": 0.6} 
                                    for c in broader
                                )
                        except Exception as e:
                            self.logger.warning(f"Error getting broader concepts: {str(e)}")
                    
                    # Add a subset of related concepts to the query
                    # Sort by weight and take the top N to prevent query explosion
                    related_concepts.sort(key=lambda x: x["weight"], reverse=True)
                    for concept in related_concepts[:5]:  # Limit to top 5 related concepts
                        if concept["term"] and concept["term"].lower() != entity.text.lower():
                            enhanced_terms.append(concept["term"])
            
            # Combine original query with enhanced terms
            if enhanced_terms:
                # Deduplicate terms
                enhanced_terms = list(set(enhanced_terms))
                # Build the enhanced query
                enhanced_query = query + " " + " ".join(enhanced_terms)
                self.logger.info(f"Enhanced query with ontology terms: {enhanced_query}")
                return enhanced_query
            
            return query
            
        except Exception as e:
            self.logger.error(f"Query enhancement failed: {str(e)}")
            return query
    
    async def enhance_graph_retrieval(
        self,
        graph_service,
        original_query: str,
        initial_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Enhance retrieval results using ontology relationships.
        
        This method reranks and augments the initial retrieval results
        using ontology relationships, boosting results that are more
        ontologically relevant to the query concepts.
        
        Args:
            graph_service: Service for graph database operations
            original_query: Original search query
            initial_results: Initial retrieval results
            
        Returns:
            Enhanced and reranked retrieval results
        """
        if not initial_results or not self.ontology_service:
            return initial_results
            
        try:
            # Extract concepts from the original query
            query_entities = await self.ontology_service.extract_biomedical_entities(original_query)
            query_concepts = [e.concept_id for e in query_entities if e.concept_id]
            
            if not query_concepts:
                return initial_results
                
            # Enhance results based on ontology relationships
            enhanced_results = []
            
            for result in initial_results:
                # Get concepts mentioned in the result
                result_concepts = result.get("concepts", [])
                
                # Calculate ontology relevance score
                ontology_score = 0.0
                
                # Direct concept match
                for qc in query_concepts:
                    for rc in result_concepts:
                        rc_id = rc.get("id") or rc.get("concept_id")
                        if qc == rc_id:
                            ontology_score += 1.0
                            break
                
                # Check for related concepts
                if ontology_score == 0 and self.ontology_service:
                    for qc in query_concepts:
                        for rc in result_concepts:
                            rc_id = rc.get("id") or rc.get("concept_id")
                            
                            # Check if concepts are related in the ontology
                            try:
                                if await self._are_concepts_related(qc, rc_id):
                                    ontology_score += 0.5
                                    break
                            except Exception as e:
                                self.logger.warning(f"Error checking concept relatedness: {str(e)}")
                
                # Combine original score with ontology score
                original_score = result.get("score", 0.0)
                if isinstance(original_score, str):
                    try:
                        original_score = float(original_score)
                    except ValueError:
                        original_score = 0.0
                
                combined_score = (1 - self.ontology_weight) * original_score + self.ontology_weight * ontology_score
                
                # Create enhanced result
                enhanced_result = result.copy()
                enhanced_result["original_score"] = original_score
                enhanced_result["ontology_score"] = ontology_score
                enhanced_result["score"] = combined_score
                
                enhanced_results.append(enhanced_result)
            
            # Sort by combined score
            enhanced_results.sort(key=lambda x: x["score"], reverse=True)
            
            return enhanced_results
            
        except Exception as e:
            self.logger.error(f"Graph retrieval enhancement failed: {str(e)}")
            return initial_results
    
    async def _are_concepts_related(self, concept_id1: str, concept_id2: str) -> bool:
        """
        Check if two concepts are related in the ontology.
        
        Args:
            concept_id1: First concept ID
            concept_id2: Second concept ID
            
        Returns:
            True if concepts are related, False otherwise
        """
        if not concept_id1 or not concept_id2:
            return False
            
        # Check if one is a parent/child of the other
        try:
            # Check SNOMED relationships
            if hasattr(self.ontology_service, "snomed_client") and self.ontology_service.snomed_client:
                if await self.ontology_service.snomed_client.is_a(concept_id1, concept_id2):
                    return True
                if await self.ontology_service.snomed_client.is_a(concept_id2, concept_id1):
                    return True
            
            # Check UMLS relationships
            if hasattr(self.ontology_service, "umls_client") and self.ontology_service.umls_client:
                parents1 = await self.ontology_service.umls_client.get_parent_concepts(concept_id1)
                parents2 = await self.ontology_service.umls_client.get_parent_concepts(concept_id2)
                
                # Check if either concept is a parent of the other
                parent_ids1 = [p.get("id") for p in parents1]
                parent_ids2 = [p.get("id") for p in parents2]
                
                if concept_id1 in parent_ids2 or concept_id2 in parent_ids1:
                    return True
                
                # Check for shared parents (siblings)
                common_parents = set(parent_ids1).intersection(parent_ids2)
                if common_parents:
                    return True
        except Exception as e:
            self.logger.warning(f"Error checking ontology relationship: {str(e)}")
        
        return False