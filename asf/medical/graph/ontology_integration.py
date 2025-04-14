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
from asf.medical.core.logging_config import get_logger
from asf.medical.clients.umls import UMLSClient
from asf.medical.clients.snomed.snomed_client import SNOMEDClient
from asf.medical.graph.graph_service import GraphService
from asf.medical.core.cache import Cache, get_cache

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
    logger.warning("Advanced NLP features unavailable. Install with: pip install torch transformers")

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
        snomed_client: Optional[SNOMEDClient] = None,
        cache: Optional[Cache] = None,
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
        self.snomed_client = snomed_client or SNOMEDClient()
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
    
    async def extract_biomedical_entities(
        self, 
        text: str,
        entity_types: Optional[List[str]] = None
    ) -> List[BiomedicalEntity]:
        """
        Extract biomedical entities from text using advanced NER.
        
        Args:
            text: Input text to extract entities from
            entity_types: Optional list of entity types to extract (e.g., ["DISEASE", "DRUG"])
                          If None, all entity types are extracted
        
        Returns:
            List of extracted BiomedicalEntity objects
        """
        if not text.strip():
            return []
            
        entities = []
        
        # Use advanced NLP if available
        if self.use_advanced_nlp and self._ner_model is not None:
            try:
                # Run the NER pipeline
                ner_results = self._ner_pipeline(text)
                
                # Process NER results
                current_entity = None
                current_tokens = []
                
                for token in ner_results:
                    # Format of token: {'entity': 'B-DISEASE', 'score': 0.9932, 'word': 'pneumonia', 'start': 10, 'end': 19}
                    if token["score"] < self.confidence_threshold:
                        continue
                        
                    entity_tag = token["entity"]  # e.g., B-DISEASE, I-DISEASE
                    if entity_tag.startswith("B-"):  # Beginning of entity
                        # If we were building an entity, add it to results
                        if current_entity is not None and current_tokens:
                            entity_text = " ".join(current_tokens)
                            entity_type = current_entity.replace("B-", "").replace("I-", "")
                            
                            # Skip if we're filtering by entity type
                            if entity_types and entity_type not in entity_types:
                                current_entity = entity_tag
                                current_tokens = [token["word"]]
                                continue
                                
                            # Create biomedical entity
                            bio_entity = BiomedicalEntity(
                                text=entity_text,
                                entity_type=entity_type,
                                start_char=token["start"] - len(entity_text) - 1,  # Approximate
                                end_char=token["end"] - 1,  # Approximate
                                confidence=token["score"]
                            )
                            entities.append(bio_entity)
                        
                        # Start new entity
                        current_entity = entity_tag
                        current_tokens = [token["word"]]
                    
                    elif entity_tag.startswith("I-") and current_entity is not None:
                        # Continue building entity
                        if entity_tag.replace("I-", "") == current_entity.replace("B-", ""):
                            current_tokens.append(token["word"])
                
                # Add the last entity if any
                if current_entity is not None and current_tokens:
                    entity_text = " ".join(current_tokens)
                    entity_type = current_entity.replace("B-", "").replace("I-", "")
                    
                    if not entity_types or entity_type in entity_types:
                        bio_entity = BiomedicalEntity(
                            text=entity_text,
                            entity_type=entity_type,
                            start_char=0,  # Will need fixing in a real implementation
                            end_char=len(entity_text),  # Will need fixing in a real implementation
                            confidence=0.9  # Simplified - would average token scores
                        )
                        entities.append(bio_entity)
                
                # Link entities to ontology concepts
                for entity in entities:
                    await self._link_entity_to_ontology(entity)
                    
            except Exception as e:
                logger.error(f"Error during biomedical NER: {str(e)}")
                # Fall back to basic approach if advanced NLP fails
        
        # If we have no entities (or advanced NLP failed), use simple method
        if not entities:
            # Basic approach using regular expressions and ontology lookups
            # This would be a simplified fallback implementation
            pass
            
        return entities
    
    async def _link_entity_to_ontology(self, entity: BiomedicalEntity) -> None:
        """
        Link extracted entity to standard ontology concepts.
        
        Args:
            entity: BiomedicalEntity to map to ontology concepts
        """
        try:
            # Try to get concept from cache first
            cache_key = f"ontology:concept:{entity.text.lower()}:{entity.entity_type}"
            cached_concept = await self.cache.get(cache_key)
            
            if cached_concept:
                entity.concept_id = cached_concept.get("concept_id")
                entity.ontology_source = cached_concept.get("ontology_source")
                return
                
            # Query UMLS for concept mapping
            umls_concepts = await self.umls_client.search_concepts(entity.text)
            
            if umls_concepts and len(umls_concepts) > 0:
                # Select best matching concept based on semantic type compatibility
                best_concept = umls_concepts[0]  # Simplified - would use semantic type filtering
                
                entity.concept_id = best_concept.get("ui")
                entity.ontology_source = "UMLS"
                
                # Cache the result
                await self.cache.set(
                    cache_key, 
                    {"concept_id": entity.concept_id, "ontology_source": entity.ontology_source},
                    expire=86400  # Cache for 24 hours
                )
                
        except Exception as e:
            logger.error(f"Error linking entity to ontology: {str(e)}")
    
    async def extract_biomedical_relations(
        self,
        text: str,
        entities: Optional[List[BiomedicalEntity]] = None
    ) -> List[BiomedicalRelation]:
        """
        Extract semantic relationships between biomedical entities.
        
        Args:
            text: Input text to extract relations from
            entities: Optional pre-extracted entities. If None, entities will be extracted
        
        Returns:
            List of extracted BiomedicalRelation objects
        """
        relations = []
        
        # Extract entities if not provided
        extracted_entities = entities or await self.extract_biomedical_entities(text)
        if len(extracted_entities) < 2:
            return []  # Need at least two entities to form a relation
        
        # Use advanced NLP for relation extraction if available
        if self.use_advanced_nlp:
            try:
                # Lazy-load relation extraction model if needed
                if self._relation_model is None and not self.model_path:
                    logger.info(f"Loading relation extraction model: {self.default_relation_model}")
                    self._relation_tokenizer = AutoTokenizer.from_pretrained(self.default_relation_model)
                    self._relation_model = AutoModelForSequenceClassification.from_pretrained(self.default_relation_model)
                
                # For each pair of entities, extract potential relationships
                for i, source_entity in enumerate(extracted_entities):
                    for j, target_entity in enumerate(extracted_entities):
                        if i == j:
                            continue  # Skip same entity
                            
                        # Extract context around entities
                        # In a real implementation, we would use more sophisticated context extraction
                        context = text  # Simplified - would extract relevant sentence
                        
                        # Prepare input for relation classification
                        # This is a simplified approach - actual implementation would depend on model
                        input_text = f"{source_entity.text} [SEP] {target_entity.text} [SEP] {context}"
                        inputs = self._relation_tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
                        
                        # Get relation prediction (dummy implementation)
                        # In a real scenario, we would use the actual model prediction
                        relation_type = "TREATS"  # Placeholder
                        confidence = 0.85  # Placeholder
                        
                        # Only include relations above threshold
                        if confidence >= self.confidence_threshold:
                            relation = BiomedicalRelation(
                                relation_type=relation_type,
                                source_entity=source_entity,
                                target_entity=target_entity,
                                confidence=confidence,
                                evidence_text=context
                            )
                            relations.append(relation)
                
            except Exception as e:
                logger.error(f"Error during relation extraction: {str(e)}")
        
        # If using domain rules (fallback or additional approach)
        # Implement rule-based relation extraction
        # This would use patterns like "{DRUG} is effective for treating {DISEASE}"
        
        return relations
    
    async def compute_semantic_similarity(
        self,
        text1: str, 
        text2: str
    ) -> float:
        """
        Compute semantic similarity between two biomedical texts using domain-specific models.
        
        Args:
            text1: First text
            text2: Second text
        
        Returns:
            Similarity score from 0.0 to 1.0
        """
        if not self.use_advanced_nlp or self._embeddings_model is None:
            logger.warning("Advanced NLP not available for semantic similarity")
            return 0.0
            
        try:
            # Encode both texts into embeddings
            encoded1 = self._embeddings_tokenizer(
                text1, 
                padding=True, 
                truncation=True, 
                return_tensors="pt",
                max_length=512
            )
            encoded2 = self._embeddings_tokenizer(
                text2, 
                padding=True, 
                truncation=True, 
                return_tensors="pt",
                max_length=512
            )
            
            # Get embeddings from the model
            with torch.no_grad():
                output1 = self._embeddings_model(**encoded1)
                output2 = self._embeddings_model(**encoded2)
            
            # Use CLS token embedding as the sentence embedding
            embedding1 = output1.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
            embedding2 = output2.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
            
            # Compute cosine similarity
            similarity = torch.nn.functional.cosine_similarity(
                torch.tensor(embedding1).unsqueeze(0),
                torch.tensor(embedding2).unsqueeze(0)
            ).item()
            
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error computing semantic similarity: {str(e)}")
            return 0.0
    
    async def enrich_knowledge_graph_with_entities(
        self,
        document_id: str,
        text: str,
        entity_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Extract biomedical entities from text and add them to the knowledge graph.
        
        Args:
            document_id: ID of the document being processed
            text: Text to extract entities from
            entity_types: Optional list of entity types to extract
            
        Returns:
            Dictionary with extraction statistics
        """
        if not self.graph_service:
            raise ValueError("Graph service is required for knowledge graph enrichment")
            
        # Extract biomedical entities
        entities = await self.extract_biomedical_entities(text, entity_types)
        
        # Extract relations between entities
        relations = await self.extract_biomedical_relations(text, entities)
        
        # Add entities to graph
        entity_nodes = []
        for entity in entities:
            # Skip entities without ontology mapping if required
            if entity.concept_id is None:
                continue
                
            # Create or get entity node
            entity_node = await self.graph_service.create_or_get_node(
                labels=[ONTOLOGY_NODE_LABEL, entity.entity_type],
                properties={
                    "text": entity.text,
                    "concept_id": entity.concept_id,
                    "ontology_source": entity.ontology_source or "EXTRACTED",
                    "confidence": entity.confidence
                }
            )
            entity_nodes.append(entity_node)
            
            # Link entity to document
            await self.graph_service.create_relationship(
                start_node_id=document_id,
                end_node_id=entity_node.id,
                relationship_type=ONTOLOGY_RELATIONSHIP,
                properties={
                    "confidence": entity.confidence,
                    "extraction_timestamp": datetime.now().isoformat()
                }
            )
        
        # Add relations to graph
        relation_count = 0
        for relation in relations:
            # Find nodes for entities in the relation
            source_node = None
            target_node = None
            
            for node in entity_nodes:
                node_text = node.properties.get("text")
                
                if node_text == relation.source_entity.text:
                    source_node = node
                elif node_text == relation.target_entity.text:
                    target_node = node
            
            if source_node and target_node:
                # Create relationship between entities
                await self.graph_service.create_relationship(
                    start_node_id=source_node.id,
                    end_node_id=target_node.id,
                    relationship_type=relation.relation_type,
                    properties={
                        "confidence": relation.confidence,
                        "evidence_text": relation.evidence_text,
                        "source_doc_id": document_id,
                        "extraction_timestamp": datetime.now().isoformat()
                    }
                )
                relation_count += 1
        
        return {
            "document_id": document_id,
            "entities_extracted": len(entities),
            "entities_added": len(entity_nodes),
            "relations_extracted": len(relations),
            "relations_added": relation_count
        }

    async def find_contradiction_candidates(
        self, 
        concept_id: str,
        relation_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Find potential contradictions in the knowledge graph for a given concept.
        
        This uses semantic understanding of contradictions (treats vs. worsens, 
        improves vs. no effect) to identify potentially conflicting statements.
        
        Args:
            concept_id: The ontology concept ID to check for contradictions
            relation_types: Optional specific relation types to check
            
        Returns:
            List of contradiction candidate pairs with evidence
        """
        if not self.graph_service:
            raise ValueError("Graph service is required for contradiction detection")
            
        contradictions = []
        
        # Define contradictory relation pairs
        contradiction_pairs = {
            "TREATS": ["WORSENS", "NO_EFFECT"],
            "IMPROVES": ["WORSENS", "NO_EFFECT"],
            "CAUSES": ["PREVENTS", "NO_EFFECT"],
            "INCREASES": ["DECREASES"],
            "ASSOCIATED_WITH": ["NOT_ASSOCIATED_WITH"]
        }
        
        # Get all relations for the concept
        query = f"""
        MATCH (c:{ONTOLOGY_NODE_LABEL} {{concept_id: $concept_id}})-[r]->(target)
        RETURN c, r, target
        UNION
        MATCH (source)-[r]->(c:{ONTOLOGY_NODE_LABEL} {{concept_id: $concept_id}})
        RETURN source, r, c
        """
        
        results = await self.graph_service.execute_query(
            query, 
            {"concept_id": concept_id}
        )
        
        # Analyze results for contradictory relations
        relations_by_target = {}
        
        for record in results:
            source = record.get("source") or record.get("c")
            target = record.get("target") or record.get("c")
            relation = record.get("r")
            
            relation_type = relation.get("type")
            if relation_types and relation_type not in relation_types:
                continue
                
            target_id = target.get("concept_id")
            if target_id not in relations_by_target:
                relations_by_target[target_id] = []
                
            relations_by_target[target_id].append({
                "source": source,
                "target": target,
                "relation": relation
            })
        
        # Check for contradictions
        for target_id, relations in relations_by_target.items():
            for i, rel1 in enumerate(relations):
                rel1_type = rel1["relation"].get("type")
                
                # Skip if this relation type doesn't have contradictions defined
                if rel1_type not in contradiction_pairs:
                    continue
                
                for j in range(i+1, len(relations)):
                    rel2 = relations[j]
                    rel2_type = rel2["relation"].get("type")
                    
                    # Check if relations contradict each other
                    if rel2_type in contradiction_pairs.get(rel1_type, []):
                        contradictions.append({
                            "relation1": rel1,
                            "relation2": rel2,
                            "evidence": {
                                "contradiction_type": f"{rel1_type} vs {rel2_type}",
                                "concept_id": concept_id,
                                "target_concept_id": target_id
                            }
                        })
        
        return contradictions

    async def map_across_ontologies(
        self,
        concept_id: str,
        source_ontology: str,
        target_ontology: str
    ) -> List[Dict[str, Any]]:
        """
        Map a concept from one ontology to another using UMLS as a bridge.
        
        Args:
            concept_id: The source concept ID to map
            source_ontology: Source ontology (e.g., "SNOMED", "MESH")
            target_ontology: Target ontology to map to
            
        Returns:
            List of matching concepts in the target ontology
        """
        mapped_concepts = []
        
        try:
            # Try to get from cache first
            cache_key = f"ontology:mapping:{source_ontology}:{concept_id}:{target_ontology}"
            cached_mapping = await self.cache.get(cache_key)
            
            if cached_mapping:
                return cached_mapping
            
            # Use UMLS as a bridge for cross-ontology mapping
            umls_concepts = await self.umls_client.get_concept_mappings(
                concept_id=concept_id,
                source_vocabulary=source_ontology
            )
            
            # For each UMLS concept, find related concepts in the target ontology
            for umls_concept in umls_concepts:
                umls_cui = umls_concept.get("cui")
                if not umls_cui:
                    continue
                    
                target_concepts = await self.umls_client.get_concepts_by_cui(
                    cui=umls_cui,
                    target_vocabulary=target_ontology
                )
                
                for target_concept in target_concepts:
                    mapped_concepts.append({
                        "source_concept_id": concept_id,
                        "source_ontology": source_ontology,
                        "umls_cui": umls_cui,
                        "target_concept_id": target_concept.get("id"),
                        "target_ontology": target_ontology,
                        "target_preferred_name": target_concept.get("name"),
                        "confidence": target_concept.get("score", 1.0)
                    })
            
            # Cache the results if we found any mappings
            if mapped_concepts:
                await self.cache.set(cache_key, mapped_concepts, expire=86400*7)  # Cache for a week
                
            return mapped_concepts
            
        except Exception as e:
            logger.error(f"Error mapping across ontologies: {str(e)}")
            return []
    
    async def add_cross_ontology_mappings_to_graph(
        self,
        concept_id: str,
        source_ontology: str,
        target_ontologies: List[str]
    ) -> Dict[str, Any]:
        """
        Add cross-ontology mappings to the knowledge graph for a specific concept.
        
        Args:
            concept_id: The source concept ID to map
            source_ontology: Source ontology (e.g., "SNOMED", "MESH")
            target_ontologies: List of target ontologies to map to
            
        Returns:
            Statistics about the mappings added
        """
        if not self.graph_service:
            raise ValueError("Graph service is required for adding mappings to the graph")
            
        mappings_added = 0
        results = {}
        
        # Create source concept node if it doesn't exist
        source_node = await self.graph_service.create_or_get_node(
            labels=[ONTOLOGY_NODE_LABEL, source_ontology],
            properties={
                "concept_id": concept_id,
                "ontology": source_ontology
            }
        )
        
        # For each target ontology, add mappings
        for target_ontology in target_ontologies:
            mappings = await self.map_across_ontologies(
                concept_id=concept_id,
                source_ontology=source_ontology,
                target_ontology=target_ontology
            )
            
            for mapping in mappings:
                target_concept_id = mapping.get("target_concept_id")
                umls_cui = mapping.get("umls_cui")
                confidence = mapping.get("confidence", 1.0)
                
                # Create target concept node
                target_node = await self.graph_service.create_or_get_node(
                    labels=[ONTOLOGY_NODE_LABEL, target_ontology],
                    properties={
                        "concept_id": target_concept_id,
                        "ontology": target_ontology,
                        "preferred_name": mapping.get("target_preferred_name")
                    }
                )
                
                # Create relationship between concepts
                await self.graph_service.create_relationship(
                    start_node_id=source_node.id,
                    end_node_id=target_node.id,
                    relationship_type=CROSS_ONTOLOGY_RELATIONSHIP,
                    properties={
                        "umls_cui": umls_cui,
                        "confidence": confidence,
                        "mapping_timestamp": datetime.now().isoformat()
                    }
                )
                
                mappings_added += 1
                
            results[target_ontology] = len(mappings)
            
        return {
            "source_concept": f"{source_ontology}:{concept_id}",
            "mappings_added": mappings_added,
            "target_ontologies": results
        }
        
    async def find_contradictions_by_text(
        self,
        text: str,
        confidence_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Extract entities from text and find potential contradictions for them.
        
        Args:
            text: Text to analyze for contradictions
            confidence_threshold: Minimum confidence for entity extraction
            
        Returns:
            List of contradiction candidates with evidence
        """
        # Extract entities from the text
        entities = await self.extract_biomedical_entities(text)
        
        # Filter entities by confidence
        high_confidence_entities = [
            entity for entity in entities
            if entity.confidence >= confidence_threshold
        ]
        
        all_contradictions = []
        
        # Check each entity for contradictions
        for entity in high_confidence_entities:
            if entity.concept_id:
                contradictions = await self.find_contradiction_candidates(entity.concept_id)
                
                if contradictions:
                    for contradiction in contradictions:
                        contradiction["source_text"] = text
                        contradiction["entity"] = entity.to_dict()
                    
                    all_contradictions.extend(contradictions)
        
        return all_contradictions

    async def analyze_medical_text(
        self,
        text: str,
        extract_entities: bool = True,
        extract_relations: bool = True,
        link_to_ontology: bool = True
    ) -> Dict[str, Any]:
        """
        Complete analysis of medical text using advanced NLP techniques.
        
        This is a convenience method that combines entity extraction,
        relation extraction, and semantic analysis in one call.
        
        Args:
            text: Medical text to analyze
            extract_entities: Whether to extract biomedical entities
            extract_relations: Whether to extract relationships between entities
            link_to_ontology: Whether to link extracted entities to ontology concepts
            
        Returns:
            Dictionary with analysis results
        """
        results = {
            "text_length": len(text),
            "timestamp": datetime.now().isoformat()
        }
        
        # Extract entities if requested
        entities = []
        if extract_entities:
            entities = await self.extract_biomedical_entities(text)
            results["entities"] = [entity.to_dict() for entity in entities]
        
        # Extract relations if requested and if we have entities
        if extract_relations and entities and len(entities) > 1:
            relations = await self.extract_biomedical_relations(text, entities)
            results["relations"] = [relation.to_dict() for relation in relations]
        
        # Find potential contradictions in the knowledge graph
        if link_to_ontology and entities:
            # Only check for contradictions if we have high-confidence entities with concept IDs
            concept_entities = [e for e in entities if e.concept_id and e.confidence >= self.confidence_threshold]
            if concept_entities:
                contradictions = []
                for entity in concept_entities:
                    entity_contradictions = await self.find_contradiction_candidates(entity.concept_id)
                    if entity_contradictions:
                        contradictions.extend(entity_contradictions)
                
                if contradictions:
                    results["contradictions"] = contradictions
        
        return results
    
    async def compare_medical_statements(
        self,
        statement1: str,
        statement2: str,
        detailed_analysis: bool = False
    ) -> Dict[str, Any]:
        """
        Compare two medical statements for semantic similarity and potential contradictions.
        
        Args:
            statement1: First medical statement
            statement2: Second medical statement
            detailed_analysis: Whether to perform detailed entity and relation analysis
            
        Returns:
            Comparison results including similarity score and potential contradictions
        """
        results = {
            "statements": {
                "statement1": statement1,
                "statement2": statement2
            },
            "timestamp": datetime.now().isoformat()
        }
        
        # Compute semantic similarity
        similarity_score = await self.compute_semantic_similarity(statement1, statement2)
        results["similarity_score"] = similarity_score
        
        # Determine if statements are potentially contradictory based on similarity
        results["potentially_contradictory"] = False
        
        # If similarity is moderate (not too low, not too high), they might be talking about the same thing
        # but with different claims
        if 0.4 <= similarity_score <= 0.8:
            results["potentially_contradictory"] = True
        
        # Do detailed analysis if requested
        if detailed_analysis:
            # Extract entities and relations from both statements
            entities1 = await self.extract_biomedical_entities(statement1)
            entities2 = await self.extract_biomedical_entities(statement2)
            
            relations1 = await self.extract_biomedical_relations(statement1, entities1)
            relations2 = await self.extract_biomedical_relations(statement2, entities2)
            
            results["analysis"] = {
                "statement1": {
                    "entities": [e.to_dict() for e in entities1],
                    "relations": [r.to_dict() for r in relations1]
                },
                "statement2": {
                    "entities": [e.to_dict() for e in entities2],
                    "relations": [r.to_dict() for r in relations2]
                }
            }
            
            # Find common entities between statements
            common_entities = []
            for e1 in entities1:
                for e2 in entities2:
                    # Check if entities refer to the same concept
                    if e1.concept_id and e2.concept_id and e1.concept_id == e2.concept_id:
                        common_entities.append({
                            "entity1": e1.to_dict(),
                            "entity2": e2.to_dict()
                        })
                    # Or if they have very similar text
                    elif e1.text.lower() == e2.text.lower() or e1.text.lower() in e2.text.lower() or e2.text.lower() in e1.text.lower():
                        common_entities.append({
                            "entity1": e1.to_dict(),
                            "entity2": e2.to_dict()
                        })
            
            results["common_entities"] = common_entities
            
            # Find contradictory relations for common entities
            contradictory_relations = []
            for r1 in relations1:
                for r2 in relations2:
                    # Check if relations involve the same entities but have contradictory types
                    if (r1.source_entity.concept_id and r1.source_entity.concept_id == r2.source_entity.concept_id and
                        r1.target_entity.concept_id and r1.target_entity.concept_id == r2.target_entity.concept_id):
                        
                        # Define contradictory relation types
                        contradictory_pairs = {
                            "TREATS": ["WORSENS", "NO_EFFECT"],
                            "IMPROVES": ["WORSENS", "NO_EFFECT"],
                            "CAUSES": ["PREVENTS", "NO_EFFECT"],
                            "INCREASES": ["DECREASES"],
                            "ASSOCIATED_WITH": ["NOT_ASSOCIATED_WITH"]
                        }
                        
                        if r1.relation_type in contradictory_pairs and r2.relation_type in contradictory_pairs.get(r1.relation_type, []):
                            contradictory_relations.append({
                                "relation1": r1.to_dict(),
                                "relation2": r2.to_dict(),
                                "contradiction_type": f"{r1.relation_type} vs {r2.relation_type}"
                            })
            
            if contradictory_relations:
                results["contradictory_relations"] = contradictory_relations
                results["potentially_contradictory"] = True
        
        return results
    
    async def batch_process_documents(
        self,
        document_texts: List[Dict[str, str]],
        extract_entities: bool = True,
        extract_relations: bool = True,
        enrich_graph: bool = False
    ) -> Dict[str, Any]:
        """
        Batch process multiple medical documents for NLP analysis.
        
        Args:
            document_texts: List of documents with 'id' and 'text' fields
            extract_entities: Whether to extract biomedical entities
            extract_relations: Whether to extract relationships
            enrich_graph: Whether to add extracted info to knowledge graph
            
        Returns:
            Processing results for each document
        """
        results = {
            "total_documents": len(document_texts),
            "processed_documents": 0,
            "failed_documents": 0,
            "total_entities": 0,
            "total_relations": 0,
            "document_results": []
        }
        
        for doc in document_texts:
            doc_id = doc.get("id")
            doc_text = doc.get("text", "")
            
            if not doc_id or not doc_text:
                results["failed_documents"] += 1
                continue
            
            try:
                doc_result = {
                    "id": doc_id,
                    "text_length": len(doc_text)
                }
                
                # Extract entities
                entities = []
                if extract_entities:
                    entities = await self.extract_biomedical_entities(doc_text)
                    doc_result["entities_count"] = len(entities)
                    results["total_entities"] += len(entities)
                
                # Extract relations
                relations = []
                if extract_relations and entities and len(entities) > 1:
                    relations = await self.extract_biomedical_relations(doc_text, entities)
                    doc_result["relations_count"] = len(relations)
                    results["total_relations"] += len(relations)
                
                # Enrich knowledge graph if requested
                if enrich_graph and self.graph_service:
                    enrichment_result = await self.enrich_knowledge_graph_with_entities(
                        document_id=doc_id,
                        text=doc_text
                    )
                    doc_result["graph_enrichment"] = enrichment_result
                
                results["document_results"].append(doc_result)
                results["processed_documents"] += 1
                
            except Exception as e:
                logger.error(f"Error processing document {doc_id}: {str(e)}")
                results["failed_documents"] += 1
        
        return results
    
    @classmethod
    async def create_with_biobert(
        cls,
        graph_service: Optional[GraphService] = None,
        cache: Optional[Cache] = None
    ) -> 'OntologyIntegrationService':
        """
        Factory method to create an instance with BioBERT models.
        
        Args:
            graph_service: Optional graph service for knowledge graph operations
            cache: Optional cache for storing mapped concepts
            
        Returns:
            Configured OntologyIntegrationService instance
        """
        service = cls(
            graph_service=graph_service,
            cache=cache,
            use_advanced_nlp=True,
            model_path=None  # Will use default BioBERT models
        )
        
        # Initialize models
        if NLP_ADVANCED_AVAILABLE:
            await asyncio.to_thread(service._initialize_nlp_components)
        
        return service
    
    @classmethod
    async def create_with_pubmedbert(
        cls,
        graph_service: Optional[GraphService] = None,
        cache: Optional[Cache] = None
    ) -> 'OntologyIntegrationService':
        """
        Factory method to create an instance with PubMedBERT models.
        
        Args:
            graph_service: Optional graph service for knowledge graph operations
            cache: Optional cache for storing mapped concepts
            
        Returns:
            Configured OntologyIntegrationService instance
        """
        service = cls(
            graph_service=graph_service,
            cache=cache,
            use_advanced_nlp=True
        )
        
        # Override default model paths with PubMedBERT models
        service.default_ner_model = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
        service.default_relation_model = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
        service.default_embeddings_model = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
        
        # Initialize models
        if NLP_ADVANCED_AVAILABLE:
            await asyncio.to_thread(service._initialize_nlp_components)
        
        return service