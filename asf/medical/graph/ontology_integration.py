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

Dependencies:
- UMLS API access for ontology concept resolution
- Graph database connection (Neo4j or Memgraph)
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

# Configure logger
logger = get_logger(__name__)

# Constants
DEFAULT_CONFIDENCE_THRESHOLD = 0.7
ONTOLOGY_NODE_LABEL = "OntologyConcept"
ONTOLOGY_RELATIONSHIP = "MENTIONS_CONCEPT"
CROSS_ONTOLOGY_RELATIONSHIP = "MAPS_TO"


@dataclass
class OntologyConcept:
    """Represents a concept from a medical ontology (SNOMED, MeSH, etc.)."""
    concept_id: str
    ontology: str
    name: str
    definition: Optional[str] = None
    semantic_types: List[str] = field(default_factory=list)
    synonyms: List[str] = field(default_factory=list)
    ancestors: List[str] = field(default_factory=list)
    descendants: List[str] = field(default_factory=list)
    relationships: Dict[str, List[str]] = field(default_factory=dict)
    source_uri: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "concept_id": self.concept_id,
            "ontology": self.ontology,
            "name": self.name,
            "definition": self.definition,
            "semantic_types": self.semantic_types,
            "synonyms": self.synonyms,
            "ancestors": self.ancestors,
            "descendants": self.descendants,
            "relationships": self.relationships,
            "source_uri": self.source_uri
        }


class OntologyIntegrationService:
    """
    Service for integrating medical ontologies into the knowledge graph.
    
    This service provides methods to:
    1. Map text to ontology concepts
    2. Link concepts across different ontologies
    3. Enrich graph with ontology relationships
    4. Navigate concept hierarchies
    """
    
    def __init__(
        self, 
        graph_service: GraphService,
        umls_client: Optional[UMLSClient] = None,
        snomed_client: Optional[SNOMEDClient] = None,
        cache: Optional[Cache] = None,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD
    ):
        """
        Initialize the ontology integration service.
        
        Args:
            graph_service: Graph database service for storing and querying concepts
            umls_client: Client for accessing UMLS API (optional)
            snomed_client: Client for accessing SNOMED CT API (optional)
            cache: Cache instance for storing ontology lookups (optional)
            confidence_threshold: Minimum confidence for concept mapping
        """
        self.graph_service = graph_service
        self.umls_client = umls_client or UMLSClient()
        self.snomed_client = snomed_client or SNOMEDClient()
        self.cache = cache or get_cache("ontology_service")
        self.confidence_threshold = confidence_threshold
        
        # Track statistics for monitoring
        self.stats = {
            "text_mappings": 0,
            "concepts_added": 0,
            "relationships_added": 0,
            "cross_mappings_created": 0
        }
    
    async def map_text_to_ontologies(
        self, 
        text: str,
        source_ontologies: List[str] = None,
        confidence_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Map free text to standardized concepts from medical ontologies.
        
        Args:
            text: Free text to map to ontology concepts
            source_ontologies: List of ontology sources to use (e.g., ["SNOMED", "MESH"])
            confidence_threshold: Minimum mapping confidence (0.0-1.0)
            
        Returns:
            List of mapped concepts with metadata
        """
        if not source_ontologies:
            source_ontologies = ["SNOMED", "MESH"]
        
        threshold = confidence_threshold or self.confidence_threshold
        
        # Cache key for this request
        cache_key = f"text_map:{text}:{'-'.join(sorted(source_ontologies))}"
        cached = await self.cache.get(cache_key)
        if cached:
            return json.loads(cached)
        
        # Use UMLS to map text to concepts
        try:
            concepts = await self.umls_client.search_concepts(
                text,
                sources=source_ontologies,
                include_semantic_types=True,
                include_synonyms=True
            )
            
            # Filter by confidence threshold
            filtered_concepts = [
                {
                    "concept_id": concept["cui"],
                    "ontology": source,
                    "text": concept.get("name", ""),
                    "confidence": concept.get("score", 0) / 1000.0,
                    "definition": concept.get("definition", ""),
                    "semantic_types": concept.get("semanticTypes", []),
                    "synonyms": concept.get("synonyms", []),
                    "span": concept.get("span", {"begin": 0, "end": len(text)}),
                    "source_uri": concept.get("uri", "")
                }
                for concept in concepts
                for source in concept.get("sources", [])
                if source in source_ontologies and concept.get("score", 0) / 1000.0 >= threshold
            ]
            
            # Update statistics
            self.stats["text_mappings"] += 1
            
            # Cache results
            await self.cache.set(cache_key, json.dumps(filtered_concepts), expire=86400)  # 24-hour cache
            
            return filtered_concepts
            
        except Exception as e:
            logger.error(f"Error mapping text to ontologies: {e}")
            return []

    async def get_concept_by_id(self, concept_id: str, ontology: str) -> Optional[OntologyConcept]:
        """
        Retrieve a concept by its ID and ontology source.
        
        Args:
            concept_id: The concept identifier
            ontology: The ontology source (SNOMED, MESH, etc.)
            
        Returns:
            OntologyConcept object if found, None otherwise
        """
        # Check cache first
        cache_key = f"concept:{ontology}:{concept_id}"
        cached = await self.cache.get(cache_key)
        if cached:
            return OntologyConcept(**json.loads(cached))
        
        # Check graph database
        concept_node = await self.graph_service.get_node(
            ONTOLOGY_NODE_LABEL,
            properties={
                "concept_id": concept_id,
                "ontology": ontology
            }
        )
        
        if concept_node:
            concept = OntologyConcept(
                concept_id=concept_node.get("concept_id"),
                ontology=concept_node.get("ontology"),
                name=concept_node.get("name"),
                definition=concept_node.get("definition"),
                semantic_types=concept_node.get("semantic_types", []),
                synonyms=concept_node.get("synonyms", []),
                source_uri=concept_node.get("source_uri")
            )
            
            # Cache the result
            await self.cache.set(cache_key, json.dumps(concept.to_dict()), expire=86400)
            return concept
        
        # If not in graph, fetch from UMLS
        try:
            if ontology == "SNOMED":
                concept_data = await self.snomed_client.get_concept(concept_id)
            else:
                concept_data = await self.umls_client.get_concept_by_id(
                    concept_id, 
                    source=ontology
                )
            
            if concept_data:
                concept = OntologyConcept(
                    concept_id=concept_id,
                    ontology=ontology,
                    name=concept_data.get("name", ""),
                    definition=concept_data.get("definition", ""),
                    semantic_types=concept_data.get("semanticTypes", []),
                    synonyms=concept_data.get("synonyms", []),
                    source_uri=concept_data.get("uri", "")
                )
                
                # Store in graph for future use
                await self.add_concept_to_graph(concept)
                
                # Cache the result
                await self.cache.set(cache_key, json.dumps(concept.to_dict()), expire=86400)
                return concept
            
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving concept {concept_id} from {ontology}: {e}")
            return None
    
    async def add_concept_to_graph(self, concept: Union[OntologyConcept, Dict[str, Any]]) -> bool:
        """
        Add an ontology concept to the knowledge graph.
        
        Args:
            concept: OntologyConcept object or dictionary representation
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if isinstance(concept, OntologyConcept):
                concept_dict = concept.to_dict()
            else:
                concept_dict = concept
            
            # Ensure required properties
            if not concept_dict.get("concept_id") or not concept_dict.get("ontology"):
                logger.error("Cannot add concept without concept_id and ontology")
                return False
            
            # Add concept node to graph
            node_id = await self.graph_service.create_node(
                ONTOLOGY_NODE_LABEL,
                properties=concept_dict
            )
            
            if node_id:
                self.stats["concepts_added"] += 1
                logger.debug(f"Added concept {concept_dict.get('concept_id')} to graph")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error adding concept to graph: {e}")
            return False
    
    async def get_related_concepts(
        self,
        concept_id: str,
        ontology: str,
        relationship_types: List[str] = None,
        max_depth: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Get concepts related to a given concept.
        
        Args:
            concept_id: The concept identifier
            ontology: The ontology source (SNOMED, MESH, etc.)
            relationship_types: Types of relationships to follow (e.g., ["broader", "narrower"])
            max_depth: Maximum traversal depth in concept hierarchy
            
        Returns:
            List of related concepts with relationship information
        """
        if not relationship_types:
            relationship_types = ["broader", "narrower", "related"]
        
        # Check cache first
        cache_key = f"related:{ontology}:{concept_id}:{'-'.join(relationship_types)}:{max_depth}"
        cached = await self.cache.get(cache_key)
        if cached:
            return json.loads(cached)
        
        related_concepts = []
        
        # Get concept from graph
        source_concept = await self.get_concept_by_id(concept_id, ontology)
        if not source_concept:
            return []
        
        # Query graph for directly related concepts
        for rel_type in relationship_types:
            graph_rel_type = self._map_relationship_type(rel_type, ontology)
            
            related_nodes = await self.graph_service.get_related_nodes(
                ONTOLOGY_NODE_LABEL,
                start_properties={"concept_id": concept_id, "ontology": ontology},
                relationship_type=graph_rel_type,
                direction="outgoing" if rel_type == "narrower" else "incoming",
                depth=max_depth
            )
            
            for node in related_nodes:
                related_concepts.append({
                    "concept_id": node.get("concept_id"),
                    "ontology": node.get("ontology"),
                    "text": node.get("name", ""),
                    "relationship": rel_type,
                    "definition": node.get("definition", ""),
                    "semantic_types": node.get("semantic_types", []),
                })
        
        # If not enough found in graph, enrich from API
        if len(related_concepts) < 3 and ontology in ["SNOMED", "MESH"]:
            try:
                if ontology == "SNOMED":
                    api_related = await self.snomed_client.get_related_concepts(
                        concept_id,
                        relationship_types=relationship_types
                    )
                else:
                    api_related = await self.umls_client.get_related_concepts(
                        concept_id,
                        source=ontology,
                        relationship_types=relationship_types
                    )
                
                # Add API results to related concepts if not already present
                existing_ids = {c["concept_id"] for c in related_concepts}
                for rel in api_related:
                    if rel.get("concept_id") not in existing_ids:
                        related_concepts.append(rel)
                        
                        # Add to graph for future queries
                        rel_concept = OntologyConcept(
                            concept_id=rel["concept_id"],
                            ontology=rel["ontology"],
                            name=rel["text"],
                            definition=rel.get("definition", ""),
                            semantic_types=rel.get("semantic_types", [])
                        )
                        await self.add_concept_to_graph(rel_concept)
                        
                        # Add relationship in graph
                        rel_type = self._map_relationship_type(rel["relationship"], ontology)
                        if rel["relationship"] == "narrower":
                            await self.graph_service.create_relationship(
                                ONTOLOGY_NODE_LABEL,
                                {"concept_id": concept_id, "ontology": ontology},
                                rel_type,
                                ONTOLOGY_NODE_LABEL,
                                {"concept_id": rel["concept_id"], "ontology": rel["ontology"]}
                            )
                        else:
                            await self.graph_service.create_relationship(
                                ONTOLOGY_NODE_LABEL,
                                {"concept_id": rel["concept_id"], "ontology": rel["ontology"]},
                                rel_type,
                                ONTOLOGY_NODE_LABEL,
                                {"concept_id": concept_id, "ontology": ontology}
                            )
                            
                        self.stats["relationships_added"] += 1
                
            except Exception as e:
                logger.error(f"Error getting related concepts from API: {e}")
        
        # Cache the results
        await self.cache.set(cache_key, json.dumps(related_concepts), expire=86400)
        
        return related_concepts
    
    async def map_concepts_across_ontologies(
        self,
        concepts: List[Dict[str, Any]],
        target_ontologies: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Map concepts from source ontologies to equivalent concepts in target ontologies.
        
        Args:
            concepts: List of source concepts to map
            target_ontologies: List of target ontology sources
            
        Returns:
            List of mapped concepts with relationship to source concepts
        """
        if not target_ontologies:
            target_ontologies = ["SNOMED", "MESH", "ICD10"]
        
        mapped_concepts = []
        
        for concept in concepts:
            # Skip if concept lacks required fields
            if not concept.get("concept_id") or not concept.get("ontology"):
                continue
                
            # Skip if target includes source ontology
            if concept["ontology"] in target_ontologies:
                target_list = [t for t in target_ontologies if t != concept["ontology"]]
            else:
                target_list = target_ontologies
                
            # Skip if no remaining targets
            if not target_list:
                continue
                
            # Check cache first
            cache_key = f"cross_map:{concept['ontology']}:{concept['concept_id']}:{'-'.join(sorted(target_list))}"
            cached = await self.cache.get(cache_key)
            if cached:
                cached_results = json.loads(cached)
                for result in cached_results:
                    result["mapped_from"] = concept
                    mapped_concepts.append(result)
                continue
                
            # First check graph for existing mappings
            graph_mappings = await self.graph_service.get_related_nodes(
                ONTOLOGY_NODE_LABEL,
                start_properties={"concept_id": concept["concept_id"], "ontology": concept["ontology"]},
                relationship_type=CROSS_ONTOLOGY_RELATIONSHIP,
                direction="outgoing",
                end_properties={"ontology": {"$in": target_list}}
            )
            
            cross_mappings = []
            
            for mapping in graph_mappings:
                cross_mappings.append({
                    "concept_id": mapping.get("concept_id"),
                    "ontology": mapping.get("ontology"),
                    "text": mapping.get("name", ""),
                    "definition": mapping.get("definition", ""),
                    "semantic_types": mapping.get("semantic_types", []),
                    "mapping_type": "existing"
                })
            
            # If insufficient results from graph, use UMLS API
            if len(cross_mappings) < len(target_list):
                try:
                    api_mappings = await self.umls_client.map_to_ontologies(
                        concept["concept_id"],
                        source_ontology=concept["ontology"],
                        target_ontologies=target_list
                    )
                    
                    # Add API results if not already present
                    existing_ids = {f"{m['ontology']}:{m['concept_id']}" for m in cross_mappings}
                    for mapping in api_mappings:
                        mapping_key = f"{mapping['ontology']}:{mapping['concept_id']}"
                        if mapping_key not in existing_ids:
                            cross_mappings.append(mapping)
                            
                            # Add mapped concept to graph
                            await self.add_concept_to_graph({
                                "concept_id": mapping["concept_id"],
                                "ontology": mapping["ontology"],
                                "name": mapping["text"],
                                "definition": mapping.get("definition", ""),
                                "semantic_types": mapping.get("semantic_types", [])
                            })
                            
                            # Add cross-ontology relationship
                            await self.graph_service.create_relationship(
                                ONTOLOGY_NODE_LABEL,
                                {"concept_id": concept["concept_id"], "ontology": concept["ontology"]},
                                CROSS_ONTOLOGY_RELATIONSHIP,
                                ONTOLOGY_NODE_LABEL,
                                {"concept_id": mapping["concept_id"], "ontology": mapping["ontology"]}
                            )
                            
                            self.stats["cross_mappings_created"] += 1
                    
                except Exception as e:
                    logger.error(f"Error mapping concept across ontologies: {e}")
            
            # Cache the mappings
            await self.cache.set(cache_key, json.dumps(cross_mappings), expire=86400 * 7)  # 7-day cache
            
            # Add source concept reference to results
            for mapping in cross_mappings:
                mapping["mapped_from"] = concept
                mapped_concepts.append(mapping)
        
        return mapped_concepts
    
    async def enrich_graph_with_ontologies(
        self,
        node_type: str,
        content_field: str,
        limit: int = 100,
        batch_size: int = 10
    ) -> int:
        """
        Enrich nodes in the graph with ontology concept annotations.
        
        Args:
            node_type: Type of nodes to enrich
            content_field: Field containing text to analyze
            limit: Maximum number of nodes to process
            batch_size: Number of nodes to process in parallel
            
        Returns:
            Number of nodes enriched
        """
        # Get nodes that need ontology enrichment
        nodes = await self.graph_service.get_nodes(
            node_type,
            properties={},
            limit=limit,
            order_by="created_at",
            order_direction="DESC"
        )
        
        if not nodes:
            logger.info(f"No {node_type} nodes found for ontology enrichment")
            return 0
            
        # Process in batches
        total_processed = 0
        for i in range(0, min(len(nodes), limit), batch_size):
            batch = nodes[i:i+batch_size]
            tasks = []
            
            for node in batch:
                if content_field in node and node[content_field]:
                    task = self._enrich_node_with_ontologies(
                        node["id"],
                        node_type,
                        node[content_field]
                    )
                    tasks.append(task)
            
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                successful = sum(1 for r in results if r is True)
                total_processed += successful
                
                logger.info(f"Enriched batch of {successful}/{len(tasks)} nodes with ontology concepts")
        
        return total_processed
    
    async def _enrich_node_with_ontologies(
        self,
        node_id: str,
        node_type: str, 
        content: str
    ) -> bool:
        """
        Enrich a single node with ontology concepts.
        
        Args:
            node_id: ID of the node to enrich
            node_type: Type of the node
            content: Text content to analyze
            
        Returns:
            True if enrichment was successful
        """
        try:
            # Map content to ontology concepts
            concepts = await self.map_text_to_ontologies(
                content,
                source_ontologies=["SNOMED", "MESH"],
                confidence_threshold=0.7  # Higher threshold for enrichment
            )
            
            if not concepts:
                return False
                
            # Add concepts that don't exist
            for concept in concepts:
                # Check if concept exists
                existing = await self.graph_service.get_node(
                    ONTOLOGY_NODE_LABEL,
                    properties={
                        "concept_id": concept["concept_id"],
                        "ontology": concept["ontology"]
                    }
                )
                
                if not existing:
                    concept_obj = {
                        "concept_id": concept["concept_id"],
                        "ontology": concept["ontology"],
                        "name": concept["text"],
                        "definition": concept.get("definition", ""),
                        "semantic_types": concept.get("semantic_types", []),
                        "synonyms": concept.get("synonyms", []),
                        "created_at": datetime.now().isoformat()
                    }
                    await self.add_concept_to_graph(concept_obj)
            
            # Create relationships between node and concepts
            for concept in concepts:
                rel_properties = {
                    "confidence": concept.get("confidence", 0.0),
                    "span_begin": concept.get("span", {}).get("begin", 0),
                    "span_end": concept.get("span", {}).get("end", 0),
                    "created_at": datetime.now().isoformat()
                }
                
                await self.graph_service.create_relationship(
                    node_type,
                    {"id": node_id},
                    ONTOLOGY_RELATIONSHIP,
                    ONTOLOGY_NODE_LABEL,
                    {"concept_id": concept["concept_id"], "ontology": concept["ontology"]},
                    properties=rel_properties
                )
                
                self.stats["relationships_added"] += 1
            
            return True
            
        except Exception as e:
            logger.error(f"Error enriching node {node_id} with ontologies: {e}")
            return False
            
    def _map_relationship_type(self, generic_type: str, ontology: str) -> str:
        """Maps generic relationship types to ontology-specific relationship types."""
        if ontology == "SNOMED":
            mapping = {
                "broader": "IS_A",
                "narrower": "HAS_DESCENDANT",
                "related": "ASSOCIATED_WITH"
            }
        elif ontology == "MESH":
            mapping = {
                "broader": "BROADER",
                "narrower": "NARROWER",
                "related": "RELATED_TO"
            }
        else:
            mapping = {
                "broader": "BROADER",
                "narrower": "NARROWER",
                "related": "RELATED_TO"
            }
            
        return mapping.get(generic_type.lower(), generic_type)


class GraphRAGOntologyEnhancer:
    """
    Enhances GraphRAG retrieval with ontology awareness.
    
    This class adds ontology-based capabilities to GraphRAG:
    1. Query expansion using related medical concepts
    2. Re-ranking results based on ontology concept matching
    3. Enriching results with ontology metadata
    """
    
    def __init__(
        self,
        ontology_service: OntologyIntegrationService,
        expansion_depth: int = 1,
        include_narrower_concepts: bool = True,
        include_broader_concepts: bool = True,
        include_related_concepts: bool = False,
        ontology_weight: float = 0.3
    ):
        """
        Initialize the GraphRAG ontology enhancer.
        
        Args:
            ontology_service: Service for ontology integration
            expansion_depth: Depth for query expansion
            include_narrower_concepts: Include more specific concepts
            include_broader_concepts: Include more general concepts
            include_related_concepts: Include related concepts
            ontology_weight: Weight of ontology factors in ranking (0.0-1.0)
        """
        self.ontology_service = ontology_service
        self.expansion_depth = expansion_depth
        self.include_narrower = include_narrower_concepts
        self.include_broader = include_broader_concepts
        self.include_related = include_related_concepts
        self.ontology_weight = max(0.0, min(1.0, ontology_weight))
        
    async def enhance_search_query(self, query: str) -> str:
        """
        Expand search query with ontology concepts.
        
        Args:
            query: Original search query
            
        Returns:
            Enhanced query with additional ontology terms
        """
        # Map query to ontology concepts
        concepts = await self.ontology_service.map_text_to_ontologies(
            query,
            source_ontologies=["SNOMED", "MESH"],
            confidence_threshold=0.75
        )
        
        if not concepts:
            return query
            
        # Get related concepts based on configuration
        expansion_concepts = []
        relationship_types = []
        
        if self.include_narrower:
            relationship_types.append("narrower")
        if self.include_broader:
            relationship_types.append("broader")
        if self.include_related:
            relationship_types.append("related")
        
        if relationship_types:
            for concept in concepts[:3]:  # Limit to top 3 concepts for expansion
                related = await self.ontology_service.get_related_concepts(
                    concept["concept_id"],
                    concept["ontology"],
                    relationship_types=relationship_types,
                    max_depth=self.expansion_depth
                )
                expansion_concepts.extend(related)
        
        # Build enhanced query
        enhanced_terms = []
        
        # Add original query first
        enhanced_terms.append(query.strip())
        
        # Add synonyms from direct concept matches
        for concept in concepts:
            if "text" in concept and concept["text"] != query:
                enhanced_terms.append(concept["text"])
        
        # Add related concepts with high confidence
        for concept in expansion_concepts:
            # Only add terms that aren't already in query or too similar
            term = concept["text"]
            if term and self._is_significant_term(term, enhanced_terms):
                enhanced_terms.append(term)
                
        # Combine into final query (limit to reasonable size)
        enhanced_query = " ".join(enhanced_terms[:7])  # Limit to original + 6 enhanced terms
        return enhanced_query
    
    async def enhance_graph_retrieval(
        self,
        graph_service: GraphService,
        query: str,
        base_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Enhance graph retrieval results with ontology information.
        
        Args:
            graph_service: Graph database service
            query: Original search query
            base_results: Base retrieval results
            
        Returns:
            Enhanced results with ontology information and re-ranking
        """
        if not base_results:
            return []
            
        # Map query to ontology concepts
        query_concepts = await self.ontology_service.map_text_to_ontologies(
            query,
            source_ontologies=["SNOMED", "MESH"]
        )
        
        if not query_concepts:
            return base_results
            
        enhanced_results = []
        
        # Process each result
        for result in base_results:
            result_copy = result.copy()
            node_id = result_copy.get("id")
            
            if not node_id:
                enhanced_results.append(result_copy)
                continue
                
            # Get ontology concepts related to this node
            related_concepts = await graph_service.get_related_nodes(
                "OntologyConcept",
                start_properties={"id": node_id},
                relationship_type=ONTOLOGY_RELATIONSHIP,
                direction="outgoing"
            )
            
            if not related_concepts:
                enhanced_results.append(result_copy)
                continue
                
            # Calculate concept overlap score
            ontology_score = 0.0
            matched_concept = None
            
            for node_concept in related_concepts:
                node_concept_id = node_concept.get("concept_id")
                node_ontology = node_concept.get("ontology")
                
                if not node_concept_id or not node_ontology:
                    continue
                    
                # Check for direct matches
                for qc in query_concepts:
                    if qc["concept_id"] == node_concept_id and qc["ontology"] == node_ontology:
                        ontology_score += 1.0
                        matched_concept = node_concept
                        break
                        
                # If no direct match and no matched concept yet, check for related matches
                if not matched_concept:
                    for qc in query_concepts:
                        # Check if they're related by cross-ontology mapping
                        if qc["ontology"] != node_ontology:
                            mappings = await self.ontology_service.map_concepts_across_ontologies(
                                [qc],
                                target_ontologies=[node_ontology]
                            )
                            
                            for mapping in mappings:
                                if mapping["concept_id"] == node_concept_id:
                                    ontology_score += 0.8
                                    matched_concept = node_concept
                                    break
                        
                        # Check for hierarchical relationship
                        elif qc["ontology"] == node_ontology:
                            related_to_query = await self.ontology_service.get_related_concepts(
                                qc["concept_id"],
                                qc["ontology"],
                                relationship_types=["broader", "narrower"],
                                max_depth=1
                            )
                            
                            for related in related_to_query:
                                if related["concept_id"] == node_concept_id:
                                    relationship = related["relationship"]
                                    # Narrower concepts (more specific) get higher weight
                                    weight = 0.9 if relationship == "narrower" else 0.7
                                    ontology_score += weight
                                    matched_concept = node_concept
                                    break
            
            # Re-rank based on ontology score
            original_score = result_copy.get("score", 0.0)
            ontology_factor = min(ontology_score, 1.0)  # Cap at 1.0
            
            # Weighted average of original score and ontology factor
            result_copy["score"] = (1 - self.ontology_weight) * original_score + \
                                  self.ontology_weight * ontology_factor
                                  
            # Add ontology enhancement information
            result_copy["ontology_enhanced"] = ontology_score > 0
            
            if matched_concept:
                result_copy["matched_concept"] = {
                    "name": matched_concept.get("name", ""),
                    "concept_id": matched_concept.get("concept_id", ""),
                    "ontology": matched_concept.get("ontology", ""),
                    "semantic_types": matched_concept.get("semantic_types", [])
                }
                
            enhanced_results.append(result_copy)
        
        # Sort by new scores
        enhanced_results.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        
        return enhanced_results
    
    def _is_significant_term(self, new_term: str, existing_terms: List[str]) -> bool:
        """
        Check if a term is significantly different from existing terms.
        Helps avoid adding redundant terms to queries.
        """
        new_term_lower = new_term.lower()
        
        # Check if term is too short or too long
        if len(new_term_lower) < 3 or len(new_term_lower) > 50:
            return False
            
        # Check if term is already in existing terms (case insensitive)
        for term in existing_terms:
            if new_term_lower == term.lower():
                return False
                
            # Check if term is a substring of another term
            if new_term_lower in term.lower() or term.lower() in new_term_lower:
                # If one is significantly longer than the other, it might still be useful
                length_ratio = len(new_term_lower) / len(term.lower())
                if 0.8 <= length_ratio <= 1.2:
                    return False
        
        return True


async def create_ontology_integration_service(
    graph_service: Optional[GraphService] = None
) -> OntologyIntegrationService:
    """
    Factory function to create and initialize an ontology integration service.
    
    Args:
        graph_service: Optional graph service instance
        
    Returns:
        Initialized OntologyIntegrationService
    """
    if graph_service is None:
        graph_service = GraphService()
        await graph_service.connect()
    
    # Initialize clients
    umls_client = UMLSClient()
    snomed_client = SNOMEDClient()
    
    # Create cache
    cache = get_cache("ontology_integration")
    
    # Create service
    service = OntologyIntegrationService(
        graph_service=graph_service,
        umls_client=umls_client,
        snomed_client=snomed_client,
        cache=cache,
        confidence_threshold=0.7
    )
    
    return service