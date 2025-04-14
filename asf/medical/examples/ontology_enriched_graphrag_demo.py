"""
Ontology-Enriched GraphRAG Demonstration

This script demonstrates how to use the ontology integration service
with the GraphRAG system to enhance medical research capabilities.

The demonstration:
1. Initializes the ontology integration service
2. Shows how to map text to ontology concepts
3. Demonstrates query expansion with ontology concepts
4. Performs ontology-enriched graph retrieval
5. Shows how to visualize the ontology-enhanced results

Usage:
    python -m asf.medical.examples.ontology_enriched_graphrag_demo

Requirements:
    - Configured UMLS credentials for UMLS API access
    - Running graph database (Neo4j or Memgraph)
"""

import os
import asyncio
import logging
from pprint import pprint
from typing import Dict, List, Any, Optional

# Import modules
from asf.medical.core.logging_config import get_logger, configure_logging
from asf.medical.graph.graph_service import GraphService
from asf.medical.graph.ontology_integration import (
    OntologyIntegrationService,
    GraphRAGOntologyEnhancer,
    create_ontology_integration_service
)
from asf.medical.graph.graph_rag import GraphRAGRetriever

# Configure logging
configure_logging()
logger = get_logger(__name__)


async def run_demo():
    """Run the ontology-enriched GraphRAG demonstration."""
    logger.info("Starting Ontology-Enriched GraphRAG demonstration")
    
    # Connect to graph database
    graph_service = GraphService()
    await graph_service.connect()
    
    # Initialize ontology integration service
    logger.info("Initializing ontology integration service...")
    ontology_service = await create_ontology_integration_service(graph_service)
    
    # Create GraphRAG ontology enhancer
    ontology_enhancer = GraphRAGOntologyEnhancer(
        ontology_service=ontology_service,
        expansion_depth=1,
        include_narrower_concepts=True,
        include_broader_concepts=True,
        ontology_weight=0.4
    )
    
    # Create GraphRAG retriever
    graph_rag = GraphRAGRetriever(graph_service=graph_service)
    
    # Demo 1: Map text to ontology concepts
    demo_text = "The effect of ACE inhibitors on patients with diabetic nephropathy and hypertension"
    logger.info(f"Demo 1: Mapping text to ontology concepts: '{demo_text}'")
    
    concepts = await ontology_service.map_text_to_ontologies(
        demo_text,
        source_ontologies=["SNOMED", "MESH"]
    )
    
    print("\n=== Mapped Ontology Concepts ===")
    for concept in concepts:
        print(f"• {concept['text']} ({concept['ontology']}:{concept['concept_id']}) - confidence: {concept['confidence']:.2f}")
    
    # Demo 2: Query expansion with ontology concepts
    demo_query = "COVID-19 lung inflammation treatment"
    logger.info(f"Demo 2: Expanding search query: '{demo_query}'")
    
    enhanced_query = await ontology_enhancer.enhance_search_query(demo_query)
    
    print("\n=== Query Expansion ===")
    print(f"Original query: {demo_query}")
    print(f"Enhanced query: {enhanced_query}")
    
    # Demo 3: Retrieve related concepts
    if concepts and len(concepts) > 0:
        demo_concept = concepts[0]
        logger.info(f"Demo 3: Finding related concepts for {demo_concept['text']}")
        
        related_concepts = await ontology_service.get_related_concepts(
            demo_concept['concept_id'],
            demo_concept['ontology'],
            relationship_types=["broader", "narrower", "related"]
        )
        
        print("\n=== Related Concepts ===")
        print(f"For concept: {demo_concept['text']} ({demo_concept['ontology']}:{demo_concept['concept_id']})")
        
        for concept in related_concepts:
            print(f"• {concept['relationship'].upper()}: {concept['text']} ({concept['ontology']}:{concept['concept_id']})")
    
    # Demo 4: Enrich graph with ontology annotations (if graph not yet enriched)
    enrichment_node_type = "Article"
    enrichment_content_field = "abstract"
    
    # Check if graph needs enrichment
    has_ontology_relationships = await graph_service.check_relationship_exists(
        "MENTIONS_CONCEPT", 
        from_type=enrichment_node_type,
        to_type="OntologyConcept"
    )
    
    if not has_ontology_relationships:
        logger.info(f"Demo 4: Enriching graph with ontology annotations")
        print("\n=== Enriching Graph with Ontology Annotations ===")
        print(f"Processing {enrichment_node_type} nodes, field: {enrichment_content_field}")
        
        # Enrich a small batch as a demo
        enriched_count = await ontology_service.enrich_graph_with_ontologies(
            enrichment_node_type,
            enrichment_content_field,
            limit=10  # Small number for demo purposes
        )
        
        print(f"Enriched {enriched_count} nodes with ontology concepts")
    else:
        logger.info("Graph already has ontology annotations, skipping enrichment")
    
    # Demo 5: Perform GraphRAG retrieval with ontology enhancement
    demo_research_query = "Treatments for severe respiratory conditions in elderly patients"
    logger.info(f"Demo 5: Performing GraphRAG retrieval with ontology enhancement")
    
    print("\n=== GraphRAG Retrieval with Ontology Enhancement ===")
    print(f"Research query: {demo_research_query}")
    
    # Get base results
    base_results = await graph_rag.retrieve(
        demo_research_query,
        node_types=["Article"],
        max_results=5
    )
    
    # Get ontology-enhanced results
    enhanced_results = await ontology_enhancer.enhance_graph_retrieval(
        graph_service,
        demo_research_query,
        base_results
    )
    
    # Compare results
    print("\n--- Base Results ---")
    for i, result in enumerate(base_results[:3], 1):
        print(f"{i}. {result.get('title', 'Untitled')} (score: {result.get('score', 0):.3f})")
    
    print("\n--- Ontology-Enhanced Results ---")
    for i, result in enumerate(enhanced_results[:3], 1):
        title = result.get('title', 'Untitled')
        score = result.get('score', 0)
        
        if result.get('ontology_enhanced'):
            concept = result.get('matched_concept', {})
            concept_name = concept.get('name', '')
            concept_ont = concept.get('ontology', '')
            print(f"{i}. {title} (score: {score:.3f}) - matched {concept_ont} concept: {concept_name}")
        else:
            print(f"{i}. {title} (score: {score:.3f})")
    
    # Demo 6: Generate cross-ontology mappings
    logger.info("Demo 6: Generating cross-ontology mappings")
    
    print("\n=== Cross-Ontology Mapping ===")
    print("Generating mappings between SNOMED and MeSH...")
    
    # Generate a small sample of mappings for demonstration
    mappings = await ontology_service.map_concepts_across_ontologies(
        concepts[:2],  # Take first 2 concepts only for demo
        target_ontologies=["MESH" if concepts[0]["ontology"] == "SNOMED" else "SNOMED"]
    )
    
    # Show mappings
    for mapping in mappings:
        if "mapped_from" in mapping:
            source = mapping["mapped_from"]
            print(f"• {source['text']} ({source['ontology']}:{source['concept_id']}) → " +
                  f"{mapping['text']} ({mapping['ontology']}:{mapping['concept_id']})")
    
    logger.info("Ontology-Enriched GraphRAG demonstration completed")


if __name__ == "__main__":
    asyncio.run(run_demo())