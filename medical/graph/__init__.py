"""
Graph module for the Medical Research Synthesizer.

This module provides components for working with graph databases to store and
query medical knowledge graphs. It includes clients for different graph database
backends (Neo4j, Memgraph), a unified graph service interface, and a GraphRAG
(Graph Retrieval-Augmented Generation) service for advanced knowledge retrieval.

Components:
- graph_service: Unified interface for interacting with graph databases
- neo4j_client: Client for Neo4j graph database
- memgraph_client: Client for Memgraph graph database
- graph_rag: Service for graph-based retrieval-augmented generation

The graph module enables storing medical articles, concepts, and their relationships
in a knowledge graph, which can be queried for contradiction detection, concept
relationships, and semantic search.
"""

