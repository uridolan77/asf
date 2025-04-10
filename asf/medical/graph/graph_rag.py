"""
GraphRAG service for the Medical Research Synthesizer.

This module provides a service for graph-based retrieval-augmented generation.
"""

import logging
from typing import Dict, List, Optional, Any, Union, Tuple
import numpy as np

from asf.medical.graph.graph_service import GraphService
from asf.medical.ml.models import BioMedLMService
from asf.medical.core.config import settings

# Set up logging
logger = logging.getLogger(__name__)

class GraphRAG:
    """
    Service for graph-based retrieval-augmented generation.
    
    This service provides methods for retrieving and generating content using a graph database.
    """
    
    _instance = None
    
    def __new__(cls):
        """
        Create a singleton instance of the GraphRAG service.
        
        Returns:
            GraphRAG: The singleton instance
        """
        if cls._instance is None:
            cls._instance = super(GraphRAG, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the GraphRAG service."""
        self.graph_service = None
        self.biomedlm_service = None
        
        logger.info("GraphRAG service initialized")
    
    def _get_graph_service(self) -> GraphService:
        """
        Get the graph service.
        
        Returns:
            GraphService: The graph service
        """
        if self.graph_service is None:
            logger.info("Initializing graph service")
            self.graph_service = GraphService()
        return self.graph_service
    
    def _get_biomedlm_service(self) -> BioMedLMService:
        """
        Get the BioMedLM service.
        
        Returns:
            BioMedLMService: The BioMedLM service
        """
        if self.biomedlm_service is None:
            logger.info("Initializing BioMedLM service")
            self.biomedlm_service = BioMedLMService()
        return self.biomedlm_service
    
    def retrieve_articles_by_concept(self, concept: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve articles that mention a concept.
        
        Args:
            concept: Concept name or CUI
            max_results: Maximum number of results to return
            
        Returns:
            List of articles
        """
        logger.info(f"Retrieving articles for concept: {concept}")
        
        # Get graph service
        graph_service = self._get_graph_service()
        
        # Connect to the graph database
        graph_service.connect()
        
        # Check if concept is a CUI
        if concept.startswith("C") and concept[1:].isdigit():
            # Concept is a CUI
            cui = concept
        else:
            # Concept is a name, need to find the CUI
            query = """
            MATCH (c:Concept)
            WHERE c.name CONTAINS $concept
            RETURN c.cui AS cui, c.name AS name
            LIMIT 1
            """
            
            params = {
                "concept": concept
            }
            
            result = graph_service.execute_query(query, params)
            
            if not result:
                logger.warning(f"Concept not found: {concept}")
                return []
            
            cui = result[0]["cui"]
        
        # Get articles that mention the concept
        articles = graph_service.get_concept_articles(cui)
        
        # Sort by frequency
        articles = sorted(articles, key=lambda x: x.get("frequency", 0), reverse=True)
        
        # Limit results
        articles = articles[:max_results]
        
        logger.info(f"Retrieved {len(articles)} articles for concept: {concept}")
        
        return articles
    
    def retrieve_concepts_by_article(self, pmid: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve concepts mentioned in an article.
        
        Args:
            pmid: Article PMID
            max_results: Maximum number of results to return
            
        Returns:
            List of concepts
        """
        logger.info(f"Retrieving concepts for article: {pmid}")
        
        # Get graph service
        graph_service = self._get_graph_service()
        
        # Connect to the graph database
        graph_service.connect()
        
        # Get concepts mentioned in the article
        concepts = graph_service.get_article_concepts(pmid)
        
        # Sort by frequency
        concepts = sorted(concepts, key=lambda x: x.get("frequency", 0), reverse=True)
        
        # Limit results
        concepts = concepts[:max_results]
        
        logger.info(f"Retrieved {len(concepts)} concepts for article: {pmid}")
        
        return concepts
    
    def retrieve_related_concepts(self, concept: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve concepts related to a concept.
        
        Args:
            concept: Concept name or CUI
            max_results: Maximum number of results to return
            
        Returns:
            List of related concepts
        """
        logger.info(f"Retrieving related concepts for concept: {concept}")
        
        # Get graph service
        graph_service = self._get_graph_service()
        
        # Connect to the graph database
        graph_service.connect()
        
        # Check if concept is a CUI
        if concept.startswith("C") and concept[1:].isdigit():
            # Concept is a CUI
            cui = concept
        else:
            # Concept is a name, need to find the CUI
            query = """
            MATCH (c:Concept)
            WHERE c.name CONTAINS $concept
            RETURN c.cui AS cui, c.name AS name
            LIMIT 1
            """
            
            params = {
                "concept": concept
            }
            
            result = graph_service.execute_query(query, params)
            
            if not result:
                logger.warning(f"Concept not found: {concept}")
                return []
            
            cui = result[0]["cui"]
        
        # Get related concepts
        related_concepts = graph_service.get_related_concepts(cui)
        
        # Limit results
        related_concepts = related_concepts[:max_results]
        
        logger.info(f"Retrieved {len(related_concepts)} related concepts for concept: {concept}")
        
        return related_concepts
    
    def retrieve_contradictions(self, pmid: str = None, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve contradictions in the graph.
        
        Args:
            pmid: Article PMID (optional)
            max_results: Maximum number of results to return
            
        Returns:
            List of contradictions
        """
        logger.info(f"Retrieving contradictions for article: {pmid}" if pmid else "Retrieving all contradictions")
        
        # Get graph service
        graph_service = self._get_graph_service()
        
        # Connect to the graph database
        graph_service.connect()
        
        # Get contradictions
        contradictions = graph_service.get_contradictions(pmid)
        
        # Sort by contradiction score
        contradictions = sorted(contradictions, key=lambda x: x.get("contradiction_score", 0), reverse=True)
        
        # Limit results
        contradictions = contradictions[:max_results]
        
        logger.info(f"Retrieved {len(contradictions)} contradictions")
        
        return contradictions
    
    def retrieve_path_between_concepts(
        self,
        concept1: str,
        concept2: str,
        max_path_length: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the shortest path between two concepts.
        
        Args:
            concept1: First concept name or CUI
            concept2: Second concept name or CUI
            max_path_length: Maximum path length
            
        Returns:
            List of path elements
        """
        logger.info(f"Retrieving path between concepts: {concept1} and {concept2}")
        
        # Get graph service
        graph_service = self._get_graph_service()
        
        # Connect to the graph database
        graph_service.connect()
        
        # Check if concepts are CUIs
        if concept1.startswith("C") and concept1[1:].isdigit():
            # Concept is a CUI
            cui1 = concept1
        else:
            # Concept is a name, need to find the CUI
            query = """
            MATCH (c:Concept)
            WHERE c.name CONTAINS $concept
            RETURN c.cui AS cui, c.name AS name
            LIMIT 1
            """
            
            params = {
                "concept": concept1
            }
            
            result = graph_service.execute_query(query, params)
            
            if not result:
                logger.warning(f"Concept not found: {concept1}")
                return []
            
            cui1 = result[0]["cui"]
        
        if concept2.startswith("C") and concept2[1:].isdigit():
            # Concept is a CUI
            cui2 = concept2
        else:
            # Concept is a name, need to find the CUI
            query = """
            MATCH (c:Concept)
            WHERE c.name CONTAINS $concept
            RETURN c.cui AS cui, c.name AS name
            LIMIT 1
            """
            
            params = {
                "concept": concept2
            }
            
            result = graph_service.execute_query(query, params)
            
            if not result:
                logger.warning(f"Concept not found: {concept2}")
                return []
            
            cui2 = result[0]["cui"]
        
        # Get path between concepts
        query = f"""
        MATCH path = shortestPath((c1:Concept {{cui: $cui1}})-[*1..{max_path_length}]-(c2:Concept {{cui: $cui2}}))
        RETURN path
        """
        
        params = {
            "cui1": cui1,
            "cui2": cui2
        }
        
        result = graph_service.execute_query(query, params)
        
        if not result:
            logger.warning(f"No path found between concepts: {concept1} and {concept2}")
            return []
        
        # Extract path elements
        path = result[0]["path"]
        
        # Convert path to a list of elements
        path_elements = []
        
        # Note: The exact format of the path depends on the graph database client
        # This is a simplified example
        for i in range(0, len(path), 2):
            if i < len(path) - 1:
                path_elements.append({
                    "source": path[i],
                    "relationship": path[i+1],
                    "target": path[i+2] if i+2 < len(path) else None
                })
        
        logger.info(f"Retrieved path with {len(path_elements)} elements between concepts: {concept1} and {concept2}")
        
        return path_elements
    
    def retrieve_subgraph(self, concept: str, depth: int = 2, max_nodes: int = 20) -> Dict[str, Any]:
        """
        Retrieve a subgraph around a concept.
        
        Args:
            concept: Concept name or CUI
            depth: Subgraph depth
            max_nodes: Maximum number of nodes
            
        Returns:
            Subgraph data
        """
        logger.info(f"Retrieving subgraph for concept: {concept}")
        
        # Get graph service
        graph_service = self._get_graph_service()
        
        # Connect to the graph database
        graph_service.connect()
        
        # Check if concept is a CUI
        if concept.startswith("C") and concept[1:].isdigit():
            # Concept is a CUI
            cui = concept
        else:
            # Concept is a name, need to find the CUI
            query = """
            MATCH (c:Concept)
            WHERE c.name CONTAINS $concept
            RETURN c.cui AS cui, c.name AS name
            LIMIT 1
            """
            
            params = {
                "concept": concept
            }
            
            result = graph_service.execute_query(query, params)
            
            if not result:
                logger.warning(f"Concept not found: {concept}")
                return {"nodes": [], "edges": []}
            
            cui = result[0]["cui"]
        
        # Get subgraph
        query = f"""
        MATCH (c:Concept {{cui: $cui}})
        CALL apoc.path.subgraphNodes(c, {{maxLevel: {depth}, limit: {max_nodes}}})
        YIELD node
        RETURN collect(node) AS nodes
        """
        
        params = {
            "cui": cui
        }
        
        result = graph_service.execute_query(query, params)
        
        if not result or not result[0]["nodes"]:
            logger.warning(f"No subgraph found for concept: {concept}")
            return {"nodes": [], "edges": []}
        
        # Extract nodes
        nodes = result[0]["nodes"]
        
        # Get edges between nodes
        node_ids = [node["id"] for node in nodes]
        
        query = """
        MATCH (n)-[r]->(m)
        WHERE id(n) IN $node_ids AND id(m) IN $node_ids
        RETURN collect(r) AS edges
        """
        
        params = {
            "node_ids": node_ids
        }
        
        result = graph_service.execute_query(query, params)
        
        # Extract edges
        edges = result[0]["edges"] if result and result[0]["edges"] else []
        
        logger.info(f"Retrieved subgraph with {len(nodes)} nodes and {len(edges)} edges for concept: {concept}")
        
        # Convert to a format suitable for visualization
        subgraph = {
            "nodes": nodes,
            "edges": edges
        }
        
        return subgraph
    
    def generate_summary(self, query: str, max_articles: int = 5) -> Dict[str, Any]:
        """
        Generate a summary of articles related to a query.
        
        Args:
            query: Query string
            max_articles: Maximum number of articles to include
            
        Returns:
            Summary data
        """
        logger.info(f"Generating summary for query: {query}")
        
        # Get graph service
        graph_service = self._get_graph_service()
        
        # Connect to the graph database
        graph_service.connect()
        
        # Get BioMedLM service
        biomedlm_service = self._get_biomedlm_service()
        
        # Extract concepts from query
        # This is a simplified example; in a real implementation, you would use NLP
        concepts = query.split()
        
        # Get articles for each concept
        all_articles = []
        
        for concept in concepts:
            articles = self.retrieve_articles_by_concept(concept, max_results=max_articles)
            all_articles.extend(articles)
        
        # Remove duplicates
        unique_articles = {}
        for article in all_articles:
            pmid = article.get("pmid", "")
            if pmid and pmid not in unique_articles:
                unique_articles[pmid] = article
        
        articles = list(unique_articles.values())
        
        # Sort by relevance (using BioMedLM to calculate similarity to query)
        article_scores = []
        
        for article in articles:
            title = article.get("title", "")
            similarity = biomedlm_service.calculate_similarity(query, title)
            article_scores.append((article, similarity))
        
        # Sort by similarity
        article_scores = sorted(article_scores, key=lambda x: x[1], reverse=True)
        
        # Limit to max_articles
        article_scores = article_scores[:max_articles]
        
        # Extract articles
        articles = [article for article, _ in article_scores]
        
        # Generate summary
        summary = {
            "query": query,
            "articles": articles,
            "article_count": len(articles)
        }
        
        logger.info(f"Generated summary with {len(articles)} articles for query: {query}")
        
        return summary
