"""
GraphRAG service for the Medical Research Synthesizer.

This module provides a service for graph-based retrieval-augmented generation.
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any, Callable, TypeVar
from datetime import datetime

from asf.medical.graph.graph_service import GraphService
from asf.medical.ml.models.biomedlm import BioMedLMService
from asf.medical.core.enhanced_cache import enhanced_cache_manager as cache_manager, enhanced_cached as cached, cache_manager

logger = logging.getLogger(__name__)

class GraphRAG:
    """
    Service for graph-based retrieval-augmented generation.

    This service provides methods for retrieving and generating content using a graph database.
    """

    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs) -> 'GraphRAG':  # pylint: disable=unused-argument
        """
        Create a singleton instance of the GraphRAG service.

        This implementation ensures that dependencies passed to the constructor
        are properly handled even when using the singleton pattern.

        Returns:
            GraphRAG: The singleton instance
        """
        if cls._instance is None:
            cls._instance = super(GraphRAG, cls).__new__(cls)
            cls._initialized = False
        return cls._instance

    def __init__(self, graph_service: Optional[GraphService] = None, biomedlm_service: Optional[BioMedLMService] = None):
        """Initialize the GraphRAG service.

        This method is called every time an instance is created or retrieved,
        but the initialization is only performed once unless new dependencies are provided.

        Args:
            graph_service: Graph service instance (optional)
            biomedlm_service: BioMedLM service instance (optional)
        """
        if not self._initialized or graph_service is not None or biomedlm_service is not None:
            self.graph_service = graph_service
            self.biomedlm_service = biomedlm_service
            self._initialized = True
            asyncio.create_task(self._clear_cache())
            logger.info("GraphRAG service initialized with new dependencies")
        else:
            logger.debug("GraphRAG service already initialized, reusing existing instance")

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

        Raises:
            RuntimeError: If the BioMedLM service is not available
        """
        if self.biomedlm_service is None:
            try:
                logger.info("Initializing BioMedLM service")
                self.biomedlm_service = BioMedLMService()
            except Exception as e:
    logger.error(f\"Failed to initialize BioMedLM service: {str(e)}\")
    raise DatabaseError(f\"Failed to initialize BioMedLM service: {str(e)}\") RuntimeError(f"BioMedLM service not available: {str(e)}") from e
        return self.biomedlm_service

    async def _clear_cache(self) -> None:
        try:
            await enhanced_cache_manager.clear(namespace="graphrag")
            logger.info("GraphRAG cache cleared")
        except Exception as e:
            logger.warning(f"Failed to clear GraphRAG cache: {str(e)}")

    T = TypeVar('T')

    def _safe_execute(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Execute a function safely with error handling.

        Args:
            func: Function to execute
            *args: Positional arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function

        Returns:
            Result of the function

        Raises:
            Exception: If the function fails
        """
        try:
            return func(*args, **kwargs)
        except Exception as e:
    logger.error(f\"Error executing {func.__name__}: {str(e)}\")
    raise DatabaseError(f\"Error executing {func.__name__}: {str(e)}\")

    async def _safe_execute_async(self, func: Callable[..., T], *args, **kwargs) -> T:
        try:
            return await func(*args, **kwargs)
        except Exception as e:
    logger.error(f\"Error executing {func.__name__}: {str(e)}\")
    raise DatabaseError(f\"Error executing {func.__name__}: {str(e)}\")

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

        graph_service = self._get_graph_service()

        graph_service.connect()

        if concept.startswith("C") and concept[1:].isdigit():
            cui = concept
        else:
            query = """
            MATCH (c:Concept)
            WHERE c.name CONTAINS $concept
            RETURN c.cui AS cui, c.name AS name
            LIMIT 1
        Retrieve concepts mentioned in an article.

        Args:
            pmid: Article PMID
            max_results: Maximum number of results to return

        Returns:
            List of concepts
        Retrieve concepts related to a concept.

        Args:
            concept: Concept name or CUI
            max_results: Maximum number of results to return

        Returns:
            List of related concepts
            MATCH (c:Concept)
            WHERE c.name CONTAINS $concept
            RETURN c.cui AS cui, c.name AS name
            LIMIT 1
        Retrieve contradictions in the graph.

        Args:
            pmid: Article PMID (optional)
            max_results: Maximum number of results to return

        Returns:
            List of contradictions
        Retrieve the shortest path between two concepts.

        Args:
            concept1: First concept name or CUI
            concept2: Second concept name or CUI
            max_path_length: Maximum path length

        Returns:
            List of path elements
            MATCH (c:Concept)
            WHERE c.name CONTAINS $concept
            RETURN c.cui AS cui, c.name AS name
            LIMIT 1
            MATCH (c:Concept)
            WHERE c.name CONTAINS $concept
            RETURN c.cui AS cui, c.name AS name
            LIMIT 1
        MATCH path = shortestPath((c1:Concept {{cui: $cui1}})-[*1..{max_path_length}]-(c2:Concept {{cui: $cui2}}))
        RETURN path
        Retrieve a subgraph around a concept.

        Args:
            concept: Concept name or CUI
            depth: Subgraph depth
            max_nodes: Maximum number of nodes

        Returns:
            Subgraph data
            MATCH (c:Concept)
            WHERE c.name CONTAINS $concept
            RETURN c.cui AS cui, c.name AS name
            LIMIT 1
        MATCH (c:Concept {{cui: $cui}})
        CALL apoc.path.subgraphNodes(c, {{maxLevel: {depth}, limit: {max_nodes}}})
        YIELD node
        RETURN collect(node) AS nodes
        MATCH (n)-[r]->(m)
        WHERE id(n) IN $node_ids AND id(m) IN $node_ids
        RETURN collect(r) AS edges
        Search for articles using graph-based retrieval-augmented generation.

        This method combines vector search and graph traversal to find relevant articles.

        Args:
            query: Search query
            max_results: Maximum number of results to return
            use_vector_search: Whether to use vector search
            use_graph_search: Whether to use graph search

        Returns:
            Search results
        Search for articles using graph-based retrieval-augmented generation.

        This method combines vector search and graph traversal to find relevant articles.

        Args:
            query: Search query
            max_results: Maximum number of results to return
            use_vector_search: Whether to use vector search
            use_graph_search: Whether to use graph search

        Returns:
            Search results

        Raises:
            ValueError: If the query is empty or invalid
            ConnectionError: If the graph database connection fails
            RuntimeError: If the BioMedLM service is not available
        Generate a summary of articles related to a query.

        Args:
            query: Query string
            max_articles: Maximum number of articles to include

        Returns:
            Summary data