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
from asf.medical.core.cache import cached, cache_manager

# Set up logging
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
        # We ignore the arguments here, they will be handled in __init__
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
        # Only initialize once unless new dependencies are provided
        if not self._initialized or graph_service is not None or biomedlm_service is not None:
            self.graph_service = graph_service
            self.biomedlm_service = biomedlm_service
            self._initialized = True
            # Clear cache when dependencies change
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
                logger.error(f"Failed to initialize BioMedLM service: {str(e)}")
                raise RuntimeError(f"BioMedLM service not available: {str(e)}") from e
        return self.biomedlm_service

    async def _clear_cache(self) -> None:
        """
        Clear the GraphRAG cache.

        This method is called when the GraphRAG service is initialized with new dependencies.
        """
        try:
            await cache_manager.clear(namespace="graphrag")
            logger.info("GraphRAG cache cleared")
        except Exception as e:
            logger.warning(f"Failed to clear GraphRAG cache: {str(e)}")

    # Define a type variable for the return type of the wrapped function
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
            logger.error(f"Error executing {func.__name__}: {str(e)}")
            raise

    async def _safe_execute_async(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Execute an async function safely with error handling.

        Args:
            func: Async function to execute
            *args: Positional arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function

        Returns:
            Result of the function

        Raises:
            Exception: If the function fails
        """
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error executing {func.__name__}: {str(e)}")
            raise

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

    @cached(prefix="graphrag_search", ttl=3600)
    async def search(self, query: str, max_results: int = 20, use_vector_search: bool = True, use_graph_search: bool = True) -> Dict[str, Any]:
        """
        Search for articles using graph-based retrieval-augmented generation.

        This method combines vector search and graph traversal to find relevant articles.

        Args:
            query: Search query
            max_results: Maximum number of results to return
            use_vector_search: Whether to use vector search
            use_graph_search: Whether to use graph search

        Returns:
            Search results
        """
        logger.info(f"GraphRAG search: query='{query}', max_results={max_results}, use_vector_search={use_vector_search}, use_graph_search={use_graph_search}")

        # Get graph service
        graph_service = self._get_graph_service()

        # Connect to the graph database
        graph_service.connect()

        # Get BioMedLM service
        biomedlm_service = self._get_biomedlm_service()

        # Extract concepts from query
        # This is a simplified example; in a real implementation, you would use NLP
        concepts = query.split()

        results = []

        # Vector search
        if use_vector_search:
            # Encode the query
            query_embedding = biomedlm_service.encode(query)

            # Get articles with similar embeddings
            vector_results = graph_service.vector_search(query_embedding, max_results=max_results)
            results.extend(vector_results)

        # Graph search
        if use_graph_search:
            # Get articles for each concept
            for concept in concepts:
                concept_articles = self.retrieve_articles_by_concept(concept, max_results=max_results // len(concepts) if concepts else max_results)
                results.extend(concept_articles)

            # Get related concepts and their articles
            for concept in concepts:
                related_concepts = self.retrieve_related_concepts(concept, max_results=5)
                for related_concept in related_concepts:
                    related_articles = self.retrieve_articles_by_concept(related_concept.get("cui", ""), max_results=3)
                    for article in related_articles:
                        article["connection"] = {
                            "type": "related_concept",
                            "concept": related_concept.get("name", ""),
                            "cui": related_concept.get("cui", "")
                        }
                    results.extend(related_articles)

        # Remove duplicates
        unique_results = {}
        for result in results:
            pmid = result.get("pmid", "")
            if pmid and pmid not in unique_results:
                unique_results[pmid] = result

        results = list(unique_results.values())

        # Sort by relevance (using BioMedLM to calculate similarity to query)
        result_scores = []

        for result in results:
            title = result.get("title", "")
            abstract = result.get("abstract", "")
            text = f"{title}. {abstract}"
            similarity = biomedlm_service.calculate_similarity(query, text)
            result_scores.append((result, similarity))

        # Sort by similarity
        result_scores = sorted(result_scores, key=lambda x: x[1], reverse=True)

        # Limit to max_results
        result_scores = result_scores[:max_results]

        # Extract results
        results = [result for result, _ in result_scores]

        # Create response
        response = {
            "query": query,
            "results": results,
            "result_count": len(results),
            "search_time": datetime.now().isoformat(),
            "source": "graphrag",
            "search_methods": {
                "vector_search": use_vector_search,
                "graph_search": use_graph_search
            }
        }

        logger.info(f"GraphRAG search found {len(results)} results for query: {query}")

        return response

    @cached(prefix="graphrag_search", ttl=3600, namespace="graphrag")
    async def search(self, query: str, max_results: int = 20, use_vector_search: bool = True, use_graph_search: bool = True) -> Dict[str, Any]:
        """
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
        """
        if not query or not query.strip():
            raise ValueError("Search query cannot be empty")

        if max_results < 1:
            raise ValueError("max_results must be at least 1")

        logger.info(f"GraphRAG search: query='{query}', max_results={max_results}, use_vector_search={use_vector_search}, use_graph_search={use_graph_search}")

        # Get services
        graph_service = self._safe_execute(self._get_graph_service)

        # Connect to the graph database
        connected = self._safe_execute(graph_service.connect)
        if not connected:
            raise ConnectionError("Failed to connect to graph database")

        # Get BioMedLM service
        biomedlm_service = self._safe_execute(self._get_biomedlm_service)

        # Extract concepts from query
        # This is a simplified example; in a real implementation, you would use NLP
        concepts = query.split()

        results = []

        # Vector search
        if use_vector_search:
            try:
                # Encode the query
                query_embedding = self._safe_execute(biomedlm_service.encode, query)

                # Get articles with similar embeddings
                vector_results = self._safe_execute(graph_service.vector_search, query_embedding, max_results=max_results)
                logger.info(f"Vector search found {len(vector_results)} results")
                results.extend(vector_results)
            except Exception as e:
                logger.warning(f"Vector search failed: {str(e)}")
                # Continue with graph search even if vector search fails

        # Graph search
        if use_graph_search:
            try:
                # Get articles for each concept
                for concept in concepts:
                    try:
                        concept_max_results = max_results // len(concepts) if concepts else max_results
                        concept_articles = self._safe_execute(
                            self.retrieve_articles_by_concept,
                            concept,
                            max_results=concept_max_results
                        )
                        logger.info(f"Found {len(concept_articles)} articles for concept '{concept}'")
                        results.extend(concept_articles)
                    except Exception as e:
                        logger.warning(f"Failed to retrieve articles for concept '{concept}': {str(e)}")
                        # Continue with other concepts

                # Get related concepts and their articles
                for concept in concepts:
                    try:
                        related_concepts = self._safe_execute(
                            self.retrieve_related_concepts,
                            concept,
                            max_results=5
                        )
                        logger.info(f"Found {len(related_concepts)} related concepts for '{concept}'")

                        for related_concept in related_concepts:
                            try:
                                concept_cui = related_concept.get("cui", "")
                                if not concept_cui:
                                    continue

                                related_articles = self._safe_execute(
                                    self.retrieve_articles_by_concept,
                                    concept_cui,
                                    max_results=3
                                )

                                for article in related_articles:
                                    article["connection"] = {
                                        "type": "related_concept",
                                        "concept": related_concept.get("name", ""),
                                        "cui": concept_cui
                                    }

                                results.extend(related_articles)
                            except Exception as e:
                                logger.warning(f"Failed to retrieve articles for related concept '{related_concept.get('name', '')}': {str(e)}")
                                # Continue with other related concepts
                    except Exception as e:
                        logger.warning(f"Failed to retrieve related concepts for '{concept}': {str(e)}")
                        # Continue with other concepts
            except Exception as e:
                logger.warning(f"Graph search failed: {str(e)}")
                # If both vector and graph search fail, raise an exception
                if not results and not use_vector_search:
                    raise RuntimeError(f"Both vector and graph search failed: {str(e)}") from e

        # Check if we have any results
        if not results:
            logger.warning(f"No results found for query: {query}")
            return {
                "query": query,
                "results": [],
                "result_count": 0,
                "search_time": datetime.now().isoformat(),
                "source": "graphrag",
                "search_methods": {
                    "vector_search": use_vector_search,
                    "graph_search": use_graph_search
                },
                "error": "No results found"
            }

        try:
            # Remove duplicates
            unique_results = {}
            for result in results:
                pmid = result.get("pmid", "")
                if pmid and pmid not in unique_results:
                    unique_results[pmid] = result

            results = list(unique_results.values())
            logger.info(f"Found {len(results)} unique results after deduplication")

            # Sort by relevance (using BioMedLM to calculate similarity to query)
            result_scores = []

            try:
                for result in results:
                    try:
                        title = result.get("title", "")
                        abstract = result.get("abstract", "")
                        text = f"{title}. {abstract}"
                        similarity = self._safe_execute(biomedlm_service.calculate_similarity, query, text)
                        result_scores.append((result, similarity))
                    except Exception as e:
                        # If similarity calculation fails for a result, use a default score
                        logger.warning(f"Failed to calculate similarity for result {result.get('pmid', '')}: {str(e)}")
                        result_scores.append((result, 0.0))

                # Sort by similarity
                result_scores = sorted(result_scores, key=lambda x: x[1], reverse=True)
            except Exception as e:
                logger.warning(f"Failed to sort results by similarity: {str(e)}")
                # If sorting fails, use the original order
                result_scores = [(result, 0.0) for result in results]

            # Limit to max_results
            result_scores = result_scores[:max_results]

            # Extract results
            results = [result for result, _ in result_scores]
        except Exception as e:
            logger.error(f"Failed to process search results: {str(e)}")
            # Return the original results if processing fails
            results = results[:max_results]

        # Create response
        response = {
            "query": query,
            "results": results,
            "result_count": len(results),
            "search_time": datetime.now().isoformat(),
            "source": "graphrag",
            "search_methods": {
                "vector_search": use_vector_search,
                "graph_search": use_graph_search
            }
        }

        logger.info(f"GraphRAG search found {len(results)} results for query: {query}")

        return response

    @cached(prefix="graphrag_summary", ttl=3600, namespace="graphrag")
    async def generate_summary(self, query: str, max_articles: int = 5) -> Dict[str, Any]:
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
            "article_count": len(articles),
            "generated_at": datetime.now().isoformat(),
            "source": "graphrag"
        }

        logger.info(f"Generated summary with {len(articles)} articles for query: {query}")

        return summary
