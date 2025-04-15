"""
GraphRAG service for the Medical Research Synthesizer.

This module provides a service for graph-based retrieval-augmented generation,
combining the power of knowledge graphs with large language models. It enhances
the quality and factual accuracy of generated text by retrieving relevant
information from the knowledge graph and using it to augment the generation process.

The GraphRAG service uses the graph database to retrieve relevant articles,
concepts, and their relationships based on a query, and then uses this information
to generate more accurate and contextually relevant responses. It supports various
retrieval strategies including semantic similarity, graph traversal, and hybrid
approaches.

Features:
- Knowledge graph-based retrieval for medical information
- Integration with large language models for text generation
- Support for various retrieval strategies
- Context augmentation with graph relationships
- Explanation generation for retrieved information
- Ontology-enhanced retrieval with SNOMED CT and MeSH integration
"""
import logging
import asyncio
from typing import Dict, List, Optional, Any, Callable, TypeVar, Union
from medical.graph.graph_service import GraphService
from medical.ml.models.biomedlm import BioMedLMService
from medical.graph.ontology_integration import OntologyIntegrationService, GraphRAGOntologyEnhancer

# Set up logger
logger = logging.getLogger(__name__)

# Import cache manager
try:
    from medical.core.enhanced_cache import enhanced_cache_manager
except ImportError:
    # Fallback for when enhanced_cache is not available
    logger.warning("Enhanced cache module not available, using dummy implementation")

    class DummyCacheManager:
        """Dummy cache manager for when the real one is not available."""
        async def clear(self, _=None):
            """Dummy clear method."""
            pass

    enhanced_cache_manager = DummyCacheManager()

class GraphRAG:
    """
    Service for graph-based retrieval-augmented generation.

    This service combines knowledge graphs with large language models for enhanced
    text generation. It retrieves relevant information from the knowledge graph
    based on a query and uses this information to augment the generation process,
    resulting in more accurate and contextually relevant responses.

    The service supports various retrieval strategies:
    - Semantic similarity: Retrieving articles with similar embeddings
    - Graph traversal: Following relationships in the knowledge graph
    - Hybrid approaches: Combining multiple retrieval strategies

    It also provides methods for explaining the retrieved information and its
    relevance to the query, enhancing the transparency of the generation process.
    """
    _instance = None
    _initialized = False
    def __new__(cls, *_, **__) -> 'GraphRAG':  # pylint: disable=unused-argument
        """
        Create a singleton instance of the GraphRAG service.

        This implementation ensures that dependencies passed to the constructor
        are properly handled even when using the singleton pattern. It follows
        the singleton pattern to ensure a single instance is used throughout
        the application.

        Args:
            *args: Positional arguments (ignored in this implementation)
            **kwargs: Keyword arguments (ignored in this implementation)

        Returns:
            GraphRAG: The singleton instance
        """
        if cls._instance is None:
            cls._instance = super(GraphRAG, cls).__new__(cls)
            cls._initialized = False
        return cls._instance

    def __init__(
        self,
        graph_service: Optional[GraphService] = None,
        biomedlm_service: Optional[BioMedLMService] = None,
        ontology_service: Optional[OntologyIntegrationService] = None
    ):
        """
        Initialize the GraphRAG service.

        This method is called every time an instance is created or retrieved,
        but the initialization is only performed once unless new dependencies are provided.
        This ensures that the singleton pattern works correctly while still allowing
        dependency injection for testing and flexibility.

        The service can be initialized with optional dependencies:
        - graph_service: For interacting with the knowledge graph
        - biomedlm_service: For text generation and analysis
        - ontology_service: For integrating external medical ontologies

        If dependencies are not provided, they will be lazily initialized when needed.

        Args:
            graph_service: Graph service instance (optional)
            biomedlm_service: BioMedLM service instance (optional)
            ontology_service: Ontology integration service instance (optional)
        """
        if (not self._initialized or 
            graph_service is not None or 
            biomedlm_service is not None or
            ontology_service is not None):
            
            self.graph_service = graph_service
            self.biomedlm_service = biomedlm_service
            self.ontology_service = ontology_service
            self.ontology_enhancer = None
            
            self._initialized = True
            asyncio.create_task(self._clear_cache())
            asyncio.create_task(self._initialize_ontology_services())
            logger.info("GraphRAG service initialized with new dependencies")
        else:
            logger.debug("GraphRAG service already initialized, reusing existing instance")
    
    async def _initialize_ontology_services(self) -> None:
        """Initialize ontology services if not already initialized."""
        if self.ontology_service is None:
            try:
                logger.info("Initializing ontology integration service")
                graph_service = self._get_graph_service()
                self.ontology_service = OntologyIntegrationService(graph_client=graph_service)
                await self.ontology_service.initialize()
            except Exception as e:
                logger.error(f"Failed to initialize ontology service: {str(e)}")
                self.ontology_service = None
        
        if self.ontology_service and self.ontology_enhancer is None:
            logger.info("Initializing GraphRAG ontology enhancer")
            self.ontology_enhancer = GraphRAGOntologyEnhancer(
                ontology_service=self.ontology_service,
                expansion_depth=1,
                include_narrower_concepts=True,
                include_broader_concepts=False,
                ontology_weight=0.3
            )

    def _get_graph_service(self) -> GraphService:
        """
        Get the graph service.

        This method returns the graph service instance. If the service has not
        been initialized yet, it creates a new instance. This lazy initialization
        pattern allows the GraphRAG service to be created without immediately
        requiring a graph service instance.

        Returns:
            GraphService: The graph service instance
        """
        if self.graph_service is None:
            logger.info("Initializing graph service")
            self.graph_service = GraphService()
        return self.graph_service

    def _get_biomedlm_service(self) -> BioMedLMService:
        """
        Get the BioMedLM service.

        This method returns the BioMedLM service instance. If the service has not
        been initialized yet, it attempts to create a new instance. If initialization
        fails, a RuntimeError is raised with details about the failure.

        Returns:
            BioMedLMService: The BioMedLM service instance

        Raises:
            RuntimeError: If the BioMedLM service is not available or initialization fails
        """
        if self.biomedlm_service is None:
            try:
                logger.info("Initializing BioMedLM service")
                self.biomedlm_service = BioMedLMService()
            except Exception as e:
                logger.error(f"Failed to initialize BioMedLM service: {str(e)}")
                raise RuntimeError(f"BioMedLM service not available: {str(e)}") from e
        return self.biomedlm_service

    def _get_ontology_service(self) -> Optional[OntologyIntegrationService]:
        """
        Get the ontology integration service.

        This method returns the ontology integration service instance if available.
        Unlike other service getters, this does not attempt to create the service
        if it doesn't exist, since initialization is asynchronous and handled
        separately.

        Returns:
            OntologyIntegrationService: The ontology service instance, or None if not available
        """
        return self.ontology_service

    async def _clear_cache(self) -> None:
        """
        Clear the GraphRAG cache.

        This method clears the cache used by the GraphRAG service. It is called
        automatically when the service is initialized with new dependencies to
        ensure that cached results are invalidated when the underlying services
        change.

        The method is asynchronous and handles exceptions internally to prevent
        cache-clearing failures from affecting the service initialization.
        """
        try:
            await enhanced_cache_manager.clear(namespace="graphrag")
            logger.info("GraphRAG cache cleared")
        except Exception as e:
            logger.warning(f"Failed to clear GraphRAG cache: {str(e)}")

    T = TypeVar('T')
    def _safe_execute(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Execute a function safely with error handling.

        This method provides a wrapper for executing functions with consistent
        error handling. It catches any exceptions raised by the function, logs them,
        and re-raises them as RuntimeError with additional context.

        Args:
            func: Function to execute
            *args: Positional arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function

        Returns:
            Result of the function

        Raises:
            RuntimeError: If the function execution fails
        """
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error executing {func.__name__}: {str(e)}")
            raise RuntimeError(f"Error executing {func.__name__}: {str(e)}") from e

    async def _safe_execute_async(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Execute an async function safely with error handling.

        This method provides a wrapper for executing asynchronous functions with
        consistent error handling. It catches any exceptions raised by the function,
        logs them, and re-raises them as RuntimeError with additional context.

        Args:
            func: Async function to execute
            *args: Positional arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function

        Returns:
            Result of the async function

        Raises:
            RuntimeError: If the function execution fails
        """
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error executing {func.__name__}: {str(e)}")
            raise RuntimeError(f"Error executing {func.__name__}: {str(e)}") from e

    def retrieve_articles_by_concept(self, concept: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve articles that mention a concept.

        This method retrieves articles from the knowledge graph that mention a
        specific concept. The concept can be specified by its Concept Unique
        Identifier (CUI) or by its name. If a name is provided, the method first
        looks up the corresponding CUI.

        Args:
            concept: Concept name or CUI
            max_results: Maximum number of results to return

        Returns:
            List of articles that mention the concept

        Raises:
            RuntimeError: If the query fails
        """
        logger.info(f"Retrieving articles for concept: {concept}")

        graph_service = self._get_graph_service()
        return self._safe_execute(
            graph_service.get_articles_by_concept,
            concept,
            max_results
        )

    def retrieve_concepts_by_article(self, pmid: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve concepts mentioned in an article.

        This method retrieves concepts from the knowledge graph that are mentioned
        in a specific article. The article is identified by its PubMed ID (PMID).

        Args:
            pmid: Article PMID
            max_results: Maximum number of results to return

        Returns:
            List of concepts mentioned in the article

        Raises:
            RuntimeError: If the query fails
        """
        logger.info(f"Retrieving concepts for article: {pmid}")

        graph_service = self._get_graph_service()
        return self._safe_execute(
            graph_service.get_concepts_by_article,
            pmid,
            max_results
        )

    def retrieve_related_concepts(self, concept: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve concepts related to a concept.

        This method retrieves concepts from the knowledge graph that are related
        to a specific concept. The concept can be specified by its Concept Unique
        Identifier (CUI) or by its name. If a name is provided, the method first
        looks up the corresponding CUI.

        Args:
            concept: Concept name or CUI
            max_results: Maximum number of results to return

        Returns:
            List of concepts related to the specified concept

        Raises:
            RuntimeError: If the query fails
        """
        logger.info(f"Retrieving concepts related to: {concept}")

        graph_service = self._get_graph_service()
        return self._safe_execute(
            graph_service.get_related_concepts,
            concept,
            max_results
        )

    async def search_articles(
        self, 
        query: str, 
        max_results: int = 10, 
        use_vector_search: bool = True, 
        use_graph_search: bool = True,
        use_ontology_enhancement: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Search for articles using graph-based retrieval-augmented generation.

        This method combines vector search and graph traversal to find relevant articles.
        It can use vector similarity search, graph-based search, ontology enhancement,
        or a combination of these methods depending on the parameters provided.

        Args:
            query: Search query
            max_results: Maximum number of results to return
            use_vector_search: Whether to use vector search
            use_graph_search: Whether to use graph search
            use_ontology_enhancement: Whether to use ontology enhancement

        Returns:
            List of articles matching the search criteria

        Raises:
            ValueError: If the query is empty or invalid
            RuntimeError: If the search fails
        """
        logger.info(f"Searching articles with query: {query}")

        if not query.strip():
            raise ValueError("Search query cannot be empty")

        graph_service = self._get_graph_service()
        
        # First, perform the base search
        results = await self._safe_execute_async(
            graph_service.search_articles,
            query,
            max_results,
            use_vector_search,
            use_graph_search
        )
        
        # Apply ontology enhancement if requested and available
        if use_ontology_enhancement and self.ontology_enhancer:
            try:
                # Ensure ontology services are initialized
                if self.ontology_service is None:
                    await self._initialize_ontology_services()
                
                if self.ontology_enhancer:
                    # Enhance query with ontology concepts
                    enhanced_query = await self.ontology_enhancer.enhance_search_query(query)
                    
                    # If the query was enhanced, perform a search with the enhanced query
                    if enhanced_query != query:
                        logger.info(f"Searching with ontology-enhanced query: {enhanced_query}")
                        enhanced_results = await self._safe_execute_async(
                            graph_service.search_articles,
                            enhanced_query,
                            max_results,
                            use_vector_search,
                            use_graph_search
                        )
                        
                        # Enhance retrieval with ontology graph
                        final_results = await self.ontology_enhancer.enhance_graph_retrieval(
                            graph_service,
                            query,
                            enhanced_results
                        )
                        
                        return final_results
            except Exception as e:
                logger.warning(f"Ontology enhancement failed, falling back to base search: {str(e)}")
        
        return results

    async def generate_summary(
        self, 
        query: str,
        max_articles: int = 5,
        use_ontology_enhancement: bool = True
    ) -> Dict[str, Any]:
        """
        Generate a summary of articles related to a query.

        This method searches for articles related to a query and generates a summary
        of the findings. It uses the BioMedLM service to generate the summary based
        on the retrieved articles. Ontology enhancement can be applied to improve
        retrieval quality.

        Args:
            query: Query string
            max_articles: Maximum number of articles to include in the summary
            use_ontology_enhancement: Whether to use ontology enhancement for retrieval

        Returns:
            Dictionary containing the summary data, including:
            - summary: The generated summary text
            - articles: List of articles used in the summary
            - query: The original query
            - ontology_enhanced: Whether ontology enhancement was applied

        Raises:
            ValueError: If the query is empty or invalid
            RuntimeError: If the summary generation fails
        """
        logger.info(f"Generating summary for query: {query}")

        if not query.strip():
            raise ValueError("Query cannot be empty")

        # Search for relevant articles with ontology enhancement if requested
        articles = await self._safe_execute_async(
            self.search_articles,
            query,
            max_articles,
            use_vector_search=True,
            use_graph_search=True,
            use_ontology_enhancement=use_ontology_enhancement
        )

        if not articles:
            return {
                "summary": "No relevant articles found.",
                "articles": [],
                "query": query,
                "ontology_enhanced": use_ontology_enhancement and self.ontology_enhancer is not None
            }

        # Generate summary using BioMedLM
        biomedlm_service = self._get_biomedlm_service()
        summary = await self._safe_execute_async(
            biomedlm_service.generate_summary,
            query,
            articles
        )

        return {
            "summary": summary,
            "articles": articles,
            "query": query,
            "ontology_enhanced": use_ontology_enhancement and self.ontology_enhancer is not None
        }

    async def enrich_knowledge_graph_with_ontologies(
        self, 
        node_type: str = "Article", 
        content_field: str = "abstract",
        limit: int = 1000
    ) -> Dict[str, Any]:
        """
        Enrich the knowledge graph with ontology references.
        
        This method processes nodes in the knowledge graph and adds references
        to ontology concepts from SNOMED CT and MeSH.
        
        Args:
            node_type: Type of node to enrich
            content_field: Field containing text to map to ontologies
            limit: Maximum number of nodes to process
            
        Returns:
            Dictionary with status information
            
        Raises:
            RuntimeError: If ontology service is not available or enrichment fails
        """
        logger.info(f"Enriching knowledge graph with ontologies: {node_type}.{content_field}")
        
        if not self.ontology_service:
            await self._initialize_ontology_services()
        
        if not self.ontology_service:
            raise RuntimeError("Ontology service not available")
        
        try:
            await self.ontology_service.enrich_graph_with_ontologies(
                node_type, content_field, limit
            )
            
            return {
                "status": "success",
                "message": f"Enriched {node_type} nodes with ontology references",
                "node_type": node_type,
                "content_field": content_field,
                "limit": limit
            }
        except Exception as e:
            logger.error(f"Error enriching knowledge graph with ontologies: {str(e)}")
            raise RuntimeError(f"Error enriching knowledge graph with ontologies: {str(e)}")
    
    async def create_ontology_mappings(self) -> Dict[str, Any]:
        """
        Create mappings between different ontologies (e.g., SNOMED CT to MeSH).
        
        This method generates and persists mappings between concepts in different
        ontologies, enabling cross-ontology searches and knowledge integration.
        
        Returns:
            Dictionary with mapping statistics
            
        Raises:
            RuntimeError: If ontology service is not available or mapping fails
        """
        logger.info("Creating ontology mappings")
        
        if not self.ontology_service:
            await self._initialize_ontology_services()
        
        if not self.ontology_service:
            raise RuntimeError("Ontology service not available")
        
        try:
            await self.ontology_service.generate_mappings()
            
            return {
                "status": "success",
                "message": "Generated ontology mappings",
                "mapping_count": len(self.ontology_service.ontology_mappings)
            }
        except Exception as e:
            logger.error(f"Error creating ontology mappings: {str(e)}")
            raise RuntimeError(f"Error creating ontology mappings: {str(e)}")
    
    async def search_ontology(
        self, 
        query: str,
        source_ontologies: Optional[List[str]] = None,
        target_ontologies: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for medical concepts in ontologies based on text.
        
        Args:
            query: Text to search for
            source_ontologies: List of ontologies to search in (e.g., ["MESH", "SNOMED"])
            target_ontologies: List of ontologies to map concepts to
            
        Returns:
            List of matched ontology concepts with mappings
            
        Raises:
            RuntimeError: If ontology service is not available or search fails
        """
        logger.info(f"Searching ontologies for: {query}")
        
        if not self.ontology_service:
            await self._initialize_ontology_services()
        
        if not self.ontology_service:
            raise RuntimeError("Ontology service not available")
        
        try:
            results = await self.ontology_service.map_text_to_ontologies(
                query, source_ontologies, target_ontologies
            )
            
            return results
        except Exception as e:
            logger.error(f"Error searching ontologies: {str(e)}")
            raise RuntimeError(f"Error searching ontologies: {str(e)}")