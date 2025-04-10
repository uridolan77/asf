"""
Search service for the Medical Research Synthesizer.

This module provides a service for searching medical literature.
"""

import logging
import uuid
import hashlib
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from enum import Enum

from asf.medical.core.cache import cached, cache_manager
from asf.medical.core.exceptions import (
    SearchError, ValidationError,
    ExternalServiceError, DatabaseError
)

from asf.medical.clients.ncbi_client import NCBIClient
from asf.medical.clients.clinical_trials_client import ClinicalTrialsClient
from asf.medical.storage.repositories.result_repository import ResultRepository
from asf.medical.storage.repositories.query_repository import QueryRepository
from asf.medical.graph.graph_rag import GraphRAG
from asf.medical.data_ingestion_layer.query_builder import (
    MedicalQueryBuilder, MedicalCondition, MedicalIntervention, OutcomeMetric, StudyDesign
)

class SearchMethod(str, Enum):
    """Search method enum."""
    PUBMED = "pubmed"
    CLINICAL_TRIALS = "clinical_trials"
    GRAPH_RAG = "graph_rag"

# Set up logging
logger = logging.getLogger(__name__)

class SearchService:
    """
    Service for searching medical literature.
    """

    def __init__(
        self,
        ncbi_client: NCBIClient,
        clinical_trials_client: ClinicalTrialsClient,
        query_repository: QueryRepository,
        result_repository: ResultRepository,
        graph_rag: Optional[GraphRAG] = None
    ):
        """
        Initialize the search service.

        Args:
            ncbi_client: NCBI client
            clinical_trials_client: ClinicalTrials.gov client
            query_repository: Query repository
            result_repository: Result repository
            graph_rag: GraphRAG service (optional)
        """
        self.ncbi_client = ncbi_client
        self.clinical_trials_client = clinical_trials_client
        self.query_repository = query_repository
        self.result_repository = result_repository
        self.graph_rag = graph_rag

    @cached(prefix="search", data_type="search")
    async def search(
        self, query: str, max_results: int = 100, page: int = 1, page_size: int = 20,
        user_id: Optional[int] = None, search_method: Union[str, SearchMethod] = SearchMethod.PUBMED,
        use_graph_rag: bool = False, use_vector_search: bool = True, use_graph_search: bool = True
    ) -> Dict[str, Any]:
        """
        Search for medical literature with the given query and return enriched results with pagination.

        Args:
            query: Search query
            max_results: Maximum number of results to return from the API
            page: Page number (1-based)
            page_size: Number of results per page
            user_id: User ID for storing the query and results
            search_method: Search method to use (pubmed, clinical_trials, or graph_rag)
            use_graph_rag: Whether to use GraphRAG for search (overrides search_method if True)
            use_vector_search: Whether to use vector search with GraphRAG
            use_graph_search: Whether to use graph search with GraphRAG

        Returns:
            Search results with pagination metadata

        Raises:
            ValidationError: If the query is invalid
            ExternalServiceError: If the external API fails
            DatabaseError: If there's an error storing the results
        """
        if not query or not query.strip():
            raise ValidationError("Search query cannot be empty")

        if max_results < 1 or max_results > 500:
            raise ValidationError("max_results must be between 1 and 500")

        if page < 1:
            raise ValidationError("page must be at least 1")

        if page_size < 1 or page_size > 100:
            raise ValidationError("page_size must be between 1 and 100")

        # Convert search_method to enum if it's a string
        if isinstance(search_method, str):
            try:
                search_method = SearchMethod(search_method.lower())
            except ValueError:
                logger.warning(f"Invalid search method: {search_method}, using default")
                search_method = SearchMethod.PUBMED

        # Override search_method if use_graph_rag is True
        if use_graph_rag:
            search_method = SearchMethod.GRAPH_RAG

        logger.info(f"Executing search: {query} (max_results={max_results}, method={search_method})")

        try:
            # Use the appropriate search method
            if search_method == SearchMethod.GRAPH_RAG:
                if self.graph_rag is None:
                    logger.warning("GraphRAG not available, falling back to PubMed search")
                    search_results = await self.ncbi_client.search_pubmed(query, max_results=max_results)
                else:
                    # Search using GraphRAG
                    logger.info(f"Using GraphRAG for search (vector_search={use_vector_search}, graph_search={use_graph_search})")
                    graph_results = await self.graph_rag.search(
                        query,
                        max_results=max_results,
                        use_vector_search=use_vector_search,
                        use_graph_search=use_graph_search
                    )

                    # Convert GraphRAG results to the same format as PubMed results
                    search_results = {
                        'esearchresult': {
                            'count': str(graph_results.get('result_count', 0)),
                            'idlist': [result.get('pmid', '') for result in graph_results.get('results', [])],
                            'translationset': [],
                            'querytranslation': query,
                            'retmax': str(max_results),
                            'retstart': '0',
                            'querykey': '1',
                            'webenv': '',
                        }
                    }

                    # Store the original GraphRAG results for later use
                    search_results['graph_rag_results'] = graph_results
            elif search_method == SearchMethod.CLINICAL_TRIALS:
                # Search ClinicalTrials.gov
                clinical_trials_results = await self.clinical_trials_client.search(query, max_results=max_results)

                # Convert ClinicalTrials.gov results to the same format as PubMed results
                search_results = {
                    'esearchresult': {
                        'count': str(len(clinical_trials_results)),
                        'idlist': [trial.get('nct_id', '') for trial in clinical_trials_results],
                        'translationset': [],
                        'querytranslation': query,
                        'retmax': str(max_results),
                        'retstart': '0',
                        'querykey': '1',
                        'webenv': '',
                    }
                }

                # Store the original ClinicalTrials.gov results for later use
                search_results['clinical_trials_results'] = clinical_trials_results
            else:  # Default to PubMed
                # Search PubMed
                search_results = await self.ncbi_client.search_pubmed(query, max_results=max_results)

            if not search_results or 'esearchresult' not in search_results:
                logger.warning(f"No results found for query: {query}")
                return {
                    "query": query,
                    "total_count": 0,
                    "results": [],
                    "result_id": str(uuid.uuid4())
                }
        except Exception as e:
            logger.error(f"Error searching PubMed: {str(e)}")
            raise ExternalServiceError("NCBI PubMed", f"Failed to search: {str(e)}")

        try:
            # Extract IDs and fetch abstracts
            id_list = search_results['esearchresult'].get('idlist', [])

            if not id_list:
                logger.warning(f"No article IDs found for query: {query}")
                return {
                    "query": query,
                    "total_count": 0,
                    "results": [],
                    "result_id": str(uuid.uuid4())
                }

            abstracts = await self.ncbi_client.fetch_pubmed_abstracts(id_list=id_list)

            if not abstracts:
                logger.warning(f"No abstracts found for query: {query}")
                return {
                    "query": query,
                    "total_count": 0,
                    "results": [],
                    "result_id": str(uuid.uuid4())
                }
        except Exception as e:
            logger.error(f"Error fetching abstracts: {str(e)}")
            raise ExternalServiceError("NCBI PubMed", f"Failed to fetch abstracts: {str(e)}")

        # Enrich with metadata
        enriched_articles = []
        for article in abstracts:
            # Add impact factor, citation count, etc.
            enriched = self._enrich_article(article)
            enriched_articles.append(enriched)

        # Generate a deterministic result ID based on the query and results
        # This ensures that the same query with the same results gets the same ID
        result_id_input = f"{query}:{max_results}:{len(enriched_articles)}"
        result_id = hashlib.md5(result_id_input.encode()).hexdigest()

        # Store results in cache for quick access
        self.result_cache[result_id] = {
            'query': query,
            'results': enriched_articles,
            'timestamp': datetime.now().isoformat(),
            'user_id': user_id
        }

        # Store in database
        if user_id:
            try:
                # Store query
                query_obj = await self.query_repository.create_async(
                    db=None,  # This will be handled by the repository
                    obj_in={
                        'user_id': user_id,
                        'query_text': query,
                        'query_type': 'text',
                        'parameters': {'max_results': max_results}
                    }
                )

                # Store results
                await self.result_repository.create_async(
                    db=None,  # This will be handled by the repository
                    obj_in={
                        'result_id': result_id,
                        'user_id': user_id,
                        'query_id': query_obj.id,
                        'result_type': 'search',
                        'result_data': {'articles': enriched_articles},
                        'created_at': datetime.now()
                    }
                )
                logger.info(f"Search results stored in database: result_id={result_id}, query_id={query_obj.id}")
            except Exception as e:
                logger.error(f"Error storing search results: {str(e)}")
                # Log but don't raise to avoid failing the search if storage fails
                # This allows the search to succeed even if the database is down

        logger.info(f"Search completed: {len(enriched_articles)} results found (result_id={result_id})")

        # Apply pagination
        total_count = len(enriched_articles)
        total_pages = (total_count + page_size - 1) // page_size  # Ceiling division

        # Calculate start and end indices for the current page
        start_idx = (page - 1) * page_size
        end_idx = min(start_idx + page_size, total_count)

        # Get the results for the current page
        paged_results = enriched_articles[start_idx:end_idx]

        # Create pagination metadata
        pagination_metadata = {
            "page": page,
            "page_size": page_size,
            "total_pages": total_pages,
            "total_count": total_count,
            "has_previous": page > 1,
            "has_next": page < total_pages
        }

        return {
            "query": query,
            "total_count": total_count,
            "results": paged_results,
            "result_id": result_id,
            "pagination": pagination_metadata
        }

    @cached(prefix="search_pico", data_type="search")
    async def search_pico(
        self,
        condition: str,
        interventions: List[str] = [],
        outcomes: List[str] = [],
        population: Optional[str] = None,
        study_design: Optional[str] = None,
        years: int = 5,
        max_results: int = 100,
        page: int = 1,
        page_size: int = 20,
        user_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Search PubMed using the PICO framework with pagination.

        Args:
            condition: Medical condition
            interventions: List of interventions
            outcomes: List of outcomes
            population: Target population
            study_design: Study design
            years: Number of years to search back
            max_results: Maximum number of results to return from the API
            page: Page number (1-based)
            page_size: Number of results per page
            user_id: User ID for storing the query and results

        Returns:
            Search results with pagination metadata

        Raises:
            ValidationError: If the condition is invalid or missing
            ExternalServiceError: If the NCBI API fails
            DatabaseError: If there's an error storing the results
        """
        # Validate inputs
        if not condition or not condition.strip():
            raise ValidationError("Condition is required for PICO search")

        if years < 1 or years > 50:
            raise ValidationError("Years must be between 1 and 50")

        if max_results < 1 or max_results > 500:
            raise ValidationError("max_results must be between 1 and 500")

        if page < 1:
            raise ValidationError("page must be at least 1")

        if page_size < 1 or page_size > 100:
            raise ValidationError("page_size must be between 1 and 100")

        logger.info(f"Executing PICO search: {condition} (max_results={max_results})")
        logger.debug(f"PICO search parameters: interventions={interventions}, outcomes={outcomes}, population={population}, study_design={study_design}, years={years}")

        try:
            # Build PICO query
            query_builder = MedicalQueryBuilder()

            # Add condition
            query_builder.add_condition(MedicalCondition(condition))

            # Add interventions
            for intervention in interventions:
                if intervention and intervention.strip():
                    query_builder.add_intervention(MedicalIntervention(intervention))

            # Add outcomes
            for outcome in outcomes:
                if outcome and outcome.strip():
                    query_builder.add_outcome(OutcomeMetric(outcome))

            # Add population if provided
            if population and population.strip():
                query_builder.set_population(population)

            # Add study design if provided
            if study_design and study_design.strip():
                query_builder.set_study_design(StudyDesign(study_design))

            # Set time range
            query_builder.set_years(years)

            # Build the query
            query = query_builder.build()
            logger.info(f"Built PICO query: {query}")

            # Execute the search
            return await self.search(query, max_results, page, page_size, user_id)
        except ValidationError:
            # Re-raise validation errors
            raise
        except Exception as e:
            logger.error(f"Error in PICO search: {str(e)}")
            raise SearchError(condition, f"Failed to execute PICO search: {str(e)}")

    async def get_result(self, result_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a stored search result.

        Args:
            result_id: Result ID

        Returns:
            Search result or None if not found

        Raises:
            ValidationError: If the result_id is invalid
            ResourceNotFoundError: If the result is not found
            DatabaseError: If there's an error retrieving the result
        """
        if not result_id or not result_id.strip():
            raise ValidationError("Result ID cannot be empty")

        logger.info(f"Getting search result: {result_id}")

        # Try to get from cache first
        cache_key = f"result:{result_id}"
        cached_result = await cache_manager.get(cache_key, data_type="search")
        if cached_result is not None:
            logger.debug(f"Result found in cache: {result_id}")
            return cached_result

        # If not in cache, try to get from database
        try:
            result = await self.result_repository.get_by_result_id_async(db=None, result_id=result_id)
            if result:
                logger.debug(f"Result found in database: {result_id}")
                # Convert to dictionary
                result_dict = {
                    'query': result.query.query_text if result.query else '',
                    'results': result.result_data.get('articles', []),
                    'timestamp': result.created_at.isoformat(),
                    'user_id': result.user_id
                }

                # Store in cache for future use
                await cache_manager.set(cache_key, result_dict, data_type="search")

                return result_dict
            else:
                logger.warning(f"Result not found: {result_id}")
                return None
        except Exception as e:
            logger.error(f"Error getting result from database: {str(e)}")
            raise DatabaseError(f"Failed to retrieve search result: {str(e)}")

    def _enrich_article(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enrich an article with additional metadata.

        Args:
            article: Article data

        Returns:
            Enriched article
        """
        try:
            # Make a copy of the article to avoid modifying the original
            enriched = article.copy()

            # Add impact factor (placeholder)
            enriched['impact_factor'] = 0.0

            # Add citation count (placeholder)
            enriched['citation_count'] = 0

            # Add authority score (placeholder)
            enriched['authority_score'] = 0.0

            # Standardize dates
            if 'publication_date' in enriched:
                try:
                    # Parse and format the date
                    date_str = enriched['publication_date']
                    # This is a placeholder - actual implementation would parse and standardize the date
                    enriched['publication_date_iso'] = date_str
                except Exception as e:
                    logger.warning(f"Failed to parse date: {e}")
                    # Use the original date if parsing fails
                    enriched['publication_date_iso'] = enriched['publication_date']

            # Ensure required fields are present
            for field in ['pmid', 'title', 'authors', 'journal']:
                if field not in enriched:
                    enriched[field] = ''

            # Ensure abstract is present
            if 'abstract' not in enriched:
                enriched['abstract'] = ''

            return enriched
        except Exception as e:
            logger.error(f"Error enriching article: {str(e)}")
            # Return the original article if enrichment fails
            return article
