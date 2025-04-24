"""Search service for the Medical Research Synthesizer.

This module provides a service for searching medical literature.
"""
import uuid
import hashlib
import traceback
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from enum import Enum
from ..core.cache import redis_cached
from ..core.exceptions import (
    SearchError, ValidationError,
    ExternalServiceError, DatabaseError
)
from medical.core.logging_config import get_logger
from medical.clients.ncbi.ncbi_client import NCBIClient
from medical.clients.clinical_trials_gov.clinical_trials_client import ClinicalTrialsClient
from medical.storage.repositories.result_repository import ResultRepository
from medical.storage.repositories.query_repository import QueryRepository
from medical.graph.graph_rag import GraphRAG
from medical.data_ingestion_layer.query_builder import (
    MedicalQueryBuilder, MedicalCondition, MedicalIntervention, OutcomeMetric, StudyDesign
)
class SearchMethod(str, Enum):
    """Search method enum.

    This enum defines the available search methods:
    - PUBMED: Search PubMed using the NCBI API
    - CLINICAL_TRIALS: Search ClinicalTrials.gov
    - GRAPH_RAG: Search using GraphRAG (graph-based retrieval-augmented generation)
    """
    PUBMED = "pubmed"
    CLINICAL_TRIALS = "clinical_trials"
    GRAPH_RAG = "graph_rag"
logger = get_logger(__name__)
class SearchService:
    """
    Service for searching medical literature.

    This service provides methods for searching medical literature from various sources,
    including PubMed, ClinicalTrials.gov, and using GraphRAG for enhanced search capabilities.
    """
    def __init__(
        self,
        ncbi_client: NCBIClient,
        clinical_trials_client: ClinicalTrialsClient,
        query_repository: QueryRepository,
        result_repository: ResultRepository,
        graph_rag: Optional[GraphRAG] = None
    ):
        self.ncbi_client = ncbi_client
        self.clinical_trials_client = clinical_trials_client
        self.query_repository = query_repository
        self.result_repository = result_repository
        self.graph_rag = graph_rag
        available_methods = [SearchMethod.PUBMED, SearchMethod.CLINICAL_TRIALS]
        if self.graph_rag is not None:
            available_methods.append(SearchMethod.GRAPH_RAG)
            logger.info("GraphRAG search method is available")
        else:
            logger.info("GraphRAG search method is not available")
        logger.info(f"Search service initialized with methods: {', '.join([m.value for m in available_methods])}")
    def is_graph_rag_available(self) -> bool:
        """Check if GraphRAG is available.

        Returns:
            bool: True if GraphRAG is available, False otherwise
        """
        return self.graph_rag is not None
    @redis_cached(ttl=3600, prefix="search")
    async def search(
        self, query: str, max_results: int = 100, page: int = 1, page_size: int = 20,
        user_id: Optional[int] = None, search_method: Union[str, SearchMethod] = SearchMethod.PUBMED,
        use_graph_rag: bool = False, use_vector_search: bool = True, use_graph_search: bool = True
    ) -> Dict[str, Any]:
        if not query or not query.strip():
            raise ValidationError("Search query cannot be empty")
        if max_results < 1 or max_results > 500:
            raise ValidationError("max_results must be between 1 and 500")
        if page < 1:
            raise ValidationError("page must be at least 1")
        if page_size < 1 or page_size > 100:
            raise ValidationError("page_size must be between 1 and 100")
        if isinstance(search_method, str):
            try:
                search_method = SearchMethod(search_method.lower())
            except ValueError:
                logger.warning(f"Invalid search method: {search_method}, using default")
                search_method = SearchMethod.PUBMED
        if use_graph_rag:
            search_method = SearchMethod.GRAPH_RAG
        logger.info(f"Executing search: {query} (max_results={max_results}, method={search_method})")
        try:
            if search_method == SearchMethod.GRAPH_RAG:
                if not self.is_graph_rag_available():
                    logger.warning("GraphRAG not available, falling back to PubMed search")
                    search_results = await self.ncbi_client.search_pubmed(query, max_results=max_results)
                    search_results['fallback_reason'] = "GraphRAG not available"
                else:
                    try:
                        logger.info(f"Using GraphRAG for search (vector_search={use_vector_search}, graph_search={use_graph_search})")
                        graph_results = await self.graph_rag.search(
                            query,
                            max_results=max_results,
                            use_vector_search=use_vector_search,
                            use_graph_search=use_graph_search
                        )
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
                        search_results['graph_rag_results'] = graph_results
                    except ConnectionError as e:
                        logger.error(f"GraphRAG connection error: {str(e)}")
                        logger.warning("Falling back to PubMed search due to GraphRAG connection error")
                        search_results = await self.ncbi_client.search_pubmed(query, max_results=max_results)
                        search_results['fallback_reason'] = f"GraphRAG connection error: {str(e)}"
                    except ValueError as e:
                        logger.error(f"GraphRAG value error: {str(e)}")
                        logger.warning("Falling back to PubMed search due to GraphRAG value error")
                        search_results = await self.ncbi_client.search_pubmed(query, max_results=max_results)
                        search_results['fallback_reason'] = f"GraphRAG value error: {str(e)}"
                    except Exception as e:
                        logger.error(f"GraphRAG search error: {str(e)}")
                        logger.error(traceback.format_exc())
                        logger.warning("Falling back to PubMed search due to GraphRAG error")
                        search_results = await self.ncbi_client.search_pubmed(query, max_results=max_results)
                        search_results['fallback_reason'] = f"GraphRAG error: {str(e)}"
            elif search_method == SearchMethod.CLINICAL_TRIALS:
                try:
                    clinical_trials_results = await self.clinical_trials_client.search(query, max_results=max_results)
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
                    search_results['clinical_trials_results'] = clinical_trials_results
                except ConnectionError as e:
                    logger.error(f"ClinicalTrials.gov connection error: {str(e)}")
                    logger.warning("Falling back to PubMed search due to ClinicalTrials.gov connection error")
                    search_results = await self.ncbi_client.search_pubmed(query, max_results=max_results)
                    search_results['fallback_reason'] = f"ClinicalTrials.gov connection error: {str(e)}"
                except Exception as e:
                    logger.error(f"ClinicalTrials.gov search error: {str(e)}")
                    logger.error(traceback.format_exc())
                    logger.warning("Falling back to PubMed search due to ClinicalTrials.gov error")
                    search_results = await self.ncbi_client.search_pubmed(query, max_results=max_results)
                    search_results['fallback_reason'] = f"ClinicalTrials.gov error: {str(e)}"
            else:  # Default to PubMed
                try:
                    search_results = await self.ncbi_client.search_pubmed(query, max_results=max_results)
                except Exception as e:
                    logger.error(f"PubMed search error: {str(e)}")
                    logger.error(traceback.format_exc())
                    raise SearchError(f"PubMed search failed: {str(e)}") from e
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

        enriched_articles = []
        for article in abstracts:
            enriched = self._enrich_article(article)
            enriched_articles.append(enriched)
        result_id_input = f"{query}:{max_results}:{len(enriched_articles)}"
        result_id = hashlib.md5(result_id_input.encode()).hexdigest()
        self.result_cache[result_id] = {
            'query': query,
            'results': enriched_articles,
            'timestamp': datetime.now().isoformat(),
            'user_id': user_id
        }
        if user_id:
            try:
                query_obj = await self.query_repository.create_async(
                    None,  # This will be handled by the repository
                    obj_in={
                        'user_id': user_id,
                        'query_text': query,
                        'query_type': 'text',
                        'parameters': {'max_results': max_results}
                    }
                )
                await self.result_repository.create_async(
                    None,  # This will be handled by the repository
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
        logger.info(f"Search completed: {len(enriched_articles)} results found (result_id={result_id})")
        total_count = len(enriched_articles)
        total_pages = (total_count + page_size - 1) // page_size  # Ceiling division
        start_idx = (page - 1) * page_size
        end_idx = min(start_idx + page_size, total_count)
        paged_results = enriched_articles[start_idx:end_idx]
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
    @redis_cached(ttl=3600, prefix="search_pico")
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
            query_builder = MedicalQueryBuilder()
            query_builder.add_condition(MedicalCondition(condition))
            for intervention in interventions:
                if intervention and intervention.strip():
                    query_builder.add_intervention(MedicalIntervention(intervention))
            for outcome in outcomes:
                if outcome and outcome.strip():
                    query_builder.add_outcome(OutcomeMetric(outcome))
            if population and population.strip():
                query_builder.set_population(population)
            if study_design and study_design.strip():
                query_builder.set_study_design(StudyDesign(study_design))
            query_builder.set_years(years)
            query = query_builder.build()
            logger.info(f"Built PICO query: {query}")
            return await self.search(query, max_results, page, page_size, user_id)
        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"Error in PICO search: {str(e)}")
            raise SearchError(condition, f"Failed to execute PICO search: {str(e)}")
    async def get_result(self, result_id: str) -> Optional[Dict[str, Any]]:
        if not result_id or not result_id.strip():
            raise ValidationError("Result ID cannot be empty")
        logger.info(f"Getting search result: {result_id}")
        # Use redis_cached decorator instead of direct cache access
        # This is a placeholder for actual cache implementation
        cached_result = None
        if cached_result is not None:
            logger.debug(f"Result found in cache: {result_id}")
            return cached_result
        try:
            result = await self.result_repository.get_by_result_id_async(None, result_id=result_id)
            if result:
                logger.debug(f"Result found in database: {result_id}")
                result_dict = {
                    'query': result.query.query_text if result.query else '',
                    'results': result.result_data.get('articles', []),
                    'timestamp': result.created_at.isoformat(),
                    'user_id': result.user_id
                }
                # Use redis_cached decorator instead of direct cache access
                # This is a placeholder for actual cache implementation
                pass
                return result_dict
            else:
                logger.warning(f"Result not found: {result_id}")
                return None
        except Exception as e:
            logger.error(f"Error getting result from database: {str(e)}")
            raise DatabaseError(f"Failed to retrieve search result: {str(e)}")

    def _enrich_article(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich an article with additional metadata.

        This method adds additional metadata to an article, such as impact factor,
        citation count, and authority score. It also ensures that all required
        fields are present in the article.

        Args:
            article: The article data to enrich

        Returns:
            Enriched article with additional metadata
        """
        try:
            enriched = article.copy()
            enriched['impact_factor'] = 0.0
            enriched['citation_count'] = 0
            enriched['authority_score'] = 0.0
            if 'publication_date' in enriched:
                try:
                    date_str = enriched['publication_date']
                    enriched['publication_date_iso'] = date_str
                except Exception as e:
                    logger.warning(f"Failed to parse date: {e}")
                    enriched['publication_date_iso'] = enriched['publication_date']
            for field in ['pmid', 'title', 'authors', 'journal']:
                if field not in enriched:
                    enriched[field] = ''
            if 'abstract' not in enriched:
                enriched['abstract'] = ''
            return enriched
        except Exception as e:
            logger.error(f"Error enriching article: {str(e)}")
            return article