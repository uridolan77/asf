"""Analysis service for the Medical Research Synthesizer.

This module provides a service for analyzing medical literature.
"""
import logging
import uuid
import hashlib
import time
from datetime import datetime
from typing import Dict, Any, Optional
from ..core.enhanced_cache import enhanced_cache_manager, enhanced_cached
from ..core.progress_tracker import ProgressTracker
from ..core.exceptions import (
    AnalysisError, ValidationError,
    ExternalServiceError, DatabaseError, ModelError, ResourceNotFoundError
)
from ..ml.services.contradiction_service import ContradictionService
from ..ml.services.temporal_service import TemporalService
from ..services.search_service import SearchService
from ..storage.repositories.result_repository import ResultRepository

logger = logging.getLogger(__name__)

class AnalysisProgressTracker(ProgressTracker):
    """Progress tracker for analysis operations."""
    def __init__(self, analysis_id: str, total_steps: int = 100):
        super().__init__(operation_id=analysis_id, total_steps=total_steps)
        self.analysis_id = analysis_id
        self.analysis_type = "unknown"
        self.start_time = time.time()
    def set_analysis_type(self, analysis_type: str):
        self.analysis_type = analysis_type
    def get_progress_details(self) -> Dict[str, Any]:
        details = super().get_progress_details()
        details.update({
            "analysis_id": self.analysis_id,
            "analysis_type": self.analysis_type,
            "elapsed_time": time.time() - self.start_time
        })
        return details
    async def save_progress(self):
        progress_key = f"analysis_progress:{self.analysis_id}"
        await enhanced_cache_manager.set(
            progress_key,
            self.get_progress_details(),
            ttl=3600,  # 1 hour TTL
            data_type="progress"
        )

def validate_analysis_input(func):
    async def wrapper(self, *args, **kwargs):
        query = kwargs.get('query', '')
        max_results = kwargs.get('max_results', 20)
        threshold = kwargs.get('threshold', 0.7)
        if not query or not query.strip():
            raise ValidationError("Query cannot be empty")
        if max_results is not None:
            if not isinstance(max_results, int):
                raise ValidationError("max_results must be an integer")
            if max_results < 1:
                raise ValidationError("max_results must be at least 1")
            if max_results > 500:
                raise ValidationError("max_results cannot exceed 500")
        if threshold is not None:
            if not isinstance(threshold, (int, float)):
                raise ValidationError("threshold must be a number")
            if threshold < 0.0 or threshold > 1.0:
                raise ValidationError("threshold must be between 0.0 and 1.0")
        return await func(self, *args, **kwargs)
    return wrapper

def track_analysis_progress(analysis_type: str, total_steps: int = 100):
    def decorator(func):
        async def wrapper(self, *args, **kwargs):
            func_name = func.__name__
            param_str = f"{func_name}:{args}:{kwargs}"
            analysis_id = hashlib.md5(param_str.encode()).hexdigest()
            tracker = AnalysisProgressTracker(analysis_id, total_steps)
            tracker.set_analysis_type(analysis_type)
            tracker.update(0, "Starting analysis")
            await tracker.save_progress()
            kwargs['progress_tracker'] = tracker
            try:
                result = await func(self, *args, **kwargs)
                tracker.complete("Analysis completed successfully")
                await tracker.save_progress()
                return result
            except Exception as e:
                tracker.fail(f"Analysis failed: {str(e)}")
                await tracker.save_progress()
                raise
        return wrapper
    return decorator

def enhanced_error_handling(func):
    async def wrapper(self, *args, **kwargs):
        try:
            return await func(self, *args, **kwargs)
        except ValidationError:
            raise
        except ExternalServiceError:
            raise
        except ModelError:
            raise
        except DatabaseError:
            raise
        except ResourceNotFoundError:
            raise
        except AnalysisError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {str(e)}", exc_info=True)
            raise AnalysisError(
                component=f"Analysis Service ({func.__name__})",
                message=f"Unexpected error: {str(e)}"
            )
    return wrapper

def cached_analysis(ttl: int = 3600, prefix: str = "analysis", data_type: str = "analysis"):
    def decorator(func):
        async def wrapper(self, *args, **kwargs):
            skip_cache = kwargs.pop('skip_cache', False)
            if skip_cache:
                return await func(self, *args, **kwargs)
            func_name = func.__name__
            param_str = f"{func_name}:{args}:{kwargs}"
            cache_key = f"{prefix}:{hashlib.md5(param_str.encode()).hexdigest()}"
            cached_result = await enhanced_cache_manager.get(cache_key, data_type=data_type)
            if cached_result is not None:
                logger.debug(f"Cache hit for {func_name}")
                return cached_result
            logger.debug(f"Cache miss for {func_name}")
            result = await func(self, *args, **kwargs)
            await enhanced_cache_manager.set(cache_key, result, ttl=ttl, data_type=data_type)
            return result
        return wrapper
    return decorator

class AnalysisService:
    """Service for analyzing medical literature.

    This service provides methods for analyzing medical literature, including
    contradiction detection, CAP (Community-Acquired Pneumonia) analysis,
    and retrieving previously performed analyses.
    """
    def __init__(
        self,
        contradiction_service: ContradictionService,
        temporal_service: TemporalService,
        search_service: SearchService,
        result_repository: ResultRepository
    ):
        self.contradiction_service = contradiction_service
        self.temporal_service = temporal_service
        self.search_service = search_service
        self.result_repository = result_repository
    @enhanced_cached(prefix="analyze_contradictions", data_type="analysis")
    async def analyze_contradictions(
        self,
        query: str,
        max_results: int = 20,
        threshold: float = 0.7,
        use_biomedlm: bool = True,
        use_tsmixer: bool = False,
        use_lorentz: bool = False,
        user_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """Analyze contradictions in medical literature based on a query.

        This method searches for medical literature matching the query and
        analyzes it for contradictions. It identifies statements that contradict
        each other and provides explanations for the contradictions.

        Args:
            query: The search query to find relevant medical literature
            max_results: Maximum number of search results to analyze
            threshold: Minimum contradiction score threshold (0.0-1.0)
            use_biomedlm: Whether to use BioMedLM for contradiction detection
            use_tsmixer: Whether to use TSMixer for temporal contradiction detection
            use_lorentz: Whether to use Lorentz embeddings for contradiction detection
            user_id: ID of the user performing the analysis

        Returns:
            Dictionary containing the contradiction analysis results

        Raises:
            ValidationError: If the query is empty or invalid
            AnalysisError: If an error occurs during analysis
            ExternalServiceError: If an external service call fails
        """
        if not query or not query.strip():
            raise ValidationError("Query cannot be empty")
        if max_results < 1 or max_results > 100:
            raise ValidationError("max_results must be between 1 and 100")
        if threshold < 0.0 or threshold > 1.0:
            raise ValidationError("threshold must be between 0.0 and 1.0")
        if not (use_biomedlm or use_tsmixer or use_lorentz):
            raise ValidationError("At least one detection method must be enabled")
        logger.info(f"Analyzing contradictions for query: {query}")
        logger.debug(f"Analysis parameters: max_results={max_results}, threshold={threshold}, use_biomedlm={use_biomedlm}, use_tsmixer={use_tsmixer}, use_lorentz={use_lorentz}")
        try:
            search_result = await self.search_service.search(query, max_results, user_id)
            if not search_result or not search_result.get('results'):
                logger.warning(f"No search results found for query: {query}")
                return {
                    "query": query,
                    "total_articles": 0,
                    "contradictions": [],
                    "analysis_id": str(uuid.uuid4()),
                    "detection_method": "none"
                }
        except Exception as e:
            logger.error(f"Error searching for articles: {str(e)}")
            raise ExternalServiceError("Search Service", f"Failed to search for articles: {str(e)}")
        articles = search_result['results']
        try:
            contradictions = await self.contradiction_service.detect_contradictions_in_articles(
                articles,
                threshold=threshold,
                use_all_methods=use_biomedlm or use_tsmixer or use_lorentz
            )
            detection_methods = []
            if use_biomedlm:
                detection_methods.append("biomedlm")
            if use_tsmixer:
                detection_methods.append("tsmixer")
            if use_lorentz:
                detection_methods.append("lorentz")
            detection_method = "+".join(detection_methods) if detection_methods else "none"
            analysis_id_input = f"{query}:{max_results}:{threshold}:{use_biomedlm}:{use_tsmixer}:{use_lorentz}:{len(contradictions)}"
            analysis_id = hashlib.md5(analysis_id_input.encode()).hexdigest()
            analysis_result = {
                "query": query,
                "total_articles": len(articles),
                "contradictions": contradictions,
                "analysis_id": analysis_id,
                "detection_method": detection_method
            }
            logger.info(f"Contradiction analysis completed: {len(contradictions)} contradictions found using {detection_method}")
        except Exception as e:
            logger.error(f"Error detecting contradictions: {str(e)}")
            raise ModelError("Contradiction Service", f"Failed to detect contradictions: {str(e)}")
        cache_key = f"analysis:{analysis_id}"
        cache_value = {
            'analysis': analysis_result,
            'timestamp': datetime.now().isoformat(),
            'user_id': user_id
        }
        await enhanced_cache_manager.set(cache_key, cache_value, data_type="analysis")
        if user_id:
            try:
                await self.result_repository.create_async(
                    None,  # This will be handled by the repository
                    obj_in={
                        'result_id': analysis_id,
                        'user_id': user_id,
                        'query_id': None,  # We don't have a query ID here
                        'result_type': 'contradiction_analysis',
                        'result_data': analysis_result,
                        'created_at': datetime.now()
                    }
                )
                logger.info(f"Contradiction analysis stored in database: analysis_id={analysis_id}")
            except Exception as e:
                logger.error(f"Error storing contradiction analysis: {str(e)}")
        return analysis_result
    @enhanced_cached(prefix="analyze_cap", data_type="analysis")
    async def analyze_cap(self, user_id: Optional[int] = None) -> Dict[str, Any]:
        """Perform CAP (Community-Acquired Pneumonia) analysis.

        This method searches for medical literature related to community-acquired
        pneumonia and analyzes it to identify treatment types, patient populations,
        and outcomes.

        Args:
            user_id: ID of the user performing the analysis

        Returns:
            Dictionary containing the CAP analysis results

        Raises:
            ExternalServiceError: If the search service call fails
            AnalysisError: If an error occurs during analysis
        """
        logger.info("Analyzing CAP literature")
        try:
            search_result = await self.search_service.search(
                "community acquired pneumonia treatment",
                max_results=50,
                user_id=user_id
            )
            if not search_result or not search_result.get('results'):
                logger.warning("No CAP articles found")
                return {
                    "total_articles": 0,
                    "analysis": {},
                    "analysis_id": str(uuid.uuid4())
                }
        except Exception as e:
            logger.error(f"Error searching for CAP articles: {str(e)}")
            raise ExternalServiceError("Search Service", f"Failed to search for CAP articles: {str(e)}")
        articles = search_result['results']
        try:
            analysis = {
                "treatment_types": {
                    "antibiotics": 0,
                    "supportive_care": 0,
                    "other": 0
                },
                "patient_populations": {
                    "adults": 0,
                    "children": 0,
                    "elderly": 0
                },
                "outcomes": {
                    "mortality": 0,
                    "length_of_stay": 0,
                    "complications": 0
                }
            }
            for article in articles:
                abstract = article.get('abstract', '').lower()
                if 'antibiotic' in abstract:
                    analysis['treatment_types']['antibiotics'] += 1
                elif 'supportive care' in abstract:
                    analysis['treatment_types']['supportive_care'] += 1
                else:
                    analysis['treatment_types']['other'] += 1
                if 'children' in abstract or 'pediatric' in abstract:
                    analysis['patient_populations']['children'] += 1
                elif 'elderly' in abstract or 'geriatric' in abstract:
                    analysis['patient_populations']['elderly'] += 1
                else:
                    analysis['patient_populations']['adults'] += 1
                if 'mortality' in abstract or 'death' in abstract:
                    analysis['outcomes']['mortality'] += 1
                elif 'length of stay' in abstract or 'hospital stay' in abstract:
                    analysis['outcomes']['length_of_stay'] += 1
                elif 'complication' in abstract:
                    analysis['outcomes']['complications'] += 1
            analysis_id_input = "cap_analysis"
            analysis_id = hashlib.md5(analysis_id_input.encode()).hexdigest()
            analysis_result = {
                "total_articles": len(articles),
                "analysis": analysis,
                "analysis_id": analysis_id
            }
            logger.info(f"CAP analysis completed: {len(articles)} articles analyzed")
        except Exception as e:
            logger.error(f"Error analyzing CAP literature: {str(e)}")
            raise AnalysisError("CAP Analysis", f"Failed to analyze CAP literature: {str(e)}")
        cache_key = f"analysis:{analysis_id}"
        cache_value = {
            'analysis': analysis_result,
            'timestamp': datetime.now().isoformat(),
            'user_id': user_id
        }
        await enhanced_cache_manager.set(cache_key, cache_value, data_type="analysis")
        if user_id:
            try:
                await self.result_repository.create_async(
                    None,  # This will be handled by the repository
                    obj_in={
                        'result_id': analysis_id,
                        'user_id': user_id,
                        'query_id': None,  # We don't have a query ID here
                        'result_type': 'cap_analysis',
                        'result_data': analysis_result,
                        'created_at': datetime.now()
                    }
                )
                logger.info(f"CAP analysis stored in database: analysis_id={analysis_id}")
            except Exception as e:
                logger.error(f"Error storing CAP analysis: {str(e)}")
        return analysis_result
    async def get_analysis(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a previously performed analysis by ID.

        This method retrieves the results of a previously performed analysis
        using its unique identifier. It first checks the cache and then falls
        back to the database if not found in cache.

        Args:
            analysis_id: The unique identifier of the analysis to retrieve

        Returns:
            Dictionary containing the analysis results, or None if not found

        Raises:
            ValidationError: If the analysis ID is empty or invalid
            DatabaseError: If an error occurs when retrieving from the database
        """
        if not analysis_id or not analysis_id.strip():
            raise ValidationError("Analysis ID cannot be empty")
        logger.info(f"Getting analysis: {analysis_id}")
        cache_key = f"analysis:{analysis_id}"
        cached_result = await enhanced_cache_manager.get(cache_key, data_type="analysis")
        if cached_result is not None:
            logger.debug(f"Analysis found in cache: {analysis_id}")
            return cached_result
        try:
            result = await self.result_repository.get_by_result_id_async(None, result_id=analysis_id)
            if result:
                logger.debug(f"Analysis found in database: {analysis_id}")
                result_dict = {
                    'analysis': result.result_data,
                    'timestamp': result.created_at.isoformat(),
                    'user_id': result.user_id
                }
                await enhanced_cache_manager.set(cache_key, result_dict, data_type="analysis")
                return result_dict
            else:
                logger.warning(f"Analysis not found: {analysis_id}")
                return None
        except Exception as e:
            logger.error(f"Error getting analysis from database: {str(e)}")
            raise DatabaseError(f"Failed to retrieve analysis: {str(e)}")