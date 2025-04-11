"""
Analysis service for the Medical Research Synthesizer.

This module provides a service for analyzing medical literature.
"""

import logging
import uuid
import hashlib
from datetime import datetime
from typing import Dict, List, Any, Optional

from asf.medical.core.enhanced_cache import enhanced_cache_manager as cache_manager, enhanced_cached as cached, cache_manager
from asf.medical.core.exceptions import (
    AnalysisError, ResourceNotFoundError, ValidationError,
    ExternalServiceError, DatabaseError, ModelError
)

from asf.medical.ml.services.unified_contradiction_service import UnifiedUnifiedUnifiedContradictionService
from asf.medical.ml.services.temporal_service import TemporalService
from asf.medical.services.search_service import SearchService
from asf.medical.storage.repositories.result_repository import ResultRepository

logger = logging.getLogger(__name__)

class AnalysisService:
    """
    Service for analyzing medical literature.
    """

    def __init__(
        self,
        contradiction_service: EnhancedUnifiedUnifiedContradictionService,
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
    logger.error(f\"Error searching for articles: {str(e)}\")
    raise DatabaseError(f\"Error searching for articles: {str(e)}\") ExternalServiceError("Search Service", f"Failed to search for articles: {str(e)}")

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
    logger.error(f\"Error detecting contradictions: {str(e)}\")
    raise DatabaseError(f\"Error detecting contradictions: {str(e)}\") ModelError("Contradiction Service", f"Failed to detect contradictions: {str(e)}")

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
                    db,  # This will be handled by the repository
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
    logger.error(f\"Error searching for CAP articles: {str(e)}\")
    raise DatabaseError(f\"Error searching for CAP articles: {str(e)}\") ExternalServiceError("Search Service", f"Failed to search for CAP articles: {str(e)}")

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
    logger.error(f\"Error analyzing CAP literature: {str(e)}\")
    raise DatabaseError(f\"Error analyzing CAP literature: {str(e)}\") AnalysisError("CAP Analysis", f"Failed to analyze CAP literature: {str(e)}")

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
                    db,  # This will be handled by the repository
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
        if not analysis_id or not analysis_id.strip():
            raise ValidationError("Analysis ID cannot be empty")

        logger.info(f"Getting analysis: {analysis_id}")

        cache_key = f"analysis:{analysis_id}"
        cached_result = await enhanced_cache_manager.get(cache_key, data_type="analysis")
        if cached_result is not None:
            logger.debug(f"Analysis found in cache: {analysis_id}")
            return cached_result

        try:
            result = await self.result_repository.get_by_result_id_async(db, result_id=analysis_id)
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
    logger.error(f\"Error getting analysis from database: {str(e)}\")
    raise DatabaseError(f\"Error getting analysis from database: {str(e)}\") DatabaseError(f"Failed to retrieve analysis: {str(e)}")
