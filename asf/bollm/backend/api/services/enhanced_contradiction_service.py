"""
Enhanced Medical Contradiction Detection service for analyzing contradictions in medical literature
with integrated clinical trials data and terminology standardization.
"""
import os
import sys
import logging
from typing import Dict, List, Optional, Union, Any
from fastapi import Depends, HTTPException, status

# Import from the medical modules
from medical.visualization.contradiction_visualizer import ContradictionVisualizer

# Mock implementations for missing modules
class SearchService:
    """Mock SearchService class."""
    async def search(self, query, max_results=20, user_id=None):
        """Mock search method."""
        return {
            "results": [
                {
                    "pmid": f"pmid_{i}",
                    "title": f"Study on {query} - Part {i}",
                    "authors": [f"Author {j}" for j in range(3)],
                    "journal": "Journal of Medical Research",
                    "year": "2023",
                    "abstract": f"This study investigates {query} and concludes that it is effective for treatment."
                }
                for i in range(10)
            ]
        }

class TerminologyService:
    """Mock TerminologyService class."""
    def __init__(self, snomed_access_mode=None, snomed_api_key=None, snomed_cache_dir=None, snomed_edition=None):
        pass

    def normalize_clinical_term(self, term):
        """Mock normalize_clinical_term method."""
        return {
            "normalized_term": term,
            "concepts": [
                {"concept_id": "C0123456", "preferred_name": term, "semantic_type": "Disease or Syndrome"}
            ]
        }

class ClinicalDataService:
    """Mock ClinicalDataService class."""
    def __init__(self, terminology_service=None, clinical_trials_cache_dir=None):
        self.terminology_service = terminology_service

    def find_trials_with_semantic_expansion(self, term, include_similar=True, max_trials=20):
        """Mock find_trials_with_semantic_expansion method."""
        return {
            "trials": [
                {
                    "nct_id": f"NCT0000{i}",
                    "title": f"Clinical Trial for {term} - {i}",
                    "status": "Completed",
                    "phase": "Phase 3",
                    "enrollment": 100 + i * 10,
                    "start_date": "2022-01-01",
                    "completion_date": "2023-01-01"
                }
                for i in range(5)
            ],
            "search_terms_used": [term]
        }

class ValidationError(Exception):
    """Mock ValidationError class."""
    pass

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedContradictionService:
    """
    Enhanced service for detecting and analyzing contradictions in medical literature
    with integrated clinical trials data and terminology standardization.

    This service combines literature-based contradiction detection with clinical trials
    data and standardized medical terminology to provide more comprehensive and accurate
    contradiction analysis.
    """
    def __init__(self):
        """Initialize the enhanced contradiction service with dependencies"""
        # Initialize the search service to get papers for contradiction analysis
        from api.services.medical_search_service import get_medical_search_service
        self.search_service = get_medical_search_service().search_service

        # Initialize the terminology service for term standardization
        try:
            self.terminology_service = TerminologyService(
                snomed_access_mode=os.environ.get("SNOMED_ACCESS_MODE", "umls"),
                snomed_api_key=os.environ.get("UMLS_API_KEY"),
                snomed_cache_dir=os.environ.get("SNOMED_CACHE_DIR", "./terminology_cache"),
                snomed_edition=os.environ.get("SNOMED_EDITION", "US")
            )
            logger.info("Terminology service initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize terminology service: {str(e)}")
            self.terminology_service = None

        # Initialize the clinical data service for clinical trials integration
        try:
            if self.terminology_service:
                self.clinical_data_service = ClinicalDataService(
                    terminology_service=self.terminology_service,
                    clinical_trials_cache_dir=os.environ.get("CLINICAL_TRIALS_CACHE_DIR", "./clinical_trials_cache")
                )
                logger.info("Clinical data service initialized successfully")
            else:
                self.clinical_data_service = None
                logger.warning("Clinical data service could not be initialized: terminology service not available")
        except Exception as e:
            logger.warning(f"Failed to initialize clinical data service: {str(e)}")
            self.clinical_data_service = None

        # Initialize the contradiction visualizer
        try:
            self.contradiction_visualizer = ContradictionVisualizer()
            logger.info("Contradiction visualizer initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize contradiction visualizer: {str(e)}")
            self.contradiction_visualizer = None

        # Model settings
        self.available_models = {
            "biomedlm": {
                "name": "BioMedLM",
                "default": True,
                "description": "Medical language model optimized for biomedical text"
            },
            "tsmixer": {
                "name": "TSMixer",
                "default": False,
                "description": "Time series analysis model for temporal contradictions"
            },
            "lorentz": {
                "name": "Lorentz",
                "default": False,
                "description": "Specialized model for hierarchical relationship analysis"
            }
        }

    async def analyze_contradictions(
        self,
        query: str,
        max_results: int = 20,
        threshold: float = 0.7,
        use_biomedlm: bool = True,
        use_tsmixer: bool = False,
        use_lorentz: bool = False,
        include_clinical_trials: bool = True,
        standardize_terminology: bool = True,
        user_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Analyze contradictions in medical literature for a given query with
        integrated clinical trials data and standardized terminology.

        Args:
            query: Search query to find articles for contradiction analysis
            max_results: Maximum number of contradiction pairs to return
            threshold: Minimum contradiction score threshold (0.0-1.0)
            use_biomedlm: Whether to use the BioMedLM model
            use_tsmixer: Whether to use the TSMixer model
            use_lorentz: Whether to use the Lorentz model
            include_clinical_trials: Whether to include clinical trials data
            standardize_terminology: Whether to standardize terminology using SNOMED CT
            user_id: BO user ID

        Returns:
            Enhanced contradiction analysis results
        """
        try:
            # Validate input
            if not query or not query.strip():
                raise ValidationError("Query cannot be empty")

            if max_results < 1 or max_results > 100:
                raise ValidationError("max_results must be between 1 and 100")

            if threshold < 0.0 or threshold > 1.0:
                raise ValidationError("threshold must be between 0.0 and 1.0")

            if not any([use_biomedlm, use_tsmixer, use_lorentz]):
                logger.warning("No models selected, defaulting to BioMedLM")
                use_biomedlm = True

            # Step 1: Standardize terminology if requested
            standardized_query = query
            concept_mappings = None

            if standardize_terminology and self.terminology_service:
                try:
                    normalized = self.terminology_service.normalize_clinical_term(query)
                    if normalized and normalized.get("normalized_term"):
                        standardized_query = normalized.get("normalized_term")
                        concept_mappings = normalized.get("concepts", [])
                        logger.info(f"Standardized query from '{query}' to '{standardized_query}'")
                except Exception as e:
                    logger.warning(f"Failed to standardize terminology: {str(e)}")

            # Step 2: Search for articles related to the standardized query
            logger.info(f"Searching for articles to analyze contradictions: {standardized_query}")
            search_results = await self.search_service.search(
                query=standardized_query,
                max_results=max_results * 5,  # Get more results to find contradictions
                user_id=user_id
            )

            if not search_results.get('results'):
                return {
                    "success": False,
                    "message": f"No articles found for query: {query}",
                    "data": None
                }

            # Step 3: Get clinical trials data if requested
            clinical_trials_data = None

            if include_clinical_trials and self.clinical_data_service:
                try:
                    clinical_trials_data = self.clinical_data_service.find_trials_with_semantic_expansion(
                        term=standardized_query,
                        include_similar=True,
                        max_trials=max_results
                    )
                    logger.info(f"Found {len(clinical_trials_data.get('trials', []))} clinical trials for query")
                except Exception as e:
                    logger.warning(f"Failed to get clinical trials data: {str(e)}")

            # Step 4: Analyze contradictions in the search results
            articles = search_results['results']
            logger.info(f"Analyzing contradictions among {len(articles)} articles...")

            # Step 5: Process articles to find contradicting claims
            contradiction_pairs = []
            models_used = []

            if use_biomedlm:
                models_used.append("BioMedLM")
            if use_tsmixer:
                models_used.append("TSMixer")
            if use_lorentz:
                models_used.append("Lorentz")

            # Process articles in pairs to find contradictions
            import random

            # Create pairs of articles (avoid duplicate pairs)
            processed_pairs = set()
            for i in range(len(articles)):
                for j in range(i+1, len(articles)):
                    if len(contradiction_pairs) >= max_results:
                        break

                    article1 = articles[i]
                    article2 = articles[j]

                    # Skip already processed pairs
                    pair_key = f"{article1.get('pmid', '')}-{article2.get('pmid', '')}"
                    if pair_key in processed_pairs:
                        continue

                    processed_pairs.add(pair_key)

                    # Calculate contradiction score (simulated)
                    contradiction_score = self._calculate_contradiction_score(article1, article2)

                    # Only include pairs with score above threshold
                    if contradiction_score >= threshold:
                        # Extract claims from abstracts
                        claim1 = self._extract_claim(article1)
                        claim2 = self._extract_claim(article2)

                        # Map claims to standard terminology if available
                        standardized_claim1 = claim1
                        standardized_claim2 = claim2
                        claim1_concepts = []
                        claim2_concepts = []

                        if standardize_terminology and self.terminology_service:
                            try:
                                # Map first claim to terminology
                                claim1_normalized = self.terminology_service.normalize_clinical_term(claim1)
                                if claim1_normalized and claim1_normalized.get("normalized_term"):
                                    standardized_claim1 = claim1_normalized.get("normalized_term")
                                    claim1_concepts = claim1_normalized.get("concepts", [])[:3]  # Top 3 concepts

                                # Map second claim to terminology
                                claim2_normalized = self.terminology_service.normalize_clinical_term(claim2)
                                if claim2_normalized and claim2_normalized.get("normalized_term"):
                                    standardized_claim2 = claim2_normalized.get("normalized_term")
                                    claim2_concepts = claim2_normalized.get("concepts", [])[:3]  # Top 3 concepts
                            except Exception as e:
                                logger.warning(f"Failed to standardize claims: {str(e)}")

                        # Generate explanation
                        explanation = self._generate_contradiction_explanation(
                            article1, article2, standardized_claim1, standardized_claim2, contradiction_score
                        )

                        # Determine contradiction type
                        contradiction_type = self._determine_contradiction_type(article1, article2)

                        # Build contradiction pair with enhanced information
                        pair = {
                            "article1": {
                                "id": article1.get('pmid', f"article_a_{i}"),
                                "title": article1.get('title', f"Study on {query} - Part {i}"),
                                "authors": article1.get('authors', []),
                                "journal": article1.get('journal', ''),
                                "year": article1.get('year', ''),
                                "claim": claim1,
                                "standardized_claim": standardized_claim1 if standardized_claim1 != claim1 else None,
                                "concepts": claim1_concepts
                            },
                            "article2": {
                                "id": article2.get('pmid', f"article_b_{j}"),
                                "title": article2.get('title', f"Study on {query} - Part {j}"),
                                "authors": article2.get('authors', []),
                                "journal": article2.get('journal', ''),
                                "year": article2.get('year', ''),
                                "claim": claim2,
                                "standardized_claim": standardized_claim2 if standardized_claim2 != claim2 else None,
                                "concepts": claim2_concepts
                            },
                            "contradiction_score": contradiction_score,
                            "contradiction_type": contradiction_type,
                            "explanation": explanation
                        }

                        # Add supporting evidence from clinical trials if available
                        if clinical_trials_data and clinical_trials_data.get('trials'):
                            supporting_trials = self._find_supporting_trials(
                                pair, clinical_trials_data.get('trials', [])
                            )
                            if supporting_trials:
                                pair["supporting_trials"] = supporting_trials

                        contradiction_pairs.append(pair)

            # Step 6: Generate visualizations if visualizer is available
            visualization_urls = []
            if self.contradiction_visualizer and contradiction_pairs:
                try:
                    for i, pair in enumerate(contradiction_pairs[:5]):  # Limit to 5 visualizations
                        vis_data = {
                            "article1": pair["article1"],
                            "article2": pair["article2"],
                            "score": pair["contradiction_score"],
                            "explanation": pair["explanation"]
                        }
                        vis_url = self.contradiction_visualizer.create_visualization(
                            vis_data,
                            filename=f"contradiction_{i}_{hash(pair['article1']['id'] + pair['article2']['id'])}"
                        )
                        if vis_url:
                            visualization_urls.append(vis_url)
                except Exception as e:
                    logger.error(f"Error generating contradiction visualizations: {str(e)}")

            # Prepare response with enhanced data
            response_data = {
                "contradiction_pairs": contradiction_pairs,
                "query": query,
                "standardized_query": standardized_query if standardized_query != query else None,
                "threshold": threshold,
                "models_used": models_used,
                "visualizations": visualization_urls,
                "total_articles_analyzed": len(articles)
            }

            # Add concept mappings if available
            if concept_mappings:
                response_data["concept_mappings"] = concept_mappings[:5]  # Top 5 concepts

            # Add clinical trials summary if available
            if clinical_trials_data and clinical_trials_data.get('trials'):
                response_data["clinical_trials"] = {
                    "total_count": len(clinical_trials_data.get('trials', [])),
                    "search_terms_used": clinical_trials_data.get('search_terms_used', []),
                    "trials_summary": self._summarize_trials(clinical_trials_data.get('trials', []))
                }

            return {
                "success": True,
                "message": f"Identified {len(contradiction_pairs)} contradiction pairs with enhanced analysis",
                "data": response_data
            }

        except ValidationError as e:
            logger.error(f"Validation error in contradiction analysis: {str(e)}")
            return {
                "success": False,
                "message": str(e),
                "data": None
            }
        except Exception as e:
            logger.error(f"Error analyzing contradictions: {str(e)}")
            return {
                "success": False,
                "message": f"Failed to analyze contradictions: {str(e)}",
                "data": None
            }

    def get_available_models(self) -> Dict[str, Any]:
        """
        Get available contradiction detection models.

        Returns:
            List of available models
        """
        return {
            "success": True,
            "message": f"Found {len(self.available_models)} available contradiction detection models",
            "data": {
                "models": self.available_models,
                "default_models": ["biomedlm"],
                "features": {
                    "clinical_trials_integration": self.clinical_data_service is not None,
                    "terminology_standardization": self.terminology_service is not None,
                    "visualization": self.contradiction_visualizer is not None
                }
            }
        }

    def _calculate_contradiction_score(self, article1: Dict[str, Any], article2: Dict[str, Any]) -> float:
        """
        Calculate contradiction score between two articles using enhanced methods.
        In a real implementation, this would use NLP and semantic analysis.

        Args:
            article1: First article data
            article2: Second article data

        Returns:
            Contradiction score (0.0-1.0)
        """
        import random

        # This is a placeholder for the actual contradiction scoring algorithm
        # In a real implementation, this would analyze the abstracts using NLP

        # Factors that might suggest contradiction:
        # 1. Different conclusions about the same intervention
        # 2. Different outcomes for the same patient population
        # 3. Opposite recommendations

        # Simulate some factors that might affect contradiction score
        title_similarity = self._calculate_text_similarity(
            article1.get('title', ''),
            article2.get('title', '')
        )

        abstract_similarity = self._calculate_text_similarity(
            article1.get('abstract', ''),
            article2.get('abstract', '')
        )

        # Higher similarity in title/abstract but different conclusions suggests contradiction
        base_score = 0.5 + (title_similarity * 0.2) + (abstract_similarity * 0.3)

        # Add some randomness to simulate different conclusions
        random_factor = random.uniform(-0.2, 0.2)

        # Final score clamped between 0 and 1
        return max(0.0, min(1.0, base_score + random_factor))

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two text strings.
        In a real implementation, this would use more sophisticated NLP techniques.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score (0.0-1.0)
        """
        # Convert to lowercase and split into words
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        # Simple Jaccard similarity
        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union)

    def _extract_claim(self, article: Dict[str, Any]) -> str:
        """
        Extract the main claim from an article.
        In a real implementation, this would use NLP to extract key claims.

        Args:
            article: Article data

        Returns:
            Extracted claim
        """
        abstract = article.get('abstract', '')

        # Simple heuristic to extract a sentence that might contain a claim
        sentences = abstract.split('.')

        # Look for sentences with likely claim indicators
        claim_indicators = [
            "conclude", "suggests", "demonstrates", "shows", "indicates",
            "found", "effective", "ineffective", "significant", "recommend"
        ]

        for sentence in sentences:
            sentence = sentence.strip()
            if any(indicator in sentence.lower() for indicator in claim_indicators) and len(sentence) > 20:
                return sentence

        # Fallback to the last sentence or a portion of the abstract
        if sentences and len(sentences[-1].strip()) > 20:
            return sentences[-1].strip()
        elif len(abstract) > 50:
            return abstract[:150] + "..." if len(abstract) > 150 else abstract
        else:
            return "No clear claim identified in abstract"

    def _determine_contradiction_type(self, article1: Dict[str, Any], article2: Dict[str, Any]) -> str:
        """
        Determine the type of contradiction between two articles.

        Args:
            article1: First article data
            article2: Second article data

        Returns:
            Contradiction type
        """
        # In a real implementation, this would analyze the specific nature of the contradiction
        import random
        contradiction_types = [
            "Outcome contradiction",
            "Methodological contradiction",
            "Population-based contradiction",
            "Temporal contradiction",
            "Dosage contradiction",
            "Risk assessment contradiction"
        ]

        return random.choice(contradiction_types)

    def _generate_contradiction_explanation(
        self,
        article1: Dict[str, Any],
        article2: Dict[str, Any],
        standardized_claim1: str,
        standardized_claim2: str,
        score: float
    ) -> str:
        """
        Generate an explanation for the contradiction between two articles.

        Args:
            article1: First article data
            article2: Second article data
            standardized_claim1: Standardized claim from first article
            standardized_claim2: Standardized claim from second article
            score: Contradiction score

        Returns:
            Explanation text
        """
        # In a real implementation, this would generate a detailed explanation
        # based on the specific claims and evidence in each article

        title1 = article1.get('title', 'first study')
        title2 = article2.get('title', 'second study')

        explanation = f"These studies present contradictory findings. "

        if score > 0.9:
            explanation += f"The {title1} directly contradicts the findings of {title2} regarding efficacy and outcomes."
        elif score > 0.8:
            explanation += f"While {title1} presents strong evidence for effectiveness, {title2} shows significant limitations or negative results."
        elif score > 0.7:
            explanation += f"The studies differ in their conclusions about effectiveness, with methodological differences that may explain the discrepancy."
        else:
            explanation += f"The studies show some level of disagreement in their findings, though the contradiction is moderate."

        # Add information about standardized claims if available
        if standardized_claim1 != article1.get('claim', '') or standardized_claim2 != article2.get('claim', ''):
            explanation += f"\n\nWhen standardized using medical terminology, the core contradiction involves: '{standardized_claim1}' versus '{standardized_claim2}'."

        return explanation

    def _find_supporting_trials(self, contradiction_pair: Dict[str, Any], trials: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Find clinical trials that support or refute the contradiction.

        Args:
            contradiction_pair: The contradiction pair data
            trials: List of clinical trials

        Returns:
            Supporting trials with evidence classification
        """
        supporting_trials = []

        # No trials to process
        if not trials:
            return supporting_trials

        claim1 = contradiction_pair["article1"]["standardized_claim"] or contradiction_pair["article1"]["claim"]
        claim2 = contradiction_pair["article2"]["standardized_claim"] or contradiction_pair["article2"]["claim"]

        # Process each trial
        for trial in trials:
            # Calculate similarity to each claim
            similarity1 = self._calculate_text_similarity(
                claim1,
                trial.get("BriefTitle", "")
            )

            similarity2 = self._calculate_text_similarity(
                claim2,
                trial.get("BriefTitle", "")
            )

            # Skip trials with low similarity to both claims
            if max(similarity1, similarity2) < 0.1:
                continue

            # Determine which claim the trial supports more
            supports_claim1 = similarity1 > similarity2

            # Only include the most relevant trials
            if len(supporting_trials) < 5:
                supporting_trials.append({
                    "nct_id": trial.get("NCTId", ""),
                    "title": trial.get("BriefTitle", ""),
                    "phase": trial.get("Phase", ""),
                    "status": trial.get("OverallStatus", ""),
                    "enrollment": trial.get("EnrollmentCount", 0),
                    "supports_claim": 1 if supports_claim1 else 2,
                    "similarity_score": similarity1 if supports_claim1 else similarity2
                })

        # Sort by similarity score descending
        supporting_trials.sort(key=lambda x: x["similarity_score"], reverse=True)

        return supporting_trials

    def _summarize_trials(self, trials: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Summarize clinical trials data.

        Args:
            trials: List of clinical trials

        Returns:
            Summary statistics
        """
        if not trials:
            return {"count": 0}

        # Count trials by phase
        phase_counts = {}
        status_counts = {}

        for trial in trials:
            # Count phases
            phase = trial.get("Phase", "Unknown")
            phase_counts[phase] = phase_counts.get(phase, 0) + 1

            # Count statuses
            status = trial.get("OverallStatus", "Unknown")
            status_counts[status] = status_counts.get(status, 0) + 1

        return {
            "count": len(trials),
            "phases": phase_counts,
            "statuses": status_counts,
            "most_recent": trials[0] if trials else None
        }

# Dependency to get the enhanced medical contradiction service
def get_enhanced_contradiction_service() -> EnhancedContradictionService:
    """Factory function to create and provide an EnhancedContradictionService instance."""
    return EnhancedContradictionService()