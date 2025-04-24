"""
Medical Contradiction Detection service for analyzing contradictions in medical literature.
"""
import os
import sys
import logging
from typing import Dict, Any, List, Optional, Union
from fastapi import Depends, HTTPException, status

# Add the project root directory to sys.path to import the medical module
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
if project_root not in sys.path:
    sys.path.append(project_root)

# Mock implementations to avoid import errors
class ValidationError(Exception):
    """Validation error exception."""
    pass

class SearchService:
    """Mock SearchService for the Medical Research Synthesizer."""

    async def search(self, query, method="all", max_results=100, user_id=None):
        """Search for medical literature."""
        return {
            "source": method,
            "results": [
                {"id": "mock-1", "title": "Mock Result 1"},
                {"id": "mock-2", "title": "Mock Result 2"}
            ]
        }

class ContradictionVisualizer:
    """Mock ContradictionVisualizer for the Medical Research Synthesizer."""

    async def find_contradictions(self, query, results=None):
        """Find contradictions in search results."""
        return {
            "query": query,
            "contradictions": [
                {
                    "topic": "Treatment Efficacy",
                    "statements": [
                        {
                            "text": "Treatment X is effective for condition Y.",
                            "source": "mock-1",
                            "confidence": 0.85
                        },
                        {
                            "text": "Treatment X shows no significant benefit for condition Y.",
                            "source": "mock-2",
                            "confidence": 0.78
                        }
                    ]
                }
            ]
        }

    async def generate_visualization(self, contradictions, format="json"):
        """Generate a visualization of contradictions."""
        if format == "json":
            return contradictions
        elif format == "html":
            return "<html><body><h1>Mock Contradiction Visualization</h1></body></html>"
        else:
            return contradictions

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MedicalContradictionService:
    """
    Service for detecting and analyzing contradictions in medical literature.
    This provides a bridge between the BO frontend and the Medical Research contradiction detection functionality.
    """
    def __init__(self):
        """Initialize the contradiction service with dependencies"""
        # Initialize the search service to get papers for contradiction analysis
        from api.services.medical_search_service import get_medical_search_service
        self.search_service = get_medical_search_service().search_service

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
        user_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Analyze contradictions in medical literature for a given query.

        Args:
            query: Search query to find articles for contradiction analysis
            max_results: Maximum number of contradiction pairs to return
            threshold: Minimum contradiction score threshold (0.0-1.0)
            use_biomedlm: Whether to use the BioMedLM model
            use_tsmixer: Whether to use the TSMixer model
            use_lorentz: Whether to use the Lorentz model
            user_id: BO user ID

        Returns:
            Contradiction analysis results
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

            # Step 1: Search for articles related to the query
            logger.info(f"Searching for articles to analyze contradictions: {query}")
            search_results = await self.search_service.search(
                query=query,
                max_results=max_results * 5,  # Get more results to find contradictions
                user_id=user_id
            )

            if not search_results.get('results'):
                return {
                    "success": False,
                    "message": f"No articles found for query: {query}",
                    "data": None
                }

            # Step 2: Analyze contradictions in the search results
            articles = search_results['results']
            logger.info(f"Analyzing contradictions among {len(articles)} articles...")

            # In a real implementation, this would use the actual contradiction detection API
            # For now, we'll create simulated contradiction pairs

            # Step 3: Process articles to find contradicting claims
            # This implementation is a placeholder for the actual contradiction detection algorithm
            contradiction_pairs = []
            models_used = []

            if use_biomedlm:
                models_used.append("BioMedLM")
            if use_tsmixer:
                models_used.append("TSMixer")
            if use_lorentz:
                models_used.append("Lorentz")

            # Process articles in pairs to find contradictions
            # In a real implementation, this would use NLP and semantic analysis
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
                    pair_key = f"{article1['pmid']}-{article2['pmid']}"
                    if pair_key in processed_pairs:
                        continue

                    processed_pairs.add(pair_key)

                    # Calculate contradiction score (simulated)
                    contradiction_score = self._calculate_contradiction_score(article1, article2)

                    # Only include pairs with score above threshold
                    if contradiction_score >= threshold:
                        # Extract claims from abstracts (simplified)
                        claim1 = self._extract_claim(article1)
                        claim2 = self._extract_claim(article2)

                        # Generate explanation
                        explanation = self._generate_contradiction_explanation(article1, article2, contradiction_score)

                        contradiction_pairs.append({
                            "article1": {
                                "id": article1.get('pmid', f"article_a_{i}"),
                                "title": article1.get('title', f"Study on {query} - Part {i}"),
                                "authors": article1.get('authors', []),
                                "journal": article1.get('journal', ''),
                                "year": article1.get('year', ''),
                                "claim": claim1
                            },
                            "article2": {
                                "id": article2.get('pmid', f"article_b_{j}"),
                                "title": article2.get('title', f"Study on {query} - Part {j}"),
                                "authors": article2.get('authors', []),
                                "journal": article2.get('journal', ''),
                                "year": article2.get('year', ''),
                                "claim": claim2
                            },
                            "contradiction_score": contradiction_score,
                            "contradiction_type": self._determine_contradiction_type(article1, article2),
                            "explanation": explanation
                        })

            # Step 4: Generate visualizations if visualizer is available
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

            return {
                "success": True,
                "message": f"Identified {len(contradiction_pairs)} contradiction pairs",
                "data": {
                    "contradiction_pairs": contradiction_pairs,
                    "query": query,
                    "threshold": threshold,
                    "models_used": models_used,
                    "visualizations": visualization_urls,
                    "total_articles_analyzed": len(articles)
                }
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
                "default_models": ["biomedlm"]
            }
        }

    def _calculate_contradiction_score(self, article1: Dict[str, Any], article2: Dict[str, Any]) -> float:
        """
        Calculate contradiction score between two articles.
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
        score: float
    ) -> str:
        """
        Generate an explanation for the contradiction between two articles.

        Args:
            article1: First article data
            article2: Second article data
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

        return explanation

# Dependency to get the medical contradiction service
def get_medical_contradiction_service() -> MedicalContradictionService:
    """Factory function to create and provide a MedicalContradictionService instance."""
    return MedicalContradictionService()