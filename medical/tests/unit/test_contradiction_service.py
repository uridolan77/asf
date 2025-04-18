"""Unit tests for the ContradictionService.

This module provides unit tests for the ContradictionService class.
"""

import pytest
import logging
from typing import Dict, Any, List

from ...ml.services.contradiction_service import ContradictionService

logger = logging.getLogger(__name__)


@pytest.mark.unit
@pytest.mark.service
@pytest.mark.async_test
class TestContradictionService:
    """Test cases for ContradictionService."""

    @pytest.fixture
    def sample_claims(self):
        """Create sample claims for testing."""
        return [
            {
                "id": "claim1",
                "text": "Aspirin reduces the risk of heart attack.",
                "metadata": {
                    "publication_date": "2020-01-01T00:00:00Z",
                    "study_type": "randomized controlled trial"
                }
            },
            {
                "id": "claim2",
                "text": "Aspirin does not significantly affect heart attack risk.",
                "metadata": {
                    "publication_date": "2010-01-01T00:00:00Z",
                    "study_type": "observational study"
                }
            }
        ]

    @pytest.mark.asyncio
    async def test_detect_contradiction(
        self,
        contradiction_service: ContradictionService,
        sample_claims: List[Dict[str, Any]]
    ):
        """Test detecting contradiction between two claims."""
        claim1 = sample_claims[0]["text"]
        claim2 = sample_claims[1]["text"]

        result = await contradiction_service.detect_contradiction(claim1, claim2)

        assert isinstance(result, dict)
        assert "is_contradiction" in result
        assert "score" in result
        assert "explanation" in result

    @pytest.mark.asyncio
    async def test_detect_temporal_contradiction(
        self,
        contradiction_service: ContradictionService,
        sample_claims: List[Dict[str, Any]]
    ):
        """Test detecting temporal contradiction between two claims."""
        claim1 = sample_claims[0]["text"]
        claim2 = sample_claims[1]["text"]
        date1 = sample_claims[0]["metadata"]["publication_date"]
        date2 = sample_claims[1]["metadata"]["publication_date"]

        result = await contradiction_service.detect_temporal_contradiction(
            claim1, claim2, date1, date2
        )

        assert isinstance(result, dict)
        assert "is_contradiction" in result
        assert "score" in result
        assert "explanation" in result

    @pytest.mark.asyncio
    async def test_generate_explanation(
        self,
        contradiction_service: ContradictionService,
        sample_claims: List[Dict[str, Any]]
    ):
        """Test generating explanation for a contradiction."""
        claim1 = sample_claims[0]["text"]
        claim2 = sample_claims[1]["text"]

        contradiction = await contradiction_service.detect_contradiction(claim1, claim2)
        explanation = await contradiction_service.generate_explanation(contradiction)

        assert isinstance(explanation, str)
        assert len(explanation) > 0
