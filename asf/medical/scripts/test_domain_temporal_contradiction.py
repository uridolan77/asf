import sys
import logging
from pathlib import Path
from typing import Dict, Any
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
try:
    from asf.medical.ml.services.unified_contradiction_service import ContradictionService
    from asf.medical.ml.services.temporal_service import TemporalService
    from asf.medical.ml.models.biomedlm import BioMedLMService
    USING_REAL_SERVICES = True
except ImportError:
    logger.warning("Failed to import required modules. Using mock implementations.")
    USING_REAL_SERVICES = False
    class ContradictionType:
        NONE = "none"
        DIRECT = "direct"
        NEGATION = "negation"
        STATISTICAL = "statistical"
        METHODOLOGICAL = "methodological"
        TEMPORAL = "temporal"
        UNKNOWN = "unknown"
    class ContradictionConfidence:
        HIGH = "high"
        MEDIUM = "medium"
        LOW = "low"
        UNKNOWN = "unknown"
    class MockBioMedLMService:
        """Mock BioMedLM service for testing."""
        def __init__(self):
            """Initialize the mock BioMedLM service.
    Args:
        # TODO: Add parameter descriptions
    Returns:
        # TODO: Add return description
    Args:
        # TODO: Add parameter descriptions
    Returns:
        # TODO: Add return description
    Args:
        # TODO: Add parameter descriptions
    Returns:
        # TODO: Add return description
        def __init__(self):
            """Initialize the mock temporal service.
    Args:
        # TODO: Add parameter descriptions
    Returns:
        # TODO: Add return description
    """
            logger.info("Mock temporal service initialized")
            self.domain_characteristics = {
                "oncology": {
                    "half_life": 365 * 2,  # 2 years
                    "evolution_rate": "rapid",
                    "evidence_stability": "moderate",
                    "technology_dependence": "high",
                    "description": "Cancer research evolves rapidly with new treatments and targeted therapies"
                },
                "infectious_disease": {
                    "half_life": 365 * 1,  # 1 year
                    "evolution_rate": "very_rapid",
                    "evidence_stability": "low",
                    "technology_dependence": "high",
                    "description": "Infectious disease knowledge changes quickly with emerging pathogens and resistance patterns"
                },
                "neurology": {
                    "half_life": 365 * 4,  # 4 years
                    "evolution_rate": "slow",
                    "evidence_stability": "high",
                    "technology_dependence": "moderate",
                    "description": "Neurological principles change slowly despite technological advances in imaging"
                },
                "psychiatry": {
                    "half_life": 365 * 5,  # 5 years
                    "evolution_rate": "slow",
                    "evidence_stability": "moderate",
                    "technology_dependence": "low",
                    "description": "Psychiatric knowledge evolves gradually with long-term studies and observations"
                },
                "default": {
                    "half_life": 365 * 2.5,  # 2.5 years
                    "evolution_rate": "moderate",
                    "evidence_stability": "moderate",
                    "technology_dependence": "moderate",
                    "description": "General medical knowledge with moderate evolution rate"
                }
            }
        def calculate_temporal_confidence(
            self,
            publication_date: str,
            domain: str = "default",
            reference_date: str = None,
            include_details: bool = False
        ):
        def __init__(self, biomedlm_service=None, temporal_service=None):
            """Initialize the mock contradiction service.
    Args:
        # TODO: Add parameter descriptions
    Returns:
        # TODO: Add return description
    """
            logger.info("Mock contradiction service initialized")
            self.biomedlm_service = biomedlm_service or MockBioMedLMService()
            self.temporal_service = temporal_service or MockTemporalService()
            self.thresholds = {
                ContradictionType.DIRECT: 0.7,
                ContradictionType.NEGATION: 0.8,
                ContradictionType.STATISTICAL: 0.7,
                ContradictionType.METHODOLOGICAL: 0.7,
                ContradictionType.TEMPORAL: 0.7,
            }
        async def detect_contradiction(
            self,
            claim1: str,
            claim2: str,
            metadata1: Dict[str, Any] = None,
            metadata2: Dict[str, Any] = None,
            threshold: float = 0.7,
            use_biomedlm: bool = True,
            use_temporal: bool = True,
            use_tsmixer: bool = False
        ) -> Dict[str, Any]: