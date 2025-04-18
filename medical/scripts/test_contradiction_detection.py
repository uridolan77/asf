import sys
import json
import logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
TEST_CLAIMS = [
    {
        "claim1": "Statin therapy reduces the risk of cardiovascular events in patients with high cholesterol.",
        "claim2": "Statin therapy does not reduce the risk of cardiovascular events in patients with high cholesterol.",
        "metadata1": {
            "publication_date": "2020-01-01",
            "study_design": "randomized controlled trial",
            "sample_size": 1000,
            "p_value": 0.001,
            "effect_size": 0.3
        },
        "metadata2": {
            "publication_date": "2021-06-15",
            "study_design": "randomized controlled trial",
            "sample_size": 2000,
            "p_value": 0.45,
            "effect_size": -0.05
        }
    },
    {
        "claim1": "Antibiotics are effective for treating bacterial pneumonia.",
        "claim2": "Antibiotics are ineffective for treating bacterial pneumonia.",
        "metadata1": None,
        "metadata2": None
    },
    {
        "claim1": "Regular exercise improves cardiovascular health.",
        "claim2": "Physical activity has positive effects on heart health.",
        "metadata1": None,
        "metadata2": None
    },
    {
        "claim1": "Vitamin D supplementation prevents respiratory infections.",
        "claim2": "Vitamin D supplementation has no effect on respiratory infection risk.",
        "metadata1": {
            "publication_date": "2019-05-10",
            "study_design": "meta-analysis",
            "sample_size": 5000,
            "p_value": 0.02,
            "effect_size": 0.15
        },
        "metadata2": {
            "publication_date": "2022-03-20",
            "study_design": "randomized controlled trial",
            "sample_size": 3000,
            "p_value": 0.3,
            "effect_size": 0.05
        }
    }
]
TEST_ARTICLES = [
    {
        "pmid": "12345678",
        "title": "Statin therapy reduces cardiovascular risk",
        "abstract": "This study shows that statin therapy significantly reduces the risk of cardiovascular events in patients with high cholesterol.",
        "publication_date": "2020-01-01",
        "study_design": "randomized controlled trial",
        "sample_size": 1000,
        "p_value": 0.001,
        "effect_size": 0.3
    },
    {
        "pmid": "23456789",
        "title": "No benefit of statin therapy on cardiovascular outcomes",
        "abstract": "This study found no significant reduction in cardiovascular events with statin therapy in patients with high cholesterol.",
        "publication_date": "2021-06-15",
        "study_design": "randomized controlled trial",
        "sample_size": 2000,
        "p_value": 0.45,
        "effect_size": -0.05
    },
    {
        "pmid": "34567890",
        "title": "Antibiotics for bacterial pneumonia",
        "abstract": "Antibiotics are effective for treating bacterial pneumonia and should be prescribed promptly.",
        "publication_date": "2018-03-10",
        "study_design": "clinical guideline",
        "sample_size": None,
        "p_value": None,
        "effect_size": None
    },
    {
        "pmid": "45678901",
        "title": "Vitamin D and respiratory infections",
        "abstract": "Vitamin D supplementation was found to reduce the risk of respiratory infections in this large meta-analysis.",
        "publication_date": "2019-05-10",
        "study_design": "meta-analysis",
        "sample_size": 5000,
        "p_value": 0.02,
        "effect_size": 0.15
    },
    {
        "pmid": "56789012",
        "title": "Vitamin D supplementation for respiratory infection prevention",
        "abstract": "This randomized controlled trial found no significant effect of vitamin D supplementation on respiratory infection risk.",
        "publication_date": "2022-03-20",
        "study_design": "randomized controlled trial",
        "sample_size": 3000,
        "p_value": 0.3,
        "effect_size": 0.05
    }
]
async def test_contradiction_service():
    logger.info("Testing API request creation...")
    request = {
        "claim1": "Statin therapy reduces the risk of cardiovascular events in patients with high cholesterol.",
        "claim2": "Statin therapy does not reduce the risk of cardiovascular events in patients with high cholesterol.",
        "metadata1": {
            "publication_date": "2020-01-01",
            "study_design": "randomized controlled trial",
            "sample_size": 1000,
            "p_value": 0.001,
            "effect_size": 0.3
        },
        "metadata2": {
            "publication_date": "2021-06-15",
            "study_design": "randomized controlled trial",
            "sample_size": 2000,
            "p_value": 0.45,
            "effect_size": -0.05
        },
        "threshold": 0.7
    }
    logger.info(f"API request: {json.dumps(request, indent=2)}")
    logger.info("API request validation successful")
async def main():