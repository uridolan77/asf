from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
import random

from .auth import get_current_user, get_db
from models.user import User
from .models import ContradictionAnalysisRequest, ScreeningRequest, BiasAssessmentRequest

router = APIRouter()

@router.post("/api/medical/analysis/contradictions")
def analyze_contradictions(
    request: ContradictionAnalysisRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Analyze contradictions in medical research for a given query.
    """
    # Mock data for contradictions analysis
    models_used = []
    if request.use_biomedlm:
        models_used.append("BioMedLM")
    if request.use_tsmixer:
        models_used.append("TSMixer")
    if request.use_lorentz:
        models_used.append("Lorentz")

    contradiction_pairs = [
        {
            "article1": {
                "id": f"article_a_{i}",
                "title": f"Study supporting {request.query} - Part {i}",
                "claim": f"Evidence shows that {request.query} is effective for treating certain conditions."
            },
            "article2": {
                "id": f"article_b_{i}",
                "title": f"Study refuting {request.query} - Part {i}",
                "claim": f"No significant evidence was found to support {request.query} as an effective treatment."
            },
            "contradiction_score": round(random.uniform(request.threshold, 0.99), 2),
            "explanation": f"These studies present contradictory findings about the efficacy of {request.query}."
        }
        for i in range(1, min(request.max_results + 1, 11))
    ]

    return {
        "success": True,
        "message": f"Identified {len(contradiction_pairs)} contradiction pairs",
        "data": {
            "contradiction_pairs": contradiction_pairs,
            "query": request.query,
            "threshold": request.threshold,
            "models_used": models_used
        }
    }

@router.get("/api/medical/analysis/cap")
def analyze_cap(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get analysis specific to Community Acquired Pneumonia (CAP).
    """
    # Mock data for CAP analysis
    treatments = [
        {
            "name": "Amoxicillin",
            "efficacy_score": 0.85,
            "recommendation_level": "Strong",
            "patient_groups": ["Adults", "Children > 5 years"],
            "contraindications": ["Penicillin allergy"]
        },
        {
            "name": "Azithromycin",
            "efficacy_score": 0.78,
            "recommendation_level": "Moderate",
            "patient_groups": ["Adults", "Children > 2 years"],
            "contraindications": ["Macrolide allergy", "Certain cardiac conditions"]
        },
        {
            "name": "Respiratory support",
            "efficacy_score": 0.92,
            "recommendation_level": "Strong",
            "patient_groups": ["All patients with respiratory distress"],
            "contraindications": []
        },
        {
            "name": "Corticosteroids",
            "efficacy_score": 0.65,
            "recommendation_level": "Conditional",
            "patient_groups": ["Severe cases", "Patients with inflammatory response"],
            "contraindications": ["Active untreated infections"]
        }
    ]

    diagnostic_criteria = [
        {
            "criterion": "Chest X-ray confirmation",
            "sensitivity": 0.87,
            "specificity": 0.83,
            "recommendation": "Strongly recommended for diagnosis"
        },
        {
            "criterion": "Clinical symptoms (fever, cough, dyspnea)",
            "sensitivity": 0.92,
            "specificity": 0.61,
            "recommendation": "Essential for initial assessment"
        },
        {
            "criterion": "Sputum culture",
            "sensitivity": 0.65,
            "specificity": 0.95,
            "recommendation": "Recommended for pathogen identification"
        },
        {
            "criterion": "Blood tests (WBC count, CRP)",
            "sensitivity": 0.81,
            "specificity": 0.74,
            "recommendation": "Recommended to assess severity"
        }
    ]

    recent_findings = [
        {
            "title": "Antibiotic resistance trends in CAP",
            "summary": "Increasing resistance to macrolides observed in Streptococcus pneumoniae isolates.",
            "year": 2024,
            "impact": "High",
            "source": "International Journal of Antimicrobial Agents"
        },
        {
            "title": "Procalcitonin-guided therapy in CAP",
            "summary": "Procalcitonin-guided antibiotic therapy reduced antibiotic exposure without affecting outcomes.",
            "year": 2024,
            "impact": "Moderate",
            "source": "American Journal of Respiratory and Critical Care Medicine"
        },
        {
            "title": "CAP in the post-COVID era",
            "summary": "Changes in pathogen distribution and disease severity noted since the COVID-19 pandemic.",
            "year": 2023,
            "impact": "High",
            "source": "Lancet Respiratory Medicine"
        }
    ]

    return {
        "success": True,
        "message": "Retrieved CAP analysis data",
        "data": {
            "treatments": treatments,
            "diagnostic_criteria": diagnostic_criteria,
            "recent_findings": recent_findings,
            "meta": {
                "last_updated": "2025-04-15",
                "guidelines_source": "Infectious Diseases Society of America / American Thoracic Society"
            }
        }
    }

@router.post("/api/medical/screening/prisma")
def screen_articles(
    request: ScreeningRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Screen articles using PRISMA framework.
    """
    # Mock data for article screening
    stages = {
        "identification": "Initial database search",
        "screening": "Title/abstract screening",
        "eligibility": "Full-text assessment",
        "included": "Final inclusion"
    }

    articles = [
        {
            "id": f"screening_{i}",
            "title": f"Screening Study on {request.query} - Part {i}",
            "source": "PubMed",
            "year": 2023 - i % 5,
            "stage": request.stage,
            "status": random.choice(["included", "excluded"]),
            "exclusion_reason": None if random.random() > 0.4 else random.choice([
                "Wrong population", "Wrong intervention", "Wrong outcome", "Wrong study design"
            ]),
            "screening_score": round(random.uniform(0.5, 0.99), 2)
        }
        for i in range(1, min(request.max_results + 1, 21))
    ]

    # Filter out excluded articles if at the included stage
    if request.stage == "included":
        articles = [a for a in articles if a["status"] == "included"]

    prisma_stats = {
        "identification": random.randint(200, 500),
        "screening": random.randint(100, 200),
        "eligibility": random.randint(30, 100),
        "included": len([a for a in articles if a["status"] == "included"])
    }

    return {
        "success": True,
        "message": f"Screened articles at {stages.get(request.stage, request.stage)} stage",
        "data": {
            "articles": articles,
            "query": request.query,
            "stage": request.stage,
            "prisma_stats": prisma_stats,
            "total_results": len(articles)
        }
    }

@router.post("/api/medical/screening/bias-assessment")
def assess_bias(
    request: BiasAssessmentRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Assess bias in medical research articles.
    """
    # Default domains if not provided
    domains = request.domains or [
        "Selection bias",
        "Performance bias",
        "Detection bias",
        "Attrition bias",
        "Reporting bias"
    ]

    articles = [
        {
            "id": f"bias_{i}",
            "title": f"Bias Assessment Study on {request.query} - Part {i}",
            "journal": "Journal of Evidence-Based Medicine",
            "year": 2023 - i % 5,
            "bias_assessment": {
                domain: {
                    "risk": random.choice(["Low", "Moderate", "High"]),
                    "explanation": f"Assessment of {domain.lower()} for this study."
                }
                for domain in domains
            },
            "overall_risk": random.choice(["Low", "Moderate", "High"]),
            "assessment_tool": "Cochrane Risk of Bias Tool"
        }
        for i in range(1, min(request.max_results + 1, 21))
    ]

    summary = {
        domain: {
            "Low": len([a for a in articles if a["bias_assessment"][domain]["risk"] == "Low"]),
            "Moderate": len([a for a in articles if a["bias_assessment"][domain]["risk"] == "Moderate"]),
            "High": len([a for a in articles if a["bias_assessment"][domain]["risk"] == "High"])
        }
        for domain in domains
    }

    return {
        "success": True,
        "message": f"Assessed bias in {len(articles)} articles",
        "data": {
            "articles": articles,
            "query": request.query,
            "domains": domains,
            "summary": summary,
            "total_results": len(articles)
        }
    }
