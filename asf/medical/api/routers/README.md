# ASF Medical Research Synthesizer API Routers

This directory contains the API routers for the ASF Medical Research Synthesizer.

## Authentication Router (`auth.py`)

The authentication router provides endpoints for user authentication and authorization.

### Endpoints

- `POST /auth/token`: Get an access token
- `POST /auth/register`: Register a new user
- `GET /auth/me`: Get current user information
- `PUT /auth/me`: Update current user information

## Search Router (`search.py`)

The search router provides endpoints for searching medical literature.

### Endpoints

- `POST /search`: Search for medical literature
- `POST /search/pico`: Search using PICO framework
- `GET /search/result/{result_id}`: Get a search result
- `GET /search/history`: Get search history

## Analysis Router (`analysis.py`)

The analysis router provides endpoints for analyzing medical literature.

### Endpoints

- `POST /analysis/contradictions`: Analyze contradictions in literature
- `POST /analysis/cap`: Analyze clinical applicability
- `GET /analysis/result/{analysis_id}`: Get an analysis result
- `GET /analysis/history`: Get analysis history

## Knowledge Base Router (`knowledge_base.py`)

The knowledge base router provides endpoints for managing knowledge bases.

### Endpoints

- `POST /knowledge-base`: Create a knowledge base
- `GET /knowledge-base/{name}`: Get a knowledge base
- `GET /knowledge-base`: List knowledge bases
- `PUT /knowledge-base/{kb_id}`: Update a knowledge base
- `DELETE /knowledge-base/{kb_id}`: Delete a knowledge base

## Export Router (`export.py`)

The export router provides endpoints for exporting data.

### Endpoints

- `POST /export/csv`: Export to CSV
- `POST /export/excel`: Export to Excel
- `POST /export/pdf`: Export to PDF
- `POST /export/json`: Export to JSON

## Screening Router (`screening.py`)

The screening router provides endpoints for PRISMA-guided screening and bias assessment.

### Endpoints

- `POST /screening/prisma`: Screen articles according to PRISMA guidelines
- `POST /screening/bias-assessment`: Assess risk of bias in articles
- `GET /screening/flow-diagram`: Get PRISMA flow diagram data

### PRISMA Screening

The PRISMA screening endpoint (`POST /screening/prisma`) screens articles according to PRISMA guidelines, returning the screening results and PRISMA flow data.

#### Request

```json
{
  "query": "statin therapy cardiovascular",
  "max_results": 20,
  "stage": "screening",
  "criteria": {
    "include": ["randomized controlled trial", "cardiovascular outcomes"],
    "exclude": ["animal study", "in vitro"]
  }
}
```

#### Response

```json
{
  "query": "statin therapy cardiovascular",
  "stage": "screening",
  "total_articles": 15,
  "included": 8,
  "excluded": 5,
  "uncertain": 2,
  "results": [
    {
      "article_id": "12345",
      "title": "Randomized controlled trial of statin therapy",
      "stage": "screening",
      "decision": "include",
      "confidence": 0.8,
      "matched_include_criteria": ["randomized controlled trial"],
      "matched_exclude_criteria": [],
      "notes": "Included due to: randomized controlled trial"
    },
    ...
  ],
  "flow_data": {
    "identification": {
      "records_identified": 20,
      "records_removed_before_screening": 5
    },
    "screening": {
      "records_screened": 15,
      "records_excluded": 5
    },
    "eligibility": {
      "full_text_assessed": 10,
      "full_text_excluded": 2,
      "exclusion_reasons": {}
    },
    "included": {
      "studies_included": 8
    }
  }
}
```

### Bias Assessment

The bias assessment endpoint (`POST /screening/bias-assessment`) assesses the risk of bias in articles, returning the assessment results.

#### Request

```json
{
  "query": "statin therapy cardiovascular",
  "max_results": 20,
  "domains": ["randomization", "blinding", "allocation_concealment", "sample_size", "attrition"]
}
```

#### Response

```json
{
  "query": "statin therapy cardiovascular",
  "total_articles": 15,
  "low_risk": 5,
  "moderate_risk": 7,
  "high_risk": 2,
  "unclear_risk": 1,
  "results": [
    {
      "study_id": "12345",
      "title": "Randomized controlled trial of statin therapy",
      "assessment": {
        "randomization": {
          "risk": "low",
          "positive_score": 1.0,
          "negative_score": 0.0,
          "evidence": [
            {
              "text": "randomized",
              "context": "We conducted a randomized controlled trial with 1000 patients.",
              "type": "positive",
              "weight": 1.0
            }
          ]
        },
        ...
        "overall": {
          "risk": "low",
          "summary": "0 domains at high risk of bias, 1 domains unclear",
          "high_risk_domains": [],
          "unclear_domains": ["allocation_concealment"]
        }
      }
    },
    ...
  ]
}
```

## Contradiction Router (`contradiction.py`)

The contradiction router provides endpoints for enhanced contradiction detection.

### Endpoints

- `POST /contradiction/analyze`: Analyze contradictions in literature
- `POST /contradiction/detect`: Detect contradiction between two claims

### Contradiction Analysis

The contradiction analysis endpoint (`POST /contradiction/analyze`) analyzes contradictions in literature matching the query, using enhanced contradiction detection methods.

#### Request

```json
{
  "query": "statin therapy cardiovascular",
  "max_results": 20,
  "threshold": 0.7,
  "use_all_methods": true
}
```

#### Response

```json
{
  "query": "statin therapy cardiovascular",
  "total_articles": 15,
  "contradictions_found": 3,
  "contradiction_types": {
    "direct": 1,
    "negation": 1,
    "methodological": 1
  },
  "contradictions": [
    {
      "article1": {
        "id": "12345",
        "title": "Statin therapy reduces cardiovascular events",
        "claim": "Statin therapy reduces the risk of cardiovascular events in patients with high cholesterol."
      },
      "article2": {
        "id": "67890",
        "title": "No benefit of statin therapy in low-risk patients",
        "claim": "Statin therapy does not reduce the risk of cardiovascular events in patients with low cardiovascular risk."
      },
      "contradiction_score": 0.85,
      "contradiction_type": "methodological",
      "confidence": "high",
      "explanation": "The claims contradict each other with methodological differences: different populations (high cholesterol vs low cardiovascular risk)."
    },
    ...
  ],
  "analysis_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

### Contradiction Detection

The contradiction detection endpoint (`POST /contradiction/detect`) detects contradiction between two claims using enhanced contradiction detection methods.

#### Request

```json
{
  "claim1": "Statin therapy reduces the risk of cardiovascular events in patients with high cholesterol.",
  "claim2": "Statin therapy does not reduce the risk of cardiovascular events in patients with high cholesterol.",
  "metadata1": {
    "publication_date": "2020-01-01",
    "study_design": "randomized controlled trial",
    "sample_size": 1000,
    "p_value": 0.001
  },
  "metadata2": {
    "publication_date": "2021-06-15",
    "study_design": "randomized controlled trial",
    "sample_size": 2000,
    "p_value": 0.45
  },
  "use_all_methods": true
}
```

#### Response

```json
{
  "claim1": "Statin therapy reduces the risk of cardiovascular events in patients with high cholesterol.",
  "claim2": "Statin therapy does not reduce the risk of cardiovascular events in patients with high cholesterol.",
  "is_contradiction": true,
  "contradiction_score": 0.92,
  "contradiction_type": "negation",
  "confidence": "high",
  "explanation": "Claim 2 is a negation of Claim 1 with similarity 0.92.",
  "methods_used": ["biomedlm", "negation", "temporal", "methodological", "statistical"],
  "details": {
    "direct": {
      "is_contradiction": true,
      "score": 0.85,
      "confidence": "high",
      "explanation": "The claims directly contradict each other with a score of 0.85."
    },
    "negation": {
      "is_contradiction": true,
      "score": 0.92,
      "confidence": "high",
      "explanation": "Claim 2 is a negation of Claim 1 with similarity 0.92."
    },
    ...
  }
}
```
