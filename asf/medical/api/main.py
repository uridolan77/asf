"""
RESTful API for Medical Research Synthesizer

This module provides a comprehensive API for accessing all features of the
enhanced medical research synthesizer.
"""

import os
import json
from typing import List, Dict, Optional, Any
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel
import tempfile
import shutil

# Import our enhanced synthesizer components
from asf.medical.data_ingestion_layer.enhanced_medical_research_synthesizer import EnhancedMedicalResearchSynthesizer
from asf.medical.api.export_utils import export_to_csv, export_to_excel, export_to_pdf, export_to_json
from asf.medical.data_ingestion_layer.query_builder import MedicalCondition, MedicalIntervention, OutcomeMetric, StudyDesign

# Initialize the API
app = FastAPI(
    title="Medical Research Synthesizer API",
    description="API for searching, analyzing and synthesizing medical research literature",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Initialize our synthesizer
synthesizer = EnhancedMedicalResearchSynthesizer(
    email=os.getenv("NCBI_EMAIL", "your.email@example.com"),
    api_key=os.getenv("NCBI_API_KEY"),
    impact_factor_source=os.getenv("IMPACT_FACTOR_SOURCE", "journal_impact_factors.csv")
)

# Define Pydantic models for request/response
class QueryRequest(BaseModel):
    query: str
    max_results: int = 20

class SearchResponse(BaseModel):
    query: str
    total_count: int
    results: List[Dict[str, Any]]

class ContradictionAnalysisRequest(BaseModel):
    query: str
    max_results: int = 20
    use_biomedlm: bool = True
    threshold: float = 0.7

class ContradictionAnalysisResponse(BaseModel):
    total_articles: int
    num_contradictions: int
    contradictions: List[Dict[str, Any]]
    by_topic: Dict[str, List[Dict[str, Any]]]
    detection_method: str = "keyword"

class KnowledgeBaseRequest(BaseModel):
    name: str
    query: str
    schedule: str = "weekly"
    max_results: int = 100

class KnowledgeBaseResponse(BaseModel):
    name: str
    query: str
    kb_file: str
    initial_results: int
    update_schedule: str
    created_date: str

class PICORequest(BaseModel):
    condition: str
    interventions: List[str] = []
    outcomes: List[str] = []
    population: Optional[str] = None
    study_design: Optional[str] = None
    years: int = 5
    max_results: int = 20

# Define API routes
@app.get("/", tags=["General"])
async def root():
    """Welcome endpoint with basic API information"""
    return {
        "message": "Welcome to the Medical Research Synthesizer API",
        "version": "1.0.0",
        "documentation": "/docs",
        "endpoints": {
            "search": "POST /search",
            "analyze-contradictions": "POST /analyze-contradictions",
            "knowledge-base": "POST /knowledge-base, GET /knowledge-base/{name}",
            "cap-analysis": "GET /cap-analysis",
            "export": "GET /export/{format}"
        }
    }

@app.post("/search", response_model=SearchResponse, tags=["Search"])
async def search(request: QueryRequest):
    """
    Search PubMed with the given query and return enriched results.

    This endpoint performs a search using the enhanced NCBIClient and enriches
    the results with metadata such as impact factors, authority scores,
    and standardized dates.
    """
    try:
        results = synthesizer.search_and_enrich(query=request.query, max_results=request.max_results)
        return {
            "query": request.query,
            "total_count": len(results),
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-contradictions", response_model=ContradictionAnalysisResponse, tags=["Analysis"])
async def analyze_contradictions(request: ContradictionAnalysisRequest):
    """
    Analyze contradictions in literature matching the query.

    This endpoint searches for and identifies potential contradictions in the
    medical literature, comparing the authority of contradictory findings.

    It can use BioMedLM for more accurate contradiction detection if enabled.
    """
    try:
        analysis = synthesizer.search_and_analyze_contradictions(
            query=request.query,
            max_results=request.max_results,
            use_biomedlm=request.use_biomedlm,
            threshold=request.threshold
        )
        return analysis
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/pico-search", response_model=SearchResponse, tags=["Search"])
async def pico_search(request: PICORequest):
    """
    Search using PICO (Population, Intervention, Comparison, Outcome) framework.

    This endpoint allows users to construct a search using the structured PICO
    framework common in evidence-based medicine.
    """
    try:
        # Create query builder
        query_builder = synthesizer.create_query()

        # Add condition
        query_builder.add_condition(MedicalCondition(request.condition))

        # Add interventions
        for intervention in request.interventions:
            query_builder.add_intervention(MedicalIntervention(intervention))

        # Add outcomes
        for outcome in request.outcomes:
            query_builder.add_outcome(OutcomeMetric(outcome))

        # Add population if provided
        if request.population:
            query_builder.add_population(request.population)

        # Add study design if provided
        if request.study_design:
            query_builder.add_study_design(StudyDesign(request.study_design))

        # Set date range
        query_builder.last_n_years(request.years)

        # English only
        query_builder.english_only()

        # Execute search
        results = synthesizer.search_and_enrich(
            query_builder=query_builder,
            max_results=request.max_results
        )

        # Get the generated query
        query = query_builder.build_pico_query()

        return {
            "query": query,
            "total_count": len(results),
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/knowledge-base", response_model=KnowledgeBaseResponse, tags=["Knowledge Base"])
async def create_knowledge_base(request: KnowledgeBaseRequest, background_tasks: BackgroundTasks):
    """
    Create and schedule updates for a knowledge base.

    This endpoint creates a new knowledge base for tracking publications on a
    specific topic and schedules regular updates to keep it current.
    """
    try:
        # Create knowledge base
        kb_info = synthesizer.create_and_update_knowledge_base(
            name=request.name,
            query=request.query,
            schedule=request.schedule,
            max_results=request.max_results
        )
        return kb_info
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/knowledge-base/{name}", tags=["Knowledge Base"])
async def get_knowledge_base(name: str):
    """
    Get articles from a knowledge base.

    This endpoint retrieves all articles stored in a specific knowledge base.
    """
    try:
        articles = synthesizer.get_knowledge_base(name)
        return {
            "name": name,
            "articles": articles,
            "count": len(articles)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/knowledge-bases", tags=["Knowledge Base"])
async def list_knowledge_bases():
    """
    List all available knowledge bases.

    This endpoint returns a list of all knowledge bases that have been created.
    """
    try:
        # In a real implementation, this would scan the knowledge base directory
        # For now, return a simple response based on the storage directory
        kb_dir = synthesizer.kb_dir
        knowledge_bases = []

        if os.path.exists(kb_dir):
            for filename in os.listdir(kb_dir):
                if filename.endswith('.json'):
                    kb_name = filename.replace('.json', '')

                    # Get basic stats
                    kb_path = os.path.join(kb_dir, filename)
                    stat = os.stat(kb_path)

                    kb_info = {
                        "name": kb_name,
                        "file": kb_path,
                        "size": stat.st_size,
                        "last_modified": stat.st_mtime
                    }

                    # Try to get more details
                    try:
                        articles = synthesizer.get_knowledge_base(kb_name)
                        kb_info["article_count"] = len(articles)
                    except:
                        kb_info["article_count"] = "unknown"

                    knowledge_bases.append(kb_info)

        return {
            "knowledge_bases": knowledge_bases,
            "count": len(knowledge_bases)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/cap-analysis", tags=["Specialized Analysis"])
async def cap_analysis():
    """
    Perform specialized analysis of CAP literature.

    This endpoint executes a predefined analysis of contradictions in
    Community-Acquired Pneumonia treatment literature.
    """
    try:
        analysis = synthesizer.search_cap_contradictory_treatments()
        return analysis
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/cap-duration-vs-agent", tags=["Specialized Analysis"])
async def cap_duration_vs_agent():
    """
    Compare treatment duration vs agent choice for CAP.

    This endpoint performs specialized analysis comparing treatment duration
    versus agent choice for Community-Acquired Pneumonia.
    """
    try:
        analysis = synthesizer.cap_duration_vs_agent_analysis()
        return analysis
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/export/{format}", tags=["Export"])
async def export_results(
    format: str,
    query: str = Query(..., description="Search query"),
    max_results: int = Query(20, description="Maximum number of results")
):
    """
    Export search results in various formats.

    This endpoint allows downloading search results in JSON, CSV, Excel, or PDF formats.
    """
    if format not in ["json", "csv", "excel", "pdf"]:
        raise HTTPException(status_code=400, detail=f"Unsupported format: {format}")

    try:
        # Get search results
        results = synthesizer.search_and_enrich(query=query, max_results=max_results)

        # Export based on format
        if format == "json":
            json_data = export_to_json(results)
            return StreamingResponse(
                iter([json_data]),
                media_type="application/json",
                headers={"Content-Disposition": f"attachment; filename=results_{query.replace(' ', '_')}.json"}
            )
        elif format == "csv":
            csv_data = export_to_csv(results)
            return StreamingResponse(
                iter([csv_data.getvalue()]),
                media_type="text/csv",
                headers={"Content-Disposition": f"attachment; filename=results_{query.replace(' ', '_')}.csv"}
            )
        elif format == "excel":
            excel_data = export_to_excel(results)
            return StreamingResponse(
                excel_data,
                media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                headers={"Content-Disposition": f"attachment; filename=results_{query.replace(' ', '_')}.xlsx"}
            )
        elif format == "pdf":
            pdf_data = export_to_pdf(results, title=f"Search Results: {query}")
            return StreamingResponse(
                iter([pdf_data.getvalue()]),
                media_type="application/pdf",
                headers={"Content-Disposition": f"attachment; filename=results_{query.replace(' ', '_')}.pdf"}
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-impact-factors", tags=["Configuration"])
async def upload_impact_factors(file: UploadFile = File(...)):
    """
    Upload a new journal impact factors CSV file.

    This endpoint allows uploading a custom impact factor database to enhance
    the authority scoring.
    """
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_path = temp_file.name

        # Update the impact factor source in the synthesizer
        synthesizer.metadata_extractor = PublicationMetadataExtractor(
            impact_factor_source=temp_path
        )

        return {
            "message": "Impact factor file uploaded successfully",
            "filename": file.filename,
            "journal_count": len(synthesizer.metadata_extractor.impact_factors)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        file.file.close()

# Run with: uvicorn api:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)