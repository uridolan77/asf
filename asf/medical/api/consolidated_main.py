"""
Consolidated RESTful API for Medical Research Synthesizer

This module provides a comprehensive FastAPI-based API for the Enhanced Medical Research Synthesizer.
It combines functionality from both the original FastAPI and Flask implementations,
including authentication, query building, search, contradiction analysis,
knowledge base management, and export capabilities.
"""

import os
import json
import uuid
import logging
import tempfile
import shutil
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks, File, UploadFile, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, Field

# Import our enhanced synthesizer components
from asf.medical.data_ingestion_layer.enhanced_medical_research_synthesizer import EnhancedMedicalResearchSynthesizer
from asf.medical.api.export_utils import export_to_csv, export_to_excel, export_to_pdf, export_to_json
from asf.medical.data_ingestion_layer.query_builder import MedicalCondition, MedicalIntervention, OutcomeMetric, StudyDesign

# Import our API components
from asf.medical.api.ray_orchestrator_api import router as ray_router
from asf.medical.api.temporal_rollback_api import router as temporal_router
from asf.medical.api.auth_unified import (
    Token, User, authenticate_user, create_access_token, 
    get_current_active_user, register_user, users_db,
    ACCESS_TOKEN_EXPIRE_MINUTES
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("medical_api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("medical_research_api")

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

# Include our routers
app.include_router(ray_router)
app.include_router(temporal_router)

# Initialize our synthesizer
synthesizer = EnhancedMedicalResearchSynthesizer(
    email=os.getenv("NCBI_EMAIL", "your.email@example.com"),
    api_key=os.getenv("NCBI_API_KEY"),
    impact_factor_source=os.getenv("IMPACT_FACTOR_SOURCE", "journal_impact_factors.csv")
)

# In-memory storage for queries, results, and knowledge bases
query_storage = {}
result_storage = {}
kb_storage = {}

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

class UserRegistrationRequest(BaseModel):
    email: str
    password: str

# Authentication routes
@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    """User login endpoint."""
    user = authenticate_user(users_db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer", "role": user.role}

@app.post("/register", status_code=status.HTTP_201_CREATED)
async def register_new_user(request: UserRegistrationRequest):
    """User registration endpoint (only available to admins in production)."""
    if register_user(request.email, request.password):
        return {"message": "User registered successfully"}
    else:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email already registered"
        )

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
async def search(request: QueryRequest, current_user: User = Depends(get_current_active_user)):
    """
    Search PubMed with the given query and return enriched results.

    This endpoint performs a search using the enhanced NCBIClient and enriches
    the results with metadata such as impact factors, authority scores,
    and standardized dates.
    """
    try:
        results = synthesizer.search_and_enrich(query=request.query, max_results=request.max_results)
        
        # Store results for later use
        result_id = str(uuid.uuid4())
        result_storage[result_id] = {
            'query': request.query,
            'results': results,
            'timestamp': datetime.now().isoformat(),
            'user': current_user.email
        }
        
        return {
            "query": request.query,
            "total_count": len(results),
            "results": results,
            "result_id": result_id  # Return result_id for later reference
        }
    except Exception as e:
        logger.error(f"Error executing search: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-contradictions", response_model=ContradictionAnalysisResponse, tags=["Analysis"])
async def analyze_contradictions(request: ContradictionAnalysisRequest, current_user: User = Depends(get_current_active_user)):
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
        
        # Store the analysis
        analysis_id = str(uuid.uuid4())
        result_storage[analysis_id] = {
            'query': request.query,
            'analysis': analysis,
            'timestamp': datetime.now().isoformat(),
            'user': current_user.email
        }
        
        return {
            **analysis,
            "analysis_id": analysis_id  # Return analysis_id for later reference
        }
    except Exception as e:
        logger.error(f"Error analyzing contradictions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/pico-search", response_model=SearchResponse, tags=["Search"])
async def pico_search(request: PICORequest, current_user: User = Depends(get_current_active_user)):
    """
    Search using PICO (Population, Intervention, Comparison, Outcome) framework.

    This endpoint builds a structured query using the PICO framework and returns
    enriched search results.
    """
    try:
        # Create condition
        condition = MedicalCondition(request.condition)
        
        # Create interventions
        interventions = [MedicalIntervention(i) for i in request.interventions]
        
        # Create outcomes
        outcomes = [OutcomeMetric(o) for o in request.outcomes]
        
        # Create population and study design if provided
        population = None
        if request.population:
            population = request.population
            
        study_design = None
        if request.study_design:
            study_design = StudyDesign(request.study_design)
        
        # Build query
        builder = synthesizer.create_pico_query(
            condition=condition,
            interventions=interventions,
            outcomes=outcomes,
            population=population,
            study_design=study_design,
            years=request.years
        )
        
        # Store the query and builder
        query_id = str(uuid.uuid4())
        query = builder.build_pico_query(use_mesh=True)
        query_storage[query_id] = {
            'query': query,
            'builder': builder,
            'created_at': datetime.now().isoformat(),
            'user': current_user.email
        }
        
        # Execute search
        results = synthesizer.search_and_enrich(query_builder=builder, max_results=request.max_results)
        
        # Store results
        result_id = str(uuid.uuid4())
        result_storage[result_id] = {
            'query': query,
            'results': results,
            'timestamp': datetime.now().isoformat(),
            'user': current_user.email
        }
        
        return {
            "query": query,
            "total_count": len(results),
            "results": results,
            "query_id": query_id,
            "result_id": result_id
        }
    except Exception as e:
        logger.error(f"Error executing PICO search: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/query/template/{template_id}", tags=["Query"])
async def get_query_from_template(template_id: str, current_user: User = Depends(get_current_active_user)):
    """Create a query from a template."""
    try:
        builder = synthesizer.create_query_from_template(template_id)
        
        # Build the query
        query = builder.build_pico_query(use_mesh=True)
        
        # Store the query and builder
        query_id = str(uuid.uuid4())
        query_storage[query_id] = {
            'query': query,
            'builder': builder,
            'created_at': datetime.now().isoformat(),
            'user': current_user.email,
            'template': template_id
        }
        
        # Get explanation for this query
        explanation = synthesizer.query_interface.explain_query(query_type='pico', use_mesh=True)
        
        return {
            'query_id': query_id,
            'query': query,
            'components': explanation['components']
        }
    except Exception as e:
        logger.error(f"Error creating query from template: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/knowledge-base", response_model=KnowledgeBaseResponse, tags=["Knowledge Base"])
async def create_knowledge_base(request: KnowledgeBaseRequest, current_user: User = Depends(get_current_active_user)):
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
        
        # Store KB info
        kb_id = str(uuid.uuid4())
        kb_storage[kb_id] = {
            'kb_info': kb_info,
            'created_at': datetime.now().isoformat(),
            'user': current_user.email
        }
        
        return kb_info
    except Exception as e:
        logger.error(f"Error creating knowledge base: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/knowledge-base/{name}", tags=["Knowledge Base"])
async def get_knowledge_base(name: str, current_user: User = Depends(get_current_active_user)):
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
        logger.error(f"Error getting knowledge base: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/knowledge-bases", tags=["Knowledge Base"])
async def list_knowledge_bases(current_user: User = Depends(get_current_active_user)):
    """
    List all available knowledge bases.

    This endpoint returns a list of all knowledge bases that have been created.
    """
    try:
        # First check in-memory storage
        kb_list = []
        for kb_id, kb_data in kb_storage.items():
            kb_info = kb_data['kb_info']
            kb_list.append({
                'kb_id': kb_id,
                'name': kb_info['name'],
                'query': kb_info['query'],
                'initial_results': kb_info['initial_results'],
                'update_schedule': kb_info['update_schedule'],
                'created_date': kb_info['created_date']
            })
        
        # Then check file system
        kb_dir = synthesizer.kb_dir
        knowledge_bases = []
        
        if os.path.exists(kb_dir):
            for filename in os.listdir(kb_dir):
                if filename.endswith('.json'):
                    kb_name = filename.replace('.json', '')
                    
                    # Skip if already in memory
                    if any(kb['name'] == kb_name for kb in kb_list):
                        continue

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
            "in_memory_knowledge_bases": kb_list,
            "file_knowledge_bases": knowledge_bases,
            "total_count": len(kb_list) + len(knowledge_bases)
        }
    except Exception as e:
        logger.error(f"Error listing knowledge bases: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/knowledge-base/{name}/update", tags=["Knowledge Base"])
async def update_knowledge_base(name: str, current_user: User = Depends(get_current_active_user)):
    """Manually update a knowledge base."""
    try:
        # Find the KB in storage
        kb_id = None
        kb_data = None
        kb_query = None
        
        for id, data in kb_storage.items():
            if data['kb_info']['name'] == name:
                kb_id = id
                kb_data = data
                kb_query = data['kb_info']['query']
                break
        
        if not kb_query:
            # Try to find in file system
            kb_path = os.path.join(synthesizer.kb_dir, f"{name}.json")
            if not os.path.exists(kb_path):
                raise HTTPException(status_code=404, detail=f"Knowledge base '{name}' not found")
            
            # Get query from file
            try:
                with open(kb_path, 'r') as f:
                    kb_data = json.load(f)
                    kb_query = kb_data.get('query', '')
            except:
                raise HTTPException(status_code=500, detail=f"Error reading knowledge base file")
            
            if not kb_query:
                raise HTTPException(status_code=500, detail=f"Knowledge base query not found")
            
            kb_file = kb_path
        else:
            kb_file = kb_data['kb_info']['kb_file']
        
        # Update the KB
        result = synthesizer.incremental_client.search_and_update_knowledge_base(
            kb_query,
            kb_file,
            max_results=100
        )
        
        return {
            'name': name,
            'query': kb_query,
            'total_count': result['total_count'],
            'new_count': result['new_count'],
            'update_time': result['update_time']
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating knowledge base: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/cap-analysis", tags=["Specialized Analysis"])
async def cap_analysis(current_user: User = Depends(get_current_active_user)):
    """
    Perform specialized analysis of CAP literature.

    This endpoint executes a predefined analysis of contradictions in
    Community-Acquired Pneumonia treatment literature.
    """
    try:
        analysis = synthesizer.search_cap_contradictory_treatments()
        
        # Store the analysis
        analysis_id = str(uuid.uuid4())
        result_storage[analysis_id] = {
            'analysis': analysis,
            'timestamp': datetime.now().isoformat(),
            'user': current_user.email
        }
        
        return {
            'analysis_id': analysis_id,
            'total_articles': analysis['total_articles'],
            'num_contradictions': analysis['num_contradictions'],
            'contradictions_by_intervention': {k: len(v) for k, v in analysis['contradictions_by_intervention'].items()},
            'authority_analysis': analysis['authority_analysis']
        }
    except Exception as e:
        logger.error(f"Error analyzing CAP treatments: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/cap-analysis/detailed", tags=["Specialized Analysis"])
async def cap_analysis_detailed(current_user: User = Depends(get_current_active_user)):
    """
    Detailed analysis of CAP treatment duration vs agent.

    This endpoint performs a specialized analysis comparing treatment duration
    and agent effectiveness for Community-Acquired Pneumonia.
    """
    try:
        analysis = synthesizer.cap_duration_vs_agent_analysis()
        
        # Store the analysis
        analysis_id = str(uuid.uuid4())
        result_storage[analysis_id] = {
            'analysis': analysis,
            'timestamp': datetime.now().isoformat(),
            'user': current_user.email
        }
        
        return {
            'analysis_id': analysis_id,
            'duration_analysis': analysis['duration_analysis'],
            'agent_analysis': analysis['agent_analysis'],
            'cross_analysis': analysis['cross_analysis']
        }
    except Exception as e:
        logger.error(f"Error performing detailed CAP analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/export/{format}", tags=["Export"])
async def export_results(
    format: str,
    result_id: Optional[str] = None,
    query: Optional[str] = None,
    max_results: int = 20,
    current_user: User = Depends(get_current_active_user)
):
    """
    Export search results in various formats.

    This endpoint allows downloading search results in JSON, CSV, Excel, or PDF formats.
    It can use either a stored result_id or execute a new search with the provided query.
    """
    if format not in ["json", "csv", "excel", "pdf"]:
        raise HTTPException(status_code=400, detail=f"Unsupported format: {format}")

    try:
        # Get results
        if result_id and result_id in result_storage:
            # Use stored results
            result_data = result_storage[result_id]
            if 'results' in result_data:
                results = result_data['results']
                query_text = result_data.get('query', 'Stored query')
            elif 'analysis' in result_data:
                # This is a contradiction analysis
                return await export_contradiction_analysis(format, result_id, current_user)
            else:
                raise HTTPException(status_code=400, detail="Invalid result_id")
        elif query:
            # Execute a new search
            results = synthesizer.search_and_enrich(query=query, max_results=max_results)
            query_text = query
        else:
            raise HTTPException(status_code=400, detail="Either result_id or query must be provided")

        # Export based on format
        if format == "json":
            return export_to_json(results, query_text)
        elif format == "csv":
            return export_to_csv(results, query_text)
        elif format == "excel":
            return export_to_excel(results, query_text)
        elif format == "pdf":
            return export_to_pdf(results, query_text)
    except Exception as e:
        logger.error(f"Error exporting results: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def export_contradiction_analysis(
    format: str,
    analysis_id: str,
    current_user: User
):
    """Export contradiction analysis results."""
    try:
        # Get analysis
        if analysis_id not in result_storage or 'analysis' not in result_storage[analysis_id]:
            raise HTTPException(status_code=400, detail="Invalid analysis_id")
        
        analysis = result_storage[analysis_id]['analysis']
        query_text = result_storage[analysis_id].get('query', 'Contradiction Analysis')
        
        # Export based on format
        if format == "json":
            return JSONResponse(content=analysis)
        elif format == "pdf":
            # Create a PDF report
            from asf.medical.api.export_utils import export_contradiction_analysis_to_pdf
            pdf_path = export_contradiction_analysis_to_pdf(analysis, query_text)
            return FileResponse(
                path=pdf_path,
                filename=f"contradiction_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                media_type="application/pdf"
            )
        else:
            raise HTTPException(status_code=400, detail=f"Format {format} not supported for contradiction analysis")
    except Exception as e:
        logger.error(f"Error exporting contradiction analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-impact-factors", tags=["Admin"])
async def upload_impact_factors(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_active_user)
):
    """
    Upload a CSV file with journal impact factors.

    This endpoint allows administrators to upload a new impact factor file
    to update the system's journal impact factor data.
    """
    # Check if user is admin
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to perform this action"
        )
    
    try:
        # Save the uploaded file to a temporary location
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        try:
            contents = await file.read()
            with open(temp_file.name, 'wb') as f:
                f.write(contents)
        except Exception:
            raise HTTPException(status_code=500, detail="Could not save the file")
        finally:
            await file.close()
        
        # Update the metadata extractor with the new impact factors
        try:
            synthesizer.metadata_extractor.load_impact_factors(temp_file.name)
            return {"message": "Impact factors updated successfully"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error updating impact factors: {str(e)}")
        finally:
            # Clean up the temporary file
            os.unlink(temp_file.name)
    except Exception as e:
        logger.error(f"Error uploading impact factors: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Run the API with uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("consolidated_main:app", host="0.0.0.0", port=8000, reload=True)
