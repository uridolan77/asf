from typing import Dict, List, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, Body, status
from asf.medical.services.knowledge_base_service import KnowledgeBaseService
from asf.medical.services.search_service import SearchService, SearchMethod
from asf.medical.core.exceptions import ResourceNotFoundError, ValidationError

router = APIRouter(prefix="/api/knowledge-base", tags=["knowledge-base"])

# This would be properly injected in a real application
def get_kb_service() -> KnowledgeBaseService:
    """Dependency to get the knowledge base service."""
    from asf.medical.clients.ncbi.ncbi_client import NCBIClient
    from asf.medical.clients.clinical_trials_client import ClinicalTrialsClient
    from asf.medical.storage.repositories.result_repository import ResultRepository
    from asf.medical.storage.repositories.query_repository import QueryRepository
    from asf.medical.graph.graph_rag import GraphRAG
    
    # Initialize dependencies
    ncbi_client = NCBIClient()
    clinical_trials_client = ClinicalTrialsClient()
    query_repository = QueryRepository()
    result_repository = ResultRepository()
    graph_rag = GraphRAG()
    
    # Create and return search service
    search_service = SearchService(
        ncbi_client=ncbi_client,
        clinical_trials_client=clinical_trials_client,
        query_repository=query_repository,
        result_repository=result_repository,
        graph_rag=graph_rag
    )
    
    # Create and return knowledge base service
    return KnowledgeBaseService(
        search_service=search_service,
        kb_repository=None  # This would be properly initialized in a real app
    )

def get_search_service() -> SearchService:
    """Dependency to get the search service."""
    from asf.medical.clients.ncbi.ncbi_client import NCBIClient
    from asf.medical.clients.clinical_trials_client import ClinicalTrialsClient
    from asf.medical.storage.repositories.result_repository import ResultRepository
    from asf.medical.storage.repositories.query_repository import QueryRepository
    from asf.medical.graph.graph_rag import GraphRAG
    
    # Initialize dependencies
    ncbi_client = NCBIClient()
    clinical_trials_client = ClinicalTrialsClient()
    query_repository = QueryRepository()
    result_repository = ResultRepository()
    graph_rag = GraphRAG()
    
    # Create and return search service
    return SearchService(
        ncbi_client=ncbi_client,
        clinical_trials_client=clinical_trials_client,
        query_repository=query_repository,
        result_repository=result_repository,
        graph_rag=graph_rag
    )

@router.post("/create", response_model=Dict[str, Any], status_code=status.HTTP_201_CREATED)
async def create_knowledge_base(
    name: str = Body(..., description="Knowledge base name"),
    query: str = Body(..., description="Search query"),
    update_schedule: str = Body("weekly", description="Update schedule (daily, weekly, monthly)"),
    kb_service: KnowledgeBaseService = Depends(get_kb_service)
):
    """Create a new knowledge base."""
    try:
        kb = await kb_service.create_knowledge_base(
            name=name,
            query=query,
            update_schedule=update_schedule
        )
        return kb
    except ValidationError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to create knowledge base: {str(e)}")

@router.get("/list", response_model=List[Dict[str, Any]])
async def list_knowledge_bases(
    kb_service: KnowledgeBaseService = Depends(get_kb_service)
):
    """List all knowledge bases."""
    try:
        kbs = await kb_service.list_knowledge_bases()
        return kbs
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to retrieve knowledge bases: {str(e)}")

@router.get("/{kb_id}", response_model=Dict[str, Any])
async def get_knowledge_base(
    kb_id: str,
    kb_service: KnowledgeBaseService = Depends(get_kb_service)
):
    """Get a knowledge base by ID."""
    try:
        kb = await kb_service.get_knowledge_base_by_id(kb_id)
        if not kb:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Knowledge base {kb_id} not found")
        return kb
    except ResourceNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Knowledge base {kb_id} not found")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to retrieve knowledge base: {str(e)}")

@router.post("/update/{kb_id}", response_model=Dict[str, Any])
async def update_knowledge_base(
    kb_id: str,
    kb_service: KnowledgeBaseService = Depends(get_kb_service)
):
    """Update a knowledge base."""
    try:
        kb = await kb_service.update_knowledge_base(kb_id)
        return kb
    except ResourceNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Knowledge base {kb_id} not found")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to update knowledge base: {str(e)}")

@router.delete("/{kb_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_knowledge_base(
    kb_id: str,
    kb_service: KnowledgeBaseService = Depends(get_kb_service)
):
    """Delete a knowledge base."""
    try:
        await kb_service.delete_knowledge_base(kb_id)
        return None
    except ResourceNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Knowledge base {kb_id} not found")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to delete knowledge base: {str(e)}")

@router.post("/search", response_model=Dict[str, Any])
async def search(
    query: str = Body(..., description="Search query"),
    max_results: int = Body(100, description="Maximum number of results"),
    page: int = Body(1, description="Page number"),
    page_size: int = Body(20, description="Results per page"),
    search_method: str = Body(SearchMethod.PUBMED.value, description="Search method"),
    use_graph_rag: bool = Body(False, description="Whether to use GraphRAG"),
    search_service: SearchService = Depends(get_search_service)
):
    """Search medical literature."""
    try:
        results = await search_service.search(
            query=query,
            max_results=max_results,
            page=page,
            page_size=page_size,
            search_method=search_method,
            use_graph_rag=use_graph_rag
        )
        return results
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@router.post("/search-pico", response_model=Dict[str, Any])
async def search_pico(
    condition: str = Body(..., description="Medical condition"),
    interventions: List[str] = Body([], description="Interventions"),
    outcomes: List[str] = Body([], description="Outcomes"),
    population: Optional[str] = Body(None, description="Population"),
    study_design: Optional[str] = Body(None, description="Study design"),
    years: int = Body(5, description="Number of years to include"),
    max_results: int = Body(100, description="Maximum number of results"),
    page: int = Body(1, description="Page number"),
    page_size: int = Body(20, description="Results per page"),
    search_service: SearchService = Depends(get_search_service)
):
    """Search using PICO framework."""
    try:
        results = await search_service.search_pico(
            condition=condition,
            interventions=interventions,
            outcomes=outcomes,
            population=population,
            study_design=study_design,
            years=years,
            max_results=max_results,
            page=page,
            page_size=page_size
        )
        return results
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@router.post("/{kb_id}/export", response_model=Dict[str, Any])
async def export_knowledge_base(
    kb_id: str,
    format: str = Body("json", description="Export format (json, csv, pdf)"),
    kb_service: KnowledgeBaseService = Depends(get_kb_service)
):
    """Export knowledge base data to specified format."""
    try:
        export_data = await kb_service.export_knowledge_base(kb_id, format)
        return {"url": export_data, "format": format}
    except ResourceNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Knowledge base {kb_id} not found")
    except ValidationError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to export knowledge base: {str(e)}")