"""
Ray Orchestrator API for Medical Research

This module provides a Ray-based orchestration API for distributed processing
of medical research tasks, including contradiction detection, knowledge base
management, and data analysis.
"""

import os
import json
import logging
import time
from typing import Dict, List, Optional, Tuple, Union, Any
from fastapi import APIRouter, HTTPException, BackgroundTasks, Query, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from asf.orchestration.ray_orchestrator import (
    RayOrchestrator, RayConfig, RayTaskScheduler, Task, TaskStatus
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ray-orchestrator-api")

# Create API router
router = APIRouter(
    prefix="/ray",
    tags=["Ray Orchestration"],
    responses={404: {"description": "Not found"}},
)

# Initialize Ray orchestrator
ray_config = RayConfig(
    use_ray=os.environ.get("USE_RAY", "true").lower() == "true",
    address=os.environ.get("RAY_ADDRESS"),
    num_cpus=int(os.environ.get("RAY_NUM_CPUS", "0")) or None,
    num_gpus=int(os.environ.get("RAY_NUM_GPUS", "0")) or None,
    include_dashboard=os.environ.get("RAY_INCLUDE_DASHBOARD", "true").lower() == "true",
    dashboard_port=int(os.environ.get("RAY_DASHBOARD_PORT", "8265")),
    logging_level=os.environ.get("RAY_LOGGING_LEVEL", "INFO"),
)

orchestrator = RayOrchestrator(config=ray_config)
scheduler = RayTaskScheduler(orchestrator=orchestrator)

# Start scheduler
scheduler.start()

# Define request/response models
class TaskRequest(BaseModel):
    name: str
    function_name: str
    args: List[Any] = []
    kwargs: Dict[str, Any] = {}
    dependencies: List[str] = []
    timeout: Optional[float] = 600.0  # 10 minutes default timeout
    max_retries: int = 3
    priority: int = 0
    resources: Dict[str, float] = {}
    schedule_time: Optional[float] = None

class TaskResponse(BaseModel):
    task_id: str
    status: str
    name: str
    function_name: str
    scheduled: bool = False
    schedule_time: Optional[float] = None

class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    result: Optional[Any] = None
    error: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    retry_count: int = 0

class WorkflowRequest(BaseModel):
    tasks: List[str]
    async_execution: bool = False

class WorkflowResponse(BaseModel):
    workflow_id: str
    tasks: List[str]
    status: str
    results: Optional[Dict[str, Any]] = None

class ContradictionAnalysisRequest(BaseModel):
    claim1: str
    claim2: str
    use_tsmixer: bool = True
    use_lorentz: bool = True
    use_shap: bool = True
    threshold: float = 0.7
    priority: int = 1

class ContradictionAnalysisResponse(BaseModel):
    task_id: str
    status: str
    result: Optional[Dict[str, Any]] = None

# Register functions
from asf.medical.models.biomedlm_wrapper import BioMedLMScorer
from asf.medical.models.tsmixer_contradiction_detector import TSMixerContradictionDetector
from asf.medical.models.lorentz_embedding_detector import LorentzEmbeddingContradictionDetector
from asf.medical.models.shap_explainer import ContradictionExplainer

# Initialize BioMedLM scorer
try:
    biomedlm_scorer = BioMedLMScorer(
        model_name=os.environ.get("BIOMEDLM_MODEL", "microsoft/BioMedLM"),
        use_negation_detection=True,
        use_multimodal_fusion=True,
        use_shap_explainer=True
    )
    logger.info("BioMedLM scorer initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize BioMedLM scorer: {e}")
    biomedlm_scorer = None

# Initialize TSMixer contradiction detector
try:
    tsmixer_detector = TSMixerContradictionDetector(
        biomedlm_scorer=biomedlm_scorer,
        config={"use_tsmixer": True}
    )
    logger.info("TSMixer contradiction detector initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize TSMixer contradiction detector: {e}")
    tsmixer_detector = None

# Initialize Lorentz embedding contradiction detector
try:
    lorentz_detector = LorentzEmbeddingContradictionDetector(
        biomedlm_scorer=biomedlm_scorer,
        config={"use_lorentz": True}
    )
    logger.info("Lorentz embedding contradiction detector initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Lorentz embedding contradiction detector: {e}")
    lorentz_detector = None

# Register functions with orchestrator
def detect_contradiction(claim1: str, claim2: str, use_tsmixer: bool = True, use_lorentz: bool = True, threshold: float = 0.7) -> Dict[str, Any]:
    """
    Detect contradiction between medical claims.
    
    Args:
        claim1: First medical claim
        claim2: Second medical claim
        use_tsmixer: Whether to use TSMixer for temporal analysis
        use_lorentz: Whether to use Lorentz embeddings for hierarchical analysis
        threshold: Threshold for contradiction detection
        
    Returns:
        Dictionary with contradiction detection results
    """
    result = {
        "claim1": claim1,
        "claim2": claim2,
        "has_contradiction": False,
        "contradiction_score": 0.0,
        "methods_used": []
    }
    
    # Use BioMedLM for base contradiction detection
    if biomedlm_scorer is not None:
        try:
            biomedlm_result = biomedlm_scorer.detect_contradiction(claim1, claim2)
            result["biomedlm_result"] = biomedlm_result
            result["contradiction_score"] = biomedlm_result.get("contradiction_score", 0.0)
            result["methods_used"].append("biomedlm")
        except Exception as e:
            logger.error(f"Error in BioMedLM contradiction detection: {e}")
    
    # Use TSMixer for temporal analysis if requested
    if use_tsmixer and tsmixer_detector is not None:
        try:
            tsmixer_result = tsmixer_detector.detect_contradiction(claim1, claim2)
            result["tsmixer_result"] = tsmixer_result
            
            # Update contradiction score with TSMixer result
            if "contradiction_score" in tsmixer_result:
                result["contradiction_score"] = max(
                    result["contradiction_score"],
                    tsmixer_result["contradiction_score"]
                )
            
            result["methods_used"].append("tsmixer")
        except Exception as e:
            logger.error(f"Error in TSMixer contradiction detection: {e}")
    
    # Use Lorentz embeddings for hierarchical analysis if requested
    if use_lorentz and lorentz_detector is not None:
        try:
            lorentz_result = lorentz_detector.detect_contradiction(claim1, claim2)
            result["lorentz_result"] = lorentz_result
            
            # Update contradiction score with Lorentz result
            if "contradiction_score" in lorentz_result:
                result["contradiction_score"] = max(
                    result["contradiction_score"],
                    lorentz_result["contradiction_score"]
                )
            
            result["methods_used"].append("lorentz")
        except Exception as e:
            logger.error(f"Error in Lorentz embedding contradiction detection: {e}")
    
    # Determine if contradiction is detected
    result["has_contradiction"] = result["contradiction_score"] > threshold
    
    return result

def explain_contradiction(claim1: str, claim2: str, use_shap: bool = True) -> Dict[str, Any]:
    """
    Explain contradiction between medical claims.
    
    Args:
        claim1: First medical claim
        claim2: Second medical claim
        use_shap: Whether to use SHAP for explanation
        
    Returns:
        Dictionary with contradiction explanation
    """
    # Detect contradiction first
    contradiction_result = detect_contradiction(claim1, claim2)
    
    # If no contradiction detected, return result
    if not contradiction_result.get("has_contradiction", False):
        return {
            "claim1": claim1,
            "claim2": claim2,
            "contradiction_detected": False,
            "explanation": "No contradiction detected between the claims."
        }
    
    # Use BioMedLM for explanation if available
    if biomedlm_scorer is not None and hasattr(biomedlm_scorer, 'explain_contradiction'):
        try:
            explanation = biomedlm_scorer.explain_contradiction(claim1, claim2)
            return explanation
        except Exception as e:
            logger.error(f"Error in BioMedLM contradiction explanation: {e}")
    
    # Use SHAP for explanation if requested
    if use_shap and biomedlm_scorer is not None and biomedlm_scorer.model is not None and biomedlm_scorer.tokenizer is not None:
        try:
            from asf.medical.models.shap_explainer import ContradictionExplainer
            
            explainer = ContradictionExplainer(
                model=biomedlm_scorer.model,
                tokenizer=biomedlm_scorer.tokenizer
            )
            
            explanation = explainer.explain_contradiction(
                claim1=claim1,
                claim2=claim2,
                contradiction_score=contradiction_result.get("contradiction_score", 0.0),
                use_shap=True,
                use_negation_detection=True
            )
            
            return explanation.to_dict()
        except Exception as e:
            logger.error(f"Error in SHAP contradiction explanation: {e}")
    
    # Fallback to basic explanation
    return {
        "claim1": claim1,
        "claim2": claim2,
        "contradiction_detected": True,
        "contradiction_score": contradiction_result.get("contradiction_score", 0.0),
        "explanation": "Contradiction detected between the claims.",
        "methods_used": contradiction_result.get("methods_used", [])
    }

# Register functions with orchestrator
orchestrator.register_function(detect_contradiction, "detect_contradiction")
orchestrator.register_function(explain_contradiction, "explain_contradiction")

# Define API endpoints
@router.post("/tasks", response_model=TaskResponse)
async def create_task(task_request: TaskRequest):
    """
    Create a new task.
    
    This endpoint creates a new task with the specified parameters.
    The task will be executed asynchronously.
    """
    try:
        # Create task
        if task_request.schedule_time:
            # Schedule task
            task_id = scheduler.schedule_task(
                name=task_request.name,
                function_name=task_request.function_name,
                args=task_request.args,
                kwargs=task_request.kwargs,
                dependencies=task_request.dependencies,
                timeout=task_request.timeout,
                max_retries=task_request.max_retries,
                priority=task_request.priority,
                resources=task_request.resources,
                schedule_time=task_request.schedule_time
            )
            
            scheduled = True
            schedule_time = task_request.schedule_time
        else:
            # Create task without scheduling
            task_id = orchestrator.create_task(
                name=task_request.name,
                function_name=task_request.function_name,
                args=task_request.args,
                kwargs=task_request.kwargs,
                dependencies=task_request.dependencies,
                timeout=task_request.timeout,
                max_retries=task_request.max_retries,
                priority=task_request.priority,
                resources=task_request.resources
            )
            
            scheduled = False
            schedule_time = None
        
        # Get task
        task = orchestrator.get_task(task_id)
        
        return {
            "task_id": task_id,
            "status": task.status,
            "name": task.name,
            "function_name": task.function_name,
            "scheduled": scheduled,
            "schedule_time": schedule_time
        }
    except Exception as e:
        logger.error(f"Error creating task: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/tasks/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    """
    Get task status.
    
    This endpoint returns the status of the specified task.
    """
    try:
        # Get task
        task = orchestrator.get_task(task_id)
        
        return {
            "task_id": task_id,
            "status": task.status,
            "result": task.result,
            "error": task.error,
            "start_time": task.start_time,
            "end_time": task.end_time,
            "retry_count": task.retry_count
        }
    except Exception as e:
        logger.error(f"Error getting task status: {e}")
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

@router.post("/tasks/{task_id}/execute", response_model=TaskStatusResponse)
async def execute_task(task_id: str, background_tasks: BackgroundTasks):
    """
    Execute a task.
    
    This endpoint executes the specified task asynchronously.
    """
    try:
        # Check if task exists
        task = orchestrator.get_task(task_id)
        
        # Execute task asynchronously
        background_tasks.add_task(orchestrator.execute_task, task_id)
        
        return {
            "task_id": task_id,
            "status": TaskStatus.RUNNING,
            "result": None,
            "error": None,
            "start_time": time.time(),
            "end_time": None,
            "retry_count": task.retry_count
        }
    except Exception as e:
        logger.error(f"Error executing task: {e}")
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

@router.post("/workflows", response_model=WorkflowResponse)
async def execute_workflow(workflow_request: WorkflowRequest, background_tasks: BackgroundTasks):
    """
    Execute a workflow.
    
    This endpoint executes a workflow of tasks asynchronously.
    """
    try:
        # Generate workflow ID
        workflow_id = f"workflow_{int(time.time() * 1000)}"
        
        # Execute workflow asynchronously
        if workflow_request.async_execution:
            background_tasks.add_task(
                orchestrator.execute_workflow_async, 
                workflow_request.tasks
            )
            
            return {
                "workflow_id": workflow_id,
                "tasks": workflow_request.tasks,
                "status": "RUNNING",
                "results": None
            }
        else:
            # Execute workflow synchronously
            results = orchestrator.execute_workflow(workflow_request.tasks)
            
            return {
                "workflow_id": workflow_id,
                "tasks": workflow_request.tasks,
                "status": "COMPLETED",
                "results": results
            }
    except Exception as e:
        logger.error(f"Error executing workflow: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/contradiction-analysis", response_model=ContradictionAnalysisResponse)
async def analyze_contradiction(request: ContradictionAnalysisRequest):
    """
    Analyze contradiction between medical claims.
    
    This endpoint analyzes contradiction between two medical claims using
    BioMedLM, TSMixer, and Lorentz embeddings.
    """
    try:
        # Create task for contradiction detection
        task_id = orchestrator.create_task(
            name="contradiction_analysis",
            function_name="detect_contradiction",
            args=[request.claim1, request.claim2],
            kwargs={
                "use_tsmixer": request.use_tsmixer,
                "use_lorentz": request.use_lorentz,
                "threshold": request.threshold
            },
            priority=request.priority
        )
        
        # Execute task
        result = orchestrator.execute_task(task_id)
        
        return {
            "task_id": task_id,
            "status": "COMPLETED",
            "result": result
        }
    except Exception as e:
        logger.error(f"Error analyzing contradiction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/contradiction-explanation", response_model=ContradictionAnalysisResponse)
async def explain_contradiction_endpoint(request: ContradictionAnalysisRequest):
    """
    Explain contradiction between medical claims.
    
    This endpoint explains contradiction between two medical claims using
    SHAP-based explainability.
    """
    try:
        # Create task for contradiction explanation
        task_id = orchestrator.create_task(
            name="contradiction_explanation",
            function_name="explain_contradiction",
            args=[request.claim1, request.claim2],
            kwargs={"use_shap": request.use_shap},
            priority=request.priority
        )
        
        # Execute task
        result = orchestrator.execute_task(task_id)
        
        return {
            "task_id": task_id,
            "status": "COMPLETED",
            "result": result
        }
    except Exception as e:
        logger.error(f"Error explaining contradiction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status")
async def get_orchestrator_status():
    """
    Get orchestrator status.
    
    This endpoint returns the status of the Ray orchestrator.
    """
    try:
        # Get list of processes
        processes = orchestrator.list_processes()
        
        return {
            "status": "running" if orchestrator.initialized else "not_initialized",
            "ray_address": ray_config.address,
            "num_cpus": ray_config.num_cpus,
            "num_gpus": ray_config.num_gpus,
            "processes": processes
        }
    except Exception as e:
        logger.error(f"Error getting orchestrator status: {e}")
        raise HTTPException(status_code=500, detail=str(e))
