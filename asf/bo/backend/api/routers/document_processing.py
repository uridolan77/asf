"""
Document Processing Router for BO backend.

This module provides endpoints for document processing functionality,
including single document processing, batch processing, and processing history.
"""

from fastapi import APIRouter, Depends, HTTPException, status, File, UploadFile, Form, BackgroundTasks
from typing import Any, List, Optional
from pydantic import BaseModel
import logging
import os
import json
import uuid
import asyncio
from datetime import datetime
import tempfile
import shutil
from threading import Thread

from api.auth import get_current_user, User
from api.websockets.connection_manager import connection_manager

# Import the document processing module
from asf.medical.ml.document_processing import MedicalResearchSynthesizer, EnhancedMedicalResearchSynthesizer

# Create the router
router = APIRouter(prefix="/api/document-processing", tags=["document-processing"])

# Configure logging
logger = logging.getLogger(__name__)

# Define models
class ProcessingSettings(BaseModel):
    """Settings for document processing."""
    prefer_pdfminer: bool = True
    use_enhanced_section_classifier: bool = True
    use_gliner: bool = True
    confidence_threshold: float = 0.6
    use_hgt: bool = True
    encoder_model: str = "microsoft/biogpt"
    use_enhanced_summarizer: bool = True
    check_factual_consistency: bool = True
    consistency_method: str = "qafacteval"
    consistency_threshold: float = 0.6
    use_cache: bool = True
    use_parallel: bool = True
    use_enhanced_synthesizer: bool = True
    use_streaming: bool = False
    use_sentence_segmentation: bool = True
    cache_dir: str = "cache"
    model_dir: Optional[str] = None

class ProcessingResponse(BaseModel):
    """Response for document processing."""
    task_id: str
    status: str
    message: str
    created_at: str
    streaming_url: Optional[str] = None

class ProcessingTask(BaseModel):
    """Document processing task."""
    task_id: str
    status: str
    file_name: str
    created_at: str
    completed_at: Optional[str] = None
    result_path: Optional[str] = None
    error_message: Optional[str] = None
    processing_time: Optional[float] = None
    entity_count: Optional[int] = None
    relation_count: Optional[int] = None
    progress: Optional[float] = None
    current_stage: Optional[str] = None

class BatchProcessingRequest(BaseModel):
    """Request for batch document processing."""
    settings: Optional[ProcessingSettings] = None
    batch_size: int = 4

# In-memory storage for processing tasks (in production, use a database)
processing_tasks = {}

# Directory for storing processed documents
RESULTS_DIR = os.path.join(tempfile.gettempdir(), "document_processing_results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Helper function to get synthesizer with settings
def get_synthesizer(settings: Optional[ProcessingSettings] = None) -> MedicalResearchSynthesizer:
    """Get a document synthesizer with the specified settings."""
    if settings is None:
        settings = ProcessingSettings()

    # Use the appropriate synthesizer based on settings
    if settings.use_enhanced_synthesizer:
        return EnhancedMedicalResearchSynthesizer(
            document_processor_args={
                "prefer_pdfminer": settings.prefer_pdfminer,
                "use_enhanced_section_classifier": settings.use_enhanced_section_classifier,
                "use_advanced_reference_parser": True,
                "use_anystyle": False,  # Could be a setting
                "use_grobid": False     # Could be a setting
            },
            entity_extractor_args={
                "use_gliner": settings.use_gliner,
                "confidence_threshold": settings.confidence_threshold
            },
            relation_extractor_args={
                "use_hgt": settings.use_hgt,
                "encoder_model": settings.encoder_model,
                "use_sentence_segmentation": settings.use_sentence_segmentation
            },
            summarizer_args={
                "use_enhanced": settings.use_enhanced_summarizer,
                "check_factual_consistency": settings.check_factual_consistency,
                "consistency_method": settings.consistency_method,
                "consistency_threshold": settings.consistency_threshold
            },
            use_cache=settings.use_cache,
            cache_dir=settings.cache_dir,
            model_dir=settings.model_dir
        )
    else:
        # Use original synthesizer if specifically requested
        return MedicalResearchSynthesizer(
            document_processor_args={
                "prefer_pdfminer": settings.prefer_pdfminer,
                "use_enhanced_section_classifier": settings.use_enhanced_section_classifier
            },
            entity_extractor_args={
                "use_gliner": settings.use_gliner,
                "confidence_threshold": settings.confidence_threshold
            },
            relation_extractor_args={
                "use_hgt": settings.use_hgt,
                "encoder_model": settings.encoder_model
            },
            summarizer_args={
                "use_enhanced": settings.use_enhanced_summarizer,
                "check_factual_consistency": settings.check_factual_consistency,
                "consistency_method": settings.consistency_method,
                "consistency_threshold": settings.consistency_threshold
            },
            use_cache=settings.use_cache
        )

# Helper function to process a document in the background
def process_document_task(file_path: str, task_id: str, is_pdf: bool = True, settings: Optional[ProcessingSettings] = None, use_parallel: bool = False):
    """Process a document in the background."""
    try:
        # Update task status
        processing_tasks[task_id]["status"] = "processing"
        processing_tasks[task_id]["progress"] = 0.0
        processing_tasks[task_id]["current_stage"] = "initializing"

        # Get synthesizer
        synthesizer = get_synthesizer(settings)

        # Create result directory
        result_dir = os.path.join(RESULTS_DIR, task_id)
        os.makedirs(result_dir, exist_ok=True)

        # Define progress callback
        def progress_callback(stage: str, progress: float):
            processing_tasks[task_id]["progress"] = progress
            processing_tasks[task_id]["current_stage"] = stage
            logger.debug(f"Task {task_id}: {stage} - {progress * 100:.1f}%")

            # Broadcast progress via WebSocket
            asyncio.run(connection_manager.broadcast_task_progress(
                task_id=task_id,
                progress=progress,
                stage=stage,
                metrics={
                    "file_name": os.path.basename(file_path),
                    "is_pdf": is_pdf
                }
            ))

        # Define streaming callback
        def streaming_callback(stage: str, result: Any):
            # Save intermediate results if needed
            if settings and settings.use_streaming:
                try:
                    stage_path = os.path.join(result_dir, f"{stage}.json")
                    with open(stage_path, 'w') as f:
                        if hasattr(result, 'to_dict'):
                            json.dump(result.to_dict(), f, indent=2)
                        elif hasattr(result, '__dict__'):
                            json.dump(result.__dict__, f, indent=2)
                        else:
                            json.dump(str(result), f, indent=2)

                    # Broadcast intermediate result via WebSocket
                    result_data = {}
                    if hasattr(result, 'to_dict'):
                        result_data = result.to_dict()
                    elif hasattr(result, '__dict__'):
                        result_data = result.__dict__

                    asyncio.run(connection_manager.broadcast_to_task_subscribers(
                        task_id=task_id,
                        message={
                            "type": "intermediate_result",
                            "task_id": task_id,
                            "stage": stage,
                            "result": result_data,
                            "timestamp": datetime.now().isoformat()
                        }
                    ))
                except Exception as e:
                    logger.warning(f"Failed to save intermediate result: {str(e)}")

        # Process the document based on settings
        start_time = datetime.now()

        # Check if we should use the enhanced synthesizer with streaming
        if settings and settings.use_enhanced_synthesizer and settings.use_streaming:
            # Use streaming processing
            doc_structure, metrics = synthesizer.process_streaming(
                file_path,
                is_pdf=is_pdf,
                streaming_callback=streaming_callback,
                progress_callback=progress_callback
            )
        elif settings and settings.use_enhanced_synthesizer:
            # Use enhanced synthesizer with progress tracking
            doc_structure, metrics = synthesizer.process_with_progress(
                file_path,
                is_pdf=is_pdf,
                progress_callback=progress_callback
            )
        elif use_parallel:
            # Use parallel processing with original synthesizer
            doc_structure, metrics = synthesizer.process_parallel(file_path, is_pdf=is_pdf)
        else:
            # Use standard processing with original synthesizer
            doc_structure, metrics = synthesizer.process(file_path, is_pdf=is_pdf)

        # Save final results
        synthesizer.save_results(doc_structure, result_dir)

        # Update task status
        processing_tasks[task_id].update({
            "status": "completed",
            "completed_at": datetime.now().isoformat(),
            "result_path": result_dir,
            "processing_time": metrics.get("total_processing_time"),
            "entity_count": len(doc_structure.entities) if hasattr(doc_structure, 'entities') else 0,
            "relation_count": len(doc_structure.relations) if hasattr(doc_structure, 'relations') else 0,
            "progress": 1.0,
            "current_stage": "completed"
        })

        # Broadcast completion via WebSocket
        result_data = {}
        if hasattr(doc_structure, 'to_dict'):
            result_data = doc_structure.to_dict()
        elif hasattr(doc_structure, '__dict__'):
            result_data = doc_structure.__dict__

        asyncio.run(connection_manager.broadcast_task_completed(
            task_id=task_id,
            result={
                "entity_count": len(doc_structure.entities) if hasattr(doc_structure, 'entities') else 0,
                "relation_count": len(doc_structure.relations) if hasattr(doc_structure, 'relations') else 0,
                "processing_time": metrics.get("total_processing_time"),
                "result_path": result_dir
            }
        ))

        # Close synthesizer to release resources
        if hasattr(synthesizer, 'close'):
            synthesizer.close()

        logger.info(f"Document processing completed for task {task_id}")
    except Exception as e:
        logger.error(f"Error processing document for task {task_id}: {str(e)}")
        processing_tasks[task_id].update({
            "status": "failed",
            "completed_at": datetime.now().isoformat(),
            "error_message": str(e),
            "progress": 0.0,
            "current_stage": "failed"
        })

        # Broadcast failure via WebSocket
        asyncio.run(connection_manager.broadcast_task_failed(
            task_id=task_id,
            error=str(e)
        ))

        # Try to close synthesizer if it exists
        try:
            if 'synthesizer' in locals() and hasattr(synthesizer, 'close'):
                synthesizer.close()
        except Exception as close_error:
            logger.warning(f"Error closing synthesizer: {str(close_error)}")

# Endpoints
@router.get("/")
async def document_processing_root(current_user: User = Depends(get_current_user)):
    """
    Root endpoint for document processing API.

    Returns information about available document processing endpoints.
    """
    return {
        "status": "ok",
        "endpoints": [
            {
                "path": "/process",
                "method": "POST",
                "description": "Process a single document"
            },
            {
                "path": "/batch",
                "method": "POST",
                "description": "Process multiple documents in batch"
            },
            {
                "path": "/tasks",
                "method": "GET",
                "description": "Get all processing tasks"
            },
            {
                "path": "/tasks/{task_id}",
                "method": "GET",
                "description": "Get a specific processing task"
            },
            {
                "path": "/settings",
                "method": "GET",
                "description": "Get default processing settings"
            },
            {
                "path": "/tasks/{task_id}/progress",
                "method": "GET",
                "description": "Get progress of a processing task"
            }
        ]
    }

@router.post("/process", response_model=ProcessingResponse)
async def process_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    settings_json: Optional[str] = Form(None),
    use_parallel: bool = Form(False),
    current_user: User = Depends(get_current_user)
):
    """
    Process a single document.

    Args:
        file: The document file to process
        settings_json: JSON string with processing settings
        use_parallel: Whether to use parallel processing
        current_user: The authenticated user

    Returns:
        ProcessingResponse: Information about the processing task
    """
    # Parse settings if provided
    settings = None
    if settings_json:
        try:
            settings_dict = json.loads(settings_json)
            settings = ProcessingSettings(**settings_dict)
        except Exception as e:
            logger.error(f"Error parsing settings: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid settings format: {str(e)}"
            )

    # Check file type
    file_extension = os.path.splitext(file.filename)[1].lower()
    is_pdf = file_extension == ".pdf"

    if not is_pdf and file_extension not in [".txt", ".md", ".json"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type: {file_extension}. Supported types: .pdf, .txt, .md, .json"
        )

    # Create a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_extension)

    try:
        # Write the file content
        with open(temp_file.name, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Create a task ID
        task_id = str(uuid.uuid4())

        # Create a task entry
        processing_tasks[task_id] = {
            "task_id": task_id,
            "status": "queued",
            "file_name": file.filename,
            "created_at": datetime.now().isoformat()
        }

        # Start processing in the background
        background_tasks.add_task(
            process_document_task,
            temp_file.name,
            task_id,
            is_pdf,
            settings,
            use_parallel
        )

        return ProcessingResponse(
            task_id=task_id,
            status="queued",
            message="Document processing started",
            created_at=processing_tasks[task_id]["created_at"]
        )
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing document: {str(e)}"
        )

@router.post("/batch", response_model=List[ProcessingResponse])
async def batch_process_documents(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    settings_json: Optional[str] = Form(None),
    batch_size: int = Form(4),
    current_user: User = Depends(get_current_user)
):
    """
    Process multiple documents in batch.

    Args:
        files: The document files to process
        settings_json: JSON string with processing settings
        batch_size: Number of documents to process in parallel
        current_user: The authenticated user

    Returns:
        List[ProcessingResponse]: Information about the processing tasks
    """
    # Parse settings if provided
    settings = None
    if settings_json:
        try:
            settings_dict = json.loads(settings_json)
            settings = ProcessingSettings(**settings_dict)
        except Exception as e:
            logger.error(f"Error parsing settings: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid settings format: {str(e)}"
            )

    # Process each file
    responses = []
    temp_files = []

    try:
        for file in files:
            # Check file type
            file_extension = os.path.splitext(file.filename)[1].lower()
            is_pdf = file_extension == ".pdf"

            if not is_pdf and file_extension not in [".txt", ".md", ".json"]:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Unsupported file type: {file_extension}. Supported types: .pdf, .txt, .md, .json"
                )

            # Create a temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_extension)
            temp_files.append(temp_file.name)

            # Write the file content
            with open(temp_file.name, "wb") as f:
                shutil.copyfileobj(file.file, f)

            # Create a task ID
            task_id = str(uuid.uuid4())

            # Create a task entry
            processing_tasks[task_id] = {
                "task_id": task_id,
                "status": "queued",
                "file_name": file.filename,
                "created_at": datetime.now().isoformat()
            }

            # Start processing in the background
            background_tasks.add_task(
                process_document_task,
                temp_file.name,
                task_id,
                is_pdf,
                settings,
                False  # Don't use parallel for individual files in batch
            )

            responses.append(ProcessingResponse(
                task_id=task_id,
                status="queued",
                message="Document processing started",
                created_at=processing_tasks[task_id]["created_at"]
            ))

        return responses
    except Exception as e:
        logger.error(f"Error batch processing documents: {str(e)}")
        # Clean up temporary files
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error batch processing documents: {str(e)}"
        )

@router.get("/tasks", response_model=List[ProcessingTask])
async def get_processing_tasks(
    status: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    current_user: User = Depends(get_current_user)
):
    """
    Get all processing tasks.

    Args:
        status: Filter tasks by status (queued, processing, completed, failed)
        limit: Maximum number of tasks to return
        offset: Number of tasks to skip
        current_user: The authenticated user

    Returns:
        List[ProcessingTask]: List of processing tasks
    """
    tasks = list(processing_tasks.values())

    # Filter by status if provided
    if status:
        tasks = [task for task in tasks if task["status"] == status]

    # Sort by created_at (newest first)
    tasks.sort(key=lambda x: x["created_at"], reverse=True)

    # Apply pagination
    tasks = tasks[offset:offset + limit]

    return [ProcessingTask(**task) for task in tasks]

@router.get("/tasks/{task_id}", response_model=ProcessingTask)
async def get_processing_task(
    task_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Get a specific processing task.

    Args:
        task_id: The ID of the task to get
        current_user: The authenticated user

    Returns:
        ProcessingTask: The processing task
    """
    if task_id not in processing_tasks:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task with ID {task_id} not found"
        )

    return ProcessingTask(**processing_tasks[task_id])

@router.get("/settings", response_model=ProcessingSettings)
async def get_default_settings(current_user: User = Depends(get_current_user)):
    """
    Get default processing settings.

    Args:
        current_user: The authenticated user

    Returns:
        ProcessingSettings: Default processing settings
    """
    return ProcessingSettings()

@router.get("/tasks/{task_id}/progress")
async def get_processing_progress(task_id: str):
    """
    Get the progress of a processing task.

    Args:
        task_id: The ID of the task to get progress for

    Returns:
        Dict: The processing progress
    """
    if task_id not in processing_tasks:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task with ID {task_id} not found"
        )

    task = processing_tasks[task_id]

    return {
        "task_id": task_id,
        "status": task["status"],
        "progress": task.get("progress", 0.0),
        "current_stage": task.get("current_stage", "unknown"),
        "entity_count": task.get("entity_count", 0),
        "relation_count": task.get("relation_count", 0)
    }

@router.get("/results/{task_id}")
async def get_processing_results(
    task_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Get the results of a processing task.

    Args:
        task_id: The ID of the task to get results for
        current_user: The authenticated user

    Returns:
        Dict: The processing results
    """
    if task_id not in processing_tasks:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task with ID {task_id} not found"
        )

    task = processing_tasks[task_id]

    if task["status"] != "completed":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Task with ID {task_id} is not completed"
        )

    if not task.get("result_path") or not os.path.exists(task["result_path"]):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Results for task with ID {task_id} not found"
        )

    # Read the document structure JSON
    try:
        with open(os.path.join(task["result_path"], "document_structure.json"), "r") as f:
            results = json.load(f)

        return {
            "task_id": task_id,
            "file_name": task["file_name"],
            "processing_time": task.get("processing_time"),
            "entity_count": task.get("entity_count"),
            "relation_count": task.get("relation_count"),
            "results": results
        }
    except Exception as e:
        logger.error(f"Error reading results for task {task_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error reading results: {str(e)}"
        )
