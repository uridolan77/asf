Export Tasks

This module defines background tasks for exporting data in various formats.

import logging
import shutil
from typing import List, Dict, Any, Optional

import dramatiq

from asf.medical.api.export_utils_consolidated import (
    export_to_pdf, export_contradiction_analysis_to_pdf
)

logger = logging.getLogger(__name__)

task_results = {}

@dramatiq.actor(max_retries=3, time_limit=300000)  # 5 minutes
def generate_pdf_report(results: List[Dict[str, Any]], query_text: str, output_path: str):
    """
    Generate a PDF report for search results in the background.
    
    Args:
        results: The search results to export
        query_text: The query that produced the results
        output_path: The path to save the PDF to
    """
    try:
        logger.info(f"Starting PDF generation for query: {query_text}")
        
        task_id = dramatiq.middleware.current_message.message_id
        task_results[task_id] = {
            "status": "processing", 
            "progress": 0,
            "output_path": output_path
        }
        
        pdf_output = export_to_pdf(results, query_text)
        
        shutil.copy(pdf_output.path, output_path)
        
        task_results[task_id] = {
            "status": "completed", 
            "progress": 100,
            "output_path": output_path
        }
        
        logger.info(f"PDF generation completed: {output_path}")
        return {"status": "completed", "file_path": output_path}
    
    except Exception as e:
        task_id = dramatiq.middleware.current_message.message_id
        task_results[task_id] = {
            "status": "failed", 
            "error": str(e),
            "output_path": output_path
        }
        
        logger.error(f"Error generating PDF: {str(e)}")
        raise

@dramatiq.actor(max_retries=3, time_limit=300000)  # 5 minutes
def generate_contradiction_pdf_report(analysis: Dict[str, Any], query_text: str, output_path: str):
    """
    Generate a PDF report for contradiction analysis in the background.
    
    Args:
        analysis: The contradiction analysis to export
        query_text: The query that produced the analysis
        output_path: The path to save the PDF to
    """
    try:
        logger.info(f"Starting contradiction PDF generation for query: {query_text}")
        
        task_id = dramatiq.middleware.current_message.message_id
        task_results[task_id] = {
            "status": "processing", 
            "progress": 0,
            "output_path": output_path
        }
        
        pdf_output = export_contradiction_analysis_to_pdf(analysis, query_text)
        
        shutil.copy(pdf_output.path, output_path)
        
        task_results[task_id] = {
            "status": "completed", 
            "progress": 100,
            "output_path": output_path
        }
        
        logger.info(f"Contradiction PDF generation completed: {output_path}")
        return {"status": "completed", "file_path": output_path}
    
    except Exception as e:
        task_id = dramatiq.middleware.current_message.message_id
        task_results[task_id] = {
            "status": "failed", 
            "error": str(e),
            "output_path": output_path
        }
        
        logger.error(f"Error generating contradiction PDF: {str(e)}")
        raise

def get_task_result(task_id: str) -> Optional[Dict[str, Any]]:
    """
    Get the result of a task by ID.
    
    Args:
        task_id: The ID of the task
        
    Returns:
        Task result information or None if not found
    """
    return task_results.get(task_id)
