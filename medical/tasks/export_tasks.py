"""
Export Tasks for the Medical Research Synthesizer.

This module defines background tasks for exporting data in various formats,
including PDF reports for search results and contradiction analysis. These tasks
are designed to run asynchronously in the background using Dramatiq, allowing
the API to respond quickly to requests while the export operations continue
in separate worker processes.

The module includes:
- Tasks for generating PDF reports for search results
- Tasks for generating PDF reports for contradiction analysis
- Utilities for tracking task status and retrieving results

Each task follows a consistent pattern with proper error handling, logging,
and status tracking through a shared task storage mechanism.
"""

import logging
import shutil
from typing import List, Dict, Any, Optional

import dramatiq

from ..api.export_utils import (
    export_to_pdf, export_contradiction_analysis_to_pdf
)

logger = logging.getLogger(__name__)

task_results = {}

@dramatiq.actor(max_retries=3, time_limit=300000)  # 5 minutes
def generate_pdf_report(results: List[Dict[str, Any]], query_text: str, output_path: str):
    """
    Generate a PDF report for search results in the background.

    This task generates a PDF report from the provided search results using the
    export_to_pdf utility function. It updates the task status as it progresses
    and handles any errors that occur during the export process.

    Args:
        results: List of dictionaries containing the search results to export
        query_text: The original query text that produced these results
        output_path: File path where the generated PDF should be saved

    Returns:
        Dictionary with status information if successful

    Raises:
        Exception: Any exceptions are caught, logged, and the task status is updated
                  before re-raising
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

    This task generates a PDF report from the provided contradiction analysis using
    the export_contradiction_analysis_to_pdf utility function. It updates the task
    status as it progresses and handles any errors that occur during the export process.

    Args:
        analysis: Dictionary containing the contradiction analysis results to export
        query_text: The original query text that produced this analysis
        output_path: File path where the generated PDF should be saved

    Returns:
        Dictionary with status information if successful

    Raises:
        Exception: Any exceptions are caught, logged, and the task status is updated
                  before re-raising
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

    This function retrieves the current status and result information for a task
    from the shared task_results dictionary. It's used by API endpoints to check
    the progress of background tasks and retrieve their results.

    Args:
        task_id: The unique identifier of the task to retrieve

    Returns:
        Dictionary containing task status information, including:
        - status: 'processing', 'completed', or 'failed'
        - progress: Percentage of completion (0-100)
        - output_path: Path to the output file
        - error: Error message if status is 'failed'
        Returns None if the task_id is not found
    """
    return task_results.get(task_id)
