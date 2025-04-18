"""
Enhanced Export Service for the Medical Research Synthesizer.
This module provides enhancements to the Export Service, including:
- Support for more export formats
- Better error handling for export errors
- Validation of input data
- Progress tracking for large exports
"""
import json
import csv
import logging
import time
import hashlib
from typing import Dict, Optional, Any
from enum import Enum
from ..core.exceptions import (
    ExportError, ValidationError, FileError
)
from ..core.progress_tracker import ProgressTracker
logger = logging.getLogger(__name__)
class ExportFormat(str, Enum):
    """Supported export formats.

    This enum defines the available export formats for search results and analyses.
    """
    JSON = "json"
    CSV = "csv"
    EXCEL = "excel"
    PDF = "pdf"
    XML = "xml"
    HTML = "html"
    MARKDOWN = "markdown"
    BIBTEX = "bibtex"
    RIS = "ris"
    DOCX = "docx"
class ExportProgressTracker(ProgressTracker):
    """
    Progress tracker for export operations.
    This class extends the base ProgressTracker to provide export-specific
    progress tracking functionality.
    """
    def __init__(self, export_id: str, total_steps: int = 100):
        """
        Initialize the export progress tracker.
        Args:
            export_id: Export ID
            total_steps: Total number of steps in the export
        """
        super().__init__(operation_id=export_id, total_steps=total_steps)
        self.export_id = export_id
        self.export_format = "unknown"
        self.start_time = time.time()
        self.file_path = None
    def set_export_format(self, export_format: str):
        """
        Set the export format.
        Args:
            export_format: Format of the export
        """
        self.export_format = export_format
    def set_file_path(self, file_path: str):
        """
        Set the export file path.
        Args:
            file_path: Path to the exported file
        """
        self.file_path = file_path
    def get_progress_details(self) -> Dict[str, Any]:
        """
        Get detailed progress information.
        Returns:
            Dictionary with progress details
        """
        details = super().get_progress_details()
        details.update({
            "export_id": self.export_id,
            "export_format": self.export_format,
            "elapsed_time": time.time() - self.start_time,
            "file_path": self.file_path
        })
        return details
def validate_export_input(func):
    """
    Decorator for validating export input data.
    This decorator validates input parameters for export methods.
    """
    async def wrapper(self, *args, **kwargs):
        data = kwargs.get('data', {})
        export_format = kwargs.get('export_format', None)
        if not data:
            raise ValidationError("Data cannot be empty")
        if export_format is not None:
            try:
                export_format = ExportFormat(export_format.lower())
            except ValueError:
                valid_formats = ", ".join([f.value for f in ExportFormat])
                raise ValidationError(f"Invalid export format: {export_format}. Valid formats: {valid_formats}")
        return await func(self, *args, **kwargs)
    return wrapper
def track_export_progress(export_format: str, total_steps: int = 100):
    """
    Decorator for tracking export progress.
    This decorator adds progress tracking to export methods.
    Args:
        export_format: Format of the export
        total_steps: Total number of steps in the export
    """
    def decorator(func):
        async def wrapper(self, *args, **kwargs):
            func_name = func.__name__
            param_str = f"{func_name}:{args}:{kwargs}"
            export_id = hashlib.md5(param_str.encode()).hexdigest()
            tracker = ExportProgressTracker(export_id, total_steps)
            tracker.set_export_format(export_format)
            tracker.update(0, "Starting export")
            kwargs['progress_tracker'] = tracker
            try:
                result = await func(self, *args, **kwargs)
                if isinstance(result, str):
                    tracker.set_file_path(result)
                tracker.complete("Export completed successfully")
                return result
            except Exception as e:
                tracker.fail(f"Export failed: {str(e)}")
                raise
        return wrapper
    return decorator
def enhanced_export_error_handling(func):
    """
    Decorator for enhanced error handling in export methods.
    This decorator adds detailed error handling to export methods.
    """
    async def wrapper(self, *args, **kwargs):
        try:
            return await func(self, *args, **kwargs)
        except ValidationError:
            raise
        except FileError:
            raise
        except ExportError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {str(e)}", exc_info=True)
            raise ExportError(
                format=kwargs.get('export_format', 'unknown'),
                message=f"Unexpected error: {str(e)}"
            )
    return wrapper
def export_to_xml(data: Dict[str, Any], output_path: str) -> str:
    """
    Export data to XML.
    Args:
        data: Data to export
        output_path: Path to save the XML file
    Returns:
        Path to the exported file
    """
    try:
        import xml.etree.ElementTree as ET
        from xml.dom import minidom
    except ImportError:
        raise ImportError("xml.etree.ElementTree is required for XML export")
    root = ET.Element("export")
    metadata = ET.SubElement(root, "metadata")
    if "query" in data:
        query_elem = ET.SubElement(metadata, "query")
        query_elem.text = data["query"]
    if "timestamp" in data:
        timestamp_elem = ET.SubElement(metadata, "timestamp")
        timestamp_elem.text = data["timestamp"]
    results_elem = ET.SubElement(root, "results")
    for result in data.get("results", []):
        result_elem = ET.SubElement(results_elem, "result")
        for key, value in result.items():
            if isinstance(value, (dict, list)):
                continue
            field_elem = ET.SubElement(result_elem, key)
            field_elem.text = str(value)
    xml_str = ET.tostring(root, encoding="utf-8")
    pretty_xml = minidom.parseString(xml_str).toprettyxml(indent="  ")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(pretty_xml)
    return output_path
def export_to_html(data: Dict[str, Any], output_path: str, template: Optional[str] = None) -> str:
    """
    Export data to HTML.
    Args:
        data: Data to export
        output_path: Path to save the HTML file
        template: Path to HTML template file (optional)
    Returns:
        Path to the exported file
    """
    try:
        from jinja2 import Template
    except ImportError:
        raise ImportError("jinja2 is required for HTML export")
    default_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Medical Research Synthesizer Export</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1 { color: #2c3e50; }
            .result { margin-bottom: 20px; padding: 10px; border: 1px solid #ddd; }
            .title { font-weight: bold; color: #3498db; }
            .abstract { color: #555; }
            .metadata { color: #7f8c8d; font-size: 0.9em; }
        </style>
    </head>
    <body>
        <h1>Medical Research Synthesizer Export</h1>
        {% if query %}
        <p><strong>Query:</strong> {{ query }}</p>
        {% endif %}
        <p><strong>Results:</strong> {{ results|length }}</p>
        <div class="results">
        {% for result in results %}
            <div class="result">
                <div class="title">{{ result.title }}</div>
                <div class="metadata">
                    {% if result.authors %}
                    <p><strong>Authors:</strong> {{ result.authors|join(', ') }}</p>
                    {% endif %}
                    {% if result.journal %}
                    <p><strong>Journal:</strong> {{ result.journal }}</p>
                    {% endif %}
                    {% if result.publication_date %}
                    <p><strong>Date:</strong> {{ result.publication_date }}</p>
                    {% endif %}
                </div>
                {% if result.abstract %}
                <div class="abstract">
                    <p><strong>Abstract:</strong> {{ result.abstract }}</p>
                </div>
                {% endif %}
            </div>
        {% endfor %}
        </div>
    </body>
    </html>
    """