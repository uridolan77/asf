"""
Export service for the Medical Research Synthesizer.

This module provides a service for exporting search results and analyses
in various formats (JSON, CSV, Excel, PDF).
"""

import os
import json
import csv
import tempfile
import logging
from typing import Dict, Any, Optional
from datetime import datetime
import time
import hashlib
from enum import Enum
from ..core.progress_tracker import ProgressTracker

from ..core.exceptions import (
    ExportError, ValidationError, FileError
)

logger = logging.getLogger(__name__)

class ExportFormat(str, Enum):
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
    def __init__(self, export_id: str, total_steps: int = 100):
        super().__init__(operation_id=export_id, total_steps=total_steps)
        self.export_id = export_id
        self.export_format = "unknown"
        self.start_time = time.time()
        self.file_path = None
    def set_export_format(self, export_format: str):
        self.export_format = export_format
    def set_file_path(self, file_path: str):
        self.file_path = file_path
    def get_progress_details(self) -> Dict[str, Any]:
        details = super().get_progress_details()
        details.update({
            "export_id": self.export_id,
            "export_format": self.export_format,
            "elapsed_time": time.time() - self.start_time,
            "file_path": self.file_path
        })
        return details

def validate_export_input(func):
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
    html_template = default_template
    if template:
        with open(template, "r", encoding="utf-8") as f:
            html_template = f.read()
    tmpl = Template(html_template)
    html_content = tmpl.render(**data)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    return output_path

class ExportService:
    """Service for exporting search results and analyses.

    This service provides methods for exporting search results and analyses
    in various formats (JSON, CSV, Excel, PDF).
    """

    def __init__(self, export_dir: str = "exports"):
        """Initialize the export service.

        Args:
            export_dir: Directory to store exported files

        Raises:
            FileError: If the export directory cannot be created
        """
        self.export_dir = export_dir

        try:
            os.makedirs(self.export_dir, exist_ok=True)
            logger.info(f"Export directory initialized: {self.export_dir}")
        except Exception as e:
            logger.error(f"Failed to create export directory: {str(e)}")
            raise FileError(self.export_dir, f"Failed to create export directory: {str(e)}")

    async def export_to_json(
        self,
        data: Dict[str, Any],
        include_abstracts: bool = True,
        include_metadata: bool = True
    ) -> str:
        """Export data to a JSON file.

        This method exports the provided data to a JSON file in the export directory.

        Args:
            data: The data to export (typically search results or analysis)
            include_abstracts: Whether to include article abstracts
            include_metadata: Whether to include metadata (timestamps, etc.)

        Returns:
            Path to the exported file

        Raises:
            ValidationError: If the data is invalid
            ExportError: If an error occurs during export
        """
        if not data:
            raise ValidationError("Data cannot be empty")

        logger.info("Exporting data to JSON")

        try:
            filtered_data = self._filter_data(data, include_abstracts, include_metadata)

            fd, temp_path = tempfile.mkstemp(suffix=".json")
            os.close(fd)

            with open(temp_path, 'w') as f:
                json.dump(filtered_data, f, indent=2)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_name = f"export_{timestamp}.json"
            file_path = os.path.join(self.export_dir, file_name)

            os.rename(temp_path, file_path)

            logger.info(f"Data exported to JSON: {file_path}")

            return file_path
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON data: {str(e)}")
            if 'temp_path' in locals() and os.path.exists(temp_path):
                os.unlink(temp_path)
            raise ExportError("JSON", f"Invalid JSON data: {str(e)}")
        except OSError as e:
            logger.error(f"File error during JSON export: {str(e)}")
            if 'temp_path' in locals() and os.path.exists(temp_path):
                os.unlink(temp_path)
            raise FileError(file_path if 'file_path' in locals() else self.export_dir, f"File error during JSON export: {str(e)}")
        except Exception as e:
            logger.error(f"Error exporting to JSON: {str(e)}")
            if 'temp_path' in locals() and os.path.exists(temp_path):
                os.unlink(temp_path)
            raise ExportError("JSON", f"Failed to export to JSON: {str(e)}")

    async def export_to_csv(
        self,
        data: Dict[str, Any],
        include_abstracts: bool = True,
        include_metadata: bool = True
    ) -> str:
        """Export data to a CSV file.

        This method exports the provided data to a CSV file in the export directory.

        Args:
            data: The data to export (typically search results or analysis)
            include_abstracts: Whether to include article abstracts
            include_metadata: Whether to include metadata (timestamps, etc.)

        Returns:
            Path to the exported file

        Raises:
            ValidationError: If the data is invalid
            ExportError: If an error occurs during export
        """
        logger.info("Exporting data to CSV")

        filtered_data = self._filter_data(data, include_abstracts, include_metadata)

        fd, temp_path = tempfile.mkstemp(suffix=".csv")
        os.close(fd)

        results = filtered_data.get('results', [])

        if not results:
            os.unlink(temp_path)
            raise ValueError("No results to export")

        fields = ['pmid', 'title', 'authors', 'journal', 'publication_date']

        if include_abstracts:
            fields.append('abstract')

        if include_metadata:
            fields.extend(['impact_factor', 'citation_count', 'authority_score'])

        with open(temp_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()

            for result in results:
                row = {field: result.get(field, '') for field in fields}

                if 'authors' in row and isinstance(row['authors'], list):
                    row['authors'] = ', '.join(row['authors'])

                writer.writerow(row)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"export_{timestamp}.csv"
        file_path = os.path.join(self.export_dir, file_name)

        os.rename(temp_path, file_path)

        logger.info(f"Data exported to CSV: {file_path}")

        return file_path

    async def export_to_excel(
        self,
        data: Dict[str, Any],
        include_abstracts: bool = True,
        include_metadata: bool = True
    ) -> str:
        """Export data to an Excel file.

        This method exports the provided data to an Excel file in the export directory.

        Args:
            data: The data to export (typically search results or analysis)
            include_abstracts: Whether to include article abstracts
            include_metadata: Whether to include metadata (timestamps, etc.)

        Returns:
            Path to the exported file

        Raises:
            ValidationError: If the data is invalid
            ImportError: If pandas or openpyxl is not installed
            ExportError: If an error occurs during export
        """
        logger.info("Exporting data to Excel")

        try:
            import pandas as pd
            from openpyxl import Workbook
            from openpyxl.utils.dataframe import dataframe_to_rows
        except ImportError:
            logger.error("Failed to import pandas or openpyxl")
            raise ImportError("pandas and openpyxl are required for Excel export")

        filtered_data = self._filter_data(data, include_abstracts, include_metadata)

        fd, temp_path = tempfile.mkstemp(suffix=".xlsx")
        os.close(fd)

        results = filtered_data.get('results', [])

        if not results:
            os.unlink(temp_path)
            raise ValueError("No results to export")

        df = pd.DataFrame(results)

        columns = ['pmid', 'title', 'authors', 'journal', 'publication_date']

        if include_abstracts:
            columns.append('abstract')

        if include_metadata:
            columns.extend(['impact_factor', 'citation_count', 'authority_score'])

        df = df[[col for col in columns if col in df.columns]]

        if 'authors' in df.columns:
            df['authors'] = df['authors'].apply(
                lambda x: ', '.join(x) if isinstance(x, list) else x
            )

        wb = Workbook()
        ws = wb.active
        ws.title = "Results"

        for r in dataframe_to_rows(df, index=False, header=True):
            ws.append(r)

        wb.save(temp_path)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"export_{timestamp}.xlsx"
        file_path = os.path.join(self.export_dir, file_name)

        os.rename(temp_path, file_path)

        logger.info(f"Data exported to Excel: {file_path}")

        return file_path

    async def export_to_pdf(
        self,
        data: Dict[str, Any],
        include_abstracts: bool = True,
        include_metadata: bool = True
    ) -> str:
        """Export data to a PDF file.

        This method exports the provided data to a PDF file in the export directory.

        Args:
            data: The data to export (typically search results or analysis)
            include_abstracts: Whether to include article abstracts
            include_metadata: Whether to include metadata (timestamps, etc.)

        Returns:
            Path to the exported file

        Raises:
            ValidationError: If the data is invalid
            ImportError: If reportlab is not installed
            ExportError: If an error occurs during export
        """
        logger.info("Exporting data to PDF")

        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
            from reportlab.lib.styles import getSampleStyleSheet
        except ImportError:
            logger.error("Failed to import reportlab")
            raise ImportError("reportlab is required for PDF export")

        filtered_data = self._filter_data(data, include_abstracts, include_metadata)

        fd, temp_path = tempfile.mkstemp(suffix=".pdf")
        os.close(fd)

        results = filtered_data.get('results', [])

        if not results:
            os.unlink(temp_path)
            raise ValueError("No results to export")

        doc = SimpleDocTemplate(temp_path, pagesize=letter)
        styles = getSampleStyleSheet()
        elements = []

        elements.append(Paragraph("Medical Research Synthesizer Export", styles['Title']))
        elements.append(Spacer(1, 12))

        if 'query' in filtered_data:
            elements.append(Paragraph(f"Query: {filtered_data['query']}", styles['Heading2']))
            elements.append(Spacer(1, 12))

        elements.append(Paragraph(f"Results ({len(results)})", styles['Heading2']))
        elements.append(Spacer(1, 12))

        for i, result in enumerate(results, 1):
            elements.append(Paragraph(f"{i}. {result.get('title', 'No title')}", styles['Heading3']))
            elements.append(Spacer(1, 6))

            authors = result.get('authors', [])
            if authors:
                if isinstance(authors, list):
                    authors_str = ', '.join(authors)
                else:
                    authors_str = authors
                elements.append(Paragraph(f"Authors: {authors_str}", styles['Normal']))
                elements.append(Spacer(1, 6))

            journal = result.get('journal', 'Unknown journal')
            date = result.get('publication_date', 'Unknown date')
            elements.append(Paragraph(f"Journal: {journal}, Date: {date}", styles['Normal']))
            elements.append(Spacer(1, 6))

            pmid = result.get('pmid', 'Unknown PMID')
            elements.append(Paragraph(f"PMID: {pmid}", styles['Normal']))
            elements.append(Spacer(1, 6))

            if include_metadata:
                impact_factor = result.get('impact_factor', 'N/A')
                citation_count = result.get('citation_count', 'N/A')
                authority_score = result.get('authority_score', 'N/A')

                elements.append(Paragraph(f"Impact Factor: {impact_factor}", styles['Normal']))
                elements.append(Paragraph(f"Citation Count: {citation_count}", styles['Normal']))
                elements.append(Paragraph(f"Authority Score: {authority_score}", styles['Normal']))
                elements.append(Spacer(1, 6))

            if include_abstracts and 'abstract' in result:
                elements.append(Paragraph("Abstract:", styles['Heading4']))
                elements.append(Paragraph(result['abstract'], styles['Normal']))
                elements.append(Spacer(1, 6))

            elements.append(Spacer(1, 12))

        doc.build(elements)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"export_{timestamp}.pdf"
        file_path = os.path.join(self.export_dir, file_name)

        os.rename(temp_path, file_path)

        logger.info(f"Data exported to PDF: {file_path}")

        return file_path

    def _filter_data(
        self,
        data: Dict[str, Any],
        include_abstracts: bool,
        include_metadata: bool
    ) -> Dict[str, Any]:
        """Filter data based on inclusion flags.

        This internal method filters the data based on the include_abstracts and
        include_metadata flags, removing fields that should not be included in the export.

        Args:
            data: The data to filter
            include_abstracts: Whether to include article abstracts
            include_metadata: Whether to include metadata (timestamps, etc.)

        Returns:
            Filtered data dictionary

        Raises:
            ValidationError: If the data structure is invalid
        """
        if not isinstance(data, dict):
            raise ValidationError("Data must be a dictionary")

        filtered_data = data.copy()

        if 'results' in filtered_data:
            if not isinstance(filtered_data['results'], list):
                raise ValidationError("Results must be a list")

            filtered_results = []

            for result in filtered_data['results']:
                if not isinstance(result, dict):
                    raise ValidationError("Each result must be a dictionary")

                filtered_result = result.copy()

                if not include_abstracts and 'abstract' in filtered_result:
                    del filtered_result['abstract']

                if not include_metadata:
                    for field in ['impact_factor', 'citation_count', 'authority_score']:
                        if field in filtered_result:
                            del filtered_result[field]

                filtered_results.append(filtered_result)

            filtered_data['results'] = filtered_results

        return filtered_data
