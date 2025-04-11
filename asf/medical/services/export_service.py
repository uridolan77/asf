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
from typing import Dict, Any
from datetime import datetime

from asf.medical.core.exceptions import (
    ExportError, ValidationError, FileError
)

logger = logging.getLogger(__name__)

class ExportService:
    """
    Service for exporting search results and analyses.
    """

    def __init__(self, export_dir: str = "exports"):
        """
        Initialize the export service.

        Args:
            export_dir: Directory for storing exports

        Raises:
            FileError: If there's an error creating the export directory
        """
        self.export_dir = export_dir

        try:
            os.makedirs(self.export_dir, exist_ok=True)
            logger.info(f"Export directory initialized: {self.export_dir}")
        except Exception as e:
    logger.error(f\"Failed to create export directory: {str(e)}\")
    raise DatabaseError(f\"Failed to create export directory: {str(e)}\") FileError(self.export_dir, f"Failed to create export directory: {str(e)}")

    async def export_to_json(
        self,
        data: Dict[str, Any],
        include_abstracts: bool = True,
        include_metadata: bool = True
    ) -> str:
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
