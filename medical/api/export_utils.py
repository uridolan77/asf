"""
Consolidated Export Utilities for Medical Research Synthesizer

This module provides functions to export research data in various formats:
- JSON
- CSV
- Excel
- PDF

It combines all export functionality into a single, comprehensive implementation
with common utilities for data validation, cleaning, and metadata handling.
"""

import io
import csv
import json
import os
import tempfile
import logging
import pandas as pd
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fpdf import FPDF

from ..core.exceptions import ValidationError, ExportError

logger = logging.getLogger(__name__)


# Common Utility Functions

def validate_export_data(data: Any) -> List[Dict[str, Any]]:
    """
    Validate export data to ensure it's a list of dictionaries.
    
    Args:
        data: Data to validate
    
    Returns:
        Validated list of dictionaries
    
    Raises:
        ValidationError: If data is not a list or contains invalid items
    """
    if not isinstance(data, list):
        raise ValidationError("Data must be a list of dictionaries")
    
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValidationError(f"Item at index {i} is not a dictionary")
    
    return data


def clean_export_data(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Clean export data by handling non-serializable values.
    
    Args:
        data: List of dictionaries to clean
    
    Returns:
        Cleaned list of dictionaries
    """
    cleaned_data = []
    
    for item in data:
        cleaned_item = {}
        for key, value in item.items():
            if value is not None:
                if isinstance(value, (dict, list, str, int, float, bool, type(None))):
                    cleaned_item[key] = value
                else:
                    cleaned_item[key] = str(value)
        cleaned_data.append(cleaned_item)
    
    return cleaned_data


def filter_export_data(
    data: List[Dict[str, Any]],
    include_fields: Optional[List[str]] = None,
    exclude_fields: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    Filter export data to include or exclude specific fields.
    
    Args:
        data: List of dictionaries to filter
        include_fields: Fields to include (if None, include all)
        exclude_fields: Fields to exclude
    
    Returns:
        Filtered list of dictionaries
    """
    exclude_fields = exclude_fields or []
    filtered_data = []
    
    for item in data:
        filtered_item = {}
        
        if include_fields:
            # Include only specified fields
            for field in include_fields:
                if field in item and field not in exclude_fields:
                    filtered_item[field] = item[field]
        else:
            # Include all fields except excluded ones
            for key, value in item.items():
                if key not in exclude_fields:
                    filtered_item[key] = value
        
        filtered_data.append(filtered_item)
    
    return filtered_data


def get_export_metadata(query_text: Optional[str] = None) -> Dict[str, Any]:
    """
    Generate metadata for export files.
    
    Args:
        query_text: Optional query text that produced the results
    
    Returns:
        Dictionary containing export metadata
    """
    metadata = {
        "exported_at": datetime.now().isoformat(),
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
    }
    
    if query_text:
        metadata["query"] = query_text
    
    return metadata


def get_common_fields() -> Dict[str, List[str]]:
    """
    Get common field lists for different export types.
    
    Returns:
        Dictionary containing field lists for different export types
    """
    return {
        "basic": [
            'pmid', 'title', 'journal', 'publication_date', 'authors'
        ],
        "standard": [
            'pmid', 'title', 'journal', 'publication_date', 'iso_date',
            'authors', 'abstract', 'doi'
        ],
        "detailed": [
            'pmid', 'title', 'journal', 'publication_date', 'iso_date',
            'authors', 'abstract', 'impact_factor', 'journal_quartile',
            'citation_count', 'authority_score', 'publication_types',
            'doi', 'mesh_terms'
        ],
        "analysis": [
            'pmid', 'title', 'journal', 'publication_date', 'authors',
            'abstract', 'contradiction_score', 'contradiction_type',
            'evidence_level'
        ]
    }


def handle_export_error(e: Exception, operation: str) -> None:
    """
    Handle export errors consistently.
    
    Args:
        e: The exception that occurred
        operation: The export operation that failed
    
    Raises:
        ExportError: Always raised with consistent formatting
    """
    error_message = f"Failed to export data to {operation}: {str(e)}"
    logger.error(error_message)
    raise ExportError(error_message) from e


# Format-specific Export Functions

def export_to_json(data: List[Dict[str, Any]], query_text: Optional[str] = None) -> JSONResponse:
    """
    Export data to JSON response.

    Args:
        data: List of dictionaries containing publication data
        query_text: Optional query text that produced the results

    Returns:
        JSONResponse object

    Raises:
        ValidationError: If data is not a list or contains invalid items
        ExportError: If there's an issue exporting the data
    """
    try:
        # Validate and clean the data
        validated_data = validate_export_data(data)
        cleaned_data = clean_export_data(validated_data)

        # Prepare the response data
        metadata = get_export_metadata(query_text)

        response_data = {
            "count": len(cleaned_data),
            "results": cleaned_data,
            **metadata
        }

        return JSONResponse(content=response_data)
    except Exception as e:
        handle_export_error(e, "JSON")


def export_to_csv(data: List[Dict[str, Any]], query_text: Optional[str] = None) -> StreamingResponse:
    """
    Export data to CSV.

    Args:
        data: List of dictionaries containing publication data
        query_text: Optional query text that produced the results

    Returns:
        StreamingResponse object containing CSV data

    Raises:
        ValidationError: If data is not a list or contains invalid items
        ExportError: If there's an issue exporting the data
    """
    try:
        # Validate and clean the data
        validated_data = validate_export_data(data)
        cleaned_data = clean_export_data(validated_data)

        output = io.StringIO()

        if not cleaned_data:
            writer = csv.writer(output)
            writer.writerow([])

            response = StreamingResponse(
                iter([output.getvalue()]),
                media_type="text/csv"
            )
            response.headers["Content-Disposition"] = f"attachment; filename=search_results.csv"
            return response

        # Add metadata as comments
        metadata = get_export_metadata(query_text)
        if query_text:
            output.write(f"# Query: {query_text}\n")
        output.write(f"# Exported at: {metadata['exported_at']}\n")

        # Get fields to include
        fields_to_include = get_common_fields()['detailed']

        all_keys = set()
        for item in cleaned_data:
            all_keys.update(item.keys())

        available_fields = [field for field in fields_to_include if field in all_keys]
        remaining_fields = sorted(all_keys - set(available_fields))
        available_fields.extend(remaining_fields)

        writer = csv.DictWriter(output, fieldnames=available_fields)
        writer.writeheader()

        for item in cleaned_data:
            row = {}
            for field in available_fields:
                if field in item:
                    value = item[field]
                    if isinstance(value, list):
                        row[field] = '; '.join(str(v) for v in value)
                    elif isinstance(value, dict):
                        row[field] = json.dumps(value)
                    else:
                        row[field] = value
                else:
                    row[field] = ''
            writer.writerow(row)

        response = StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv"
        )
        response.headers["Content-Disposition"] = f"attachment; filename=search_results.csv"

        return response
    except Exception as e:
        handle_export_error(e, "CSV")


def export_to_excel(data: List[Dict[str, Any]], query_text: Optional[str] = None) -> StreamingResponse:
    """
    Export data to Excel file.

    Args:
        data: List of dictionaries containing publication data
        query_text: Optional query text that produced the results

    Returns:
        StreamingResponse object containing Excel data

    Raises:
        ValidationError: If data is not a list or contains invalid items
        ExportError: If there's an issue exporting the data
    """
    try:
        # Validate and clean the data
        validated_data = validate_export_data(data)
        cleaned_data = clean_export_data(validated_data)

        output = io.BytesIO()

        if not cleaned_data:
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                pd.DataFrame().to_excel(writer, sheet_name='Results', index=False)
            output.seek(0)

            response = StreamingResponse(
                iter([output.getvalue()]),
                media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            response.headers["Content-Disposition"] = f"attachment; filename=search_results.xlsx"
            return response

        # Get fields to include
        fields_to_include = get_common_fields()['standard']

        field_names = {
            'pmid': 'PMID',
            'title': 'Title',
            'journal': 'Journal',
            'publication_date': 'Publication Date',
            'iso_date': 'ISO Date',
            'authors': 'Authors',
            'abstract': 'Abstract',
            'impact_factor': 'Impact Factor',
            'journal_quartile': 'Journal Quartile',
            'citation_count': 'Citation Count',
            'authority_score': 'Authority Score'
        }

        excel_data = []
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                raise ValueError(f"Item at index {i} is not a dictionary")

            row = {}
            for field in fields_to_include:
                if field in item:
                    value = item[field]
                    if field == 'authors' and isinstance(value, list):
                        row[field] = ', '.join(value)
                    elif isinstance(value, (dict, list)):
                        row[field] = str(value)
                    else:
                        row[field] = value
                else:
                    row[field] = ''
            excel_data.append(row)

        df = pd.DataFrame(excel_data)

        df.rename(columns=field_names, inplace=True)

        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name='Results', index=False)

            workbook = writer.book
            worksheet = writer.sheets['Results']

            header_format = workbook.add_format({
                'bold': True,
                'text_wrap': True,
                'valign': 'top',
                'bg_color': '#D9EAD3',
                'border': 1
            })

            for i, col in enumerate(df.columns):
                if col in ['Title', 'Abstract']:
                    width = 50
                elif col in ['Authors']:
                    width = 30
                elif col in ['Journal']:
                    width = 25
                else:
                    width = 15

                worksheet.set_column(i, i, width)

            for i, col in enumerate(df.columns):
                worksheet.write(0, i, col, header_format)

            if query_text:
                info_sheet = workbook.add_worksheet("Query Info")
                info_sheet.write(0, 0, "Query")
                info_sheet.write(0, 1, query_text)
                info_sheet.write(1, 0, "Results Count")
                info_sheet.write(1, 1, len(data))

        if any('publication_types' in item or 'mesh_terms' in item for item in data):
            with pd.ExcelWriter(output, mode='a', engine='openpyxl') as writer:
                terms_data = []
                for item in data:
                    pmid = item.get('pmid', '')
                    title = item.get('title', '')
                    pub_types = item.get('publication_types', [])
                    mesh_terms = item.get('mesh_terms', [])

                    if isinstance(pub_types, list):
                        pub_types_str = '; '.join(pub_types)
                    else:
                        pub_types_str = str(pub_types)

                    if isinstance(mesh_terms, list):
                        mesh_terms_str = '; '.join(mesh_terms)
                    else:
                        mesh_terms_str = str(mesh_terms)

                    terms_data.append({
                        'PMID': pmid,
                        'Title': title,
                        'Publication Types': pub_types_str,
                        'MeSH Terms': mesh_terms_str
                    })

                terms_df = pd.DataFrame(terms_data)
                terms_df.to_excel(writer, sheet_name='Types & Terms', index=False)

        output.seek(0)

        response = StreamingResponse(
            iter([output.getvalue()]),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        response.headers["Content-Disposition"] = f"attachment; filename=search_results.xlsx"

        return response
    except Exception as e:
        handle_export_error(e, "Excel")


def export_to_pdf(data: List[Dict[str, Any]], query_text: Optional[str] = None) -> FileResponse:
    """
    Export data to PDF.

    Args:
        data: List of dictionaries containing publication data
        query_text: Optional query text that produced the results

    Returns:
        FileResponse object containing PDF data

    Raises:
        ValidationError: If data is not a list or contains invalid items
        ExportError: If there's an issue exporting the data
    """
    try:
        # Validate and clean the data
        validated_data = validate_export_data(data)
        cleaned_data = clean_export_data(validated_data)
        pdf = FPDF()
        pdf.add_page()

        pdf.set_font("Arial", "B", 16)

        pdf.cell(0, 10, "Search Results", 0, 1, "C")

        if query_text:
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 10, f"Query: {query_text}", 0, 1)

        pdf.cell(0, 10, f"Results Count: {len(cleaned_data)}", 0, 1)

        pdf.set_font("Arial", "", 10)

        for i, item in enumerate(cleaned_data):

            pdf.set_font("Arial", "B", 12)
            title = item.get("title", "No Title")
            pdf.multi_cell(0, 10, f"{i+1}. {title}")

            pdf.set_font("Arial", "", 10)

            authors = item.get("authors", [])
            if authors:
                if isinstance(authors, list):
                    authors_str = ", ".join(authors)
                else:
                    authors_str = str(authors)
                pdf.multi_cell(0, 6, f"Authors: {authors_str}")

            journal = item.get("journal", "Unknown Journal")
            date = item.get("publication_date", "Unknown Date")
            pdf.multi_cell(0, 6, f"Journal: {journal}, Date: {date}")

            pmid = item.get("pmid", "")
            doi = item.get("doi", "")
            if pmid or doi:
                pdf.multi_cell(0, 6, f"PMID: {pmid}, DOI: {doi}")

            impact_factor = item.get("impact_factor", "")
            citation_count = item.get("citation_count", "")
            if impact_factor or citation_count:
                pdf.multi_cell(0, 6, f"Impact Factor: {impact_factor}, Citations: {citation_count}")

            abstract = item.get("abstract", "")
            if abstract:
                pdf.set_font("Arial", "I", 10)
                pdf.multi_cell(0, 6, f"Abstract: {abstract}")

            if i < len(data) - 1:
                pdf.add_page()

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        temp_file.close()

        pdf.output(temp_file.name)

        return FileResponse(
            path=temp_file.name,
            filename="search_results.pdf",
            media_type="application/pdf",
            background=lambda: os.unlink(temp_file.name)
        )
    except Exception as e:
        handle_export_error(e, "PDF")


def export_contradiction_analysis_to_pdf(
    analysis: Dict[str, Any], 
    query_text: str, 
    output_path: Optional[str] = None
) -> FileResponse:
    """
    Export contradiction analysis to PDF.

    Args:
        analysis: Contradiction analysis
        query_text: Query text
        output_path: Optional output file path

    Returns:
        FileResponse object containing PDF data

    Raises:
        ValidationError: If analysis is not a dictionary or contains invalid data
        ExportError: If there's an issue exporting the data
    """
    try:
        # Validate the analysis
        if not isinstance(analysis, dict):
            raise ValidationError("Analysis must be a dictionary")
        pdf = FPDF()
        pdf.add_page()

        pdf.set_font("Arial", "B", 16)

        pdf.cell(0, 10, "Contradiction Analysis", 0, 1, "C")

        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, f"Query: {query_text}", 0, 1)
        pdf.cell(0, 10, f"Total Articles: {analysis.get('total_articles', 0)}", 0, 1)
        pdf.cell(0, 10, f"Contradictions Found: {analysis.get('num_contradictions', 0)}", 0, 1)

        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 15, "Contradictions by Topic", 0, 1)

        pdf.set_font("Arial", "", 10)

        contradictions = analysis.get("contradictions", [])
        if not isinstance(contradictions, list):
            raise ValueError("Contradictions must be a list")

        for i, contradiction in enumerate(contradictions):
            if not isinstance(contradiction, dict):
                raise ValueError(f"Contradiction at index {i} is not a dictionary")

            pdf.set_font("Arial", "B", 12)
            topic = contradiction.get("topic", f"Contradiction {i+1}")
            pdf.multi_cell(0, 10, f"{i+1}. {topic}")

            pdf.set_font("Arial", "", 10)

            claim1 = contradiction.get("claim1", "")
            claim2 = contradiction.get("claim2", "")
            pdf.multi_cell(0, 6, f"Claim 1: {claim1}")
            pdf.multi_cell(0, 6, f"Claim 2: {claim2}")

            source1 = contradiction.get("source1", {})
            source2 = contradiction.get("source2", {})

            source1_title = source1.get("title", "Unknown Source")
            source1_authors = source1.get("authors", [])
            if isinstance(source1_authors, list):
                source1_authors_str = ", ".join(source1_authors)
            else:
                source1_authors_str = str(source1_authors)

            source2_title = source2.get("title", "Unknown Source")
            source2_authors = source2.get("authors", [])
            if isinstance(source2_authors, list):
                source2_authors_str = ", ".join(source2_authors)
            else:
                source2_authors_str = str(source2_authors)

            pdf.multi_cell(0, 6, f"Source 1: {source1_title} by {source1_authors_str}")
            pdf.multi_cell(0, 6, f"Source 2: {source2_title} by {source2_authors_str}")

            contradiction_type = contradiction.get("type", "Unknown")
            confidence = contradiction.get("confidence", 0.0)
            pdf.multi_cell(0, 6, f"Type: {contradiction_type}, Confidence: {confidence:.2f}")

            explanation = contradiction.get("explanation", "")
            if explanation:
                pdf.set_font("Arial", "I", 10)
                pdf.multi_cell(0, 6, f"Explanation: {explanation}")

            resolution = contradiction.get("resolution", "")
            if resolution:
                pdf.set_font("Arial", "B", 10)
                pdf.multi_cell(0, 6, f"Resolution: {resolution}")

            if i < len(contradictions) - 1:
                pdf.add_page()

        if not output_path:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            temp_file.close()
            output_path = temp_file.name

        pdf.output(output_path)

        return FileResponse(
            path=output_path,
            filename="contradiction_analysis.pdf",
            media_type="application/pdf",
            background=lambda: os.unlink(output_path)
        )
    except Exception as e:
        handle_export_error(e, "PDF")