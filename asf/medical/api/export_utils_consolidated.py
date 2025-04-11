"""
Export Utilities for Medical Research Synthesizer

This module provides functions to export research data in various formats:
- JSON
- CSV
- Excel
- PDF

It combines the functionality of the original export_utils.py and export_utils_v2.py
into a single, comprehensive implementation.
"""

import io
import csv
import json
import os
import tempfile
import logging
from datetime import datetime
import pandas as pd
from typing import List, Dict, Any, Optional
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fpdf import FPDF

# Set up logging
logger = logging.getLogger(__name__)

def export_to_json(data: List[Dict[str, Any]], query_text: Optional[str] = None) -> JSONResponse:
    """
    Export data to JSON response.

    Args:
        data: List of dictionaries containing publication data
        query_text: Optional query text that produced the results

    Returns:
        JSONResponse object

    Raises:
        ValueError: If data is not a list or contains invalid items
    """
    # Validate input
    if not isinstance(data, list):
        raise ValueError("Data must be a list of dictionaries")

    try:
        # First, clean the data to ensure it's JSON serializable
        cleaned_data = []
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                raise ValueError(f"Item at index {i} is not a dictionary")

            cleaned_item = {}
            for key, value in item.items():
                # Skip None values and ensure all values are JSON serializable
                if value is not None:
                    if isinstance(value, (dict, list, str, int, float, bool, type(None))):
                        cleaned_item[key] = value
                    else:
                        cleaned_item[key] = str(value)
            cleaned_data.append(cleaned_item)

        # Create response data
        response_data = {
            "count": len(cleaned_data),
            "results": cleaned_data,
            "exported_at": datetime.now().isoformat()
        }

        # Add query information if provided
        if query_text:
            response_data["query"] = query_text

        return JSONResponse(content=response_data)
    except Exception as e:
        # Log the error and re-raise with a more informative message
        logger.error(f"Error exporting to JSON: {str(e)}")
        raise ValueError(f"Failed to export data to JSON: {str(e)}") from e

def export_to_csv(data: List[Dict[str, Any]], query_text: Optional[str] = None) -> StreamingResponse:
    """
    Export data to CSV.

    Args:
        data: List of dictionaries containing publication data
        query_text: Optional query text that produced the results

    Returns:
        StreamingResponse object containing CSV data

    Raises:
        ValueError: If data is not a list or contains invalid items
    """
    # Validate input
    if not isinstance(data, list):
        raise ValueError("Data must be a list of dictionaries")

    try:
        output = io.StringIO()

        if not data:
            # Create empty CSV
            writer = csv.writer(output)
            writer.writerow([])

            # Create response
            response = StreamingResponse(
                iter([output.getvalue()]),
                media_type="text/csv"
            )
            response.headers["Content-Disposition"] = f"attachment; filename=search_results.csv"
            return response

        # Add query information as a comment if provided
        if query_text:
            output.write(f"# Query: {query_text}\n")
            output.write(f"# Exported at: {datetime.now().isoformat()}\n")

        # Define which fields to include and their display names
        fields_to_include = [
            'pmid', 'title', 'journal', 'publication_date', 'iso_date',
            'authors', 'abstract', 'impact_factor', 'journal_quartile',
            'citation_count', 'authority_score', 'publication_types',
            'doi', 'mesh_terms'
        ]

        # Get all unique keys from all results
        all_keys = set()
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                raise ValueError(f"Item at index {i} is not a dictionary")
            all_keys.update(item.keys())

        # Prioritize fields_to_include, then add any remaining fields
        available_fields = [field for field in fields_to_include if field in all_keys]
        remaining_fields = sorted(all_keys - set(available_fields))
        available_fields.extend(remaining_fields)

        writer = csv.DictWriter(output, fieldnames=available_fields)
        writer.writeheader()

        for item in data:
            # Convert complex fields to strings
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

        # Create response
        response = StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv"
        )
        response.headers["Content-Disposition"] = f"attachment; filename=search_results.csv"

        return response
    except Exception as e:
        # Log the error and re-raise with a more informative message
        logger.error(f"Error exporting to CSV: {str(e)}")
        raise ValueError(f"Failed to export data to CSV: {str(e)}") from e

def export_to_excel(data: List[Dict[str, Any]], query_text: Optional[str] = None) -> StreamingResponse:
    """
    Export data to Excel file.

    Args:
        data: List of dictionaries containing publication data
        query_text: Optional query text that produced the results

    Returns:
        StreamingResponse object containing Excel data

    Raises:
        ValueError: If data is not a list or contains invalid items
    """
    # Validate input
    if not isinstance(data, list):
        raise ValueError("Data must be a list of dictionaries")

    try:
        output = io.BytesIO()

        if not data:
            # Create an empty Excel file
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                pd.DataFrame().to_excel(writer, sheet_name='Results', index=False)
            output.seek(0)

            # Create response
            response = StreamingResponse(
                iter([output.getvalue()]),
                media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            response.headers["Content-Disposition"] = f"attachment; filename=search_results.xlsx"
            return response

        # Define which fields to include and their display names
        fields_to_include = [
            'pmid', 'title', 'journal', 'publication_date', 'iso_date',
            'authors', 'abstract', 'impact_factor', 'journal_quartile',
            'citation_count', 'authority_score'
        ]

        # Field display names
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

        # Prepare data for Excel
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

        # Convert to DataFrame
        df = pd.DataFrame(excel_data)

        # Rename columns using field_names
        df.rename(columns=field_names, inplace=True)

        # Write to Excel
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name='Results', index=False)

            # Get the workbook and worksheet objects
            workbook = writer.book
            worksheet = writer.sheets['Results']

            # Add formats
            header_format = workbook.add_format({
                'bold': True,
                'text_wrap': True,
                'valign': 'top',
                'bg_color': '#D9EAD3',
                'border': 1
            })

            # Set the column width and format
            for i, col in enumerate(df.columns):
                # Set column width based on field
                if col in ['Title', 'Abstract']:
                    width = 50
                elif col in ['Authors']:
                    width = 30
                elif col in ['Journal']:
                    width = 25
                else:
                    width = 15

                worksheet.set_column(i, i, width)

            # Set the header row
            for i, col in enumerate(df.columns):
                worksheet.write(0, i, col, header_format)

            # Add query information to a new sheet if provided
            if query_text:
                info_sheet = workbook.add_worksheet("Query Info")
                info_sheet.write(0, 0, "Query")
                info_sheet.write(0, 1, query_text)
                info_sheet.write(1, 0, "Results Count")
                info_sheet.write(1, 1, len(data))

        # Add a second sheet with publication types and mesh terms
        if any('publication_types' in item or 'mesh_terms' in item for item in data):
            with pd.ExcelWriter(output, mode='a', engine='openpyxl') as writer:
                # Prepare data
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

                # Convert to DataFrame and write to sheet
                terms_df = pd.DataFrame(terms_data)
                terms_df.to_excel(writer, sheet_name='Types & Terms', index=False)

        output.seek(0)

        # Create response
        response = StreamingResponse(
            iter([output.getvalue()]),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        response.headers["Content-Disposition"] = f"attachment; filename=search_results.xlsx"

        return response
    except Exception as e:
        # Log the error and re-raise with a more informative message
        logger.error(f"Error exporting to Excel: {str(e)}")
        raise ValueError(f"Failed to export data to Excel: {str(e)}") from e

def export_to_pdf(data: List[Dict[str, Any]], query_text: Optional[str] = None) -> FileResponse:
    """
    Export data to PDF.

    Args:
        data: List of dictionaries containing publication data
        query_text: Optional query text that produced the results

    Returns:
        FileResponse object containing PDF data

    Raises:
        ValueError: If data is not a list or contains invalid items
    """
    # Validate input
    if not isinstance(data, list):
        raise ValueError("Data must be a list of dictionaries")

    try:
        # Create PDF
        pdf = FPDF()
        pdf.add_page()

        # Set font
        pdf.set_font("Arial", "B", 16)

        # Add title
        pdf.cell(0, 10, "Search Results", 0, 1, "C")

        # Add query information if provided
        if query_text:
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 10, f"Query: {query_text}", 0, 1)

        pdf.cell(0, 10, f"Results Count: {len(data)}", 0, 1)

        # Add results
        pdf.set_font("Arial", "", 10)

        for i, item in enumerate(data):
            if not isinstance(item, dict):
                raise ValueError(f"Item at index {i} is not a dictionary")

            # Add article title
            pdf.set_font("Arial", "B", 12)
            title = item.get("title", "No Title")
            pdf.multi_cell(0, 10, f"{i+1}. {title}")

            # Add article details
            pdf.set_font("Arial", "", 10)

            # Authors
            authors = item.get("authors", [])
            if authors:
                if isinstance(authors, list):
                    authors_str = ", ".join(authors)
                else:
                    authors_str = str(authors)
                pdf.multi_cell(0, 6, f"Authors: {authors_str}")

            # Journal and date
            journal = item.get("journal", "Unknown Journal")
            date = item.get("publication_date", "Unknown Date")
            pdf.multi_cell(0, 6, f"Journal: {journal}, Date: {date}")

            # PMID and DOI
            pmid = item.get("pmid", "")
            doi = item.get("doi", "")
            if pmid or doi:
                pdf.multi_cell(0, 6, f"PMID: {pmid}, DOI: {doi}")

            # Impact factor and citation count
            impact_factor = item.get("impact_factor", "")
            citation_count = item.get("citation_count", "")
            if impact_factor or citation_count:
                pdf.multi_cell(0, 6, f"Impact Factor: {impact_factor}, Citations: {citation_count}")

            # Abstract
            abstract = item.get("abstract", "")
            if abstract:
                pdf.set_font("Arial", "I", 10)
                pdf.multi_cell(0, 6, f"Abstract: {abstract}")

            # Page break between articles
            if i < len(data) - 1:
                pdf.add_page()

        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        temp_file.close()

        # Save PDF to file
        pdf.output(temp_file.name)

        # Create response
        return FileResponse(
            path=temp_file.name,
            filename="search_results.pdf",
            media_type="application/pdf",
            background=lambda: os.unlink(temp_file.name)
        )
    except Exception as e:
        # Log the error and re-raise with a more informative message
        logger.error(f"Error exporting to PDF: {str(e)}")
        raise ValueError(f"Failed to export data to PDF: {str(e)}") from e

def export_contradiction_analysis_to_pdf(analysis: Dict[str, Any], query_text: str, output_path: Optional[str] = None) -> FileResponse:
    """
    Export contradiction analysis to PDF.

    Args:
        analysis: Contradiction analysis
        query_text: Query text
        output_path: Optional output file path

    Returns:
        FileResponse object containing PDF data

    Raises:
        ValueError: If analysis is not a dictionary or contains invalid data
    """
    # Validate input
    if not isinstance(analysis, dict):
        raise ValueError("Analysis must be a dictionary")

    try:
        # Create PDF
        pdf = FPDF()
        pdf.add_page()

        # Set font
        pdf.set_font("Arial", "B", 16)

        # Add title
        pdf.cell(0, 10, "Contradiction Analysis", 0, 1, "C")

        # Add query information
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, f"Query: {query_text}", 0, 1)
        pdf.cell(0, 10, f"Total Articles: {analysis.get('total_articles', 0)}", 0, 1)
        pdf.cell(0, 10, f"Contradictions Found: {analysis.get('num_contradictions', 0)}", 0, 1)

        # Add contradictions by topic
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 15, "Contradictions by Topic", 0, 1)

        pdf.set_font("Arial", "", 10)

        # Add contradictions
        contradictions = analysis.get("contradictions", [])
        if not isinstance(contradictions, list):
            raise ValueError("Contradictions must be a list")

        for i, contradiction in enumerate(contradictions):
            if not isinstance(contradiction, dict):
                raise ValueError(f"Contradiction at index {i} is not a dictionary")

            # Add contradiction title
            pdf.set_font("Arial", "B", 12)
            topic = contradiction.get("topic", f"Contradiction {i+1}")
            pdf.multi_cell(0, 10, f"{i+1}. {topic}")

            # Add contradiction details
            pdf.set_font("Arial", "", 10)

            # Claims
            claim1 = contradiction.get("claim1", "")
            claim2 = contradiction.get("claim2", "")
            pdf.multi_cell(0, 6, f"Claim 1: {claim1}")
            pdf.multi_cell(0, 6, f"Claim 2: {claim2}")

            # Sources
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

            # Contradiction type and confidence
            contradiction_type = contradiction.get("type", "Unknown")
            confidence = contradiction.get("confidence", 0.0)
            pdf.multi_cell(0, 6, f"Type: {contradiction_type}, Confidence: {confidence:.2f}")

            # Explanation
            explanation = contradiction.get("explanation", "")
            if explanation:
                pdf.set_font("Arial", "I", 10)
                pdf.multi_cell(0, 6, f"Explanation: {explanation}")

            # Resolution
            resolution = contradiction.get("resolution", "")
            if resolution:
                pdf.set_font("Arial", "B", 10)
                pdf.multi_cell(0, 6, f"Resolution: {resolution}")

            # Page break between contradictions
            if i < len(contradictions) - 1:
                pdf.add_page()

        # Create temporary file if output_path not provided
        if not output_path:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            temp_file.close()
            output_path = temp_file.name

        # Save PDF to file
        pdf.output(output_path)

        # Create response
        return FileResponse(
            path=output_path,
            filename="contradiction_analysis.pdf",
            media_type="application/pdf",
            background=lambda: os.unlink(output_path)
        )
    except Exception as e:
        # Log the error and re-raise with a more informative message
        logger.error(f"Error exporting contradiction analysis to PDF: {str(e)}")
        raise ValueError(f"Failed to export contradiction analysis to PDF: {str(e)}") from e
