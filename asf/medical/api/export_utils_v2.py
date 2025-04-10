"""
Export utilities for the Medical Research Synthesizer API.

This module provides functions for exporting search results and analyses
in various formats (JSON, CSV, Excel, PDF).
"""

import json
import csv
import io
import tempfile
import os
from typing import Dict, List, Any
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
import pandas as pd
from fpdf import FPDF

def export_to_json(results: List[Dict[str, Any]], query_text: str) -> JSONResponse:
    """
    Export results to JSON.
    
    Args:
        results: Search results
        query_text: Query text
        
    Returns:
        JSON response
    """
    # Create JSON response
    response_data = {
        "query": query_text,
        "count": len(results),
        "results": results
    }
    
    return JSONResponse(content=response_data)

def export_to_csv(results: List[Dict[str, Any]], query_text: str) -> StreamingResponse:
    """
    Export results to CSV.
    
    Args:
        results: Search results
        query_text: Query text
        
    Returns:
        CSV response
    """
    # Create CSV
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Write header
    if results:
        # Get all unique keys from all results
        keys = set()
        for result in results:
            keys.update(result.keys())
        
        # Sort keys for consistent output
        header = sorted(keys)
        writer.writerow(header)
        
        # Write rows
        for result in results:
            row = [result.get(key, "") for key in header]
            writer.writerow(row)
    
    # Create response
    response = StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv"
    )
    response.headers["Content-Disposition"] = f"attachment; filename=search_results.csv"
    
    return response

def export_to_excel(results: List[Dict[str, Any]], query_text: str) -> StreamingResponse:
    """
    Export results to Excel.
    
    Args:
        results: Search results
        query_text: Query text
        
    Returns:
        Excel response
    """
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Create Excel
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name="Results", index=False)
        
        # Add query information
        workbook = writer.book
        worksheet = writer.sheets["Results"]
        
        # Add query information to a new sheet
        info_sheet = workbook.add_worksheet("Query Info")
        info_sheet.write(0, 0, "Query")
        info_sheet.write(0, 1, query_text)
        info_sheet.write(1, 0, "Results Count")
        info_sheet.write(1, 1, len(results))
    
    # Create response
    response = StreamingResponse(
        iter([output.getvalue()]),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    response.headers["Content-Disposition"] = f"attachment; filename=search_results.xlsx"
    
    return response

def export_to_pdf(results: List[Dict[str, Any]], query_text: str) -> FileResponse:
    """
    Export results to PDF.
    
    Args:
        results: Search results
        query_text: Query text
        
    Returns:
        PDF response
    """
    # Create PDF
    pdf = FPDF()
    pdf.add_page()
    
    # Set font
    pdf.set_font("Arial", "B", 16)
    
    # Add title
    pdf.cell(0, 10, "Search Results", 0, 1, "C")
    
    # Add query information
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, f"Query: {query_text}", 0, 1)
    pdf.cell(0, 10, f"Results Count: {len(results)}", 0, 1)
    
    # Add results
    pdf.set_font("Arial", "", 10)
    
    for i, result in enumerate(results):
        pdf.cell(0, 10, f"Result {i+1}", 0, 1)
        
        # Add result details
        for key, value in result.items():
            # Skip complex values
            if isinstance(value, (dict, list)):
                continue
            
            # Format value
            if value is None:
                value = ""
            elif not isinstance(value, str):
                value = str(value)
            
            # Truncate long values
            if len(value) > 100:
                value = value[:100] + "..."
            
            pdf.cell(0, 10, f"{key}: {value}", 0, 1)
        
        # Add separator
        pdf.cell(0, 5, "", 0, 1)
    
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

def export_contradiction_analysis_to_pdf(analysis: Dict[str, Any], query_text: str, output_path: str) -> str:
    """
    Export contradiction analysis to PDF.
    
    Args:
        analysis: Contradiction analysis
        query_text: Query text
        output_path: Output file path
        
    Returns:
        Output file path
    """
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
    
    by_topic = analysis.get("by_topic", {})
    for topic, contradictions in by_topic.items():
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, f"Topic: {topic} ({len(contradictions)} contradictions)", 0, 1)
        
        pdf.set_font("Arial", "", 10)
        for i, contradiction in enumerate(contradictions):
            pdf.cell(0, 10, f"Contradiction {i+1}:", 0, 1)
            
            # Add contradiction details
            pdf.cell(0, 10, f"Publication 1: {contradiction.get('publication1', {}).get('title', '')}", 0, 1)
            pdf.cell(0, 10, f"Publication 2: {contradiction.get('publication2', {}).get('title', '')}", 0, 1)
            pdf.cell(0, 10, f"Contradiction Score: {contradiction.get('contradiction_score', 0):.2f}", 0, 1)
            pdf.cell(0, 10, f"Confidence: {contradiction.get('confidence', '')}", 0, 1)
            
            # Add separator
            pdf.cell(0, 5, "", 0, 1)
    
    # Save PDF to file
    pdf.output(output_path)
    
    return output_path
