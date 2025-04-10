"""
Export Utilities for Medical Research Synthesizer

This module provides functions to export research data in various formats:
- JSON
- CSV
- Excel
- PDF
"""

import io
import csv
import json
import pandas as pd
from typing import List, Dict, Any, BinaryIO
from fpdf import FPDF

def export_to_json(data: List[Dict[str, Any]]) -> str:
    """
    Export data to JSON string.

    Args:
        data: List of dictionaries containing publication data

    Returns:
        JSON formatted string
    """
    # First, clean the data to ensure it's JSON serializable
    cleaned_data = []
    for item in data:
        cleaned_item = {}
        for key, value in item.items():
            # Skip None values and ensure all values are JSON serializable
            if value is not None:
                if isinstance(value, (dict, list, str, int, float, bool, type(None))):
                    cleaned_item[key] = value
                else:
                    cleaned_item[key] = str(value)
        cleaned_data.append(cleaned_item)

    return json.dumps(cleaned_data, indent=2, ensure_ascii=False)

def export_to_csv(data: List[Dict[str, Any]]) -> io.StringIO:
    """
    Export data to CSV.

    Args:
        data: List of dictionaries containing publication data

    Returns:
        StringIO object containing CSV data
    """
    output = io.StringIO()
    if not data:
        return output

    # Define which fields to include and their display names
    fields_to_include = [
        'pmid', 'title', 'journal', 'publication_date', 'iso_date',
        'authors', 'abstract', 'impact_factor', 'journal_quartile',
        'citation_count', 'authority_score', 'publication_types',
        'doi', 'mesh_terms'
    ]

    # Check which fields actually exist in the data
    if data:
        available_fields = [field for field in fields_to_include if field in data[0]]
    else:
        available_fields = []

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

    output.seek(0)
    return output

def export_to_excel(data: List[Dict[str, Any]]) -> io.BytesIO:
    """
    Export data to Excel file.

    Args:
        data: List of dictionaries containing publication data

    Returns:
        BytesIO object containing Excel data
    """
    output = io.BytesIO()

    if not data:
        # Create an empty Excel file
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            pd.DataFrame().to_excel(writer, sheet_name='Results', index=False)
        output.seek(0)
        return output

    # Define which fields to include and their display names
    fields_to_include = [
        'pmid', 'title', 'journal', 'publication_date', 'iso_date',
        'authors', 'abstract', 'impact_factor', 'journal_quartile',
        'citation_count', 'authority_score'
    ]

    # Prepare data for Excel
    excel_data = []
    for item in data:
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

    # Rename columns
    df.rename(columns={f: field_names.get(f, f) for f in df.columns}, inplace=True)

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

        # Create a table for the data
        table_options = {
            'columns': [{'header': col} for col in df.columns],
            'style': 'Table Style Medium 2'
        }
        worksheet.add_table(0, 0, len(df), len(df.columns) - 1, table_options)

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
    return output

def export_to_pdf(data: List[Dict[str, Any]], title: str = "Export Results") -> io.BytesIO:
    """
    Export data to PDF.

    Args:
        data: List of dictionaries containing publication data
        title: Title for the PDF document

    Returns:
        BytesIO object containing PDF data
    """
    output = io.BytesIO()
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Add title
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, title, 0, 1, 'C')
    pdf.ln(5)

    # Add date of export
    import datetime
    pdf.set_font("Arial", 'I', 10)
    pdf.cell(0, 10, f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 1, 'R')
    pdf.ln(5)

    # Add summary
    pdf.set_font("Arial", '', 11)
    pdf.cell(0, 10, f"Number of publications: {len(data)}", 0, 1)
    pdf.ln(5)

    # Table of contents
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Table of Contents", 0, 1)
    pdf.ln(2)

    # Add entries to table of contents
    for i, item in enumerate(data, 1):
        pdf.set_font("Arial", '', 10)
        title_text = item.get('title', f"Publication {i}")
        if len(title_text) > 80:
            title_text = title_text[:77] + "..."

        # Add page number
        pdf.cell(0, 6, f"{i}. {title_text}", 0, 1)
        pdf.ln(1)

    # Main content
    for i, item in enumerate(data, 1):
        pdf.add_page()

        # Article title
        pdf.set_font("Arial", 'B', 14)
        title_text = item.get('title', f"Publication {i}")
        pdf.multi_cell(0, 8, title_text, 0, 'L')
        pdf.ln(5)

        # Basic information table
        pdf.set_font("Arial", 'B', 11)
        pdf.cell(40, 8, "PMID:", 0, 0)
        pdf.set_font("Arial", '', 11)
        pdf.cell(0, 8, str(item.get('pmid', 'N/A')), 0, 1)

        pdf.set_font("Arial", 'B', 11)
        pdf.cell(40, 8, "Journal:", 0, 0)
        pdf.set_font("Arial", '', 11)
        pdf.cell(0, 8, str(item.get('journal', 'N/A')), 0, 1)

        pdf.set_font("Arial", 'B', 11)
        pdf.cell(40, 8, "Publication Date:", 0, 0)
        pdf.set_font("Arial", '', 11)
        pdf.cell(0, 8, str(item.get('human_date', item.get('publication_date', 'N/A'))), 0, 1)

        # Authors
        pdf.set_font("Arial", 'B', 11)
        pdf.cell(40, 8, "Authors:", 0, 0)
        pdf.set_font("Arial", '', 11)

        authors = item.get('authors', [])
        if isinstance(authors, list):
            if len(authors) > 5:
                authors_text = ", ".join(authors[:5]) + ", et al."
            else:
                authors_text = ", ".join(authors)
        else:
            authors_text = str(authors)

        pdf.multi_cell(0, 8, authors_text, 0, 'L')

        # Authority metrics
        pdf.ln(5)
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 8, "Authority Metrics", 0, 1)
        pdf.ln(2)

        metrics = [
            ("Impact Factor", item.get('impact_factor', 'N/A')),
            ("Journal Quartile", f"Q{item.get('journal_quartile', 'N/A')}" if item.get('journal_quartile') else 'N/A'),
            ("Citation Count", item.get('citation_count', 'N/A')),
            ("Authority Score", item.get('authority_score', 'N/A'))
        ]

        for label, value in metrics:
            pdf.set_font("Arial", 'B', 10)
            pdf.cell(40, 6, label + ":", 0, 0)
            pdf.set_font("Arial", '', 10)
            pdf.cell(0, 6, str(value), 0, 1)

        # Publication Types
        pub_types = item.get('publication_types', [])
        if pub_types:
            pdf.ln(5)
            pdf.set_font("Arial", 'B', 11)
            pdf.cell(0, 8, "Publication Types:", 0, 1)
            pdf.set_font("Arial", '', 10)

            if isinstance(pub_types, list):
                pub_types_text = ", ".join(pub_types)
            else:
                pub_types_text = str(pub_types)

            pdf.multi_cell(0, 6, pub_types_text, 0, 'L')

        # Abstract
        if 'abstract' in item and item['abstract']:
            pdf.ln(5)
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(0, 8, "Abstract", 0, 1)
            pdf.ln(2)

            pdf.set_font("Arial", '', 10)

            # Truncate very long abstracts
            abstract = item['abstract']
            max_chars = 2000
            if len(abstract) > max_chars:
                abstract = abstract[:max_chars] + "..."

            pdf.multi_cell(0, 6, abstract, 0, 'L')

        # MeSH Terms
        mesh_terms = item.get('mesh_terms', [])
        if mesh_terms:
            pdf.ln(5)
            pdf.set_font("Arial", 'B', 11)
            pdf.cell(0, 8, "MeSH Terms:", 0, 1)
            pdf.set_font("Arial", '', 10)

            if isinstance(mesh_terms, list):
                mesh_terms_text = ", ".join(mesh_terms[:15])
                if len(mesh_terms) > 15:
                    mesh_terms_text += ", ..."
            else:
                mesh_terms_text = str(mesh_terms)

            pdf.multi_cell(0, 6, mesh_terms_text, 0, 'L')

        # Page break between articles
        if i < len(data):
            pdf.add_page()

    # Generate PDF
    pdf.output(output)
    output.seek(0)
    return output

def export_contradiction_analysis_to_pdf(analysis: Dict[str, Any], query: str, output_path: str = None) -> Any:
    """
    Export contradiction analysis to PDF.

    Args:
        analysis: Dictionary containing contradiction analysis
        query: The search query that produced the analysis
        output_path: Optional path to save the PDF file

    Returns:
        BytesIO object containing PDF data or file path if output_path is provided
    """
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Add title
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, f"Contradiction Analysis: {query}", 0, 1, 'C')
    pdf.ln(5)

    # Add date of export
    import datetime
    pdf.set_font("Arial", 'I', 10)
    pdf.cell(0, 10, f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 1, 'R')
    pdf.ln(5)

    # Add summary
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Analysis Summary", 0, 1)
    pdf.ln(2)

    pdf.set_font("Arial", '', 11)
    pdf.cell(0, 8, f"Total articles analyzed: {analysis.get('total_articles', 0)}", 0, 1)
    pdf.cell(0, 8, f"Contradictions found: {analysis.get('num_contradictions', 0)}", 0, 1)
    pdf.ln(5)

    # Add contradictions by topic
    by_topic = analysis.get('by_topic', {})
    if by_topic:
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, "Contradictions by Topic", 0, 1)
        pdf.ln(2)

        for topic, contradictions in by_topic.items():
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(0, 8, f"{topic}: {len(contradictions)} contradictions", 0, 1)
            pdf.ln(1)

    # Table of contradictions
    contradictions = analysis.get('contradictions', [])
    if contradictions:
        pdf.add_page()
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, "Detailed Contradictions", 0, 1)
        pdf.ln(2)

        for i, contradiction in enumerate(contradictions, 1):
            pub1 = contradiction.get('publication1', {})
            pub2 = contradiction.get('publication2', {})
            authority = contradiction.get('authority_comparison', {})

            pdf.set_font("Arial", 'B', 12)
            pdf.cell(0, 8, f"Contradiction #{i}", 0, 1)
            pdf.ln(1)

            # Publication 1
            pdf.set_font("Arial", 'B', 11)
            pdf.cell(0, 8, "Publication 1:", 0, 1)

            pdf.set_font("Arial", '', 10)
            pub1_title = pub1.get('title', 'Unknown Title')
            pdf.multi_cell(0, 6, pub1_title, 0, 'L')

            pdf.set_font("Arial", 'B', 10)
            pdf.cell(40, 6, "PMID:", 0, 0)
            pdf.set_font("Arial", '', 10)
            pdf.cell(0, 6, str(pub1.get('pmid', 'N/A')), 0, 1)

            pdf.set_font("Arial", 'B', 10)
            pdf.cell(40, 6, "Authority Score:", 0, 0)
            pdf.set_font("Arial", '', 10)
            pdf.cell(0, 6, str(pub1.get('authority_score', 'N/A')), 0, 1)

            pdf.set_font("Arial", 'B', 10)
            pdf.cell(40, 6, "Abstract Snippet:", 0, 1)
            pdf.set_font("Arial", 'I', 9)
            pdf.multi_cell(0, 5, pub1.get('abstract_snippet', 'No abstract available'), 0, 'L')
            pdf.ln(3)

            # Publication 2
            pdf.set_font("Arial", 'B', 11)
            pdf.cell(0, 8, "Publication 2:", 0, 1)

            pdf.set_font("Arial", '', 10)
            pub2_title = pub2.get('title', 'Unknown Title')
            pdf.multi_cell(0, 6, pub2_title, 0, 'L')

            pdf.set_font("Arial", 'B', 10)
            pdf.cell(40, 6, "PMID:", 0, 0)
            pdf.set_font("Arial", '', 10)
            pdf.cell(0, 6, str(pub2.get('pmid', 'N/A')), 0, 1)

            pdf.set_font("Arial", 'B', 10)
            pdf.cell(40, 6, "Authority Score:", 0, 0)
            pdf.set_font("Arial", '', 10)
            pdf.cell(0, 6, str(pub2.get('authority_score', 'N/A')), 0, 1)

            pdf.set_font("Arial", 'B', 10)
            pdf.cell(40, 6, "Abstract Snippet:", 0, 1)
            pdf.set_font("Arial", 'I', 9)
            pdf.multi_cell(0, 5, pub2.get('abstract_snippet', 'No abstract available'), 0, 'L')
            pdf.ln(3)

            # Authority comparison
            pdf.set_font("Arial", 'B', 11)
            pdf.cell(0, 8, "Authority Comparison:", 0, 1)

            higher = authority.get('higher_authority', 'equal')
            diff = abs(authority.get('authority_difference', 0))

            pdf.set_font("Arial", '', 10)
            if higher == 'publication1':
                pdf.cell(0, 6, f"Publication 1 has stronger authority (+{diff:.1f} points)", 0, 1)
            elif higher == 'publication2':
                pdf.cell(0, 6, f"Publication 2 has stronger authority (+{diff:.1f} points)", 0, 1)
            else:
                pdf.cell(0, 6, "Both publications have similar authority", 0, 1)

            # Factor comparison
            factor_comparison = authority.get('factor_comparison', {})
            if factor_comparison:
                pdf.ln(2)
                pdf.set_font("Arial", 'B', 10)
                pdf.cell(0, 6, "Factor Comparison:", 0, 1)

                for factor, value in factor_comparison.items():
                    pdf.set_font("Arial", 'B', 9)
                    factor_name = factor.replace('_', ' ').title()
                    pdf.cell(60, 5, f"{factor_name}:", 0, 0)

                    pdf.set_font("Arial", '', 9)
                    if isinstance(value, dict):
                        pub1_val = value.get('publication1', 'N/A')
                        pub2_val = value.get('publication2', 'N/A')
                        pdf.cell(0, 5, f"Pub1: {pub1_val}, Pub2: {pub2_val}", 0, 1)
                    elif isinstance(value, (int, float)):
                        if value > 0:
                            pdf.cell(0, 5, f"Higher in Publication 1 (+{value})", 0, 1)
                        elif value < 0:
                            pdf.cell(0, 5, f"Higher in Publication 2 ({value})", 0, 1)
                        else:
                            pdf.cell(0, 5, "Equal", 0, 1)
                    else:
                        pdf.cell(0, 5, str(value), 0, 1)

            # Add confidence
            pdf.ln(2)
            confidence = contradiction.get('confidence', 'medium')
            pdf.set_font("Arial", 'B', 10)
            pdf.cell(40, 6, "Confidence:", 0, 0)
            pdf.set_font("Arial", '', 10)
            pdf.cell(0, 6, confidence.capitalize(), 0, 1)

            # Add separator between contradictions
            if i < len(contradictions):
                pdf.ln(5)
                pdf.cell(0, 0, "", 1, 1)
                pdf.ln(5)

    # Generate PDF
    if output_path:
        # Save to file
        pdf.output(output_path)
        return output_path
    else:
        # Return as BytesIO
        output = io.BytesIO()
        pdf.output(output)
        output.seek(0)
        return output