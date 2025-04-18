"""
Medical Research Synthesizer Result Exporter

This module provides functionality for exporting and saving the results
of the Medical Research Synthesizer in various formats.
"""

import os
import json
import logging
import pandas as pd
import networkx as nx
from typing import Dict, Optional, Any

# Local imports
from .document_structure import DocumentStructure

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ResultExporter:
    """
    Exports and saves Medical Research Synthesizer results in various formats.
    """
    
    @staticmethod
    def save_results(doc_structure: DocumentStructure, output_dir: str) -> None:
        """
        Save processing results to files.
        
        Args:
            doc_structure: Processed document structure
            output_dir: Output directory
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save document structure as JSON
        with open(os.path.join(output_dir, "document_structure.json"), "w", encoding="utf-8") as f:
            # Convert to serializable format
            doc_dict = {
                "title": doc_structure.title,
                "abstract": doc_structure.abstract,
                "sections": [
                    {
                        "section_type": section.section_type,
                        "heading": section.heading,
                        "text": section.text[:500] + "..." if len(section.text) > 500 else section.text,
                        "entities_count": len(section.entities),
                        "relations_count": len(section.relations),
                        "subsections": [
                            {"section_type": ss.section_type, "heading": ss.heading}
                            for ss in section.subsections
                        ]
                    }
                    for section in doc_structure.sections
                ],
                "entities_count": len(doc_structure.entities),
                "relations_count": len(doc_structure.relations),
                "summary": doc_structure.summary
            }
            json.dump(doc_dict, f, indent=2)
        
        # Save entities as CSV
        entities_df = pd.DataFrame([e.to_dict() for e in doc_structure.entities])
        if not entities_df.empty:
            entities_df.to_csv(os.path.join(output_dir, "entities.csv"), index=False)
        
        # Save relations as CSV
        relations_df = pd.DataFrame(doc_structure.relations)
        if not relations_df.empty:
            relations_df.to_csv(os.path.join(output_dir, "relations.csv"), index=False)
        
        # Save knowledge graph if available
        if doc_structure.knowledge_graph:
            # Save as GraphML
            nx.write_graphml(doc_structure.knowledge_graph, os.path.join(output_dir, "knowledge_graph.graphml"))
            
            # Save as JSON for visualization
            graph_data = {
                "nodes": [
                    {
                        "id": node,
                        "label": node,
                        "type": doc_structure.knowledge_graph.nodes[node].get("type", "Unknown")
                    }
                    for node in doc_structure.knowledge_graph.nodes
                ],
                "links": [
                    {
                        "source": u,
                        "target": v,
                        "type": doc_structure.knowledge_graph.edges[u, v].get("type", "Unknown"),
                        "confidence": doc_structure.knowledge_graph.edges[u, v].get("confidence", 0.0)
                    }
                    for u, v in doc_structure.knowledge_graph.edges
                ]
            }
            with open(os.path.join(output_dir, "knowledge_graph.json"), "w") as f:
                json.dump(graph_data, f, indent=2)
        
        # Save summary as markdown
        ResultExporter.save_summary_markdown(doc_structure, output_dir)
        
        logger.info(f"Results saved to {output_dir}")
    
    @staticmethod
    def save_summary_markdown(doc_structure: DocumentStructure, output_dir: str) -> None:
        """
        Save the document summary as a markdown file.
        
        Args:
            doc_structure: Processed document structure
            output_dir: Output directory
        """
        if not doc_structure.summary:
            return
            
        with open(os.path.join(output_dir, "summary.md"), "w", encoding="utf-8") as f:
            f.write(f"# {doc_structure.title}\n\n")
            
            if doc_structure.summary.get("abstract"):
                f.write(f"## Abstract\n\n{doc_structure.summary['abstract']}\n\n")
            
            if doc_structure.summary.get("key_findings"):
                f.write("## Key Findings\n\n")
                for i, finding in enumerate(doc_structure.summary["key_findings"], 1):
                    f.write(f"{i}. {finding}\n")
                f.write("\n")
            
            if doc_structure.summary.get("methods"):
                f.write(f"## Methods\n\n{doc_structure.summary['methods']}\n\n")
            
            if doc_structure.summary.get("conclusions"):
                f.write(f"## Conclusions\n\n{doc_structure.summary['conclusions']}\n\n")
            
            if doc_structure.summary.get("limitations"):
                f.write("## Limitations\n\n")
                for i, limitation in enumerate(doc_structure.summary["limitations"], 1):
                    f.write(f"{i}. {limitation}\n")
                f.write("\n")
            
            if doc_structure.summary.get("clinical_implications"):
                f.write("## Clinical Implications\n\n")
                for i, implication in enumerate(doc_structure.summary["clinical_implications"], 1):
                    f.write(f"{i}. {implication}\n")
                f.write("\n")
    
    @staticmethod
    def export_to_json(doc_structure: DocumentStructure) -> Dict[str, Any]:
        """
        Export document structure to a JSON-serializable dictionary.
        
        Args:
            doc_structure: Document structure to export
            
        Returns:
            JSON-serializable dictionary
        """
        return {
            "title": doc_structure.title,
            "abstract": doc_structure.abstract,
            "sections": [
                {
                    "section_type": section.section_type,
                    "heading": section.heading,
                    "text": section.text[:500] + "..." if len(section.text) > 500 else section.text,
                    "entities_count": len(section.entities),
                    "relations_count": len(section.relations)
                }
                for section in doc_structure.sections
            ],
            "entities": [e.to_dict() for e in doc_structure.entities],
            "relations": doc_structure.relations,
            "summary": doc_structure.summary
        }
