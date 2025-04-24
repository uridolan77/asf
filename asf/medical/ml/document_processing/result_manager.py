"""
Result Manager Module

This module provides functionality for managing, saving, and exporting
results from the medical research synthesizer.
"""

import os
import json
import logging
import csv
import pickle
from typing import Dict, List, Any, Optional
import networkx as nx
from dataclasses import asdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ResultManager:
    """
    Manager for medical research synthesizer results.
    
    This class provides functionality for saving, exporting, and
    managing results from the medical research synthesizer.
    """
    
    @staticmethod
    def save_results(doc_structure, output_dir: str, formats: List[str] = None) -> Dict[str, str]:
        """
        Save processing results to files in various formats.
        
        Args:
            doc_structure: Processed document structure
            output_dir: Output directory
            formats: List of formats to save (json, csv, pickle, graphml)
            
        Returns:
            Dictionary mapping format to file path
        """
        # Default formats
        if formats is None:
            formats = ["json", "csv", "pickle", "graphml"]
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        saved_files = {}
        
        # Convert document structure to dictionary
        try:
            doc_dict = ResultManager._structure_to_dict(doc_structure)
        except Exception as e:
            logger.error(f"Error converting document structure to dictionary: {str(e)}")
            doc_dict = {"error": str(e)}
        
        # Save in requested formats
        for fmt in formats:
            try:
                if fmt.lower() == "json":
                    file_path = os.path.join(output_dir, "results.json")
                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump(doc_dict, f, indent=2, ensure_ascii=False)
                    saved_files["json"] = file_path
                
                elif fmt.lower() == "csv":
                    # Save entities to CSV
                    entities_path = os.path.join(output_dir, "entities.csv")
                    ResultManager._save_entities_csv(doc_structure.entities, entities_path)
                    saved_files["entities_csv"] = entities_path
                    
                    # Save relations to CSV
                    relations_path = os.path.join(output_dir, "relations.csv")
                    ResultManager._save_relations_csv(doc_structure.relations, relations_path)
                    saved_files["relations_csv"] = relations_path
                
                elif fmt.lower() == "pickle":
                    file_path = os.path.join(output_dir, "results.pkl")
                    with open(file_path, 'wb') as f:
                        pickle.dump(doc_structure, f)
                    saved_files["pickle"] = file_path
                
                elif fmt.lower() == "graphml":
                    # Save knowledge graph if available
                    if hasattr(doc_structure, 'knowledge_graph') and doc_structure.knowledge_graph:
                        graph_path = os.path.join(output_dir, "knowledge_graph.graphml")
                        nx.write_graphml(doc_structure.knowledge_graph, graph_path)
                        saved_files["graphml"] = graph_path
                
                elif fmt.lower() == "txt":
                    # Save summary as text
                    if hasattr(doc_structure, 'summary') and doc_structure.summary:
                        summary_path = os.path.join(output_dir, "summary.txt")
                        with open(summary_path, 'w', encoding='utf-8') as f:
                            if isinstance(doc_structure.summary, dict):
                                for key, value in doc_structure.summary.items():
                                    f.write(f"## {key.replace('_', ' ').title()}\n\n")
                                    f.write(f"{value}\n\n")
                            else:
                                f.write(str(doc_structure.summary))
                        saved_files["txt"] = summary_path
            
            except Exception as e:
                logger.error(f"Error saving results in {fmt} format: {str(e)}")
        
        logger.info(f"Saved results to {output_dir} in formats: {', '.join(saved_files.keys())}")
        return saved_files
    
    @staticmethod
    def _structure_to_dict(doc_structure) -> Dict[str, Any]:
        """
        Convert document structure to dictionary.
        
        Args:
            doc_structure: Document structure
            
        Returns:
            Dictionary representation
        """
        # If already a dictionary, return as is
        if isinstance(doc_structure, dict):
            return doc_structure
        
        # If has to_dict method, use it
        if hasattr(doc_structure, 'to_dict') and callable(getattr(doc_structure, 'to_dict')):
            return doc_structure.to_dict()
        
        # If dataclass, convert to dict
        if hasattr(doc_structure, '__dataclass_fields__'):
            return asdict(doc_structure)
        
        # Otherwise, convert attributes to dictionary
        result = {}
        for key, value in doc_structure.__dict__.items():
            if key.startswith('_'):
                continue
            
            if hasattr(value, 'to_dict') and callable(getattr(value, 'to_dict')):
                result[key] = value.to_dict()
            elif hasattr(value, '__dataclass_fields__'):
                result[key] = asdict(value)
            elif isinstance(value, list):
                result[key] = [
                    item.to_dict() if hasattr(item, 'to_dict') and callable(getattr(item, 'to_dict'))
                    else asdict(item) if hasattr(item, '__dataclass_fields__')
                    else item
                    for item in value
                ]
            elif isinstance(value, nx.Graph):
                # Convert NetworkX graph to dictionary
                result[key] = {
                    "nodes": [{"id": n, **d} for n, d in value.nodes(data=True)],
                    "edges": [{"source": u, "target": v, **d} for u, v, d in value.edges(data=True)]
                }
            else:
                result[key] = value
        
        return result
    
    @staticmethod
    def _save_entities_csv(entities, file_path: str) -> None:
        """
        Save entities to CSV file.
        
        Args:
            entities: List of entities
            file_path: Output file path
        """
        if not entities:
            return
        
        # Get all possible fields
        fields = set()
        for entity in entities:
            if hasattr(entity, '__dict__'):
                fields.update(entity.__dict__.keys())
            elif isinstance(entity, dict):
                fields.update(entity.keys())
        
        # Remove private fields
        fields = [f for f in fields if not f.startswith('_')]
        
        # Write CSV
        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            
            for entity in entities:
                if hasattr(entity, '__dict__'):
                    row = {k: v for k, v in entity.__dict__.items() if k in fields}
                elif isinstance(entity, dict):
                    row = {k: v for k, v in entity.items() if k in fields}
                else:
                    continue
                
                writer.writerow(row)
    
    @staticmethod
    def _save_relations_csv(relations, file_path: str) -> None:
        """
        Save relations to CSV file.
        
        Args:
            relations: List of relations
            file_path: Output file path
        """
        if not relations:
            return
        
        # Get all possible fields
        fields = set()
        for relation in relations:
            if hasattr(relation, '__dict__'):
                fields.update(relation.__dict__.keys())
            elif isinstance(relation, dict):
                fields.update(relation.keys())
        
        # Remove private fields
        fields = [f for f in fields if not f.startswith('_')]
        
        # Write CSV
        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            
            for relation in relations:
                if hasattr(relation, '__dict__'):
                    row = {k: v for k, v in relation.__dict__.items() if k in fields}
                elif isinstance(relation, dict):
                    row = {k: v for k, v in relation.items() if k in fields}
                else:
                    continue
                
                writer.writerow(row)
    
    @staticmethod
    def load_results(input_dir: str, format: str = "json") -> Any:
        """
        Load processing results from files.
        
        Args:
            input_dir: Input directory
            format: Format to load (json, pickle)
            
        Returns:
            Loaded results
        """
        if format.lower() == "json":
            file_path = os.path.join(input_dir, "results.json")
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        
        elif format.lower() == "pickle":
            file_path = os.path.join(input_dir, "results.pkl")
            if os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    return pickle.load(f)
        
        elif format.lower() == "graphml":
            file_path = os.path.join(input_dir, "knowledge_graph.graphml")
            if os.path.exists(file_path):
                return nx.read_graphml(file_path)
        
        raise FileNotFoundError(f"Results not found in {input_dir} with format {format}")
    
    @staticmethod
    def export_to_format(doc_structure, output_path: str, format: str) -> str:
        """
        Export document structure to a specific format.
        
        Args:
            doc_structure: Document structure
            output_path: Output file path
            format: Export format (json, csv, pickle, graphml, txt, html, md)
            
        Returns:
            Path to exported file
        """
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        if format.lower() == "json":
            doc_dict = ResultManager._structure_to_dict(doc_structure)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(doc_dict, f, indent=2, ensure_ascii=False)
        
        elif format.lower() == "pickle":
            with open(output_path, 'wb') as f:
                pickle.dump(doc_structure, f)
        
        elif format.lower() == "graphml":
            if hasattr(doc_structure, 'knowledge_graph') and doc_structure.knowledge_graph:
                nx.write_graphml(doc_structure.knowledge_graph, output_path)
            else:
                raise ValueError("Document structure does not have a knowledge graph")
        
        elif format.lower() == "txt":
            with open(output_path, 'w', encoding='utf-8') as f:
                # Write title
                f.write(f"# {doc_structure.title}\n\n")
                
                # Write abstract
                if hasattr(doc_structure, 'abstract') and doc_structure.abstract:
                    f.write("## Abstract\n\n")
                    f.write(f"{doc_structure.abstract}\n\n")
                
                # Write summary
                if hasattr(doc_structure, 'summary') and doc_structure.summary:
                    f.write("## Summary\n\n")
                    if isinstance(doc_structure.summary, dict):
                        for key, value in doc_structure.summary.items():
                            f.write(f"### {key.replace('_', ' ').title()}\n\n")
                            f.write(f"{value}\n\n")
                    else:
                        f.write(f"{doc_structure.summary}\n\n")
                
                # Write entities
                if hasattr(doc_structure, 'entities') and doc_structure.entities:
                    f.write("## Entities\n\n")
                    for entity in doc_structure.entities:
                        if hasattr(entity, 'text') and hasattr(entity, 'label'):
                            f.write(f"- {entity.text} ({entity.label})\n")
                    f.write("\n")
                
                # Write relations
                if hasattr(doc_structure, 'relations') and doc_structure.relations:
                    f.write("## Relations\n\n")
                    for relation in doc_structure.relations:
                        if hasattr(relation, 'head_entity') and hasattr(relation, 'relation_type') and hasattr(relation, 'tail_entity'):
                            f.write(f"- {relation.head_entity} → {relation.relation_type} → {relation.tail_entity}\n")
                    f.write("\n")
        
        elif format.lower() == "html":
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("<!DOCTYPE html>\n")
                f.write("<html>\n<head>\n")
                f.write(f"<title>{doc_structure.title}</title>\n")
                f.write("<style>\n")
                f.write("body { font-family: Arial, sans-serif; margin: 20px; }\n")
                f.write("h1 { color: #2c3e50; }\n")
                f.write("h2 { color: #3498db; }\n")
                f.write("h3 { color: #2980b9; }\n")
                f.write(".entity { margin: 5px 0; }\n")
                f.write(".relation { margin: 5px 0; }\n")
                f.write(".entity-DISEASE { color: #e74c3c; }\n")
                f.write(".entity-DRUG { color: #2ecc71; }\n")
                f.write(".entity-GENE { color: #f39c12; }\n")
                f.write("</style>\n")
                f.write("</head>\n<body>\n")
                
                # Write title
                f.write(f"<h1>{doc_structure.title}</h1>\n")
                
                # Write abstract
                if hasattr(doc_structure, 'abstract') and doc_structure.abstract:
                    f.write("<h2>Abstract</h2>\n")
                    f.write(f"<p>{doc_structure.abstract}</p>\n")
                
                # Write summary
                if hasattr(doc_structure, 'summary') and doc_structure.summary:
                    f.write("<h2>Summary</h2>\n")
                    if isinstance(doc_structure.summary, dict):
                        for key, value in doc_structure.summary.items():
                            f.write(f"<h3>{key.replace('_', ' ').title()}</h3>\n")
                            f.write(f"<p>{value}</p>\n")
                    else:
                        f.write(f"<p>{doc_structure.summary}</p>\n")
                
                # Write entities
                if hasattr(doc_structure, 'entities') and doc_structure.entities:
                    f.write("<h2>Entities</h2>\n")
                    f.write("<div class='entities'>\n")
                    for entity in doc_structure.entities:
                        if hasattr(entity, 'text') and hasattr(entity, 'label'):
                            f.write(f"<div class='entity entity-{entity.label}'>{entity.text} ({entity.label})</div>\n")
                    f.write("</div>\n")
                
                # Write relations
                if hasattr(doc_structure, 'relations') and doc_structure.relations:
                    f.write("<h2>Relations</h2>\n")
                    f.write("<div class='relations'>\n")
                    for relation in doc_structure.relations:
                        if hasattr(relation, 'head_entity') and hasattr(relation, 'relation_type') and hasattr(relation, 'tail_entity'):
                            f.write(f"<div class='relation'>{relation.head_entity} → {relation.relation_type} → {relation.tail_entity}</div>\n")
                    f.write("</div>\n")
                
                f.write("</body>\n</html>")
        
        elif format.lower() == "md":
            with open(output_path, 'w', encoding='utf-8') as f:
                # Write title
                f.write(f"# {doc_structure.title}\n\n")
                
                # Write abstract
                if hasattr(doc_structure, 'abstract') and doc_structure.abstract:
                    f.write("## Abstract\n\n")
                    f.write(f"{doc_structure.abstract}\n\n")
                
                # Write summary
                if hasattr(doc_structure, 'summary') and doc_structure.summary:
                    f.write("## Summary\n\n")
                    if isinstance(doc_structure.summary, dict):
                        for key, value in doc_structure.summary.items():
                            f.write(f"### {key.replace('_', ' ').title()}\n\n")
                            f.write(f"{value}\n\n")
                    else:
                        f.write(f"{doc_structure.summary}\n\n")
                
                # Write entities
                if hasattr(doc_structure, 'entities') and doc_structure.entities:
                    f.write("## Entities\n\n")
                    for entity in doc_structure.entities:
                        if hasattr(entity, 'text') and hasattr(entity, 'label'):
                            f.write(f"- **{entity.text}** ({entity.label})\n")
                    f.write("\n")
                
                # Write relations
                if hasattr(doc_structure, 'relations') and doc_structure.relations:
                    f.write("## Relations\n\n")
                    for relation in doc_structure.relations:
                        if hasattr(relation, 'head_entity') and hasattr(relation, 'relation_type') and hasattr(relation, 'tail_entity'):
                            f.write(f"- **{relation.head_entity}** → *{relation.relation_type}* → **{relation.tail_entity}**\n")
                    f.write("\n")
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        return output_path
