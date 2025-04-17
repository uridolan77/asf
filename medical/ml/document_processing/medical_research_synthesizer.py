"""
Medical Research Synthesizer

This module provides a complete pipeline for processing and synthesizing medical research papers.
It integrates document processing, entity extraction, relation extraction, and research summarization
components into a cohesive workflow.
"""

import os
import json
import logging
import pandas as pd
import networkx as nx
from typing import Dict, List, Optional, Any

# Local imports
from .document_structure import DocumentStructure
from .document_processor import BiomedicalDocumentProcessor
from .entity_extractor import BiomedicalEntityExtractor
from .relation_extractor import MedicalRelationExtractor
from .research_summarizer import SciFiveResearchSummarizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MedicalResearchSynthesizer:
    """
    Complete pipeline for processing and synthesizing medical research papers.
    
    Integrates document processing, entity extraction, relation extraction, and
    research summarization components into a cohesive workflow.
    """
    
    def __init__(
        self,
        document_processor_args: Dict = None,
        entity_extractor_args: Dict = None,
        relation_extractor_args: Dict = None,
        summarizer_args: Dict = None,
        device: Optional[str] = None
    ):
        """
        Initialize the medical research synthesizer.
        
        Args:
            document_processor_args: Arguments for document processor
            entity_extractor_args: Arguments for entity extractor
            relation_extractor_args: Arguments for relation extractor
            summarizer_args: Arguments for summarizer
            device: Device for PyTorch models
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Initializing Medical Research Synthesizer on {self.device}")
        
        # Initialize document processor
        doc_args = document_processor_args or {}
        self.document_processor = BiomedicalDocumentProcessor(
            device=self.device, **doc_args
        )
        
        # Initialize entity extractor
        entity_args = entity_extractor_args or {}
        self.entity_extractor = BiomedicalEntityExtractor(**entity_args)
        
        # Initialize relation extractor
        rel_args = relation_extractor_args or {}
        self.relation_extractor = MedicalRelationExtractor(
            device=self.device, **rel_args
        )
        
        # Initialize summarizer
        sum_args = summarizer_args or {}
        self.summarizer = SciFiveResearchSummarizer(
            device=self.device, **sum_args
        )
        
        logger.info("Medical Research Synthesizer initialized successfully")
    
    def process(self, text_or_path: str, is_pdf: bool = False) -> DocumentStructure:
        """
        Process a medical research paper end-to-end.
        
        Args:
            text_or_path: Text or path to PDF
            is_pdf: Whether the input is a PDF path
            
        Returns:
            Processed document structure
        """
        # Step 1: Document processing
        logger.info("Step 1: Document processing")
        doc_structure = self.document_processor.process_document(text_or_path, is_pdf)
        
        # Step 2: Entity extraction
        logger.info("Step 2: Biomedical entity extraction")
        doc_structure = self.entity_extractor.process_document(doc_structure)
        
        # Step 3: Relation extraction
        logger.info("Step 3: Relation extraction")
        doc_structure = self.relation_extractor.process_document(doc_structure)
        
        # Step 4: Research summarization
        logger.info("Step 4: Research summarization")
        doc_structure = self.summarizer.process_document(doc_structure)
        
        logger.info("Processing complete")
        return doc_structure
    
    def save_results(self, doc_structure: DocumentStructure, output_dir: str) -> None:
        """
        Save processing results to files.
        
        Args:
            doc_structure: Processed document structure
            output_dir: Output directory
        """
        import os
        import json
        import pandas as pd
        import networkx as nx
        
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
        if doc_structure.summary:
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
        
        logger.info(f"Results saved to {output_dir}")
        
    def register_with_lifecycle_manager(self):
        """
        Register the synthesizer with the model lifecycle manager.
        
        This enables model versioning, monitoring, and automated retraining.
        """
        try:
            from asf.medical.ml.model_lifecycle_manager import get_model_lifecycle_manager
            from asf.medical.ml.models.model_registry import ModelFramework
            
            # Get lifecycle manager
            lifecycle_manager = get_model_lifecycle_manager()
            
            # Register components
            lifecycle_manager.registry.register_model(
                name="medical_research_synthesizer",
                version="1.0.0",
                framework=ModelFramework.CUSTOM,
                description="Medical Research Synthesizer for processing and analyzing research papers",
                status="active",
                path="",
                parameters={
                    "document_processor": self.document_processor.__class__.__name__,
                    "entity_extractor": self.entity_extractor.__class__.__name__,
                    "relation_extractor": self.relation_extractor.__class__.__name__,
                    "summarizer": self.summarizer.__class__.__name__
                }
            )
            
            logger.info("Registered Medical Research Synthesizer with model lifecycle manager")
            return True
        except Exception as e:
            logger.warning(f"Could not register with model lifecycle manager: {str(e)}")
            return False
