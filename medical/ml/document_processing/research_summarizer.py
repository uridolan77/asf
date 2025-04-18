"""
Research Paper Summarization

This module provides functionality for summarizing medical research papers
using SciFive, a T5 model specifically trained on biomedical literature.
"""

import logging
import torch
import json
from typing import Dict, List, Optional, Any
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Local imports
from .document_structure import DocumentStructure, ResearchSummary

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SciFiveResearchSummarizer:
    """
    Research paper summarization using SciFive-Large.
    
    SciFive is a T5 model specifically trained on biomedical and scientific literature,
    making it ideal for generating concise and informative summaries of research papers.
    """
    
    def __init__(
        self,
        model_name: str = "razent/SciFive-large-Pubmed-paper_summary",
        device: Optional[str] = None,
        max_length: int = 512,
        min_length: int = 50
    ):
        """
        Initialize the SciFive summarizer.
        
        Args:
            model_name: SciFive model name
            device: Device for model
            max_length: Maximum output length
            min_length: Minimum output length
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.min_length = min_length
        
        try:
            logger.info(f"Loading SciFive model: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
            self.model.eval()
            logger.info(f"SciFive model loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"Error loading SciFive model: {str(e)}")
            self.tokenizer = None
            self.model = None
    
    def summarize(
        self,
        text: str,
        prompt: str = "summarize: ",
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
        num_beams: int = 4,
        do_sample: bool = False,
        temperature: float = 1.0,
        repetition_penalty: float = 1.2
    ) -> str:
        """
        Generate an abstractive summary.
        
        Args:
            text: Text to summarize
            prompt: Prefix prompt
            max_length: Maximum output length
            min_length: Minimum output length
            num_beams: Number of beams for beam search
            do_sample: Whether to use sampling
            temperature: Sampling temperature
            repetition_penalty: Penalty for repetition
            
        Returns:
            Generated summary
        """
        if self.model is None or self.tokenizer is None:
            logger.error("Model not initialized")
            return ""
            
        # Use default lengths if not specified
        max_length = max_length or self.max_length
        min_length = min_length or self.min_length
        
        # Truncate text if too long
        if len(text) > 10000:
            logger.warning(f"Text too long ({len(text)} chars), truncating to 10000 chars")
            text = text[:10000]
        
        # Prepare input
        input_text = prompt + text
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            max_length=1024,
            truncation=True
        ).to(self.device)
        
        # Generate summary
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_length=max_length,
                    min_length=min_length,
                    num_beams=num_beams,
                    do_sample=do_sample,
                    temperature=temperature,
                    repetition_penalty=repetition_penalty,
                    early_stopping=True
                )
            
            summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return summary
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            return ""
    
    def summarize_section(
        self,
        section_text: str,
        section_type: str
    ) -> str:
        """
        Summarize a specific section based on its type.
        
        Args:
            section_text: Section text
            section_type: Section type
            
        Returns:
            Section summary
        """
        # Use different prompts based on section type
        if section_type in ["methods", "materials_and_methods"]:
            prompt = "summarize methods: "
            max_length = 200
        elif section_type == "results":
            prompt = "summarize key findings: "
            max_length = 250
        elif section_type in ["discussion", "conclusion"]:
            prompt = "summarize conclusions: "
            max_length = 200
        elif section_type == "abstract":
            prompt = "summarize concisely: "
            max_length = 150
        else:
            prompt = "summarize: "
            max_length = 200
        
        return self.summarize(
            text=section_text,
            prompt=prompt,
            max_length=max_length
        )
    
    def extract_clinical_implications(
        self,
        text: str,
        max_length: int = 200
    ) -> str:
        """
        Extract clinical implications from research text.
        
        Args:
            text: Text to analyze
            max_length: Maximum output length
            
        Returns:
            Clinical implications summary
        """
        prompt = "extract clinical implications: "
        return self.summarize(text=text, prompt=prompt, max_length=max_length)
    
    def extract_key_findings(
        self,
        text: str,
        max_length: int = 250
    ) -> List[str]:
        """
        Extract key findings from research text.
        
        Args:
            text: Text to analyze
            max_length: Maximum output length
            
        Returns:
            List of key findings
        """
        prompt = "list key findings: "
        findings_text = self.summarize(text=text, prompt=prompt, max_length=max_length)
        
        # Split into bullet points if possible
        if "- " in findings_text:
            findings = [f.strip() for f in findings_text.split("- ") if f.strip()]
        else:
            # Try to split by sentences
            import re
            findings = [f.strip() for f in re.split(r'(?<=[.!?])\s+', findings_text) if f.strip()]
        
        return findings
    
    def extract_limitations(
        self,
        text: str,
        max_length: int = 200
    ) -> List[str]:
        """
        Extract limitations from research text.
        
        Args:
            text: Text to analyze
            max_length: Maximum output length
            
        Returns:
            List of limitations
        """
        prompt = "list study limitations: "
        limitations_text = self.summarize(text=text, prompt=prompt, max_length=max_length)
        
        # Split into bullet points if possible
        if "- " in limitations_text:
            limitations = [l.strip() for l in limitations_text.split("- ") if l.strip()]
        else:
            # Try to split by sentences
            import re
            limitations = [l.strip() for l in re.split(r'(?<=[.!?])\s+', limitations_text) if l.strip()]
        
        return limitations
    
    def process_document(self, doc_structure: DocumentStructure) -> DocumentStructure:
        """
        Generate summaries for a document structure.
        
        Args:
            doc_structure: Document structure
            
        Returns:
            Updated document structure with summaries
        """
        logger.info("Generating research summaries...")
        
        # Prepare full text for summarization
        full_text = ""
        if doc_structure.abstract:
            full_text += "Abstract: " + doc_structure.abstract + "\n\n"
        
        for section in doc_structure.sections:
            full_text += section.heading + ": " + section.text + "\n\n"
        
        # Generate abstract summary
        abstract_summary = ""
        if doc_structure.abstract:
            abstract_summary = self.summarize_section(
                doc_structure.abstract, "abstract"
            )
        
        # Extract key findings
        key_findings = self.extract_key_findings(full_text)
        
        # Generate methods summary
        methods_summary = None
        methods_sections = [s for s in doc_structure.sections 
                           if s.section_type in ["methods", "materials_and_methods"]]
        if methods_sections:
            methods_text = "\n".join([s.text for s in methods_sections])
            methods_summary = self.summarize_section(
                methods_text, "methods"
            )
        
        # Generate conclusions
        conclusions = None
        conclusion_sections = [s for s in doc_structure.sections 
                              if s.section_type in ["conclusion", "discussion"]]
        if conclusion_sections:
            conclusion_text = "\n".join([s.text for s in conclusion_sections])
            conclusions = self.summarize_section(
                conclusion_text, "conclusion"
            )
        
        # Extract limitations
        limitations = None
        discussion_sections = [s for s in doc_structure.sections 
                              if s.section_type == "discussion"]
        if discussion_sections:
            discussion_text = "\n".join([s.text for s in discussion_sections])
            limitations = self.extract_limitations(discussion_text)
        
        # Extract clinical implications
        clinical_implications = None
        if discussion_sections:
            discussion_text = "\n".join([s.text for s in discussion_sections])
            clinical_implications_text = self.extract_clinical_implications(discussion_text)
            
            # Split into bullet points if possible
            if "- " in clinical_implications_text:
                clinical_implications = [c.strip() for c in clinical_implications_text.split("- ") if c.strip()]
            else:
                # Try to split by sentences
                import re
                clinical_implications = [c.strip() for c in re.split(r'(?<=[.!?])\s+', clinical_implications_text) if c.strip()]
        
        # Create research summary object
        doc_structure.summary = {
            "abstract": abstract_summary,
            "key_findings": key_findings,
            "methods": methods_summary,
            "conclusions": conclusions,
            "limitations": limitations,
            "clinical_implications": clinical_implications
        }
        
        logger.info("Research summarization complete")
        return doc_structure
