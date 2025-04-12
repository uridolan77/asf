"""
Multimodal Fusion Module

This module provides components for fusing text embeddings with metadata
for more accurate contradiction detection and medical claim analysis.
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname%s - %(message)s"
)
logger = logging.getLogger("multimodal-fusion")

class MultimodalFusionModel(nn.Module):
    """
    Multimodal fusion model for combining text embeddings with metadata.
    
    This model uses attention-based intermediate fusion to combine text embeddings
    (e.g., from BioMedLM) with structured metadata (e.g., study design, sample size).
    """
    
    def __init__(
        self, 
        text_dim: int, 
        metadata_dim: int, 
        fusion_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        """
        Initialize the multimodal fusion model.
        
        Args:
            text_dim: Dimension of text embeddings
            metadata_dim: Dimension of metadata embeddings
            fusion_dim: Dimension of fusion space
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        
        self.text_projection = nn.Linear(text_dim, fusion_dim)
        self.metadata_projection = nn.Linear(metadata_dim, fusion_dim)
        
        self.attention = nn.MultiheadAttention(
            embed_dim=fusion_dim, 
            num_heads=num_heads, 
            dropout=dropout,
            batch_first=True
        )
        
        self.layer_norm1 = nn.LayerNorm(fusion_dim)
        self.layer_norm2 = nn.LayerNorm(fusion_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim * 4, fusion_dim)
        )
        
        self.output_projection = nn.Linear(fusion_dim, fusion_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        text_embedding: torch.Tensor, 
        metadata: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for the multimodal fusion model.
        
        Args:
            text_embedding: Text embedding tensor
            metadata: Metadata tensor
            
        Returns:
            Fused output tensor
        """
        text_proj = self.text_projection(text_embedding)
        meta_proj = self.metadata_projection(metadata)
        
        sequence = torch.stack([text_proj, meta_proj], dim=1)
        
        attn_output, _ = self.attention(sequence, sequence, sequence)
        attn_output = self.dropout(attn_output)
        
        sequence = self.layer_norm1(sequence + attn_output)
        
        ffn_output = self.ffn(sequence)
        ffn_output = self.dropout(ffn_output)
        
        sequence = self.layer_norm2(sequence + ffn_output)
        
        fused = torch.mean(sequence, dim=1)
        output = self.output_projection(fused)
        
        return output


class MetadataExtractor:
    """
    Extract structured metadata from medical text.
    
    This class provides methods for extracting structured metadata from
    medical text, such as study design, sample size, and PICO elements.
    """
    
    def __init__(self, use_spacy: bool = True):
        """
        Initialize the metadata extractor.
        
        Args:
            use_spacy: Whether to use spaCy for extraction
        """
        self.use_spacy = use_spacy
        self.spacy_model = None
        
        if self.use_spacy:
            try:
                import spacy
                
                self.spacy_model = spacy.load("en_core_sci_md")
                logger.info("Initialized spaCy for metadata extraction")
            except ImportError:
                logger.warning("spaCy not available. Falling back to rule-based approach.")
                self.use_spacy = False
            except Exception as e:
                logger.warning(f"Failed to initialize spaCy: {e}. Falling back to rule-based approach.")
                self.use_spacy = False
        
        self.study_design_patterns = {
            "randomized controlled trial": 5.0,
            "rct": 5.0,
            "double-blind": 4.5,
            "double blind": 4.5,
            "placebo-controlled": 4.5,
            "placebo controlled": 4.5,
            "single-blind": 4.0,
            "single blind": 4.0,
            "controlled trial": 4.0,
            "cohort study": 3.5,
            "prospective cohort": 3.5,
            "retrospective cohort": 3.0,
            "case-control": 3.0,
            "case control": 3.0,
            "cross-sectional": 2.5,
            "cross sectional": 2.5,
            "observational": 2.5,
            "case series": 2.0,
            "case report": 1.5,
            "systematic review": 4.5,
            "meta-analysis": 5.0,
            "meta analysis": 5.0,
            "review": 3.0,
            "in vitro": 1.0,
            "animal study": 1.0,
            "preclinical": 1.0
        }
        
        self.sample_size_patterns = [
            r"n\s*=\s*(\d+)",
            r"sample size\s*(?:of|:|was)?\s*(\d+)",
            r"enrolled\s*(\d+)\s*(?:patients|participants|subjects)",
            r"(\d+)\s*(?:patients|participants|subjects)\s*(?:were|was)?\s*enrolled",
            r"included\s*(\d+)\s*(?:patients|participants|subjects)",
            r"(\d+)\s*(?:patients|participants|subjects)\s*(?:were|was)?\s*included",
            r"total of\s*(\d+)\s*(?:patients|participants|subjects)"
        ]
        
        import re
        self.sample_size_regex = [re.compile(pattern, re.IGNORECASE) for pattern in self.sample_size_patterns]
    
    def extract_study_design(self, text: str) -> Dict[str, Any]:
        """
        Extract study design information from text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with study design information
        """
        text_lower = text.lower()
        
        matches = []
        for pattern, score in self.study_design_patterns.items():
            if pattern in text_lower:
                matches.append({
                    "design": pattern,
                    "score": score
                })
        
        matches.sort(key=lambda x: x["score"], reverse=True)
        
        if matches:
            best_match = matches[0]
            return {
                "study_design": best_match["design"],
                "design_score": best_match["score"],
                "all_matches": matches
            }
        else:
            return {
                "study_design": "unknown",
                "design_score": 0.0,
                "all_matches": []
            }
    
    def extract_sample_size(self, text: str) -> Dict[str, Any]:
        """
        Extract sample size information from text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with sample size information
        """
        matches = []
        for regex in self.sample_size_regex:
            for match in regex.finditer(text):
                try:
                    sample_size = int(match.group(1))
                    matches.append({
                        "sample_size": sample_size,
                        "match": match.group(0)
                    })
                except (ValueError, IndexError):
                    continue
        
        matches.sort(key=lambda x: x["sample_size"], reverse=True)
        
        if matches:
            best_match = matches[0]
            return {
                "sample_size": best_match["sample_size"],
                "match": best_match["match"],
                "all_matches": matches
            }
        else:
            return {
                "sample_size": 0,
                "match": None,
                "all_matches": []
            }
    
    def extract_metadata(self, text: str) -> Dict[str, Any]:
        """
        Extract all metadata from text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with all extracted metadata
        """
        study_design = self.extract_study_design(text)
        
        sample_size = self.extract_sample_size(text)
        
        metadata = {
            "study_design": study_design,
            "sample_size": sample_size
        }
        
        return metadata
    
    def encode_metadata(self, metadata: Dict[str, Any]) -> torch.Tensor:
        """
        Encode metadata as a tensor for multimodal fusion.
        
        Args:
            metadata: Metadata dictionary
            
        Returns:
            Tensor representation of metadata [metadata_dim]
        """
        design_score = metadata.get("study_design", {}).get("design_score", 0.0)
        sample_size = metadata.get("sample_size", {}).get("sample_size", 0)
        
        import math
        if sample_size > 0:
            normalized_sample_size = math.log10(sample_size) / 5.0  # Assuming max sample size is 10^5
        else:
            normalized_sample_size = 0.0
        
        normalized_design_score = design_score / 5.0  # Assuming max score is 5.0
        
        metadata_tensor = torch.tensor(
            [normalized_design_score, normalized_sample_size],
            dtype=torch.float32
        )
        
        return metadata_tensor


class MultimodalContradictionDetector:
    """
    Contradiction detector that uses multimodal fusion.
    
    This class combines text embeddings with metadata for more accurate
    contradiction detection.
    """
    
    def __init__(
        self, 
        biomedlm_scorer=None, 
        metadata_extractor=None,
        text_dim: int = 768,
        metadata_dim: int = 2,
        fusion_dim: int = 128
    ):
        """
        Initialize the multimodal contradiction detector.
        
        Args:
            biomedlm_scorer: BioMedLM scorer for text embeddings
            metadata_extractor: Metadata extractor instance
            text_dim: Dimension of text embeddings
            metadata_dim: Dimension of metadata embeddings
            fusion_dim: Dimension of fusion space
        """
        self.biomedlm_scorer = biomedlm_scorer
        
        if metadata_extractor is None:
            self.metadata_extractor = MetadataExtractor()
        else:
            self.metadata_extractor = metadata_extractor
        
        self.fusion_model = MultimodalFusionModel(
            text_dim=text_dim,
            metadata_dim=metadata_dim,
            fusion_dim=fusion_dim
        )
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fusion_model.to(self.device)
    
    def get_text_embedding(self, text: str) -> torch.Tensor:
        """
        Get text embedding from BioMedLM.
        
        Args:
            text: Input text
            
        Returns:
            Text embedding tensor
        """
        if self.biomedlm_scorer is None:
            return torch.randn(768, device=self.device)
        
        try:
            import torch
            
            inputs = self.biomedlm_scorer.tokenizer(
                text, return_tensors="pt", padding=True, truncation=True, max_length=512
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.biomedlm_scorer.model(**inputs, output_hidden_states=True)
            
            last_hidden_state = outputs.hidden_states[-1]
            cls_embedding = last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
            
            return cls_embedding.squeeze(0)  # [hidden_size]
        except Exception as e:
            logger.error(f"Error getting text embedding: {e}")
            return torch.randn(768, device=self.device)
    
    def detect_contradiction(self, text1: str, text2: str) -> Dict[str, Any]:
        """
        Detect contradiction between two texts using multimodal fusion.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Dictionary with contradiction detection results
        """
        metadata1 = self.metadata_extractor.extract_metadata(text1)
        metadata2 = self.metadata_extractor.extract_metadata(text2)
        
        metadata_tensor1 = self.metadata_extractor.encode_metadata(metadata1)
        metadata_tensor2 = self.metadata_extractor.encode_metadata(metadata2)
        
        metadata_tensor1 = metadata_tensor1.to(self.device)
        metadata_tensor2 = metadata_tensor2.to(self.device)
        
        text_embedding1 = self.get_text_embedding(text1)
        text_embedding2 = self.get_text_embedding(text2)
        
        with torch.no_grad():
            fused_embedding1 = self.fusion_model(text_embedding1, metadata_tensor1)
            fused_embedding2 = self.fusion_model(text_embedding2, metadata_tensor2)
        
        similarity = F.cosine_similarity(fused_embedding1, fused_embedding2, dim=0).item()
        
        contradiction_score = 1.0 - (similarity + 1.0) / 2.0
        
        biomedlm_result = None
        if self.biomedlm_scorer is not None:
            try:
                biomedlm_result = self.biomedlm_scorer.detect_contradiction_with_negation(text1, text2)
                biomedlm_score = biomedlm_result.get("contradiction_score", 0.5)
                
                combined_score = 0.7 * biomedlm_score + 0.3 * contradiction_score
            except Exception as e:
                logger.error(f"Error getting BioMedLM contradiction score: {e}")
                combined_score = contradiction_score
        else:
            combined_score = contradiction_score
        
        has_contradiction = combined_score > 0.7
        
        result = {
            "text1": text1,
            "text2": text2,
            "metadata1": metadata1,
            "metadata2": metadata2,
            "multimodal_contradiction_score": contradiction_score,
            "combined_contradiction_score": combined_score,
            "has_contradiction": has_contradiction,
            "method": "multimodal_fusion"
        }
        
        if biomedlm_result is not None:
            result["biomedlm_result"] = biomedlm_result
        
        return result
