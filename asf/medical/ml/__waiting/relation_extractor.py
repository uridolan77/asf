import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Tuple, Optional, Union, Any
import numpy as np
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class RelationInstance:
    """A single relation instance between two entities."""
    head_entity: str
    tail_entity: str
    head_type: str
    tail_type: str
    relation_type: str
    confidence: float
    context: str
    sentence_id: Optional[int] = None

class BioBERTRelationExtractor(nn.Module):
    """
    Relation extraction model based on BioBERT.
    
    This model extracts biomedical relations between entities using a fine-tuned
    BioBERT model with a classification head.
    """
    
    def __init__(
        self,
        model_name: str = "dmis-lab/biobert-v1.1",
        num_relations: int = 8,
        dropout_prob: float = 0.1,
        device: Optional[str] = None
    ):
        """
        Initialize the BioBERT-based relation extraction model.
        
        Args:
            model_name: Pre-trained model to use
            num_relations: Number of relation types to classify
            dropout_prob: Dropout probability for regularization
            device: Device to use (cuda or cpu)
        """
        super(BioBERTRelationExtractor, self).__init__()
        
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.num_relations = num_relations
        
        # Load pre-trained model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        
        # Freeze BERT parameters (optional - can be fine-tuned if needed)
        # for param in self.bert.parameters():
        #     param.requires_grad = False
        
        # Classification head
        hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout_prob)
        self.entity_markers = nn.Embedding(4, hidden_size)  # Start/end markers for both entities
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size, num_relations)
        )
        
        self.to(self.device)
        logger.info(f"Initialized BioBERT relation extractor with {num_relations} relation types on {self.device}")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        entity1_pos: torch.Tensor,
        entity2_pos: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the relation extraction model.
        
        Args:
            input_ids: Tokenized input IDs
            attention_mask: Attention mask for padding
            entity1_pos: Position of first entity [batch_size, 2]
            entity2_pos: Position of second entity [batch_size, 2]
            
        Returns:
            Logits for relation classification
        """
        # Get BERT embeddings
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        
        batch_size = sequence_output.size(0)
        hidden_size = sequence_output.size(2)
        
        # Extract entity representations
        entity1_start_embs = torch.stack([sequence_output[i, entity1_pos[i, 0], :] for i in range(batch_size)])
        entity1_end_embs = torch.stack([sequence_output[i, entity1_pos[i, 1], :] for i in range(batch_size)])
        entity2_start_embs = torch.stack([sequence_output[i, entity2_pos[i, 0], :] for i in range(batch_size)])
        entity2_end_embs = torch.stack([sequence_output[i, entity2_pos[i, 1], :] for i in range(batch_size)])
        
        # Pool CLS token, and entity representations
        cls_emb = sequence_output[:, 0, :]  # [batch_size, hidden_size]
        entity1_emb = (entity1_start_embs + entity1_end_embs) / 2
        entity2_emb = (entity2_start_embs + entity2_end_embs) / 2
        
        # Concatenate representations
        combined_emb = torch.cat([cls_emb, entity1_emb, entity2_emb], dim=1)  # [batch_size, hidden_size*3]
        
        # Apply dropout and classification
        combined_emb = self.dropout(combined_emb)
        logits = self.classifier(combined_emb)
        
        return logits
    
    def prepare_input(
        self,
        text: str,
        entity1: Tuple[str, int, int],
        entity2: Tuple[str, int, int]
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare input for the model by adding entity markers and tokenizing.
        
        Args:
            text: Input text
            entity1: Tuple of (entity_text, start_idx, end_idx)
            entity2: Tuple of (entity_text, start_idx, end_idx)
            
        Returns:
            Dictionary with tokenized inputs ready for the model
        """
        e1_text, e1_start, e1_end = entity1
        e2_text, e2_start, e2_end = entity2
        
        # Insert entity markers
        if e1_start < e2_start:
            marked_text = (
                text[:e1_start] + "[E1] " + text[e1_start:e1_end] + " [/E1]" +
                text[e1_end:e2_start] + "[E2] " + text[e2_start:e2_end] + " [/E2]" +
                text[e2_end:]
            )
        else:
            marked_text = (
                text[:e2_start] + "[E2] " + text[e2_start:e2_end] + " [/E2]" +
                text[e2_end:e1_start] + "[E1] " + text[e1_start:e1_end] + " [/E1]" +
                text[e1_end:]
            )
        
        # Tokenize
        encoding = self.tokenizer(
            marked_text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding="max_length"
        )
        
        # Find entity marker positions
        input_ids = encoding["input_ids"][0].tolist()
        e1_start_id = self.tokenizer.convert_tokens_to_ids("[E1]")
        e1_end_id = self.tokenizer.convert_tokens_to_ids("[/E1]")
        e2_start_id = self.tokenizer.convert_tokens_to_ids("[E2]")
        e2_end_id = self.tokenizer.convert_tokens_to_ids("[/E2]")
        
        try:
            e1_start_pos = input_ids.index(e1_start_id)
            e1_end_pos = input_ids.index(e1_end_id)
            e2_start_pos = input_ids.index(e2_start_id)
            e2_end_pos = input_ids.index(e2_end_id)
        except ValueError:
            # Fall back to approximate positions if markers not found
            logger.warning(f"Entity markers not found in tokenized text, using approximations")
            tokens = self.tokenizer.tokenize(marked_text)
            e1_start_pos = tokens.index("[E1]") if "[E1]" in tokens else 1
            e1_end_pos = tokens.index("[/E1]") if "[/E1]" in tokens else 2
            e2_start_pos = tokens.index("[E2]") if "[E2]" in tokens else 3
            e2_end_pos = tokens.index("[/E2]") if "[/E2]" in tokens else 4
        
        # Create entity position tensors
        entity1_pos = torch.tensor([[e1_start_pos, e1_end_pos]], device=self.device)
        entity2_pos = torch.tensor([[e2_start_pos, e2_end_pos]], device=self.device)
        
        # Move everything to the correct device
        encoding = {k: v.to(self.device) for k, v in encoding.items()}
        
        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "entity1_pos": entity1_pos,
            "entity2_pos": entity2_pos
        }
    
    def extract_relations(
        self,
        text: str,
        entities: List[Dict[str, Any]],
        relation_types: List[str],
        threshold: float = 0.5
    ) -> List[RelationInstance]:
        """
        Extract relations between entities in the text.
        
        Args:
            text: Input text
            entities: List of entity dictionaries with text, start, end, and label fields
            relation_types: List of relation type names corresponding to model outputs
            threshold: Confidence threshold for relation extraction
            
        Returns:
            List of extracted relations
        """
        self.eval()
        relations = []
        
        # Consider all entity pairs
        for i, entity1 in enumerate(entities):
            for j, entity2 in enumerate(entities):
                if i == j:
                    continue
                
                # Prepare input
                try:
                    entity1_tuple = (entity1["text"], entity1["start"], entity1["end"])
                    entity2_tuple = (entity2["text"], entity2["start"], entity2["end"])
                    model_input = self.prepare_input(text, entity1_tuple, entity2_tuple)
                    
                    # Run model
                    with torch.no_grad():
                        logits = self(
                            model_input["input_ids"],
                            model_input["attention_mask"],
                            model_input["entity1_pos"],
                            model_input["entity2_pos"]
                        )
                        
                        # Convert logits to probabilities
                        probs = F.softmax(logits, dim=1).squeeze(0)
                        
                        # Get relation with highest probability
                        max_idx = torch.argmax(probs).item()
                        max_prob = probs[max_idx].item()
                        
                        # Skip "no relation" (usually index 0) or low confidence
                        if max_idx > 0 and max_prob > threshold:
                            relation_type = relation_types[max_idx]
                            
                            # Create context snippet
                            start_idx = max(0, min(entity1["start"], entity2["start"]) - 50)
                            end_idx = min(len(text), max(entity1["end"], entity2["end"]) + 50)
                            context = text[start_idx:end_idx]
                            
                            # Create relation instance
                            relation = RelationInstance(
                                head_entity=entity1["text"],
                                tail_entity=entity2["text"],
                                head_type=entity1["label"],
                                tail_type=entity2["label"],
                                relation_type=relation_type,
                                confidence=max_prob,
                                context=context
                            )
                            relations.append(relation)
                except Exception as e:
                    logger.error(f"Error extracting relation: {str(e)}")
                    continue
        
        return relations

class MedicalRelationExtractor:
    """
    Medical relation extractor that detects biomedical relationships between entities.
    
    Supports detection of common medical relation types like:
    - treats (medication/procedure treats condition)
    - causes (factor causes condition)
    - associates_with (factor associated with condition)
    - compares (comparison between entities)
    - contraindicates (medication/procedure contraindicated for condition)
    - diagnoses (test diagnoses condition)
    - predicts (factor predicts outcome)
    """
    
    MEDICAL_RELATION_TYPES = [
        "no_relation",
        "treats",
        "causes",
        "associates_with",
        "compares",
        "contraindicates",
        "diagnoses",
        "predicts"
    ]
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        use_pretrained: bool = True,
        device: Optional[str] = None
    ):
        """
        Initialize the medical relation extractor.
        
        Args:
            model_path: Path to saved model weights (if not using pretrained)
            use_pretrained: Whether to use pretrained weights
            device: Device to use (cuda or cpu)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.relation_types = self.MEDICAL_RELATION_TYPES
        
        # Initialize the model
        self.model = BioBERTRelationExtractor(
            model_name="dmis-lab/biobert-v1.1",
            num_relations=len(self.relation_types),
            device=self.device
        )
        
        # Load pretrained weights if specified
        if not use_pretrained and model_path:
            logger.info(f"Loading model weights from {model_path}")
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        self.model.eval()
        logger.info(f"Initialized Medical Relation Extractor with {len(self.relation_types)} relation types")
    
    def extract_relations_from_text(
        self,
        text: str,
        entities: List[Dict[str, Any]],
        threshold: float = 0.5
    ) -> List[RelationInstance]:
        """
        Extract medical relations from text given a list of detected entities.
        
        Args:
            text: Input text
            entities: List of entity dictionaries with text, start, end, and label fields
            threshold: Confidence threshold for relation extraction
            
        Returns:
            List of extracted relations
        """
        return self.model.extract_relations(
            text=text,
            entities=entities,
            relation_types=self.relation_types,
            threshold=threshold
        )
    
    def extract_relations_from_sections(
        self,
        sections: List[Any],  # Using Any to avoid circular imports with SectionInfo
        threshold: float = 0.5
    ) -> Dict[str, List[RelationInstance]]:
        """
        Extract relations from document sections.
        
        Args:
            sections: List of SectionInfo objects
            threshold: Confidence threshold for relation extraction
            
        Returns:
            Dictionary mapping section types to lists of extracted relations
        """
        section_relations = {}
        
        for section in sections:
            # Skip sections with no structured content or entities
            if not hasattr(section, "metadata") or "structured_content" not in section.metadata:
                continue
            
            structured_content = section.metadata["structured_content"]
            if "entities" not in structured_content or not structured_content["entities"]:
                continue
            
            # Convert entities to required format
            entities = structured_content["entities"]
            
            # Extract relations from this section
            relations = self.extract_relations_from_text(
                text=section.text,
                entities=entities,
                threshold=threshold
            )
            
            if relations:
                section_relations[section.section_type] = relations
        
        return section_relations
    
    def to_networkx(self, relations: List[RelationInstance]) -> Any:
        """
        Convert relations to a NetworkX graph for visualization and analysis.
        
        Args:
            relations: List of extracted relations
            
        Returns:
            NetworkX graph object
        """
        try:
            import networkx as nx
            
            G = nx.DiGraph()
            
            # Add nodes and edges
            for rel in relations:
                # Add nodes with entity types as attributes
                G.add_node(rel.head_entity, entity_type=rel.head_type)
                G.add_node(rel.tail_entity, entity_type=rel.tail_type)
                
                # Add edge with relation type and confidence as attributes
                G.add_edge(
                    rel.head_entity,
                    rel.tail_entity,
                    relation=rel.relation_type,
                    confidence=rel.confidence,
                    context=rel.context
                )
            
            return G
        except ImportError:
            logger.warning("NetworkX not installed. Cannot create graph representation.")
            return None


# Integration with the MedicalDocumentProcessor
def integrate_relation_extraction(document_processor, document_structure):
    """
    Integrate relation extraction into the MedicalDocumentProcessor pipeline.
    
    Args:
        document_processor: Existing MedicalDocumentProcessor instance
        document_structure: Processed DocumentStructure
        
    Returns:
        Updated document structure with relations
    """
    # Initialize relation extractor
    relation_extractor = MedicalRelationExtractor()
    
    # Extract relations from each section
    all_relations = []
    section_relations = relation_extractor.extract_relations_from_sections(document_structure.sections)
    
    # Add relations to document metadata
    document_structure.metadata["relations"] = {
        section_type: [
            {
                "head": rel.head_entity,
                "head_type": rel.head_type,
                "tail": rel.tail_entity,
                "tail_type": rel.tail_type,
                "relation": rel.relation_type,
                "confidence": rel.confidence,
                "context": rel.context
            }
            for rel in relations
        ]
        for section_type, relations in section_relations.items()
    }
    
    # Create a graph representation
    all_relations = [rel for relations in section_relations.values() for rel in relations]
    document_structure.metadata["relation_graph"] = relation_extractor.to_networkx(all_relations)
    
    return document_structure