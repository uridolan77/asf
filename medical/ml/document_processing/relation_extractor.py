"""
Medical Relation Extraction

This module provides functionality for extracting and analyzing relationships
between biomedical entities in medical texts using transformer models.
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from typing import List, Dict, Tuple, Optional, Any
from itertools import combinations
from transformers import AutoTokenizer, AutoModel

# Local imports
from .document_structure import Entity, DocumentStructure, RelationInstance

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BioGPTRelationExtractor(nn.Module):
    """
    Relation extraction model based on BioGPT with optional GNN for graph-based reasoning.
    
    This model extracts biomedical relations between entities using a combination
    of BioGPT for contextualized representations and optionally a GNN for relation modeling.
    """
    
    def __init__(
        self,
        encoder_model: str = "microsoft/biogpt",
        gnn_hidden_dim: int = 256,
        num_relations: int = 8,
        dropout_prob: float = 0.1,
        device: Optional[str] = None
    ):
        """
        Initialize the relation extraction model.
        
        Args:
            encoder_model: BioGPT encoder model
            gnn_hidden_dim: Hidden dimension for GNN
            num_relations: Number of relation types
            dropout_prob: Dropout probability
            device: Device for model
        """
        super(BioGPTRelationExtractor, self).__init__()
        
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = encoder_model
        self.num_relations = num_relations
        
        # Initialize BioGPT encoder
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_model)
        self.encoder = AutoModel.from_pretrained(encoder_model)
        
        # Freeze encoder parameters
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # Initialize GNN components if available
        try:
            from torch_geometric.nn import GATv2Conv
            
            # GNN layers
            self.gnn_conv1 = GATv2Conv(
                in_channels=self.encoder.config.hidden_size,
                out_channels=gnn_hidden_dim,
                heads=4,
                dropout=dropout_prob
            )
            
            self.gnn_conv2 = GATv2Conv(
                in_channels=gnn_hidden_dim * 4,  # * heads
                out_channels=gnn_hidden_dim,
                heads=1,
                dropout=dropout_prob
            )
            
            self.use_gnn = True
            logger.info("Using GNN for relation extraction")
        except ImportError:
            self.use_gnn = False
            logger.info("GNN not available, using simple relation extraction")
        
        # Relation classifier
        input_dim = gnn_hidden_dim * 2 if self.use_gnn else self.encoder.config.hidden_size * 2
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, gnn_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(gnn_hidden_dim, num_relations)
        )
        
        self.to(self.device)
        logger.info(f"Initialized BioGPT relation extractor with {num_relations} relation types on {self.device}")
    
    def encode_text(self, text: str) -> torch.Tensor:
        """
        Encode text using BioGPT encoder.
        
        Args:
            text: Input text
            
        Returns:
            Encoded text tensor
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.encoder(**inputs)
            # Use CLS token embedding
            return outputs.last_hidden_state[:, 0, :]
    
    def forward(self, node_features, edge_index):
        """
        Forward pass through the GNN.
        
        Args:
            node_features: Node features [num_nodes, hidden_size]
            edge_index: Edge indices [2, num_edges]
            
        Returns:
            Node representations after GNN layers
        """
        if not self.use_gnn:
            return node_features
            
        # Apply GNN layers
        x = self.gnn_conv1(node_features, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.1, training=self.training)
        
        x = self.gnn_conv2(x, edge_index)
        x = F.relu(x)
        
        return x
    
    def extract_relation(
        self,
        text: str,
        entity1: Entity,
        entity2: Entity,
        relation_types: List[str]
    ) -> Tuple[str, float]:
        """
        Extract relation between two entities.
        
        Args:
            text: Context text
            entity1: First entity
            entity2: Second entity
            relation_types: List of relation type names
            
        Returns:
            Tuple of (relation_type, confidence)
        """
        # Create input text with entity markers
        e1_start = entity1.start
        e1_end = entity1.end
        e2_start = entity2.start
        e2_end = entity2.end
        
        # Ensure correct order
        if e1_start > e2_start:
            entity1, entity2 = entity2, entity1
            e1_start, e2_start = e2_start, e1_start
            e1_end, e2_end = e2_end, e1_end
        
        # Insert entity markers
        marked_text = (
            text[:e1_start] + "[E1] " + text[e1_start:e1_end] + " [/E1]" +
            text[e1_end:e2_start] + "[E2] " + text[e2_start:e2_end] + " [/E2]" +
            text[e2_end:]
        )
        
        # Encode text
        encoding = self.encode_text(marked_text)
        
        # Simple classification without GNN
        logits = self.classifier(encoding)
        
        # Get prediction
        probs = F.softmax(logits, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_idx].item()
        
        # Map to relation type
        relation_type = relation_types[pred_idx]
        
        return relation_type, confidence


class MedicalRelationExtractor:
    """
    Medical relation extractor for biomedical text with UMLS-linked entities.
    """
    
    # Medical relation types
    RELATION_TYPES = [
        "no_relation",
        "treats",
        "causes",
        "diagnoses",
        "prevents",
        "complicates",
        "predisposes",
        "associated_with"
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
            model_path: Path to saved model weights
            use_pretrained: Whether to use pretrained weights
            device: Device for model
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize model
        try:
            self.model = BioGPTRelationExtractor(
                encoder_model="microsoft/biogpt",
                num_relations=len(self.RELATION_TYPES),
                device=self.device
            )
            
            # Load pretrained weights if specified
            if not use_pretrained and model_path:
                logger.info(f"Loading model weights from {model_path}")
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            
            self.model.eval()
            logger.info(f"Initialized Medical Relation Extractor with {len(self.RELATION_TYPES)} relation types")
        except Exception as e:
            logger.error(f"Failed to initialize relation extractor: {str(e)}")
            self.model = None
    
    def extract_relations_from_text(
        self,
        text: str,
        entities: List[Entity],
        threshold: float = 0.5
    ) -> List[RelationInstance]:
        """
        Extract relations between entities in text.
        
        Args:
            text: Input text
            entities: List of entities
            threshold: Confidence threshold
            
        Returns:
            List of extracted relations
        """
        if self.model is None:
            logger.error("Relation extraction model not initialized")
            return []
            
        relations = []
        
        # Generate all possible entity pairs
        for entity1, entity2 in combinations(entities, 2):
            # Skip if entities are too far apart (more than 100 tokens)
            if abs(entity1.start - entity2.start) > 500:
                continue
                
            # Extract relation
            relation_type, confidence = self.model.extract_relation(
                text, entity1, entity2, self.RELATION_TYPES
            )
            
            # Skip no_relation or low confidence
            if relation_type == "no_relation" or confidence < threshold:
                continue
                
            # Create context snippet
            start_idx = max(0, min(entity1.start, entity2.start) - 50)
            end_idx = min(len(text), max(entity1.end, entity2.end) + 50)
            context = text[start_idx:end_idx]
            
            # Create relation instance
            relation = RelationInstance(
                head_entity=entity1.text,
                tail_entity=entity2.text,
                head_type=entity1.label,
                tail_type=entity2.label,
                relation_type=relation_type,
                confidence=confidence,
                context=context,
                head_cui=entity1.cui,
                tail_cui=entity2.cui
            )
            
            relations.append(relation)
        
        return relations
    
    def process_document(self, doc_structure: DocumentStructure) -> DocumentStructure:
        """
        Process a document structure to extract relations.
        
        Args:
            doc_structure: Document structure with entities
            
        Returns:
            Updated document structure with relations
        """
        logger.info("Extracting biomedical relations...")
        all_relations = []
        
        # Process abstract
        if doc_structure.abstract and doc_structure.entities:
            # Filter entities from abstract
            abstract_entities = [
                e for e in doc_structure.entities 
                if e.start >= 0 and e.end <= len(doc_structure.abstract)
            ]
            
            if abstract_entities:
                abstract_relations = self.extract_relations_from_text(
                    doc_structure.abstract, abstract_entities
                )
                all_relations.extend(abstract_relations)
        
        # Process sections
        for section in doc_structure.sections:
            if section.entities:
                section_relations = self.extract_relations_from_text(
                    section.text, section.entities
                )
                section.relations = [rel.to_dict() for rel in section_relations]
                all_relations.extend(section_relations)
                
                # Process subsections recursively
                for subsection in section.subsections:
                    if subsection.entities:
                        subsection_relations = self.extract_relations_from_text(
                            subsection.text, subsection.entities
                        )
                        subsection.relations = [rel.to_dict() for rel in subsection_relations]
                        all_relations.extend(subsection_relations)
        
        # Add all relations to document
        doc_structure.relations = [rel.to_dict() for rel in all_relations]
        
        # Build knowledge graph
        doc_structure.knowledge_graph = self.build_knowledge_graph(all_relations)
        
        logger.info(f"Extracted {len(all_relations)} biomedical relations")
        return doc_structure
    
    def build_knowledge_graph(self, relations: List[RelationInstance]) -> nx.DiGraph:
        """
        Build a knowledge graph from extracted relations.
        
        Args:
            relations: List of relations
            
        Returns:
            NetworkX DiGraph
        """
        G = nx.DiGraph()
        
        # Add nodes and edges
        for rel in relations:
            # Add head entity node
            if not G.has_node(rel.head_entity):
                G.add_node(
                    rel.head_entity,
                    type=rel.head_type,
                    cui=rel.head_cui
                )
                
            # Add tail entity node
            if not G.has_node(rel.tail_entity):
                G.add_node(
                    rel.tail_entity,
                    type=rel.tail_type,
                    cui=rel.tail_cui
                )
                
            # Add edge
            G.add_edge(
                rel.head_entity,
                rel.tail_entity,
                type=rel.relation_type,
                confidence=rel.confidence,
                context=rel.context
            )
        
        return G
