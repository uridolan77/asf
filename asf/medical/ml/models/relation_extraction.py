"""
Medical Relation Extraction

This module provides functionality for extracting and analyzing relationships
between biomedical entities in medical texts using transformer models.
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Any
from itertools import combinations
from transformers import AutoTokenizer, AutoModel
from ..preprocessing.document_structure import Entity, DocumentStructure, RelationInstance

# Try to import graph neural network components if available
try:
    from torch_geometric.nn import GATv2Conv
    GNN_AVAILABLE = True
except ImportError:
    GNN_AVAILABLE = False

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
        
        # Load pre-trained model and tokenizer
        logger.info(f"Loading encoder model: {encoder_model}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(encoder_model)
            self.encoder = AutoModel.from_pretrained(encoder_model)
            
            # Graph neural network components
            hidden_size = self.encoder.config.hidden_size
            self.dropout = nn.Dropout(dropout_prob)
            
            # Relation classification heads
            self.classifier = nn.Sequential(
                nn.Linear(hidden_size, gnn_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_prob),
                nn.Linear(gnn_hidden_dim, num_relations)
            )
            
            # Add GNN layers if available
            if GNN_AVAILABLE:
                self.gnn_conv1 = GATv2Conv(hidden_size, gnn_hidden_dim, heads=4, dropout=dropout_prob)
                self.gnn_conv2 = GATv2Conv(gnn_hidden_dim * 4, gnn_hidden_dim, heads=1, dropout=dropout_prob)
                logger.info("GNN layers initialized")
            
            self.to(self.device)
            logger.info(f"Initialized BioGPT relation extractor with {num_relations} relation types on {self.device}")
        except Exception as e:
            logger.error(f"Error initializing relation extraction model: {str(e)}")
            raise
    
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
    
    def prepare_entity_pair_input(
        self,
        text: str,
        entity1: Entity,
        entity2: Entity
    ) -> str:
        """
        Prepare input text for relation extraction between two entities.
        
        Args:
            text: Source text
            entity1: First entity
            entity2: Second entity
            
        Returns:
            Text with entity markers
        """
        # Sort entities by position
        if entity1.start < entity2.start:
            first, second = entity1, entity2
        else:
            first, second = entity2, entity1
        
        # Insert entity markers
        marked_text = (
            text[:first.start] + 
            f"[E1] {first.text} [/E1]" + 
            text[first.end:second.start] + 
            f"[E2] {second.text} [/E2]" + 
            text[second.end:]
        )
        
        # Truncate to manageable length while preserving entities
        if len(marked_text) > 512:
            # Find positions of markers
            e1_start = marked_text.find("[E1]")
            e2_end = marked_text.find("[/E2]") + len("[/E2]")
            
            # Ensure we keep the context around entities
            context_len = min(250, (512 - (e2_end - e1_start)) // 2)
            context_start = max(0, e1_start - context_len)
            context_end = min(len(marked_text), e2_end + context_len)
            
            marked_text = marked_text[context_start:context_end]
        
        return marked_text
    
    def forward(self, node_features, edge_index):
        """
        Forward pass through the GNN.
        
        Args:
            node_features: Node features [num_nodes, hidden_size]
            edge_index: Edge indices [2, num_edges]
            
        Returns:
            Node representations after GNN layers
        """
        if not GNN_AVAILABLE:
            logger.warning("GNN components not available, skipping GNN forward pass")
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
        # Prepare input
        marked_text = self.prepare_entity_pair_input(text, entity1, entity2)
        
        # Encode with BioGPT
        encoding = self.encode_text(marked_text)
        
        # For a simple pair, we can just use the encoded representation
        # directly with the classifier, without needing the full GNN
        logits = self.classifier(encoding)
        probs = F.softmax(logits, dim=1).squeeze(0)
        
        # Get prediction
        pred_idx = torch.argmax(probs).item()
        confidence = probs[pred_idx].item()
        
        # Return relation type and confidence
        if pred_idx == 0:
            # Assuming index 0 is "no_relation"
            return "no_relation", confidence
        else:
            return relation_types[pred_idx], confidence


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
    
    # Valid entity type pairs for each relation
    VALID_RELATION_PAIRS = {
        "treats": [
            ("Drug", "Disease"),
            ("Chemical", "Disease"),
            ("Procedure", "Disease")
        ],
        "causes": [
            ("Drug", "Disease"),
            ("Chemical", "Disease"),
            ("Disease", "Disease"),
            ("Organism", "Disease")
        ],
        "diagnoses": [
            ("Procedure", "Disease")
        ],
        "prevents": [
            ("Drug", "Disease"),
            ("Chemical", "Disease"),
            ("Procedure", "Disease")
        ],
        "complicates": [
            ("Disease", "Disease"),
            ("Procedure", "Disease")
        ],
        "predisposes": [
            ("Disease", "Disease"),
            ("Gene/Protein", "Disease")
        ],
        "associated_with": [
            # Any combination is valid
        ]
    }
    
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
            
        try:
            import networkx as nx
            self.nx = nx
        except ImportError:
            logger.warning("NetworkX not available, knowledge graph functionality will be limited")
            self.nx = None
    
    def is_valid_pair(self, entity1: Entity, entity2: Entity, relation: str) -> bool:
        """
        Check if the entity pair is valid for the given relation type.
        
        Args:
            entity1: First entity
            entity2: Second entity
            relation: Relation type
            
        Returns:
            Whether the pair is valid for the relation
        """
        # Associated_with is valid for any pair
        if relation == "associated_with":
            return True
        
        # Check if the pair types are in valid combinations
        valid_pairs = self.VALID_RELATION_PAIRS.get(relation, [])
        return (entity1.label, entity2.label) in valid_pairs or (entity2.label, entity1.label) in valid_pairs
    
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
        if not self.model:
            logger.error("Model not initialized, cannot extract relations")
            return []
            
        relations = []
        
        # Consider all entity pairs
        for entity1, entity2 in combinations(entities, 2):
            # Skip self-relations
            if entity1.start == entity2.start:
                continue
            
            # Extract relation
            relation_type, confidence = self.model.extract_relation(
                text, entity1, entity2, self.RELATION_TYPES
            )
            
            # Skip no_relation or low confidence
            if relation_type == "no_relation" or confidence < threshold:
                continue
            
            # Skip invalid entity type combinations
            if not self.is_valid_pair(entity1, entity2, relation_type):
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
        
        # Process each section
        for section in doc_structure.sections:
            if section.entities:
                section_relations = self.extract_relations_from_text(
                    section.text, section.entities
                )
                section.relations = [rel.to_dict() for rel in section_relations]
                all_relations.extend(section_relations)
            
            # Process subsections
            for subsection in section.subsections:
                if subsection.entities:
                    subsection_relations = self.extract_relations_from_text(
                        subsection.text, subsection.entities
                    )
                    subsection.relations = [rel.to_dict() for rel in subsection_relations]
                    all_relations.extend(subsection_relations)
        
        # Add all relations to document
        doc_structure.relations = [rel.to_dict() for rel in all_relations]
        
        # Build knowledge graph if NetworkX is available
        if self.nx:
            doc_structure.knowledge_graph = self.build_knowledge_graph(all_relations)
        
        logger.info(f"Extracted {len(all_relations)} biomedical relations")
        return doc_structure
    
    def build_knowledge_graph(self, relations: List[RelationInstance]) -> Any:
        """
        Build a knowledge graph from extracted relations.
        
        Args:
            relations: List of relations
            
        Returns:
            NetworkX directed graph or None if NetworkX is not available
        """
        if not self.nx:
            return None
            
        G = self.nx.DiGraph()
        
        # Add nodes and edges
        for rel in relations:
            # Add nodes with entity types as attributes
            head_id = rel.head_cui if rel.head_cui else rel.head_entity
            tail_id = rel.tail_cui if rel.tail_cui else rel.tail_entity
            
            G.add_node(head_id, 
                      entity_text=rel.head_entity, 
                      entity_type=rel.head_type, 
                      cui=rel.head_cui)
            G.add_node(tail_id, 
                      entity_text=rel.tail_entity, 
                      entity_type=rel.tail_type,
                      cui=rel.tail_cui)
            
            # Add edge with relation type and confidence
            G.add_edge(
                head_id,
                tail_id,
                relation=rel.relation_type,
                confidence=rel.confidence,
                context=rel.context
            )
        
        return G