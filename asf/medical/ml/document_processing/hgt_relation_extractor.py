"""
Heterogeneous Graph Transformer Relation Extractor

This module provides functionality for extracting biomedical relations using
a Heterogeneous Graph Transformer (HGT) model, which is specifically designed
for heterogeneous graphs with different node and edge types.
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Any, Set
import networkx as nx

# Local imports
from .document_structure import DocumentStructure, Entity, RelationInstance
from .sentence_segmenter import SentenceSegmenter, Sentence

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HGTRelationExtractor:
    """
    Biomedical relation extractor using Heterogeneous Graph Transformer (HGT).

    This extractor builds a heterogeneous graph from document entities and uses
    a Heterogeneous Graph Transformer to predict relations between entities.
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
        encoder_model: str = "microsoft/biogpt",
        model_path: Optional[str] = None,
        use_pretrained: bool = True,
        use_sentence_segmentation: bool = True,
        spacy_model: str = "en_core_sci_md",
        device: Optional[str] = None
    ):
        """
        Initialize the HGT relation extractor.

        Args:
            encoder_model: Base encoder model for node features
            model_path: Path to saved model weights
            use_pretrained: Whether to use pretrained weights
            device: Device for model
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Check if PyTorch Geometric is available
        try:
            import torch_geometric
            self.torch_geometric_available = True
            logger.info("PyTorch Geometric is available")
        except ImportError:
            self.torch_geometric_available = False
            logger.warning("PyTorch Geometric is not available. Install with: pip install torch-geometric")

        # Initialize model
        if self.torch_geometric_available:
            try:
                self.model = self._initialize_model(
                    encoder_model=encoder_model,
                    model_path=model_path,
                    use_pretrained=use_pretrained
                )
                logger.info(f"Initialized HGT Relation Extractor with {len(self.RELATION_TYPES)} relation types")
            except Exception as e:
                logger.error(f"Failed to initialize HGT relation extractor: {str(e)}")
                self.model = None
        else:
            self.model = None

        # Initialize NetworkX for knowledge graph
        try:
            self.nx = nx
        except Exception:
            logger.warning("NetworkX not available, knowledge graph functionality will be limited")
            self.nx = None

        # Initialize sentence segmenter if requested
        self.use_sentence_segmentation = use_sentence_segmentation
        if use_sentence_segmentation:
            try:
                self.sentence_segmenter = SentenceSegmenter(spacy_model=spacy_model)
                logger.info("Initialized sentence segmenter for relation extraction")
            except Exception as e:
                logger.warning(f"Failed to initialize sentence segmenter: {str(e)}")
                self.use_sentence_segmentation = False

    def _initialize_model(
        self,
        encoder_model: str,
        model_path: Optional[str],
        use_pretrained: bool
    ) -> nn.Module:
        """
        Initialize the HGT relation extraction model.

        Args:
            encoder_model: Base encoder model for node features
            model_path: Path to saved model weights
            use_pretrained: Whether to use pretrained weights

        Returns:
            Initialized model
        """
        from transformers import AutoTokenizer, AutoModel
        from torch_geometric.nn import HGTConv, Linear

        # Define model class
        class HGTRelationModel(nn.Module):
            def __init__(
                self,
                encoder_model: str,
                hidden_dim: int = 256,
                num_relations: int = 8,
                num_heads: int = 4,
                num_layers: int = 2,
                dropout: float = 0.1,
                device: str = "cpu"
            ):
                super(HGTRelationModel, self).__init__()

                self.device = device
                self.hidden_dim = hidden_dim
                self.num_relations = num_relations

                # Initialize encoder for node features
                self.tokenizer = AutoTokenizer.from_pretrained(encoder_model)
                self.encoder = AutoModel.from_pretrained(encoder_model)

                # Freeze encoder parameters
                for param in self.encoder.parameters():
                    param.requires_grad = False

                # Node types
                self.node_types = ["entity", "mention", "sentence", "section"]

                # Edge types as (src_type, edge_type, dst_type)
                self.edge_types = [
                    ("entity", "has_mention", "mention"),
                    ("mention", "in_sentence", "sentence"),
                    ("sentence", "in_section", "section"),
                    ("sentence", "next", "sentence"),
                    ("mention", "co_occurs", "mention")
                ]

                # Define node type embeddings
                self.node_type_embeddings = nn.Embedding(
                    len(self.node_types), hidden_dim
                )

                # Define HGT layers
                self.hgt_layers = nn.ModuleList()

                # Input projection
                encoder_dim = self.encoder.config.hidden_size
                self.input_projection = nn.ModuleDict({
                    node_type: Linear(encoder_dim, hidden_dim)
                    for node_type in self.node_types
                })

                # HGT layers
                for _ in range(num_layers):
                    self.hgt_layers.append(
                        HGTConv(
                            in_channels=hidden_dim,
                            out_channels=hidden_dim,
                            metadata=(self.node_types, self.edge_types),
                            heads=num_heads,
                            dropout=dropout
                        )
                    )

                # Relation classifier
                self.classifier = nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, num_relations)
                )

                self.to(device)

            def encode_text(self, text: str) -> torch.Tensor:
                """Encode text using the encoder model."""
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

            def forward(self, x_dict, edge_index_dict):
                """Forward pass through the HGT model."""
                # Project node features to hidden dimension
                h_dict = {}
                for node_type, x in x_dict.items():
                    h_dict[node_type] = self.input_projection[node_type](x)

                # Apply HGT layers
                for layer in self.hgt_layers:
                    h_dict = layer(h_dict, edge_index_dict)

                return h_dict

            def extract_relation(
                self,
                text: str,
                entity1: Entity,
                entity2: Entity,
                relation_types: List[str]
            ) -> Tuple[str, float]:
                """Extract relation between two entities."""
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

                # Simple classification without HGT for direct entity pairs
                logits = self.classifier(torch.cat([encoding, encoding], dim=1))

                # Get prediction
                probs = F.softmax(logits, dim=1)
                pred_idx = torch.argmax(probs, dim=1).item()
                confidence = probs[0, pred_idx].item()

                # Map to relation type
                relation_type = relation_types[pred_idx]

                return relation_type, confidence

        # Initialize model
        model = HGTRelationModel(
            encoder_model=encoder_model,
            num_relations=len(self.RELATION_TYPES),
            device=self.device
        )

        # Load pretrained weights if specified
        if not use_pretrained and model_path:
            logger.info(f"Loading model weights from {model_path}")
            model.load_state_dict(torch.load(model_path, map_location=self.device))

        model.eval()
        return model

    def extract_relations_from_text(
        self,
        text: str,
        entities: List[Entity],
        threshold: float = 0.5,
        sentences: Optional[List[Sentence]] = None
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
            logger.error("HGT relation extraction model not initialized")
            return []

        relations = []

        # Segment text into sentences if requested and not provided
        if self.use_sentence_segmentation and not sentences:
            try:
                sentences = self.sentence_segmenter.segment_text(text)
                logger.debug(f"Segmented text into {len(sentences)} sentences")
            except Exception as e:
                logger.warning(f"Sentence segmentation failed: {str(e)}")
                sentences = None

        # Generate all possible entity pairs
        from itertools import combinations
        for entity1, entity2 in combinations(entities, 2):
            # Skip if entities are too far apart (more than 500 characters)
            if abs(entity1.start - entity2.start) > 500:
                continue

            # If we have sentences, check if entities are in the same or adjacent sentences
            if sentences:
                same_or_adjacent_sentence = False
                entity1_sentence = None
                entity2_sentence = None

                # Find sentences containing the entities
                for sent in sentences:
                    if entity1.start >= sent.start and entity1.end <= sent.end:
                        entity1_sentence = sent
                    if entity2.start >= sent.start and entity2.end <= sent.end:
                        entity2_sentence = sent

                    # If we found both, break
                    if entity1_sentence and entity2_sentence:
                        break

                # Check if entities are in the same sentence
                if entity1_sentence and entity2_sentence:
                    if entity1_sentence == entity2_sentence:
                        same_or_adjacent_sentence = True
                    else:
                        # Check if sentences are adjacent
                        sent1_idx = sentences.index(entity1_sentence)
                        sent2_idx = sentences.index(entity2_sentence)
                        if abs(sent1_idx - sent2_idx) <= 1:
                            same_or_adjacent_sentence = True

                # Skip if entities are not in the same or adjacent sentences
                if not same_or_adjacent_sentence:
                    continue

                # Use sentence text for relation extraction if entities are in the same sentence
                if entity1_sentence == entity2_sentence:
                    # Adjust entity positions relative to sentence start
                    entity1_adjusted = Entity(
                        text=entity1.text,
                        label=entity1.label,
                        start=entity1.start - entity1_sentence.start,
                        end=entity1.end - entity1_sentence.start,
                        confidence=entity1.confidence
                    )
                    entity2_adjusted = Entity(
                        text=entity2.text,
                        label=entity2.label,
                        start=entity2.start - entity1_sentence.start,
                        end=entity2.end - entity1_sentence.start,
                        confidence=entity2.confidence
                    )

                    # Extract relation using sentence text
                    relation_type, confidence = self.model.extract_relation(
                        entity1_sentence.text, entity1_adjusted, entity2_adjusted, self.RELATION_TYPES
                    )
                else:
                    # Extract relation using original text
                    relation_type, confidence = self.model.extract_relation(
                        text, entity1, entity2, self.RELATION_TYPES
                    )
            else:
                # No sentence segmentation, use original text
                relation_type, confidence = self.model.extract_relation(
                    text, entity1, entity2, self.RELATION_TYPES
                )

            # Skip no_relation or low confidence
            if relation_type == "no_relation" or confidence < threshold:
                continue

            # Create relation instance
            relation = RelationInstance(
                relation_type=relation_type,
                head_entity=entity1.text,
                head_type=entity1.label,
                head_cui=getattr(entity1, "cui", None),
                tail_entity=entity2.text,
                tail_type=entity2.label,
                tail_cui=getattr(entity2, "cui", None),
                confidence=confidence,
                context=text[max(0, min(entity1.start, entity2.start) - 50):
                            min(len(text), max(entity1.end, entity2.end) + 50)]
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
        logger.info("Extracting biomedical relations with HGT...")
        all_relations = []

        # Segment document if using sentence segmentation
        document_sentences = {}
        if self.use_sentence_segmentation:
            try:
                document_sentences = self.sentence_segmenter.segment_document(doc_structure)
                logger.info(f"Segmented document into {sum(len(sents) for sents in document_sentences.values())} sentences")
            except Exception as e:
                logger.warning(f"Document segmentation failed: {str(e)}")

        # Process abstract
        if doc_structure.abstract and doc_structure.entities:
            # Filter entities from abstract
            abstract_entities = [
                e for e in doc_structure.entities
                if e.start >= 0 and e.end <= len(doc_structure.abstract)
            ]

            if abstract_entities:
                # Get abstract sentences if available
                abstract_sentences = document_sentences.get("abstract") if document_sentences else None

                abstract_relations = self.extract_relations_from_text(
                    doc_structure.abstract, abstract_entities, sentences=abstract_sentences
                )
                all_relations.extend(abstract_relations)

        # Process sections
        for section in doc_structure.sections:
            # Filter entities from this section
            section_entities = section.entities

            if section_entities:
                # Get section sentences if available
                section_id = f"section_{doc_structure.sections.index(section)}"
                section_sentences = document_sentences.get(section_id) if document_sentences else None

                section_relations = self.extract_relations_from_text(
                    section.text, section_entities, sentences=section_sentences
                )
                section.relations = section_relations
                all_relations.extend(section_relations)

            # Process subsections
            for subsection in section.subsections:
                if subsection.entities:
                    # Get subsection sentences if available
                    section_idx = doc_structure.sections.index(section)
                    subsection_idx = section.subsections.index(subsection)
                    subsection_id = f"section_{section_idx}_subsection_{subsection_idx}"
                    subsection_sentences = document_sentences.get(subsection_id) if document_sentences else None

                    subsection_relations = self.extract_relations_from_text(
                        subsection.text, subsection.entities, sentences=subsection_sentences
                    )
                    subsection.relations = subsection_relations
                    all_relations.extend(subsection_relations)

        # Update document relations
        doc_structure.relations = all_relations

        # Build knowledge graph
        if all_relations and self.nx:
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

    def build_heterogeneous_graph(
        self,
        doc_structure: DocumentStructure
    ) -> Tuple[Dict[str, torch.Tensor], Dict[Tuple[str, str, str], torch.Tensor]]:
        """
        Build a heterogeneous graph from document structure for HGT processing.

        Args:
            doc_structure: Document structure with entities

        Returns:
            Tuple of (node_features, edge_indices)
        """
        if not self.torch_geometric_available or self.model is None:
            logger.error("PyTorch Geometric or model not available")
            return {}, {}

        try:
            import torch_geometric
            from torch_geometric.data import HeteroData

            # Create HeteroData object
            data = HeteroData()

            # Add entity nodes
            entity_features = []
            entity_map = {}  # Map from entity to index

            for i, entity in enumerate(doc_structure.entities):
                # Encode entity text
                entity_encoding = self.model.encode_text(entity.text)
                entity_features.append(entity_encoding)
                entity_map[entity.text] = i

            if entity_features:
                data["entity"].x = torch.cat(entity_features, dim=0)

            # Add mention nodes (one per entity occurrence)
            mention_features = []
            mention_to_entity = []  # (mention_idx, entity_idx) edges
            mention_map = {}  # Map from (entity, start, end) to mention index

            mention_idx = 0
            for section in doc_structure.sections:
                for entity in section.entities:
                    # Encode mention with context
                    context_start = max(0, entity.start - 50)
                    context_end = min(len(section.text), entity.end + 50)
                    context = section.text[context_start:context_end]

                    mention_encoding = self.model.encode_text(context)
                    mention_features.append(mention_encoding)

                    # Add edge to entity
                    if entity.text in entity_map:
                        mention_to_entity.append((mention_idx, entity_map[entity.text]))

                    # Store mention mapping
                    mention_map[(entity.text, entity.start, entity.end)] = mention_idx
                    mention_idx += 1

            if mention_features:
                data["mention"].x = torch.cat(mention_features, dim=0)

                # Add entity-mention edges
                if mention_to_entity:
                    src, dst = zip(*mention_to_entity)
                    data["entity", "has_mention", "mention"].edge_index = torch.tensor(
                        [src, dst], dtype=torch.long
                    )

            # TODO: Add sentence nodes and section nodes
            # This would require sentence segmentation which is not implemented here

            # Add co-occurrence edges between mentions
            cooccurrence_edges = []

            # Consider mentions co-occurring if they are within 100 characters
            for (e1, s1, e1_end), m1_idx in mention_map.items():
                for (e2, s2, e2_end), m2_idx in mention_map.items():
                    if m1_idx != m2_idx and abs(s1 - s2) < 100:
                        cooccurrence_edges.append((m1_idx, m2_idx))

            if cooccurrence_edges:
                src, dst = zip(*cooccurrence_edges)
                data["mention", "co_occurs", "mention"].edge_index = torch.tensor(
                    [src, dst], dtype=torch.long
                )

            # Convert to dictionaries for HGT
            x_dict = {}
            edge_index_dict = {}

            for node_type, node_data in data.node_items():
                x_dict[node_type] = node_data.x

            for edge_type, edge_data in data.edge_items():
                edge_index_dict[edge_type] = edge_data.edge_index

            return x_dict, edge_index_dict

        except Exception as e:
            logger.error(f"Error building heterogeneous graph: {str(e)}")
            return {}, {}
