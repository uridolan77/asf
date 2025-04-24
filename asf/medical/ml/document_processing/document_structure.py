"""
Document Structure Classes

This module defines the data structures for representing scientific documents,
including sections, entities, relations, and document metadata.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Set, Union

@dataclass
class Entity:
    """Biomedical entity with UMLS linking."""
    text: str
    label: str
    start: int
    end: int
    cui: Optional[str] = None  # UMLS Concept Unique Identifier
    umls_entity: Optional[Dict] = None
    confidence: float = 1.0
    abbreviation: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "text": self.text,
            "label": self.label,
            "start": self.start,
            "end": self.end,
            "cui": self.cui,
            "confidence": self.confidence,
            "abbreviation": self.abbreviation,
            "umls_entity": self.umls_entity
        }

@dataclass
class SectionInfo:
    """Information about a section in a scientific document."""
    section_type: str
    heading: str
    text: str
    start_pos: int
    end_pos: int
    subsections: List['SectionInfo'] = field(default_factory=list)
    entities: List[Entity] = field(default_factory=list)
    relations: List[Dict] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DocumentStructure:
    """Structured representation of a scientific document."""
    title: str
    abstract: Optional[str] = None
    sections: List[SectionInfo] = field(default_factory=list)
    references: List[Dict[str, Any]] = field(default_factory=list)
    entities: List[Entity] = field(default_factory=list)
    relations: List[Dict] = field(default_factory=list)
    knowledge_graph: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    summary: Optional[Dict[str, str]] = None

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
    head_cui: Optional[str] = None
    tail_cui: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "head_entity": self.head_entity,
            "tail_entity": self.tail_entity,
            "head_type": self.head_type,
            "tail_type": self.tail_type,
            "relation_type": self.relation_type,
            "confidence": self.confidence,
            "context": self.context,
            "sentence_id": self.sentence_id,
            "head_cui": self.head_cui,
            "tail_cui": self.tail_cui
        }

@dataclass
class ResearchSummary:
    """Structured summary of a research paper."""
    abstract: str
    key_findings: List[str]
    clinical_implications: Optional[List[str]] = None
    limitations: Optional[List[str]] = None
    future_directions: Optional[List[str]] = None
    
    def to_markdown(self) -> str:
        """Convert to markdown format."""
        md = f"## Research Summary\n\n"
        md += f"### Abstract\n{self.abstract}\n\n"
        
        md += f"### Key Findings\n"
        for i, finding in enumerate(self.key_findings, 1):
            md += f"{i}. {finding}\n"
        md += "\n"
        
        if self.clinical_implications:
            md += f"### Clinical Implications\n"
            for i, implication in enumerate(self.clinical_implications, 1):
                md += f"{i}. {implication}\n"
            md += "\n"
            
        if self.limitations:
            md += f"### Limitations\n"
            for i, limitation in enumerate(self.limitations, 1):
                md += f"{i}. {limitation}\n"
            md += "\n"
            
        if self.future_directions:
            md += f"### Future Directions\n"
            for i, direction in enumerate(self.future_directions, 1):
                md += f"{i}. {direction}\n"
            md += "\n"
            
        return md
