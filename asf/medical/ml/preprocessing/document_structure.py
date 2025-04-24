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
        """Convert to dictionary representation."""
        return {
            "text": self.text,
            "label": self.label,
            "start": self.start,
            "end": self.end,
            "cui": self.cui,
            "confidence": self.confidence,
            "abbreviation": self.abbreviation,
            "umls_entity": str(self.umls_entity) if self.umls_entity else None
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
        """Convert to dictionary representation."""
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
    """Research paper summary."""
    abstract: str
    key_findings: str
    methods: Optional[str] = None
    conclusions: Optional[str] = None
    limitations: Optional[str] = None
    clinical_implications: Optional[str] = None
    
    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary representation."""
        result = {
            "abstract": self.abstract,
            "key_findings": self.key_findings
        }
        
        if self.methods:
            result["methods"] = self.methods
        if self.conclusions:
            result["conclusions"] = self.conclusions
        if self.limitations:
            result["limitations"] = self.limitations
        if self.clinical_implications:
            result["clinical_implications"] = self.clinical_implications
            
        return result
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        import json
        return json.dumps(self.to_dict(), indent=2)
    
    def to_markdown(self) -> str:
        """Convert to markdown format."""
        md = f"# Research Summary\n\n"
        
        md += f"## Abstract\n{self.abstract}\n\n"
        md += f"## Key Findings\n{self.key_findings}\n\n"
        
        if self.methods:
            md += f"## Methods\n{self.methods}\n\n"
        if self.conclusions:
            md += f"## Conclusions\n{self.conclusions}\n\n"
        if self.limitations:
            md += f"## Limitations\n{self.limitations}\n\n"
        if self.clinical_implications:
            md += f"## Clinical Implications\n{self.clinical_implications}\n\n"
            
        return md