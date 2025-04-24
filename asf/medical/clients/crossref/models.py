"""
Data models for CrossRef API responses.
This module defines Python classes that represent the structure of CrossRef API responses.
"""
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime


@dataclass
class DateParts:
    """Date representation in CrossRef data."""
    year: Optional[int] = None
    month: Optional[int] = None
    day: Optional[int] = None


@dataclass
class CrossRefDate:
    """Date representation in CrossRef data."""
    date_parts: List[List[int]]
    date_time: Optional[str] = None
    timestamp: Optional[int] = None


@dataclass
class Author:
    """Author representation in CrossRef data."""
    family: Optional[str] = None
    given: Optional[str] = None
    ORCID: Optional[str] = None
    authenticated_orcid: Optional[bool] = None
    affiliation: Optional[List[Dict[str, str]]] = None
    sequence: Optional[str] = None
    
    @property
    def full_name(self) -> str:
        """Return the author's full name."""
        if self.given and self.family:
            return f"{self.given} {self.family}"
        elif self.family:
            return self.family
        elif self.given:
            return self.given
        else:
            return "Unknown Author"


@dataclass
class Link:
    """Link representation in CrossRef data."""
    URL: str
    content_type: Optional[str] = None
    content_version: Optional[str] = None
    intended_application: Optional[str] = None


@dataclass
class Funder:
    """Funder representation in CrossRef data."""
    name: str
    DOI: Optional[str] = None
    award: Optional[List[str]] = None
    doi_asserted_by: Optional[str] = None


@dataclass
class License:
    """License information in CrossRef data."""
    URL: str
    start: Optional[CrossRefDate] = None
    delay_in_days: Optional[int] = None
    content_version: Optional[str] = None


@dataclass
class Reference:
    """Reference representation in CrossRef data."""
    key: str
    doi: Optional[str] = None
    doi_asserted_by: Optional[str] = None
    issue: Optional[str] = None
    first_page: Optional[str] = None
    volume: Optional[str] = None
    journal_title: Optional[str] = None
    article_title: Optional[str] = None
    series_title: Optional[str] = None
    volume_title: Optional[str] = None
    author: Optional[str] = None
    year: Optional[str] = None
    unstructured: Optional[str] = None
    standard_designator: Optional[str] = None


@dataclass
class WorkMessage:
    """Main work message from CrossRef data."""
    DOI: str
    URL: str
    type: str
    created: CrossRefDate
    indexed: CrossRefDate
    issued: CrossRefDate
    title: List[str]
    container_title: Optional[List[str]] = None
    publisher: Optional[str] = None
    language: Optional[str] = None
    ISSN: Optional[List[str]] = None
    ISBN: Optional[List[str]] = None
    subject: Optional[List[str]] = None
    author: Optional[List[Author]] = None
    editor: Optional[List[Author]] = None
    reference: Optional[List[Reference]] = None
    is_referenced_by_count: Optional[int] = None
    references_count: Optional[int] = None
    funder: Optional[List[Funder]] = None
    link: Optional[List[Link]] = None
    abstract: Optional[str] = None
    journal_issue: Optional[Dict[str, Any]] = None
    volume: Optional[str] = None
    issue: Optional[str] = None
    page: Optional[str] = None
    license: Optional[List[License]] = None
    member: Optional[str] = None
    score: Optional[float] = None


@dataclass
class Work:
    """Work object returned by CrossRef API."""
    status: str
    message_type: str
    message_version: str
    message: WorkMessage


@dataclass
class JournalMessage:
    """Journal message from CrossRef data."""
    title: str
    publisher: str
    issn: List[str]
    last_status_check_time: Optional[int] = None
    coverage_type: Optional[str] = None


@dataclass
class Journal:
    """Journal object returned by CrossRef API."""
    status: str
    message_type: str
    message_version: str
    message: JournalMessage


@dataclass
class MemberMessage:
    """Member (publisher) message from CrossRef data."""
    id: str
    primary_name: str
    location: str
    names: List[str]
    prefixes: List[str]
    counts: Optional[Dict[str, int]] = None
    coverage: Optional[Dict[str, Dict[str, Any]]] = None


@dataclass
class Member:
    """Member (publisher) object returned by CrossRef API."""
    status: str
    message_type: str
    message_version: str
    message: MemberMessage


@dataclass
class FunderMessage:
    """Funder message from CrossRef data."""
    id: str
    name: str
    location: Optional[str] = None
    work_count: Optional[int] = None
    descendant_work_count: Optional[int] = None
    uri: Optional[str] = None
    tokens: Optional[List[str]] = None
    alt_names: Optional[List[str]] = None
    descendants: Optional[List[str]] = None
    hierarchy_names: Optional[Dict[str, str]] = None


@dataclass
class Funder:
    """Funder object returned by CrossRef API."""
    status: str
    message_type: str
    message_version: str
    message: FunderMessage


@dataclass
class TypeMessage:
    """Type message from CrossRef data."""
    id: str
    label: str
    works: Optional[int] = None


@dataclass
class Type:
    """Type object returned by CrossRef API."""
    status: str
    message_type: str
    message_version: str
    message: TypeMessage


@dataclass
class PrefixMessage:
    """Prefix message from CrossRef data."""
    member: str
    name: str
    prefix: str


@dataclass
class Prefix:
    """Prefix object returned by CrossRef API."""
    status: str
    message_type: str
    message_version: str
    message: PrefixMessage