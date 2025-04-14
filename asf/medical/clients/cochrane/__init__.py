"""
Cochrane Library client for the Medical Research Synthesizer.

This package provides a client for accessing the Cochrane Library,
which contains high-quality systematic reviews, meta-analyses, and
clinical trial data.
"""

from asf.medical.clients.cochrane.cochrane_client import (
    CochraneClient,
    CochraneClientError,
    PICOElement,
    EvidenceGrade
)

__all__ = [
    'CochraneClient',
    'CochraneClientError',
    'PICOElement',
    'EvidenceGrade'
]