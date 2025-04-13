"""Evidence Extraction Modules

This module provides specialized DSPy modules for extracting evidence from medical text.
"""

import logging
from typing import Dict, Any, List, Optional, Union, Tuple

import dspy

from ..dspy_module_base import MedicalDSPyModule
from ..dspy_signatures import EvidenceExtraction

# Set up logging
logger = logging.getLogger(__name__)


class EvidenceExtractionModule(MedicalDSPyModule):
    """Module for extracting evidence from medical text to support or refute claims.

    This module identifies relevant evidence in medical text and determines
    whether it supports, refutes, or is neutral to a given claim.
    """

    def __init__(
        self,
        model: Optional[dspy.Module] = None,
        **kwargs
    ):
        """
        Initialize the evidence extraction module.

        Args:
            model: Custom evidence extraction model (optional)
            **kwargs: Additional arguments for the parent class
        """
        super().__init__(**kwargs)

        # Create default model if not provided
        self.model = model or dspy.ChainOfThought(EvidenceExtraction)

    def forward(
        self,
        text: str,
        claim: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Extract evidence from text to support or refute a claim.

        Args:
            text: Medical text to analyze
            claim: The claim to find evidence for
            **kwargs: Additional arguments

        Returns:
            Dict[str, Any]: Evidence extraction result
        """
        # Log audit
        self.log_audit(
            "EVIDENCE_EXTRACTION_FORWARD",
            {"text": text, "claim": claim, **kwargs},
            {}
        )

        # Sanitize inputs
        sanitized_text = self.sanitize_input(text)
        sanitized_claim = self.sanitize_input(claim)

        # Extract evidence
        try:
            extraction_result = self.model(
                text=sanitized_text,
                claim=sanitized_claim
            )

            # Extract fields
            if hasattr(extraction_result, '__dict__'):
                evidence = getattr(extraction_result, 'evidence', "No evidence found")
                relation = getattr(extraction_result, 'relation', "neutral")
                confidence = getattr(extraction_result, 'confidence', 0.0)
            elif isinstance(extraction_result, dict):
                evidence = extraction_result.get('evidence', "No evidence found")
                relation = extraction_result.get('relation', "neutral")
                confidence = extraction_result.get('confidence', 0.0)
            else:
                evidence = "No evidence found"
                relation = "neutral"
                confidence = 0.0

        except Exception as e:
            logger.error(f"Evidence extraction failed: {str(e)}")
            evidence = f"Error in evidence extraction: {str(e)}"
            relation = "error"
            confidence = 0.0

        # Sanitize outputs
        sanitized_evidence = self.sanitize_output(evidence)

        # Prepare result
        result = {
            'text': text,
            'claim': claim,
            'evidence': sanitized_evidence,
            'relation': relation,
            'confidence': confidence
        }

        # Log audit
        self.log_audit(
            "EVIDENCE_EXTRACTION_FORWARD",
            {"text": text, "claim": claim, **kwargs},
            result
        )

        return result


class MultiEvidenceExtractionModule(MedicalDSPyModule):
    """Module for extracting multiple pieces of evidence from medical text.

    This module identifies multiple relevant pieces of evidence in medical text
    and determines whether each supports, refutes, or is neutral to a given claim.
    """

    def __init__(
        self,
        base_extractor: Optional[dspy.Module] = None,
        evidence_finder: Optional[dspy.Module] = None,
        max_evidence_pieces: int = 5,
        **kwargs
    ):
        """
        Initialize the multi-evidence extraction module.

        Args:
            base_extractor: Base evidence extraction model (optional)
            evidence_finder: Model to identify evidence locations (optional)
            max_evidence_pieces: Maximum number of evidence pieces to extract
            **kwargs: Additional arguments for the parent class
        """
        super().__init__(**kwargs)

        # Create default models if not provided
        self.base_extractor = base_extractor or dspy.ChainOfThought(EvidenceExtraction)

        # Create evidence finder signature
        EvidenceFinderSignature = dspy.Signature(
            text=dspy.InputField(desc="Medical text to analyze"),
            claim=dspy.InputField(desc="The claim to find evidence for"),
            evidence_locations=dspy.OutputField(desc="List of locations in the text that contain relevant evidence")
        )

        self.evidence_finder = evidence_finder or dspy.ChainOfThought(EvidenceFinderSignature)
        self.max_evidence_pieces = max_evidence_pieces

    def forward(
        self,
        text: str,
        claim: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Extract multiple pieces of evidence from text.

        Args:
            text: Medical text to analyze
            claim: The claim to find evidence for
            **kwargs: Additional arguments

        Returns:
            Dict[str, Any]: Multi-evidence extraction result
        """
        # Log audit
        self.log_audit(
            "MULTI_EVIDENCE_EXTRACTION_FORWARD",
            {"text": text, "claim": claim, **kwargs},
            {}
        )

        # Sanitize inputs
        sanitized_text = self.sanitize_input(text)
        sanitized_claim = self.sanitize_input(claim)

        # Step 1: Find evidence locations
        try:
            finder_result = self.evidence_finder(
                text=sanitized_text,
                claim=sanitized_claim
            )

            if hasattr(finder_result, 'evidence_locations'):
                evidence_locations = finder_result.evidence_locations
            elif isinstance(finder_result, dict) and 'evidence_locations' in finder_result:
                evidence_locations = finder_result['evidence_locations']
            else:
                evidence_locations = []

            # Convert to list if it's a string
            if isinstance(evidence_locations, str):
                try:
                    # Try to parse as a list if it looks like one
                    if evidence_locations.strip().startswith('[') and evidence_locations.strip().endswith(']'):
                        import json
                        evidence_locations = json.loads(evidence_locations)
                    else:
                        # Split by newlines or commas if it's a simple string
                        evidence_locations = [loc.strip() for loc in evidence_locations.replace('\n', ',').split(',') if loc.strip()]
                except Exception:
                    evidence_locations = [evidence_locations]

        except Exception as e:
            logger.error(f"Evidence location finding failed: {str(e)}")
            evidence_locations = []

        # Limit the number of evidence pieces
        evidence_locations = evidence_locations[:self.max_evidence_pieces]

        # Step 2: Extract evidence for each location
        evidence_results = []

        if not evidence_locations:
            # If no locations found, try once with the full text
            try:
                extraction_result = self.base_extractor(
                    text=sanitized_text,
                    claim=sanitized_claim
                )

                if hasattr(extraction_result, '__dict__'):
                    evidence = getattr(extraction_result, 'evidence', "No evidence found")
                    relation = getattr(extraction_result, 'relation', "neutral")
                    confidence = getattr(extraction_result, 'confidence', 0.0)
                elif isinstance(extraction_result, dict):
                    evidence = extraction_result.get('evidence', "No evidence found")
                    relation = extraction_result.get('relation', "neutral")
                    confidence = extraction_result.get('confidence', 0.0)
                else:
                    evidence = "No evidence found"
                    relation = "neutral"
                    confidence = 0.0

                evidence_results.append({
                    'evidence': self.sanitize_output(evidence),
                    'relation': relation,
                    'confidence': confidence,
                    'location': "full_text"
                })
            except Exception as e:
                logger.error(f"Full text evidence extraction failed: {str(e)}")
                evidence_results.append({
                    'evidence': f"Error in evidence extraction: {str(e)}",
                    'relation': "error",
                    'confidence': 0.0,
                    'location': "full_text"
                })
        else:
            # Process each evidence location
            for i, location in enumerate(evidence_locations):
                # Extract the relevant text segment if location is specific
                if isinstance(location, dict) and 'start' in location and 'end' in location:
                    start = max(0, int(location['start']))
                    end = min(len(sanitized_text), int(location['end']))
                    segment = sanitized_text[start:end]
                elif isinstance(location, str) and location.strip():
                    # If location is a text segment, use it directly
                    segment = location
                else:
                    # Use the full text as fallback
                    segment = sanitized_text

                try:
                    extraction_result = self.base_extractor(
                        text=segment,
                        claim=sanitized_claim
                    )

                    if hasattr(extraction_result, '__dict__'):
                        evidence = getattr(extraction_result, 'evidence', "No evidence found")
                        relation = getattr(extraction_result, 'relation', "neutral")
                        confidence = getattr(extraction_result, 'confidence', 0.0)
                    elif isinstance(extraction_result, dict):
                        evidence = extraction_result.get('evidence', "No evidence found")
                        relation = extraction_result.get('relation', "neutral")
                        confidence = extraction_result.get('confidence', 0.0)
                    else:
                        evidence = "No evidence found"
                        relation = "neutral"
                        confidence = 0.0

                    evidence_results.append({
                        'evidence': self.sanitize_output(evidence),
                        'relation': relation,
                        'confidence': confidence,
                        'location': f"segment_{i+1}"
                    })
                except Exception as e:
                    logger.error(f"Evidence extraction for segment {i+1} failed: {str(e)}")
                    evidence_results.append({
                        'evidence': f"Error in evidence extraction: {str(e)}",
                        'relation': "error",
                        'confidence': 0.0,
                        'location': f"segment_{i+1}"
                    })

        # Step 3: Aggregate results
        overall_relation = self._determine_overall_relation(evidence_results)
        overall_confidence = self._calculate_overall_confidence(evidence_results)

        # Prepare result
        result = {
            'text': text,
            'claim': claim,
            'evidence_pieces': evidence_results,
            'overall_relation': overall_relation,
            'overall_confidence': overall_confidence,
            'evidence_count': len(evidence_results)
        }

        # Log audit
        self.log_audit(
            "MULTI_EVIDENCE_EXTRACTION_FORWARD",
            {"text": text, "claim": claim, **kwargs},
            result
        )

        return result

    def _determine_overall_relation(self, evidence_results: List[Dict[str, Any]]) -> str:
        """
        Determine the overall relation based on multiple evidence pieces.

        Args:
            evidence_results: List of evidence results

        Returns:
            str: Overall relation (supports, refutes, neutral, mixed)
        """
        if not evidence_results:
            return "neutral"

        # Count relations
        supports = 0
        refutes = 0
        neutral = 0

        for result in evidence_results:
            relation = result.get('relation', '').lower()
            confidence = result.get('confidence', 0.0)

            # Only count if confidence is reasonable
            if confidence >= 0.3:
                if relation in ['supports', 'support', 'supporting']:
                    supports += 1
                elif relation in ['refutes', 'refute', 'refuting', 'contradicts', 'contradict']:
                    refutes += 1
                else:
                    neutral += 1

        # Determine overall relation
        if supports > 0 and refutes > 0:
            return "mixed"
        elif supports > 0:
            return "supports"
        elif refutes > 0:
            return "refutes"
        else:
            return "neutral"

    def _calculate_overall_confidence(self, evidence_results: List[Dict[str, Any]]) -> float:
        """
        Calculate overall confidence based on multiple evidence pieces.

        Args:
            evidence_results: List of evidence results

        Returns:
            float: Overall confidence score
        """
        if not evidence_results:
            return 0.0

        # Get confidence scores
        confidences = [result.get('confidence', 0.0) for result in evidence_results]

        # Calculate weighted average based on non-zero confidences
        non_zero_confidences = [c for c in confidences if c > 0]
        if non_zero_confidences:
            return sum(non_zero_confidences) / len(non_zero_confidences)
        else:
            return 0.0
