Bias assessment service for the Medical Research Synthesizer.

This module provides a service for assessing risk of bias in medical studies.

import logging
import re
import asyncio
import spacy
from enum import Enum
from typing import Dict, List, Any, Optional
from asf.medical.core.exceptions import MLError


logger = logging.getLogger(__name__)

class BiasRisk(str, Enum):
    Risk of bias levels.
        try:
            self.nlp = nlp_model or spacy.load("en_core_sci_md")
            logger.info("Loaded spaCy model for bias assessment")
        except Exception as e:
            logger.warning(f"Could not load spaCy model: {str(e)}. Falling back to basic pattern matching.")
            self.nlp = None

        self.bias_patterns = self._load_patterns()
        logger.info(f"Loaded {sum(len(patterns) for patterns in self.bias_patterns.values())} bias assessment patterns")

    def _load_patterns(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Load patterns for bias assessment.

        Returns:
            Dictionary of patterns for each bias domain
        """
        return {
            BiasDomain.RANDOMIZATION: [
                {"pattern": r"random(ly|ized|isation)", "type": "positive", "weight": 1.0},
                {"pattern": r"random number (table|generator)", "type": "positive", "weight": 1.0},
                {"pattern": r"computer(-|\s)generated randomization", "type": "positive", "weight": 1.0},
                {"pattern": r"not random", "type": "negative", "weight": 1.0},
                {"pattern": r"quasi(-|\s)random", "type": "negative", "weight": 0.8},
                {"pattern": r"alternate allocation", "type": "negative", "weight": 1.0},
                {"pattern": r"assigned based on", "type": "negative", "weight": 0.7}
            ],
            BiasDomain.BLINDING: [
                {"pattern": r"double(-|\s)blind", "type": "positive", "weight": 1.0},
                {"pattern": r"triple(-|\s)blind", "type": "positive", "weight": 1.0},
                {"pattern": r"single(-|\s)blind", "type": "positive", "weight": 0.7},
                {"pattern": r"blind(ed|ing)", "type": "positive", "weight": 0.8},
                {"pattern": r"not blind", "type": "negative", "weight": 1.0},
                {"pattern": r"open(-|\s)label", "type": "negative", "weight": 0.9},
                {"pattern": r"unblind", "type": "negative", "weight": 1.0}
            ],
            BiasDomain.ALLOCATION_CONCEALMENT: [
                {"pattern": r"concealed allocation", "type": "positive", "weight": 1.0},
                {"pattern": r"sealed envelope", "type": "positive", "weight": 0.9},
                {"pattern": r"central(ized)? allocation", "type": "positive", "weight": 1.0},
                {"pattern": r"pharmacy-controlled", "type": "positive", "weight": 0.9},
                {"pattern": r"allocation was not concealed", "type": "negative", "weight": 1.0},
                {"pattern": r"open allocation", "type": "negative", "weight": 1.0}
            ],
            BiasDomain.SAMPLE_SIZE: [
                {"pattern": r"power analysis", "type": "positive", "weight": 1.0},
                {"pattern": r"sample size calculation", "type": "positive", "weight": 1.0},
                {"pattern": r"adequately powered", "type": "positive", "weight": 0.9},
                {"pattern": r"underpowered", "type": "negative", "weight": 1.0},
                {"pattern": r"small sample size", "type": "negative", "weight": 0.8},
                {"pattern": r"pilot study", "type": "negative", "weight": 0.6}
            ],
            BiasDomain.ATTRITION: [
                {"pattern": r"intention(-|\s)to(-|\s)treat", "type": "positive", "weight": 1.0},
                {"pattern": r"no loss to follow(-|\s)up", "type": "positive", "weight": 1.0},
                {"pattern": r"complete follow(-|\s)up", "type": "positive", "weight": 0.9},
                {"pattern": r"high (dropout|attrition|withdrawal)", "type": "negative", "weight": 1.0},
                {"pattern": r"significant loss to follow(-|\s)up", "type": "negative", "weight": 1.0},
                {"pattern": r"per(-|\s)protocol analysis", "type": "negative", "weight": 0.7}
            ],
            BiasDomain.SELECTIVE_REPORTING: [
                {"pattern": r"pre(-|\s)registered", "type": "positive", "weight": 1.0},
                {"pattern": r"published protocol", "type": "positive", "weight": 1.0},
                {"pattern": r"all outcomes reported", "type": "positive", "weight": 0.9},
                {"pattern": r"selective (outcome|reporting)", "type": "negative", "weight": 1.0},
                {"pattern": r"not all outcomes reported", "type": "negative", "weight": 1.0},
                {"pattern": r"post(-|\s)hoc analysis", "type": "negative", "weight": 0.7}
            ]
        }

    async def assess_study(self, study_text: str) -> Dict[str, Any]:
        logger.info("Assessing risk of bias in study")

        if self.nlp:
            doc = await asyncio.to_thread(self.nlp, study_text)
            text_for_analysis = study_text
        else:
            doc = None
            text_for_analysis = study_text.lower()

        assessment = {
            BiasDomain.RANDOMIZATION: await self._assess_randomization(doc, text_for_analysis),
            BiasDomain.BLINDING: await self._assess_blinding(doc, text_for_analysis),
            BiasDomain.ALLOCATION_CONCEALMENT: await self._assess_allocation(doc, text_for_analysis),
            BiasDomain.SAMPLE_SIZE: await self._assess_sample_size(doc, text_for_analysis),
            BiasDomain.ATTRITION: await self._assess_attrition(doc, text_for_analysis),
            BiasDomain.SELECTIVE_REPORTING: await self._assess_selective_reporting(doc, text_for_analysis),
            BiasDomain.OVERALL: None  # Will be calculated
        }

        high_risk_count = sum(1 for domain, result in assessment.items()
                             if domain != BiasDomain.OVERALL and result["risk"] == BiasRisk.HIGH)

        unclear_risk_count = sum(1 for domain, result in assessment.items()
                               if domain != BiasDomain.OVERALL and result["risk"] == BiasRisk.UNCLEAR)

        if high_risk_count == 0 and unclear_risk_count <= 1:
            overall_risk = BiasRisk.LOW
        elif high_risk_count <= 1 and unclear_risk_count <= 2:
            overall_risk = BiasRisk.MODERATE
        else:
            overall_risk = BiasRisk.HIGH

        assessment[BiasDomain.OVERALL] = {
            "risk": overall_risk,
            "summary": f"{high_risk_count} domains at high risk of bias, {unclear_risk_count} domains unclear",
            "high_risk_domains": [domain for domain, result in assessment.items()
                                if domain != BiasDomain.OVERALL and result["risk"] == BiasRisk.HIGH],
            "unclear_domains": [domain for domain, result in assessment.items()
                              if domain != BiasDomain.OVERALL and result["risk"] == BiasRisk.UNCLEAR]
        }

        logger.info(f"Bias assessment completed: Overall risk is {overall_risk}")
        return assessment

    async def assess_studies(self, studies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        logger.info(f"Assessing risk of bias in {len(studies)} studies")

        results = []
        for study in studies:
            study_text = ""
            if "full_text" in study and study["full_text"]:
                study_text = study["full_text"]
            elif "abstract" in study and study["abstract"]:
                study_text = f"{study.get('title', '')} {study['abstract']}"
            elif "title" in study:
                study_text = study["title"]

            if not study_text:
                logger.warning(f"No text available for study {study.get('pmid', 'unknown')}")
                continue

            assessment = await self.assess_study(study_text)

            result = {
                "study_id": study.get("pmid", ""),
                "title": study.get("title", ""),
                "assessment": assessment
            }

            results.append(result)

        return results

    async def _assess_randomization(self, doc: Optional[Any], text: str) -> Dict[str, Any]:
        return await self._assess_domain(BiasDomain.RANDOMIZATION, doc, text)

    async def _assess_blinding(self, doc: Optional[Any], text: str) -> Dict[str, Any]:
        return await self._assess_domain(BiasDomain.BLINDING, doc, text)

    async def _assess_allocation(self, doc: Optional[Any], text: str) -> Dict[str, Any]:
        return await self._assess_domain(BiasDomain.ALLOCATION_CONCEALMENT, doc, text)

    async def _assess_sample_size(self, doc: Optional[Any], text: str) -> Dict[str, Any]:
        return await self._assess_domain(BiasDomain.SAMPLE_SIZE, doc, text)

    async def _assess_attrition(self, doc: Optional[Any], text: str) -> Dict[str, Any]:
        return await self._assess_domain(BiasDomain.ATTRITION, doc, text)

    async def _assess_selective_reporting(self, doc: Optional[Any], text: str) -> Dict[str, Any]:
        return await self._assess_domain(BiasDomain.SELECTIVE_REPORTING, doc, text)

    async def _assess_domain(self, domain: BiasDomain, doc: Optional[Any], text: str) -> Dict[str, Any]:
        positive_score = 0.0
        negative_score = 0.0
        evidence = []

        for pattern in self.bias_patterns.get(domain, []):
            matches = re.finditer(pattern["pattern"], text, re.IGNORECASE)
            for match in matches:
                match_text = match.group(0)
                context = self._get_context(text, match.start(), match.end())

                if pattern["type"] == "positive":
                    positive_score += pattern["weight"]
                    evidence.append({
                        "text": match_text,
                        "context": context,
                        "type": "positive",
                        "weight": pattern["weight"]
                    })
                else:
                    negative_score += pattern["weight"]
                    evidence.append({
                        "text": match_text,
                        "context": context,
                        "type": "negative",
                        "weight": pattern["weight"]
                    })

        if positive_score > 0 and negative_score == 0:
            risk = BiasRisk.LOW
        elif negative_score > 0:
            risk = BiasRisk.HIGH
        elif positive_score > 0:
            risk = BiasRisk.MODERATE
        else:
            risk = BiasRisk.UNCLEAR

        result = {
            "risk": risk,
            "positive_score": positive_score,
            "negative_score": negative_score,
            "evidence": evidence
        }

        return result

    def _get_context(self, text: str, start: int, end: int, context_size: int = 50) -> str:
        """
        Get context around a match.

        Args:
            text: Full text
            start: Start index of match
            end: End index of match
            context_size: Number of characters to include before and after match

        Returns:
            Context string
        """
        context_start = max(0, start - context_size)
        context_end = min(len(text), end + context_size)

        left_boundary = text.rfind(".", context_start, start)
        if left_boundary != -1:
            context_start = left_boundary + 1

        right_boundary = text.find(".", end, context_end)
        if right_boundary != -1:
            context_end = right_boundary + 1

        return text[context_start:context_end].strip()

    def generate_summary(self, assessment: Dict[str, Any]) -> str:
        """
        Generate a human-readable summary of the bias assessment.

        Args:
            assessment: Bias assessment result

        Returns:
            Summary string
        """
        summary = []
        summary.append(f"Overall risk of bias: {assessment[BiasDomain.OVERALL]['risk']}")
        summary.append(assessment[BiasDomain.OVERALL]['summary'])
        summary.append("")

        for domain in [d for d in BiasDomain if d != BiasDomain.OVERALL]:
            domain_result = assessment[domain]
            summary.append(f"{domain.value.replace('_', ' ').title()}: {domain_result['risk']}")

            if domain_result["evidence"]:
                for evidence in domain_result["evidence"]:
                    summary.append(f"  - {evidence['text']} ({evidence['type']})")
                    summary.append(f"    Context: \"{evidence['context']}\"")
            else:
                summary.append("  No specific evidence found")

            summary.append("")

        return "\n".join(summary)
