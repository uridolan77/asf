PRISMA-guided screening service for the Medical Research Synthesizer.

This module provides a service for screening medical literature according to
the PRISMA (Preferred Reporting Items for Systematic Reviews and Meta-Analyses) guidelines.
import logging
from enum import Enum
from typing import Dict, List, Any, Optional
from asf.medical.core.exceptions import OperationError, ValidationError
logger = logging.getLogger(__name__)
class ScreeningStage(str, Enum):
    Screening stages according to PRISMA guidelines.
        self.biomedlm_service = biomedlm_service
        self.criteria = {
            ScreeningStage.IDENTIFICATION: {
                "include": [],
                "exclude": []
            },
            ScreeningStage.SCREENING: {
                "include": [],
                "exclude": []
            },
            ScreeningStage.ELIGIBILITY: {
                "include": [],
                "exclude": []
            }
        }
        self.flow_data = {
            "identification": {
                "records_identified": 0,
                "records_removed_before_screening": 0
            },
            "screening": {
                "records_screened": 0,
                "records_excluded": 0
            },
            "eligibility": {
                "full_text_assessed": 0,
                "full_text_excluded": 0,
                "exclusion_reasons": {}
            },
            "included": {
                "studies_included": 0
            }
        }
    def set_criteria(self, stage: ScreeningStage, include_criteria: List[str], exclude_criteria: List[str]):
        """
        Set screening criteria for a specific stage.
        Args:
            stage: Screening stage
            include_criteria: List of inclusion criteria
            exclude_criteria: List of exclusion criteria
        """
        self.criteria[stage] = {
            "include": include_criteria,
            "exclude": exclude_criteria
        }
        logger.info(f"Set {len(include_criteria)} inclusion and {len(exclude_criteria)} exclusion criteria for {stage} stage")
    async def screen_article(
        self,
        article: Dict[str, Any],
        stage: ScreeningStage,
        custom_criteria: Optional[Dict[str, List[str]]] = None
    ) -> Dict[str, Any]:
        logger.info(f"Screening article {article.get('pmid', 'unknown')} at {stage} stage")
        criteria = custom_criteria if custom_criteria else self.criteria[stage]
        result = {
            "article_id": article.get("pmid", ""),
            "title": article.get("title", ""),
            "stage": stage,
            "decision": ScreeningDecision.UNCERTAIN,
            "confidence": 0.0,
            "matched_include_criteria": [],
            "matched_exclude_criteria": [],
            "notes": ""
        }
        if stage == ScreeningStage.IDENTIFICATION:
            text = article.get("title", "")
        elif stage == ScreeningStage.SCREENING:
            text = f"{article.get('title', '')} {article.get('abstract', '')}"
        elif stage == ScreeningStage.ELIGIBILITY:
            text = article.get("full_text", f"{article.get('title', '')} {article.get('abstract', '')}")
        else:
            text = f"{article.get('title', '')} {article.get('abstract', '')}"
        for criterion in criteria["include"]:
            if await self._check_criterion(text, criterion):
                result["matched_include_criteria"].append(criterion)
        for criterion in criteria["exclude"]:
            if await self._check_criterion(text, criterion):
                result["matched_exclude_criteria"].append(criterion)
        if result["matched_exclude_criteria"]:
            result["decision"] = ScreeningDecision.EXCLUDE
            result["notes"] = f"Excluded due to: {', '.join(result['matched_exclude_criteria'])}"
            result["confidence"] = 0.8  # Higher confidence for exclusion
        elif result["matched_include_criteria"]:
            result["decision"] = ScreeningDecision.INCLUDE
            result["notes"] = f"Included due to: {', '.join(result['matched_include_criteria'])}"
            result["confidence"] = 0.7  # Moderate confidence for inclusion
        else:
            result["decision"] = ScreeningDecision.UNCERTAIN
            result["notes"] = "No criteria matched"
            result["confidence"] = 0.5  # Low confidence when uncertain
        self._update_flow_data(stage, result["decision"])
        logger.info(f"Screening decision for article {article.get('pmid', 'unknown')}: {result['decision']}")
        return result
    async def screen_articles(
        self,
        articles: List[Dict[str, Any]],
        stage: ScreeningStage,
        custom_criteria: Optional[Dict[str, List[str]]] = None
    ) -> List[Dict[str, Any]]:
        logger.info(f"Screening {len(articles)} articles at {stage} stage")
        results = []
        for article in articles:
            result = await self.screen_article(article, stage, custom_criteria)
            results.append(result)
        return results
    async def _check_criterion(self, text: str, criterion: str) -> bool:
        if self.biomedlm_service:
            try:
                similarity = self.biomedlm_service.calculate_similarity(text, criterion)
                return similarity > 0.7  # Threshold for semantic matching
            except Exception as e:
                logger.error(f"Error using BioMedLM for criterion matching: {str(e)}")
                # Fall back to simple text matching if BioMedLM fails
                return criterion.lower() in text.lower()
        else:
            return criterion.lower() in text.lower()
    def _update_flow_data(self, stage: ScreeningStage, decision: ScreeningDecision):
        """
        Update PRISMA flow data based on screening decision.
        Args:
            stage: Screening stage
            decision: Screening decision
        """
        if stage == ScreeningStage.IDENTIFICATION:
            self.flow_data["identification"]["records_identified"] += 1
            if decision == ScreeningDecision.EXCLUDE:
                self.flow_data["identification"]["records_removed_before_screening"] += 1
        elif stage == ScreeningStage.SCREENING:
            self.flow_data["screening"]["records_screened"] += 1
            if decision == ScreeningDecision.EXCLUDE:
                self.flow_data["screening"]["records_excluded"] += 1
        elif stage == ScreeningStage.ELIGIBILITY:
            self.flow_data["eligibility"]["full_text_assessed"] += 1
            if decision == ScreeningDecision.EXCLUDE:
                self.flow_data["eligibility"]["full_text_excluded"] += 1
                if "notes" in decision and decision["notes"]:
                    reason = decision["notes"]
                    if reason in self.flow_data["eligibility"]["exclusion_reasons"]:
                        self.flow_data["eligibility"]["exclusion_reasons"][reason] += 1
                    else:
                        self.flow_data["eligibility"]["exclusion_reasons"][reason] = 1
        elif stage == ScreeningStage.INCLUDED:
            if decision == ScreeningDecision.INCLUDE:
                self.flow_data["included"]["studies_included"] += 1
    def get_flow_data(self) -> Dict[str, Any]:
        """
        Get PRISMA flow data.
        Returns:
            PRISMA flow data
        """
        return self.flow_data
    def generate_flow_diagram(self) -> Dict[str, Any]:
        """
        Generate PRISMA flow diagram data.
        Returns:
            Data for generating a PRISMA flow diagram
        """
        remaining_after_identification = (
            self.flow_data["identification"]["records_identified"] -
            self.flow_data["identification"]["records_removed_before_screening"]
        )
        remaining_after_screening = (
            self.flow_data["screening"]["records_screened"] -
            self.flow_data["screening"]["records_excluded"]
        )
        diagram_data = {
            "identification": {
                "records_identified": self.flow_data["identification"]["records_identified"],
                "records_removed": self.flow_data["identification"]["records_removed_before_screening"],
                "records_remaining": remaining_after_identification
            },
            "screening": {
                "records_screened": self.flow_data["screening"]["records_screened"],
                "records_excluded": self.flow_data["screening"]["records_excluded"],
                "records_remaining": remaining_after_screening
            },
            "eligibility": {
                "full_text_assessed": self.flow_data["eligibility"]["full_text_assessed"],
                "full_text_excluded": self.flow_data["eligibility"]["full_text_excluded"],
                "exclusion_reasons": self.flow_data["eligibility"]["exclusion_reasons"]
            },
            "included": {
                "studies_included": self.flow_data["included"]["studies_included"]
            }
        }
        return diagram_data