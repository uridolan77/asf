"""
Clinical Trial Data Transformer

This module transforms ClinicalTrials.gov data into a format suitable for loading
into a Neo4j graph database. It handles normalization, cleaning, and structuring
of clinical trial data.
"""

import logging
from typing import Dict, List, Any, Optional
import re
from datetime import datetime

logger = logging.getLogger("biomedical_etl.transformers.clinical_trial")

class ClinicalTrialTransformer:
    """
    Transformer for ClinicalTrials.gov data.
    
    This class provides methods for transforming clinical trial data into a format
    suitable for loading into a Neo4j graph database. It handles normalization,
    cleaning, and structuring of trial data.
    """
    
    def __init__(self):
        """Initialize the clinical trial transformer."""
        logger.info("Initialized Clinical Trial Transformer")
    
    def transform(self, trials: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Transform a list of clinical trials.
        
        Args:
            trials: List of clinical trial dictionaries
            
        Returns:
            List of transformed clinical trial dictionaries
        """
        logger.info(f"Transforming {len(trials)} clinical trials")
        transformed_trials = []
        
        for trial in trials:
            try:
                transformed = self._transform_trial(trial)
                transformed_trials.append(transformed)
            except Exception as e:
                logger.error(f"Error transforming trial {trial.get('nct_id', 'unknown')}: {str(e)}")
        
        logger.info(f"Transformed {len(transformed_trials)} clinical trials")
        return transformed_trials
    
    def _transform_trial(self, trial: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform a single clinical trial.
        
        Args:
            trial: Clinical trial dictionary
            
        Returns:
            Transformed clinical trial dictionary
        """
        # Extract basic fields
        nct_id = trial.get("nct_id", "")
        title = trial.get("title", "")
        official_title = trial.get("official_title", "")
        
        # Use official title if available, otherwise use brief title
        display_title = official_title if official_title else title
        
        # Clean and normalize text fields
        display_title = self._clean_text(display_title)
        
        # Process descriptions
        brief_summary = trial.get("brief_summary", "")
        detailed_description = trial.get("detailed_description", "")
        
        brief_summary = self._clean_text(brief_summary)
        detailed_description = self._clean_text(detailed_description)
        
        # Format dates
        start_date = trial.get("start_date", "")
        completion_date = trial.get("completion_date", "")
        
        formatted_start_date = self._format_date(start_date)
        formatted_completion_date = self._format_date(completion_date)
        
        # Extract status and phase
        status = trial.get("status", "")
        phase = trial.get("phase", "")
        
        # Process conditions
        conditions = trial.get("conditions", [])
        processed_conditions = self._process_conditions(conditions)
        
        # Process interventions
        interventions = trial.get("interventions", [])
        processed_interventions = self._process_interventions(interventions)
        
        # Process eligibility criteria
        eligibility_criteria = trial.get("eligibility_criteria", "")
        processed_eligibility = self._clean_text(eligibility_criteria)
        
        # Process sponsor information
        lead_sponsor = trial.get("lead_sponsor", "")
        
        # Process locations
        locations = trial.get("locations", [])
        processed_locations = self._process_locations(locations)
        
        # Create transformed trial
        transformed = {
            "nct_id": nct_id,
            "title": display_title,
            "brief_summary": brief_summary,
            "detailed_description": detailed_description,
            "start_date": formatted_start_date,
            "completion_date": formatted_completion_date,
            "status": status,
            "phase": phase,
            "conditions": processed_conditions,
            "interventions": processed_interventions,
            "eligibility_criteria": processed_eligibility,
            "lead_sponsor": lead_sponsor,
            "locations": processed_locations,
            "source": "clinicaltrials.gov"
        }
        
        # Extract sentences for embedding
        sentences = self._extract_sentences(display_title, brief_summary, detailed_description)
        transformed["sentences"] = sentences
        
        return transformed
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Normalize Unicode characters
        text = text.replace('\u2019', "'")  # Smart quotes
        text = text.replace('\u2018', "'")
        text = text.replace('\u201c', '"')
        text = text.replace('\u201d', '"')
        text = text.replace('\u2014', '-')  # Em dash
        text = text.replace('\u2013', '-')  # En dash
        
        return text.strip()
    
    def _format_date(self, date_str: str) -> str:
        """
        Format a date string.
        
        Args:
            date_str: Date string in various formats
            
        Returns:
            Formatted date string (YYYY-MM-DD)
        """
        if not date_str:
            return ""
        
        try:
            # Try parsing the date string
            # Handle common formats from ClinicalTrials.gov
            
            # YYYY-MM-DD
            if re.match(r'^\d{4}-\d{2}-\d{2}$', date_str):
                return date_str
            
            # YYYY/MM/DD
            elif re.match(r'^\d{4}/\d{2}/\d{2}$', date_str):
                return date_str.replace('/', '-')
            
            # Month DD, YYYY
            elif re.match(r'^[A-Za-z]+ \d{1,2}, \d{4}$', date_str):
                dt = datetime.strptime(date_str, '%B %d, %Y')
                return dt.strftime('%Y-%m-%d')
            
            # YYYY
            elif re.match(r'^\d{4}$', date_str):
                return f"{date_str}-01-01"
            
            # Default to empty string if we can't parse the date
            return date_str
        
        except Exception:
            return date_str
    
    def _process_conditions(self, conditions: List[str]) -> List[Dict[str, str]]:
        """
        Process condition names.
        
        Args:
            conditions: List of condition name strings
            
        Returns:
            List of processed condition dictionaries
        """
        processed = []
        
        for condition in conditions:
            if not condition:
                continue
            
            clean_condition = self._clean_text(condition)
            if clean_condition:
                processed.append({
                    "name": clean_condition
                })
        
        return processed
    
    def _process_interventions(self, interventions: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Process interventions.
        
        Args:
            interventions: List of intervention dictionaries
            
        Returns:
            List of processed intervention dictionaries
        """
        processed = []
        
        for intervention in interventions:
            if not intervention:
                continue
            
            intervention_type = intervention.get("type", "")
            intervention_name = intervention.get("name", "")
            intervention_desc = intervention.get("description", "")
            
            if intervention_name:
                processed.append({
                    "name": self._clean_text(intervention_name),
                    "type": intervention_type,
                    "description": self._clean_text(intervention_desc)
                })
        
        return processed
    
    def _process_locations(self, locations: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Process locations.
        
        Args:
            locations: List of location dictionaries
            
        Returns:
            List of processed location dictionaries
        """
        processed = []
        
        for location in locations:
            if not location:
                continue
            
            facility_name = location.get("name", "")
            city = location.get("city", "")
            state = location.get("state", "")
            country = location.get("country", "")
            
            if facility_name and country:
                processed.append({
                    "name": self._clean_text(facility_name),
                    "city": self._clean_text(city),
                    "state": self._clean_text(state),
                    "country": self._clean_text(country)
                })
        
        return processed
    
    def _extract_sentences(self, title: str, brief_summary: str, detailed_description: str) -> List[Dict[str, Any]]:
        """
        Extract sentences from title and descriptions.
        
        Args:
            title: Trial title
            brief_summary: Brief summary
            detailed_description: Detailed description
            
        Returns:
            List of sentence dictionaries
        """
        sentences = []
        
        # Add title as a sentence
        if title:
            sentences.append({
                "text": title,
                "section": "title",
                "position": 0
            })
        
        # Split brief summary into sentences
        if brief_summary:
            brief_sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', brief_summary)
            
            for i, sentence in enumerate(brief_sentences):
                clean_sentence = self._clean_text(sentence)
                if clean_sentence:
                    sentences.append({
                        "text": clean_sentence,
                        "section": "brief_summary",
                        "position": i
                    })
        
        # Split detailed description into sentences
        if detailed_description:
            detailed_sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', detailed_description)
            
            for i, sentence in enumerate(detailed_sentences):
                clean_sentence = self._clean_text(sentence)
                if clean_sentence:
                    sentences.append({
                        "text": clean_sentence,
                        "section": "detailed_description",
                        "position": i
                    })
        
        return sentences