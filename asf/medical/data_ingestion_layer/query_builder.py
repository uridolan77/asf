"""
Medical Query Builder for NCBI Client

This module provides a high-level query construction API that helps users build
complex Boolean queries specific to medical literature without needing to understand
PubMed's syntax.
"""

import os
import json
import logging
import datetime
import requests
from typing import Dict, List, Optional, Union, Any, Tuple, Set
from dataclasses import dataclass, field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('query_builder')

@dataclass
class MedicalTerm:
    """Class to represent a medical term with MeSH mapping."""
    term: str
    mesh_terms: List[str] = field(default_factory=list)
    exact_match: bool = False
    is_mesh: bool = False
    

class MeshMapper:
    """Class to map user terms to MeSH terms."""
    
    def __init__(self, mesh_data_file: Optional[str] = None):
        """
        Initialize the MeSH mapper.
        
        Args:
            mesh_data_file: Path to MeSH data file (optional)
        """
        self.mesh_data = {}
        self.cache_dir = os.path.join(os.path.expanduser("~"), ".pubmed_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.mesh_file = mesh_data_file or os.path.join(self.cache_dir, "mesh_data.json")
        self.load_mesh_data()
    
    def load_mesh_data(self):
        """Load MeSH data from file if available."""
        if os.path.exists(self.mesh_file):
            try:
                with open(self.mesh_file, 'r') as f:
                    self.mesh_data = json.load(f)
                logger.info(f"Loaded MeSH data with {len(self.mesh_data)} terms")
            except Exception as e:
                logger.error(f"Error loading MeSH data: {str(e)}")
    
    def save_mesh_data(self):
        """Save MeSH data to file."""
        try:
            with open(self.mesh_file, 'w') as f:
                json.dump(self.mesh_data, f)
            logger.info(f"Saved MeSH data with {len(self.mesh_data)} terms")
        except Exception as e:
            logger.error(f"Error saving MeSH data: {str(e)}")
    
    def map_term_to_mesh(self, term: str) -> MedicalTerm:
        """
        Map a term to MeSH terms.
        
        Args:
            term: Medical term to map
            
        Returns:
            MedicalTerm object with mappings
        """
        # Check if we already have this term in cache
        term_lower = term.lower()
        if term_lower in self.mesh_data:
            mesh_info = self.mesh_data[term_lower]
            return MedicalTerm(
                term=term,
                mesh_terms=mesh_info.get('mesh_terms', []),
                exact_match=mesh_info.get('exact_match', False),
                is_mesh=mesh_info.get('is_mesh', False)
            )
        
        # Check if the term is already a MeSH term
        if self._is_mesh_term(term):
            result = MedicalTerm(
                term=term,
                mesh_terms=[term],
                exact_match=True,
                is_mesh=True
            )
            # Cache this result
            self.mesh_data[term_lower] = {
                'mesh_terms': [term],
                'exact_match': True,
                'is_mesh': True,
                'timestamp': datetime.datetime.now().timestamp()
            }
            self.save_mesh_data()
            return result
        
        # Try to map to MeSH using NCBI's API
        mesh_terms = self._fetch_mesh_terms(term)
        
        result = MedicalTerm(
            term=term,
            mesh_terms=mesh_terms,
            exact_match=any(m.lower() == term_lower for m in mesh_terms),
            is_mesh=False
        )
        
        # Cache this result
        self.mesh_data[term_lower] = {
            'mesh_terms': mesh_terms,
            'exact_match': result.exact_match,
            'is_mesh': result.is_mesh,
            'timestamp': datetime.datetime.now().timestamp()
        }
        self.save_mesh_data()
        
        return result
    
    def _is_mesh_term(self, term: str) -> bool:
        """
        Check if a term is already a MeSH term.
        This is a simplified check - in a real implementation,
        you would validate against the full MeSH database.
        
        Args:
            term: Term to check
            
        Returns:
            Boolean indicating if it's a MeSH term
        """
        # Check if term contains [Mesh] or [MeSH Term]
        if '[mesh]' in term.lower() or '[mesh term]' in term.lower():
            return True
            
        # Try to verify with NCBI's API
        try:
            url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=mesh&term={term}[mesh]&retmode=json"
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                return int(data.get('esearchresult', {}).get('count', 0)) > 0
        except Exception:
            pass
            
        return False
    
    def _fetch_mesh_terms(self, term: str) -> List[str]:
        """
        Fetch MeSH terms for a given term using NCBI's API.
        
        Args:
            term: Term to map
            
        Returns:
            List of related MeSH terms
        """
        mesh_terms = []
        
        try:
            # Use ESpell to get suggested terms
            url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/espell.fcgi?db=pubmed&term={term}&retmode=json"
            response = requests.get(url)
            
            if response.status_code == 200:
                data = response.json()
                corrected_term = data.get('esearchresult', {}).get('correctedquery')
                if corrected_term and corrected_term != term:
                    mesh_terms.append(corrected_term)
            
            # Use ESearch with automatic term mapping
            url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term={term}&retmode=json&sort=relevance&retmax=1"
            response = requests.get(url)
            
            if response.status_code == 200:
                data = response.json()
                
                # Check if we have any translation stacks
                if 'esearchresult' in data and 'translationstack' in data['esearchresult']:
                    for item in data['esearchresult']['translationstack']:
                        if isinstance(item, dict) and 'term' in item:
                            term_str = item['term']
                            # Extract MeSH terms from translation stack
                            if '[mesh]' in term_str.lower() or '[mesh terms]' in term_str.lower():
                                # Remove the field specifier
                                mesh_term = term_str.split('[')[0].strip()
                                if mesh_term and mesh_term not in mesh_terms:
                                    mesh_terms.append(mesh_term)
        except Exception as e:
            logger.error(f"Error fetching MeSH terms for '{term}': {str(e)}")
        
        return mesh_terms


class MedicalCondition:
    """Class to represent a medical condition with query building capabilities."""
    
    def __init__(self, 
                 name: str, 
                 mesh_mapper: Optional[MeshMapper] = None):
        """
        Initialize a medical condition.
        
        Args:
            name: Name of the condition
            mesh_mapper: MeshMapper instance for MeSH term mapping
        """
        self.name = name
        self.mesh_mapper = mesh_mapper or MeshMapper()
        self.mapped_term = self.mesh_mapper.map_term_to_mesh(name)
        self.synonyms = []
        self.subtypes = []
    
    def add_synonym(self, synonym: str) -> 'MedicalCondition':
        """
        Add a synonym for the condition.
        
        Args:
            synonym: Synonym term
            
        Returns:
            Self for method chaining
        """
        self.synonyms.append(synonym)
        return self
    
    def add_subtype(self, subtype: str) -> 'MedicalCondition':
        """
        Add a subtype of the condition.
        
        Args:
            subtype: Subtype term
            
        Returns:
            Self for method chaining
        """
        self.subtypes.append(subtype)
        return self
    
    def build_query_term(self, 
                        use_mesh: bool = True,
                        include_synonyms: bool = True,
                        include_subtypes: bool = True) -> str:
        """
        Build a query term for this condition.
        
        Args:
            use_mesh: Whether to use MeSH terms if available
            include_synonyms: Whether to include synonyms
            include_subtypes: Whether to include subtypes
            
        Returns:
            Query term string
        """
        terms = []
        
        # Use MeSH term if available and requested
        if use_mesh and self.mapped_term.mesh_terms:
            # Add MeSH terms with the [Mesh] qualifier
            for mesh_term in self.mapped_term.mesh_terms:
                terms.append(f'"{mesh_term}"[Mesh]')
        
        # Always include the original term
        terms.append(f'"{self.name}"[Title/Abstract]')
        
        # Add synonyms if requested
        if include_synonyms and self.synonyms:
            for synonym in self.synonyms:
                terms.append(f'"{synonym}"[Title/Abstract]')
        
        # Add subtypes if requested
        if include_subtypes and self.subtypes:
            for subtype in self.subtypes:
                terms.append(f'"{subtype}"[Title/Abstract]')
        
        # Combine terms with OR
        return f"({' OR '.join(terms)})"


class MedicalIntervention:
    """Class to represent a medical intervention with query building capabilities."""
    
    def __init__(self, 
                 name: str, 
                 intervention_type: str = 'treatment',
                 mesh_mapper: Optional[MeshMapper] = None):
        """
        Initialize a medical intervention.
        
        Args:
            name: Name of the intervention
            intervention_type: Type of intervention (treatment, diagnostic, prevention)
            mesh_mapper: MeshMapper instance for MeSH term mapping
        """
        self.name = name
        self.intervention_type = intervention_type
        self.mesh_mapper = mesh_mapper or MeshMapper()
        self.mapped_term = self.mesh_mapper.map_term_to_mesh(name)
        self.alternatives = []
        self.specific_forms = []
    
    def add_alternative(self, alternative: str) -> 'MedicalIntervention':
        """
        Add an alternative intervention.
        
        Args:
            alternative: Alternative intervention term
            
        Returns:
            Self for method chaining
        """
        self.alternatives.append(alternative)
        return self
    
    def add_specific_form(self, form: str) -> 'MedicalIntervention':
        """
        Add a specific form of the intervention.
        
        Args:
            form: Specific form term
            
        Returns:
            Self for method chaining
        """
        self.specific_forms.append(form)
        return self
    
    def build_query_term(self, 
                        use_mesh: bool = True,
                        include_alternatives: bool = True,
                        include_specific_forms: bool = True) -> str:
        """
        Build a query term for this intervention.
        
        Args:
            use_mesh: Whether to use MeSH terms if available
            include_alternatives: Whether to include alternatives
            include_specific_forms: Whether to include specific forms
            
        Returns:
            Query term string
        """
        terms = []
        
        # Use MeSH term if available and requested
        if use_mesh and self.mapped_term.mesh_terms:
            # Add MeSH terms with the [Mesh] qualifier
            for mesh_term in self.mapped_term.mesh_terms:
                terms.append(f'"{mesh_term}"[Mesh]')
        
        # Always include the original term
        terms.append(f'"{self.name}"[Title/Abstract]')
        
        # Add alternatives if requested
        if include_alternatives and self.alternatives:
            for alternative in self.alternatives:
                terms.append(f'"{alternative}"[Title/Abstract]')
        
        # Add specific forms if requested
        if include_specific_forms and self.specific_forms:
            for form in self.specific_forms:
                terms.append(f'"{form}"[Title/Abstract]')
        
        # Add intervention type if it's not already part of the name
        if self.intervention_type.lower() not in self.name.lower():
            terms.append(f'"{self.intervention_type}"[Title/Abstract]')
        
        # Combine terms with OR
        return f"({' OR '.join(terms)})"


class OutcomeMetric:
    """Class to represent an outcome metric with query building capabilities."""
    
    def __init__(self, 
                 name: str,
                 outcome_type: str = 'efficacy',
                 mesh_mapper: Optional[MeshMapper] = None):
        """
        Initialize an outcome metric.
        
        Args:
            name: Name of the outcome metric
            outcome_type: Type of outcome (efficacy, safety, cost)
            mesh_mapper: MeshMapper instance for MeSH term mapping
        """
        self.name = name
        self.outcome_type = outcome_type
        self.mesh_mapper = mesh_mapper or MeshMapper()
        self.mapped_term = self.mesh_mapper.map_term_to_mesh(name)
        self.synonyms = []
        self.related_metrics = []
    
    def add_synonym(self, synonym: str) -> 'OutcomeMetric':
        """
        Add a synonym for the outcome metric.
        
        Args:
            synonym: Synonym term
            
        Returns:
            Self for method chaining
        """
        self.synonyms.append(synonym)
        return self
    
    def add_related_metric(self, metric: str) -> 'OutcomeMetric':
        """
        Add a related outcome metric.
        
        Args:
            metric: Related metric term
            
        Returns:
            Self for method chaining
        """
        self.related_metrics.append(metric)
        return self
    
    def build_query_term(self, 
                        use_mesh: bool = True,
                        include_synonyms: bool = True,
                        include_related: bool = True) -> str:
        """
        Build a query term for this outcome metric.
        
        Args:
            use_mesh: Whether to use MeSH terms if available
            include_synonyms: Whether to include synonyms
            include_related: Whether to include related metrics
            
        Returns:
            Query term string
        """
        terms = []
        
        # Use MeSH term if available and requested
        if use_mesh and self.mapped_term.mesh_terms:
            # Add MeSH terms with the [Mesh] qualifier
            for mesh_term in self.mapped_term.mesh_terms:
                terms.append(f'"{mesh_term}"[Mesh]')
        
        # Always include the original term
        terms.append(f'"{self.name}"[Title/Abstract]')
        
        # Add synonyms if requested
        if include_synonyms and self.synonyms:
            for synonym in self.synonyms:
                terms.append(f'"{synonym}"[Title/Abstract]')
        
        # Add related metrics if requested
        if include_related and self.related_metrics:
            for metric in self.related_metrics:
                terms.append(f'"{metric}"[Title/Abstract]')
        
        # Add outcome type if it's not already part of the name
        if self.outcome_type.lower() not in self.name.lower():
            terms.append(f'"{self.outcome_type}"[Title/Abstract]')
        
        # Combine terms with OR
        return f"({' OR '.join(terms)})"


class StudyDesign:
    """Class to represent a study design with query building capabilities."""
    
    def __init__(self, 
                 design_type: str, 
                 mesh_mapper: Optional[MeshMapper] = None):
        """
        Initialize a study design.
        
        Args:
            design_type: Type of study design
            mesh_mapper: MeshMapper instance for MeSH term mapping
        """
        self.design_type = design_type
        self.mesh_mapper = mesh_mapper or MeshMapper()
        self.mapped_term = self.mesh_mapper.map_term_to_mesh(design_type)
        self.related_designs = []
        self.design_characteristics = []
        
        # Map common study design types to MeSH Publication Types
        self.design_pub_types = {
            'randomized controlled trial': 'Randomized Controlled Trial[Publication Type]',
            'rct': 'Randomized Controlled Trial[Publication Type]',
            'systematic review': 'Systematic Review[Publication Type]',
            'meta-analysis': 'Meta-Analysis[Publication Type]',
            'observational study': 'Observational Study[Publication Type]',
            'case report': 'Case Reports[Publication Type]',
            'clinical trial': 'Clinical Trial[Publication Type]',
            'comparative study': 'Comparative Study[Publication Type]',
        }
    
    def add_related_design(self, design: str) -> 'StudyDesign':
        """
        Add a related study design.
        
        Args:
            design: Related design term
            
        Returns:
            Self for method chaining
        """
        self.related_designs.append(design)
        return self
    
    def add_characteristic(self, characteristic: str) -> 'StudyDesign':
        """
        Add a study design characteristic.
        
        Args:
            characteristic: Design characteristic term
            
        Returns:
            Self for method chaining
        """
        self.design_characteristics.append(characteristic)
        return self
    
    def build_query_term(self, 
                        use_mesh: bool = True,
                        include_related: bool = True,
                        include_characteristics: bool = True) -> str:
        """
        Build a query term for this study design.
        
        Args:
            use_mesh: Whether to use MeSH terms if available
            include_related: Whether to include related designs
            include_characteristics: Whether to include design characteristics
            
        Returns:
            Query term string
        """
        terms = []
        
        # Use Publication Type if available
        design_type_lower = self.design_type.lower()
        if use_mesh and design_type_lower in self.design_pub_types:
            terms.append(self.design_pub_types[design_type_lower])
        
        # Use MeSH term if available and requested
        elif use_mesh and self.mapped_term.mesh_terms:
            # Add MeSH terms with the [Mesh] qualifier
            for mesh_term in self.mapped_term.mesh_terms:
                terms.append(f'"{mesh_term}"[Mesh]')
        
        # Always include the original term
        terms.append(f'"{self.design_type}"[Title/Abstract]')
        
        # Add related designs if requested
        if include_related and self.related_designs:
            for design in self.related_designs:
                design_lower = design.lower()
                if use_mesh and design_lower in self.design_pub_types:
                    terms.append(self.design_pub_types[design_lower])
                else:
                    terms.append(f'"{design}"[Title/Abstract]')
        
        # Add design characteristics if requested
        if include_characteristics and self.design_characteristics:
            for characteristic in self.design_characteristics:
                terms.append(f'"{characteristic}"[Title/Abstract]')
        
        # Combine terms with OR
        return f"({' OR '.join(terms)})"


class PopulationGroup:
    """Class to represent a population group with query building capabilities."""
    
    def __init__(self, 
                 description: str, 
                 mesh_mapper: Optional[MeshMapper] = None):
        """
        Initialize a population group.
        
        Args:
            description: Description of the population
            mesh_mapper: MeshMapper instance for MeSH term mapping
        """
        self.description = description
        self.mesh_mapper = mesh_mapper or MeshMapper()
        self.mapped_term = self.mesh_mapper.map_term_to_mesh(description)
        self.age_groups = []
        self.demographics = []
        self.conditions = []
    
    def add_age_group(self, age_group: str) -> 'PopulationGroup':
        """
        Add an age group to the population.
        
        Args:
            age_group: Age group term
            
        Returns:
            Self for method chaining
        """
        self.age_groups.append(age_group)
        return self
    
    def add_demographic(self, demographic: str) -> 'PopulationGroup':
        """
        Add a demographic characteristic to the population.
        
        Args:
            demographic: Demographic term
            
        Returns:
            Self for method chaining
        """
        self.demographics.append(demographic)
        return self
    
    def add_condition(self, condition: str) -> 'PopulationGroup':
        """
        Add a condition to the population.
        
        Args:
            condition: Condition term
            
        Returns:
            Self for method chaining
        """
        self.conditions.append(condition)
        return self
    
    def build_query_term(self, 
                        use_mesh: bool = True,
                        include_age_groups: bool = True,
                        include_demographics: bool = True,
                        include_conditions: bool = True) -> str:
        """
        Build a query term for this population group.
        
        Args:
            use_mesh: Whether to use MeSH terms if available
            include_age_groups: Whether to include age groups
            include_demographics: Whether to include demographics
            include_conditions: Whether to include conditions
            
        Returns:
            Query term string
        """
        terms = []
        
        # Use MeSH term if available and requested
        if use_mesh and self.mapped_term.mesh_terms:
            # Add MeSH terms with the [Mesh] qualifier
            for mesh_term in self.mapped_term.mesh_terms:
                terms.append(f'"{mesh_term}"[Mesh]')
        
        # Always include the original description
        terms.append(f'"{self.description}"[Title/Abstract]')
        
        # Add age groups if requested
        if include_age_groups and self.age_groups:
            age_terms = []
            for age_group in self.age_groups:
                age_terms.append(f'"{age_group}"[Title/Abstract]')
                
                # Add special age filters for certain groups
                if age_group.lower() in ['infant', 'infants', 'newborn', 'newborns']:
                    age_terms.append('infant[MeSH]')
                elif age_group.lower() in ['child', 'children']:
                    age_terms.append('child[MeSH]')
                elif age_group.lower() in ['adolescent', 'adolescents', 'teen', 'teens', 'teenager', 'teenagers']:
                    age_terms.append('adolescent[MeSH]')
                elif age_group.lower() in ['adult', 'adults']:
                    age_terms.append('adult[MeSH]')
                elif age_group.lower() in ['elderly', 'older adult', 'older adults', 'senior', 'seniors']:
                    age_terms.append('aged[MeSH]')
            
            if age_terms:
                terms.append(f"({' OR '.join(age_terms)})")
        
        # Add demographics if requested
        if include_demographics and self.demographics:
            for demographic in self.demographics:
                terms.append(f'"{demographic}"[Title/Abstract]')
        
        # Add conditions if requested
        if include_conditions and self.conditions:
            condition_terms = []
            for condition in self.conditions:
                mapped_condition = self.mesh_mapper.map_term_to_mesh(condition)
                
                if use_mesh and mapped_condition.mesh_terms:
                    for mesh_term in mapped_condition.mesh_terms:
                        condition_terms.append(f'"{mesh_term}"[Mesh]')
                
                condition_terms.append(f'"{condition}"[Title/Abstract]')
            
            if condition_terms:
                terms.append(f"({' OR '.join(condition_terms)})")
        
        # Combine terms with AND because all these terms should apply to the population
        return f"({' AND '.join(terms)})"


class DateRange:
    """Class to represent a date range for filtering publications."""
    
    def __init__(self, start_date: Optional[str] = None, end_date: Optional[str] = None):
        """
        Initialize a date range.
        
        Args:
            start_date: Start date (format: YYYY/MM/DD, YYYY/MM, or YYYY)
            end_date: End date (format: YYYY/MM/DD, YYYY/MM, or YYYY)
        """
        self.start_date = start_date
        self.end_date = end_date or "3000"  # Default to far future
        self.date_type = "pdat"  # Default to publication date
    
    def set_date_type(self, date_type: str) -> 'DateRange':
        """
        Set the type of date to filter by.
        
        Args:
            date_type: Date type ('pdat' for publication date,
                                 'edat' for entrez date,
                                 'mdat' for modification date)
            
        Returns:
            Self for method chaining
        """
        valid_types = ['pdat', 'edat', 'mdat']
        if date_type.lower() in valid_types:
            self.date_type = date_type.lower()
        else:
            logger.warning(f"Invalid date type '{date_type}'. Using 'pdat' instead.")
        
        return self
    
    def from_years_ago(self, years: int) -> 'DateRange':
        """
        Set the start date to a specific number of years ago.
        
        Args:
            years: Number of years ago
            
        Returns:
            Self for method chaining
        """
        import datetime
        today = datetime.date.today()
        start_year = today.year - years
        self.start_date = f"{start_year}"
        
        return self
    
    def last_n_years(self, n: int) -> 'DateRange':
        """
        Set the date range to the last N years.
        
        Args:
            n: Number of years
            
        Returns:
            Self for method chaining
        """
        return self.from_years_ago(n)
    
    def year(self, year: int) -> 'DateRange':
        """
        Set the date range to a specific year.
        
        Args:
            year: The year
            
        Returns:
            Self for method chaining
        """
        self.start_date = f"{year}"
        self.end_date = f"{year}"
        
        return self
    
    def build_query_term(self) -> str:
        """
        Build a query term for this date range.
        
        Returns:
            Query term string
        """
        if not self.start_date:
            return ""
            
        if self.start_date == self.end_date:
            return f"{self.start_date}[{self.date_type}]"
        else:
            return f"{self.start_date}:{self.end_date}[{self.date_type}]"


class JournalFilter:
    """Class to represent a journal filter."""
    
    def __init__(self, journal_names: Optional[List[str]] = None):
        """
        Initialize a journal filter.
        
        Args:
            journal_names: List of journal names to include
        """
        self.journal_names = journal_names or []
        self.issns = []
        self.impact_factor_min = None
        self.include_open_access = False
    
    def add_journal(self, journal_name: str) -> 'JournalFilter':
        """
        Add a journal to the filter.
        
        Args:
            journal_name: Journal name
            
        Returns:
            Self for method chaining
        """
        self.journal_names.append(journal_name)
        return self
    
    def add_issn(self, issn: str) -> 'JournalFilter':
        """
        Add a journal ISSN to the filter.
        
        Args:
            issn: Journal ISSN
            
        Returns:
            Self for method chaining
        """
        self.issns.append(issn)
        return self
    
    def set_min_impact_factor(self, min_factor: float) -> 'JournalFilter':
        """
        Set a minimum impact factor for journals.
        Note: This can't be directly queried in PubMed,
        so it will need post-processing of results.
        
        Args:
            min_factor: Minimum impact factor
            
        Returns:
            Self for method chaining
        """
        self.impact_factor_min = min_factor
        return self
    
    def only_open_access(self, value: bool = True) -> 'JournalFilter':
        """
        Filter to only include open access journals.
        
        Args:
            value: Whether to only include open access journals
            
        Returns:
            Self for method chaining
        """
        self.include_open_access = value
        return self
    
    def build_query_term(self) -> str:
        """
        Build a query term for this journal filter.
        
        Returns:
            Query term string
        """
        terms = []
        
        # Add journal names
        for journal in self.journal_names:
            terms.append(f'"{journal}"[Journal]')
        
        # Add ISSNs
        for issn in self.issns:
            terms.append(f'"{issn}"[ISSN]')
        
        # Add open access filter
        if self.include_open_access:
            terms.append('free full text[Filter]')
        
        # Combine terms with OR (any of these journals)
        if not terms:
            return ""
        
        return f"({' OR '.join(terms)})"


class PublicationTypeFilter:
    """Class to represent a publication type filter."""
    
    def __init__(self, publication_types: Optional[List[str]] = None):
        """
        Initialize a publication type filter.
        
        Args:
            publication_types: List of publication types to include
        """
        self.publication_types = publication_types or []
        
        # Map common publication types to PubMed syntax
        self.pubtype_map = {
            'clinical trial': 'Clinical Trial[Publication Type]',
            'randomized controlled trial': 'Randomized Controlled Trial[Publication Type]',
            'meta-analysis': 'Meta-Analysis[Publication Type]',
            'systematic review': 'Systematic Review[Publication Type]',
            'review': 'Review[Publication Type]',
            'case report': 'Case Reports[Publication Type]',
            'guideline': 'Guideline[Publication Type]',
            'practice guideline': 'Practice Guideline[Publication Type]',
            'editorial': 'Editorial[Publication Type]',
            'letter': 'Letter[Publication Type]',
            'comment': 'Comment[Publication Type]',
            'journal article': 'Journal Article[Publication Type]',
            'observational study': 'Observational Study[Publication Type]',
        }
    
    def add_type(self, publication_type: str) -> 'PublicationTypeFilter':
        """
        Add a publication type to the filter.
        
        Args:
            publication_type: Publication type
            
        Returns:
            Self for method chaining
        """
        self.publication_types.append(publication_type)
        return self
    
    def clinical_trials_only(self) -> 'PublicationTypeFilter':
        """
        Filter to only include clinical trials.
        
        Returns:
            Self for method chaining
        """
        self.publication_types.append('clinical trial')
        return self
    
    def high_quality_evidence_only(self) -> 'PublicationTypeFilter':
        """
        Filter to only include high-quality evidence (RCTs, meta-analyses, systematic reviews).
        
        Returns:
            Self for method chaining
        """
        self.publication_types.extend(['randomized controlled trial', 'meta-analysis', 'systematic review'])
        return self
    
    def review_articles_only(self) -> 'PublicationTypeFilter':
        """
        Filter to only include review articles.
        
        Returns:
            Self for method chaining
        """
        self.publication_types.append('review')
        return self
    
    def exclude_case_reports(self) -> 'PublicationTypeFilter':
        """
        Exclude case reports from results.
        
        Returns:
            Self for method chaining
        """
        # We'll handle this in build_query_term
        self.publication_types.append('NOT case report')
        return self
    
    def guidelines_only(self) -> 'PublicationTypeFilter':
        """
        Filter to only include guidelines.
        
        Returns:
            Self for method chaining
        """
        self.publication_types.extend(['guideline', 'practice guideline'])
        return self
    
    def build_query_term(self) -> str:
        """
        Build a query term for this publication type filter.
        
        Returns:
            Query term string
        """
        if not self.publication_types:
            return ""
            
        include_terms = []
        exclude_terms = []
        
        for pub_type in self.publication_types:
            if pub_type.startswith('NOT '):
                # Handle exclusions
                exclude_type = pub_type[4:].strip().lower()
                if exclude_type in self.pubtype_map:
                    exclude_terms.append(self.pubtype_map[exclude_type])
                else:
                    exclude_terms.append(f'"{exclude_type}"[Publication Type]')
            else:
                # Handle inclusions
                include_type = pub_type.lower()
                if include_type in self.pubtype_map:
                    include_terms.append(self.pubtype_map[include_type])
                else:
                    include_terms.append(f'"{pub_type}"[Publication Type]')
        
        query_parts = []
        
        # Add inclusion terms (OR'd together)
        if include_terms:
            query_parts.append(f"({' OR '.join(include_terms)})")
        
        # Add exclusion terms (NOT'd)
        for term in exclude_terms:
            query_parts.append(f"NOT {term}")
        
        # Combine with AND
        return f"({' AND '.join(query_parts)})"


class LanguageFilter:
    """Class to represent a language filter."""
    
    def __init__(self, languages: Optional[List[str]] = None):
        """
        Initialize a language filter.
        
        Args:
            languages: List of languages to include
        """
        self.languages = languages or ['english']
    
    def add_language(self, language: str) -> 'LanguageFilter':
        """
        Add a language to the filter.
        
        Args:
            language: Language name
            
        Returns:
            Self for method chaining
        """
        self.languages.append(language)
        return self
    
    def english_only(self) -> 'LanguageFilter':
        """
        Filter to only include English articles.
        
        Returns:
            Self for method chaining
        """
        self.languages = ['english']
        return self
    
    def build_query_term(self) -> str:
        """
        Build a query term for this language filter.
        
        Returns:
            Query term string
        """
        if not self.languages:
            return ""
            
        language_terms = []
        for language in self.languages:
            language_terms.append(f'"{language}"[Language]')
        
        # Combine with OR
        return f"({' OR '.join(language_terms)})"


class MedicalQueryBuilder:
    """
    High-level query builder for medical literature searches.
    """
    
    def __init__(self, mesh_mapper: Optional[MeshMapper] = None):
        """
        Initialize the query builder.
        
        Args:
            mesh_mapper: MeshMapper instance for MeSH term mapping
        """
        self.mesh_mapper = mesh_mapper or MeshMapper()
        self.conditions = []
        self.interventions = []
        self.outcomes = []
        self.study_designs = []
        self.populations = []
        self.date_range = None
        self.journal_filter = None
        self.publication_type_filter = None
        self.language_filter = None
        self.additional_filters = []
    
    def add_condition(self, condition: Union[str, MedicalCondition]) -> 'MedicalQueryBuilder':
        """
        Add a medical condition to the query.
        
        Args:
            condition: Medical condition string or object
            
        Returns:
            Self for method chaining
        """
        if isinstance(condition, str):
            condition = MedicalCondition(condition, self.mesh_mapper)
        
        self.conditions.append(condition)
        return self
    
    def add_intervention(self, intervention: Union[str, MedicalIntervention]) -> 'MedicalQueryBuilder':
        """
        Add a medical intervention to the query.
        
        Args:
            intervention: Medical intervention string or object
            
        Returns:
            Self for method chaining
        """
        if isinstance(intervention, str):
            intervention = MedicalIntervention(intervention, mesh_mapper=self.mesh_mapper)
        
        self.interventions.append(intervention)
        return self
    
    def add_outcome(self, outcome: Union[str, OutcomeMetric]) -> 'MedicalQueryBuilder':
        """
        Add an outcome metric to the query.
        
        Args:
            outcome: Outcome metric string or object
            
        Returns:
            Self for method chaining
        """
        if isinstance(outcome, str):
            outcome = OutcomeMetric(outcome, mesh_mapper=self.mesh_mapper)
        
        self.outcomes.append(outcome)
        return self
    
    def add_study_design(self, design: Union[str, StudyDesign]) -> 'MedicalQueryBuilder':
        """
        Add a study design to the query.
        
        Args:
            design: Study design string or object
            
        Returns:
            Self for method chaining
        """
        if isinstance(design, str):
            design = StudyDesign(design, mesh_mapper=self.mesh_mapper)
        
        self.study_designs.append(design)
        return self
    
    def add_population(self, population: Union[str, PopulationGroup]) -> 'MedicalQueryBuilder':
        """
        Add a population group to the query.
        
        Args:
            population: Population group string or object
            
        Returns:
            Self for method chaining
        """
        if isinstance(population, str):
            population = PopulationGroup(population, mesh_mapper=self.mesh_mapper)
        
        self.populations.append(population)
        return self
    
    def set_date_range(self, date_range: DateRange) -> 'MedicalQueryBuilder':
        """
        Set a date range for the query.
        
        Args:
            date_range: DateRange object
            
        Returns:
            Self for method chaining
        """
        self.date_range = date_range
        return self
    
    def last_n_years(self, n: int) -> 'MedicalQueryBuilder':
        """
        Set the date range to the last N years.
        
        Args:
            n: Number of years
            
        Returns:
            Self for method chaining
        """
        self.date_range = DateRange().last_n_years(n)
        return self
    
    def set_journal_filter(self, filter: JournalFilter) -> 'MedicalQueryBuilder':
        """
        Set a journal filter for the query.
        
        Args:
            filter: JournalFilter object
            
        Returns:
            Self for method chaining
        """
        self.journal_filter = filter
        return self
    
    def set_publication_type_filter(self, filter: PublicationTypeFilter) -> 'MedicalQueryBuilder':
        """
        Set a publication type filter for the query.
        
        Args:
            filter: PublicationTypeFilter object
            
        Returns:
            Self for method chaining
        """
        self.publication_type_filter = filter
        return self
    
    def set_language_filter(self, filter: LanguageFilter) -> 'MedicalQueryBuilder':
        """
        Set a language filter for the query.
        
        Args:
            filter: LanguageFilter object
            
        Returns:
            Self for method chaining
        """
        self.language_filter = filter
        return self
    
    def english_only(self) -> 'MedicalQueryBuilder':
        """
        Filter to only include English articles.
        
        Returns:
            Self for method chaining
        """
        self.language_filter = LanguageFilter().english_only()
        return self
    
    def high_quality_only(self) -> 'MedicalQueryBuilder':
        """
        Filter to only include high-quality evidence.
        
        Returns:
            Self for method chaining
        """
        self.publication_type_filter = PublicationTypeFilter().high_quality_evidence_only()
        return self
    
    def add_custom_filter(self, filter_term: str) -> 'MedicalQueryBuilder':
        """
        Add a custom filter term to the query.
        
        Args:
            filter_term: Custom filter term
            
        Returns:
            Self for method chaining
        """
        self.additional_filters.append(filter_term)
        return self
    
    def humans_only(self) -> 'MedicalQueryBuilder':
        """
        Filter to only include studies on humans.
        
        Returns:
            Self for method chaining
        """
        self.add_custom_filter("humans[MeSH Terms]")
        return self
    
    def build_pico_query(self, use_mesh: bool = True) -> str:
        """
        Build a query following the PICO format
        (Population, Intervention, Comparison, Outcome).
        
        Args:
            use_mesh: Whether to use MeSH terms if available
            
        Returns:
            PubMed query string
        """
        query_parts = []
        
        # Add conditions
        if self.conditions:
            condition_terms = []
            for condition in self.conditions:
                condition_terms.append(condition.build_query_term(use_mesh=use_mesh))
            
            if condition_terms:
                query_parts.append(f"({' OR '.join(condition_terms)})")
        
        # Add populations
        if self.populations:
            population_terms = []
            for population in self.populations:
                population_terms.append(population.build_query_term(use_mesh=use_mesh))
            
            if population_terms:
                query_parts.append(f"({' OR '.join(population_terms)})")
        
        # Add interventions
        if self.interventions:
            intervention_terms = []
            for intervention in self.interventions:
                intervention_terms.append(intervention.build_query_term(use_mesh=use_mesh))
            
            if intervention_terms:
                query_parts.append(f"({' OR '.join(intervention_terms)})")
        
        # Add outcomes
        if self.outcomes:
            outcome_terms = []
            for outcome in self.outcomes:
                outcome_terms.append(outcome.build_query_term(use_mesh=use_mesh))
            
            if outcome_terms:
                query_parts.append(f"({' OR '.join(outcome_terms)})")
        
        # Add study designs
        if self.study_designs:
            design_terms = []
            for design in self.study_designs:
                design_terms.append(design.build_query_term(use_mesh=use_mesh))
            
            if design_terms:
                query_parts.append(f"({' OR '.join(design_terms)})")
        
        # Add date range
        if self.date_range:
            date_term = self.date_range.build_query_term()
            if date_term:
                query_parts.append(date_term)
        
        # Add journal filter
        if self.journal_filter:
            journal_term = self.journal_filter.build_query_term()
            if journal_term:
                query_parts.append(journal_term)
        
        # Add publication type filter
        if self.publication_type_filter:
            pubtype_term = self.publication_type_filter.build_query_term()
            if pubtype_term:
                query_parts.append(pubtype_term)
        
        # Add language filter
        if self.language_filter:
            language_term = self.language_filter.build_query_term()
            if language_term:
                query_parts.append(language_term)
        
        # Add additional filters
        for filter_term in self.additional_filters:
            query_parts.append(f"({filter_term})")
        
        # Combine all parts with AND
        return " AND ".join(query_parts)
    
    def build_simple_query(self, use_mesh: bool = True) -> str:
        """
        Build a simple query combining main terms.
        
        Args:
            use_mesh: Whether to use MeSH terms if available
            
        Returns:
            PubMed query string
        """
        query_parts = []
        
        # Combine conditions and populations
        clinical_terms = []
        
        # Add conditions
        for condition in self.conditions:
            clinical_terms.append(condition.build_query_term(use_mesh=use_mesh))
        
        # Add populations
        for population in self.populations:
            clinical_terms.append(population.build_query_term(use_mesh=use_mesh))
        
        if clinical_terms:
            query_parts.append(f"({' OR '.join(clinical_terms)})")
        
        # Combine interventions and outcomes
        intervention_outcome_terms = []
        
        # Add interventions
        for intervention in self.interventions:
            intervention_outcome_terms.append(intervention.build_query_term(use_mesh=use_mesh))
        
        # Add outcomes
        for outcome in self.outcomes:
            intervention_outcome_terms.append(outcome.build_query_term(use_mesh=use_mesh))
        
        if intervention_outcome_terms:
            query_parts.append(f"({' OR '.join(intervention_outcome_terms)})")
        
        # Add study designs
        if self.study_designs:
            design_terms = []
            for design in self.study_designs:
                design_terms.append(design.build_query_term(use_mesh=use_mesh))
            
            if design_terms:
                query_parts.append(f"({' OR '.join(design_terms)})")
        
        # Add date range
        if self.date_range:
            date_term = self.date_range.build_query_term()
            if date_term:
                query_parts.append(date_term)
        
        # Add journal filter
        if self.journal_filter:
            journal_term = self.journal_filter.build_query_term()
            if journal_term:
                query_parts.append(journal_term)
        
        # Add publication type filter
        if self.publication_type_filter:
            pubtype_term = self.publication_type_filter.build_query_term()
            if pubtype_term:
                query_parts.append(pubtype_term)
        
        # Add language filter
        if self.language_filter:
            language_term = self.language_filter.build_query_term()
            if language_term:
                query_parts.append(language_term)
        
        # Add additional filters
        for filter_term in self.additional_filters:
            query_parts.append(f"({filter_term})")
        
        # Combine all parts with AND
        return " AND ".join(query_parts)
    
    @staticmethod
    def create_from_template(template_name: str) -> 'MedicalQueryBuilder':
        """
        Create a query builder from a template.
        
        Args:
            template_name: Name of the template
            
        Returns:
            MedicalQueryBuilder instance
        """
        builder = MedicalQueryBuilder()
        
        # COVID-19 treatment template
        if template_name.lower() == 'covid-19 treatment':
            builder.add_condition(
                MedicalCondition("COVID-19")
                .add_synonym("SARS-CoV-2")
                .add_synonym("2019-nCoV")
                .add_synonym("Coronavirus disease 2019")
            )
            
            builder.add_intervention(
                MedicalIntervention("antiviral therapy", "treatment")
                .add_alternative("remdesivir")
                .add_alternative("molnupiravir")
                .add_alternative("nirmatrelvir")
                .add_alternative("paxlovid")
            )
            
            builder.add_outcome(
                OutcomeMetric("mortality")
                .add_synonym("death")
                .add_related_metric("survival")
            )
            
            builder.add_study_design(
                StudyDesign("randomized controlled trial")
                .add_related_design("clinical trial")
                .add_related_design("systematic review")
                .add_related_design("meta-analysis")
            )
            
            builder.last_n_years(3)
            builder.english_only()
            builder.humans_only()
        
        # Diabetes management template
        elif template_name.lower() == 'diabetes management':
            builder.add_condition(
                MedicalCondition("Type 2 Diabetes Mellitus")
                .add_synonym("T2DM")
                .add_synonym("adult-onset diabetes")
            )
            
            builder.add_intervention(
                MedicalIntervention("glycemic control")
                .add_alternative("metformin")
                .add_alternative("insulin")
                .add_alternative("SGLT2 inhibitors")
                .add_alternative("GLP-1 agonists")
            )
            
            builder.add_outcome(
                OutcomeMetric("HbA1c")
                .add_synonym("glycated hemoglobin")
                .add_related_metric("blood glucose")
            )
            
            builder.add_study_design(
                StudyDesign("randomized controlled trial")
            )
            
            builder.last_n_years(5)
            builder.english_only()
        
        # Hypertension treatment in elderly template
        elif template_name.lower() == 'hypertension in elderly':
            builder.add_condition(
                MedicalCondition("hypertension")
                .add_synonym("high blood pressure")
            )
            
            builder.add_population(
                PopulationGroup("elderly")
                .add_age_group("aged")
                .add_age_group("older adult")
                .add_demographic("geriatric")
            )
            
            builder.add_intervention(
                MedicalIntervention("antihypertensive therapy")
                .add_alternative("ACE inhibitor")
                .add_alternative("angiotensin receptor blocker")
                .add_alternative("calcium channel blocker")
                .add_alternative("diuretic")
            )
            
            builder.add_outcome(
                OutcomeMetric("blood pressure")
                .add_related_metric("systolic pressure")
                .add_related_metric("diastolic pressure")
            )
            
            builder.add_outcome(
                OutcomeMetric("cardiovascular events")
                .add_related_metric("stroke")
                .add_related_metric("myocardial infarction")
            )
            
            builder.high_quality_only()
            builder.last_n_years(10)
            builder.english_only()
        
        # Community-acquired pneumonia template
        elif template_name.lower() == 'community-acquired pneumonia':
            builder.add_condition(
                MedicalCondition("community-acquired pneumonia")
                .add_synonym("CAP")
            )
            
            builder.add_intervention(
                MedicalIntervention("antibiotic therapy")
                .add_alternative("beta-lactam")
                .add_alternative("macrolide")
                .add_alternative("fluoroquinolone")
                .add_alternative("doxycycline")
            )
            
            builder.add_outcome(
                OutcomeMetric("treatment failure")
                .add_synonym("clinical failure")
                .add_related_metric("mortality")
            )
            
            builder.add_study_design(
                StudyDesign("randomized controlled trial")
                .add_related_design("systematic review")
            )
            
            builder.last_n_years(5)
            builder.english_only()
            builder.humans_only()
        
        # Default blank template
        else:
            logger.warning(f"Unknown template: {template_name}")
        
        return builder


class NCBIQueryInterface:
    """
    Interface between the MedicalQueryBuilder and NCBIClient.
    """
    
    def __init__(self, ncbi_client):
        """
        Initialize the query interface.
        
        Args:
            ncbi_client: NCBIClient instance
        """
        self.client = ncbi_client
        self.query_builder = MedicalQueryBuilder()
        self.mesh_mapper = MeshMapper()
    
    def new_query(self) -> MedicalQueryBuilder:
        """
        Create a new query builder.
        
        Returns:
            MedicalQueryBuilder instance
        """
        self.query_builder = MedicalQueryBuilder(self.mesh_mapper)
        return self.query_builder
    
    def from_template(self, template_name: str) -> MedicalQueryBuilder:
        """
        Create a query builder from a template.
        
        Args:
            template_name: Name of the template
            
        Returns:
            MedicalQueryBuilder instance
        """
        self.query_builder = MedicalQueryBuilder.create_from_template(template_name)
        return self.query_builder
    
    def search(self, 
              max_results: int = 20, 
              query_type: str = 'pico',
              use_mesh: bool = True) -> Dict[str, Any]:
        """
        Execute the search using the current query builder.
        
        Args:
            max_results: Maximum number of results to return
            query_type: Type of query to build ('pico' or 'simple')
            use_mesh: Whether to use MeSH terms if available
            
        Returns:
            Dictionary with search results
        """
        # Build the query
        if query_type.lower() == 'pico':
            query = self.query_builder.build_pico_query(use_mesh=use_mesh)
        else:
            query = self.query_builder.build_simple_query(use_mesh=use_mesh)
        
        # Execute the search
        search_results = self.client.search_pubmed(
            query, 
            max_results=max_results
        )
        
        if not search_results or 'esearchresult' not in search_results:
            return {
                'query': query,
                'total_count': 0,
                'results': []
            }
        
        # Extract count
        total_count = int(search_results['esearchresult'].get('count', 0))
        
        # Fetch abstracts if we have results
        results = []
        if total_count > 0 and 'idlist' in search_results['esearchresult']:
            id_list = search_results['esearchresult']['idlist']
            
            if id_list:
                # Limit to max_results
                id_list = id_list[:max_results]
                
                # Fetch abstracts
                abstracts = self.client.fetch_pubmed_abstracts(id_list=id_list)
                
                if abstracts:
                    results = abstracts
        
        return {
            'query': query,
            'total_count': total_count,
            'results': results
        }
    
    def search_and_fetch(self, 
                        max_results: int = 20, 
                        query_type: str = 'pico',
                        use_mesh: bool = True) -> List[Dict[str, Any]]:
        """
        Execute the search and return fetched abstracts.
        
        Args:
            max_results: Maximum number of results to return
            query_type: Type of query to build ('pico' or 'simple')
            use_mesh: Whether to use MeSH terms if available
            
        Returns:
            List of fetched abstracts
        """
        result = self.search(max_results, query_type, use_mesh)
        return result['results']
    
    def explain_query(self, query_type: str = 'pico', use_mesh: bool = True) -> Dict[str, Any]:
        """
        Explain the components of the built query.
        
        Args:
            query_type: Type of query to build ('pico' or 'simple')
            use_mesh: Whether to use MeSH terms if available
            
        Returns:
            Dictionary with query explanation
        """
        # Build the query
        if query_type.lower() == 'pico':
            query = self.query_builder.build_pico_query(use_mesh=use_mesh)
        else:
            query = self.query_builder.build_simple_query(use_mesh=use_mesh)
        
        # Explain query components
        components = {
            'conditions': [c.name for c in self.query_builder.conditions],
            'interventions': [i.name for i in self.query_builder.interventions],
            'outcomes': [o.name for o in self.query_builder.outcomes],
            'study_designs': [d.design_type for d in self.query_builder.study_designs],
            'populations': [p.description for p in self.query_builder.populations],
            'filters': []
        }
        
        # Add date range
        if self.query_builder.date_range:
            components['filters'].append(f"Date range: {self.query_builder.date_range.build_query_term()}")
        
        # Add journal filter
        if self.query_builder.journal_filter:
            journals = self.query_builder.journal_filter.journal_names
            if journals:
                components['filters'].append(f"Journals: {', '.join(journals)}")
        
        # Add publication type filter
        if self.query_builder.publication_type_filter:
            pub_types = self.query_builder.publication_type_filter.publication_types
            if pub_types:
                components['filters'].append(f"Publication types: {', '.join(pub_types)}")
        
        # Add language filter
        if self.query_builder.language_filter:
            languages = self.query_builder.language_filter.languages
            if languages:
                components['filters'].append(f"Languages: {', '.join(languages)}")
        
        # Add additional filters
        if self.query_builder.additional_filters:
            components['filters'].extend(self.query_builder.additional_filters)
        
        return {
            'query': query,
            'query_type': query_type,
            'components': components,
            'use_mesh': use_mesh
        }

# # Example usage
# if __name__ == "__main__":
#     # Example using the query builder to create a complex query
#     builder = MedicalQueryBuilder()
    
#     # Build a query about COVID-19 treatment
#     query = builder.add_condition(
#         MedicalCondition("COVID-19")
#         .add_synonym("SARS-CoV-2")
#         .add_synonym("2019-nCoV")
#     ).add_intervention(
#         MedicalIntervention("antiviral therapy")
#         .add_alternative("remdesivir")
#         .add_alternative("molnupiravir")
#     ).add_outcome(
#         OutcomeMetric("mortality")
#         .add_synonym("death")
#         .add_related_metric("survival")
#     ).add_study_design(
#         StudyDesign("randomized controlled trial")
#     ).last_n_years(2).english_only().humans_only().build_pico_query()
    
#     print(f"Generated query: {query}")
    
#     # Example using a template
#     template_query = MedicalQueryBuilder.create_from_template("covid-19 treatment").build_pico_query()
#     print(f"\nTemplate query: {template_query}")


# 1. Analyzing Contradictory Findings in Medical Literature
# The enhanced system can automatically detect, quantify, and analyze contradictory findings:
# # Initialize the integrated system
# synthesizer = EnhancedMedicalResearchSynthesizer(email="your.email@example.com")

# # Analyze contradictions in CAP treatment literature
# analysis = synthesizer.search_cap_contradictory_treatments()

# print(f"Found {analysis['num_contradictions']} contradictions in {analysis['total_articles']} articles")

# # Compare authorities of contradictory findings
# for intervention, contradictions in analysis['contradictions_by_intervention'].items():
#     if contradictions:
#         print(f"\n{intervention.capitalize()} treatment contradictions: {len(contradictions)}")
        
#         # Print which publications had higher authority
#         authority = analysis['authority_analysis'][intervention]
#         print(f"  Higher authority: Publication 1: {authority['publication1']}, " 
#               f"Publication 2: {authority['publication2']}, "
#               f"Equal: {authority['equal']}")    
        

# 2. Creating and Maintaining Up-to-Date Knowledge Bases
# The system enables automatic creation and updating of specialized knowledge bases:
# pythonCopy# Create a knowledge base for CAP treatment that updates weekly
# kb_info = synthesizer.create_and_update_knowledge_base(
#     name="cap_treatment",
#     query="community acquired pneumonia[Title/Abstract] AND treatment[Title/Abstract]",
#     schedule="weekly",
#     max_results=50
# )

# print(f"Created knowledge base '{kb_info['name']}' with {kb_info['initial_results']} initial articles")
# print(f"Updates scheduled: {kb_info['update_schedule']}")      

# 3. Building Complex Medical Queries Without PubMed Syntax Knowledge
# The query builder makes it easy to construct complex queries:
# pythonCopy# Create a query builder
# query_builder = synthesizer.create_query()

# # Build a query about COVID-19 treatment effects
# query_builder.add_condition(
#     MedicalCondition("COVID-19")
#     .add_synonym("SARS-CoV-2")
# ).add_intervention(
#     MedicalIntervention("remdesivir")
# ).add_outcome(
#     OutcomeMetric("mortality")
#     .add_synonym("death")
# ).add_study_design(
#     StudyDesign("randomized controlled trial")
# ).last_n_years(2).english_only()

# # Execute search with the constructed query
# enriched_results = synthesizer.search_and_enrich(query_builder=query_builder)