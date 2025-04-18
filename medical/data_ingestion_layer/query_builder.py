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
from typing import List, Optional, Set, Union
from dataclasses import dataclass, field
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
        """Load MeSH data from file if available.
    Args:
        # TODO: Add parameter descriptions
    Returns:
        # TODO: Add return description
    """
        if os.path.exists(self.mesh_file):
            try:
                with open(self.mesh_file, 'r') as f:
                    self.mesh_data = json.load(f)
                logger.info(f"Loaded MeSH data with {len(self.mesh_data)} terms")
            except Exception as e:
                logger.error(f"Error loading MeSH data: {str(e)}")
    def save_mesh_data(self):
        """Save MeSH data to file.
    Args:
        # TODO: Add parameter descriptions
    Returns:
        # TODO: Add return description
    """
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
        term_lower = term.lower()
        if term_lower in self.mesh_data:
            mesh_info = self.mesh_data[term_lower]
            return MedicalTerm(
                term=term,
                mesh_terms=mesh_info.get('mesh_terms', []),
                exact_match=mesh_info.get('exact_match', False),
                is_mesh=mesh_info.get('is_mesh', False)
            )
        if self._is_mesh_term(term):
            result = MedicalTerm(
                term=term,
                mesh_terms=[term],
                exact_match=True,
                is_mesh=True
            )
            self.mesh_data[term_lower] = {
                'mesh_terms': [term],
                'exact_match': True,
                'is_mesh': True,
                'timestamp': datetime.datetime.now().timestamp()
            }
            self.save_mesh_data()
            return result
        mesh_terms = self._fetch_mesh_terms(term)
        result = MedicalTerm(
            term=term,
            mesh_terms=mesh_terms,
            exact_match=any(m.lower() == term_lower for m in mesh_terms),
            is_mesh=False
        )
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
        if '[mesh]' in term.lower() or '[mesh term]' in term.lower():
            return True
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
            url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/espell.fcgi?db=pubmed&term={term}&retmode=json"
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                corrected_term = data.get('esearchresult', {}).get('correctedquery')
                if corrected_term and corrected_term != term:
                    mesh_terms.append(corrected_term)
            url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term={term}&retmode=json&sort=relevance&retmax=1"
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                if 'esearchresult' in data and 'translationstack' in data['esearchresult']:
                    for item in data['esearchresult']['translationstack']:
                        if isinstance(item, dict) and 'term' in item:
                            term_str = item['term']
                            if '[mesh]' in term_str.lower() or '[mesh terms]' in term_str.lower():
                                mesh_term = term_str.split('[')[0].strip()
                                if mesh_term and mesh_term not in mesh_terms:
                                    mesh_terms.append(mesh_term)
        except Exception as e:
            logger.error(f"Error fetching MeSH terms for '{term}': {str(e)}")
        return mesh_terms
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
        design_type_lower = self.design_type.lower()
        
        if use_mesh and design_type_lower in self.design_pub_types:
            terms.append(self.design_pub_types[design_type_lower])
        elif use_mesh and self.mapped_term.mesh_terms:
            for mesh_term in self.mapped_term.mesh_terms:
                terms.append(f'"{mesh_term}"[Mesh]')
        
        terms.append(f'"{self.design_type}"[Title/Abstract]')
        
        if include_related and self.related_designs:
            for design in self.related_designs:
                design_lower = design.lower()
                if use_mesh and design_lower in self.design_pub_types:
                    terms.append(self.design_pub_types[design_lower])
                else:
                    terms.append(f'"{design}"[Title/Abstract]')
        
        if include_characteristics and self.design_characteristics:
            for characteristic in self.design_characteristics:
                terms.append(f'"{characteristic}"[Title/Abstract]')
        
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
            intervention_type: Type of intervention (treatment, drug, device, etc.)
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
        
        if use_mesh and self.mapped_term.mesh_terms:
            for mesh_term in self.mapped_term.mesh_terms:
                terms.append(f'"{mesh_term}"[Mesh]')
        
        terms.append(f'"{self.name}"[Title/Abstract]')
        
        if include_alternatives and self.alternatives:
            for alternative in self.alternatives:
                terms.append(f'"{alternative}"[Title/Abstract]')
        
        if include_specific_forms and self.specific_forms:
            for form in self.specific_forms:
                terms.append(f'"{form}"[Title/Abstract]')
        
        if self.intervention_type.lower() not in self.name.lower():
            terms.append(f'"{self.intervention_type}"[Title/Abstract]')
        
        return f"({' OR '.join(terms)})"

class DateRange:
    """Class to represent a date range with query building capabilities."""
    
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

class PublicationTypeFilter:
    """Class to represent a publication type filter."""
    
    def __init__(self, publication_types: Optional[List[str]] = None):
        """
        Initialize a publication type filter.
        
        Args:
            publication_types: List of publication types to include
        """
        self.publication_types = publication_types or []
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
                exclude_type = pub_type[4:].strip().lower()
                if exclude_type in self.pubtype_map:
                    exclude_terms.append(self.pubtype_map[exclude_type])
                else:
                    exclude_terms.append(f'"{exclude_type}"[Publication Type]')
            else:
                include_type = pub_type.lower()
                if include_type in self.pubtype_map:
                    include_terms.append(self.pubtype_map[include_type])
                else:
                    include_terms.append(f'"{pub_type}"[Publication Type]')
        query_parts = []
        if include_terms:
            query_parts.append(f"({' OR '.join(include_terms)})")
        for term in exclude_terms:
            query_parts.append(f"NOT {term}")
        return f"({' AND '.join(query_parts)})"
class LanguageFilter:
    """Class to represent a language filter."""
    def __init__(self, languages: Optional[List[str]] = None):
        """
        Initialize a language filter.
        Args:
            languages: List of languages to include
            
        Add a language to the filter.
        Args:
            language: Language name
        Returns:
            Self for method chaining
            
        Filter to only include English articles.
        Returns:
            Self for method chaining
            
        Build a query term for this language filter.
        Returns:
            Query term string
        """
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
        
        # Add MeSH terms if available and requested
        if use_mesh and self.mapped_term.mesh_terms:
            for mesh_term in self.mapped_term.mesh_terms:
                terms.append(f'"{mesh_term}"[Mesh]')
        
        # Always include the main term
        terms.append(f'"{self.name}"[Title/Abstract]')
        
        # Add synonyms if requested
        if include_synonyms and self.synonyms:
            for synonym in self.synonyms:
                terms.append(f'"{synonym}"[Title/Abstract]')
        
        # Add subtypes if requested
        if include_subtypes and self.subtypes:
            for subtype in self.subtypes:
                terms.append(f'"{subtype}"[Title/Abstract]')
        
        # Combine all terms with OR
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
        
        # Add MeSH terms if available and requested
        if use_mesh and self.mapped_term.mesh_terms:
            for mesh_term in self.mapped_term.mesh_terms:
                terms.append(f'"{mesh_term}"[Mesh]')
        
        # Always include the main term
        terms.append(f'"{self.name}"[Title/Abstract]')
        
        # Add synonyms if requested
        if include_synonyms and self.synonyms:
            for synonym in self.synonyms:
                terms.append(f'"{synonym}"[Title/Abstract]')
        
        # Add related metrics if requested
        if include_related and self.related_metrics:
            for metric in self.related_metrics:
                terms.append(f'"{metric}"[Title/Abstract]')
        
        # Include outcome type if not already in the name
        if self.outcome_type.lower() not in self.name.lower():
            terms.append(f'"{self.outcome_type}"[Title/Abstract]')
        
        # Combine all terms with OR
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
        
        # Add MeSH terms if available and requested
        if use_mesh and self.mapped_term.mesh_terms:
            for mesh_term in self.mapped_term.mesh_terms:
                terms.append(f'"{mesh_term}"[Mesh]')
        
        # Always include the main description
        terms.append(f'"{self.description}"[Title/Abstract]')
        
        # Add age groups if requested
        if include_age_groups and self.age_groups:
            for age_group in self.age_groups:
                terms.append(f'"{age_group}"[Title/Abstract]')
        
        # Add demographics if requested
        if include_demographics and self.demographics:
            for demographic in self.demographics:
                terms.append(f'"{demographic}"[Title/Abstract]')
        
        # Add conditions if requested
        if include_conditions and self.conditions:
            for condition in self.conditions:
                terms.append(f'"{condition}"[Title/Abstract]')
        
        # Combine all terms with OR
        return f"({' OR '.join(terms)})"

class MedicalQueryBuilder:
    """
    High-level query builder for medical literature searches.
    
    This class provides a fluent interface for building structured medical queries
    following formats like PICO (Population, Intervention, Comparison, Outcome).
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
        self.population = None
        self.study_design = None
        self.date_range = None
        self.journal_filter = None
        self.publication_type_filter = None
        self.language_filter = None
        self.custom_filters = []
        self.humans_only = False
    
    def add_condition(self, condition: Union[str, MedicalCondition]) -> 'MedicalQueryBuilder':
        """
        Add a medical condition to the query.
        
        Args:
            condition: Medical condition string or object
        
        Returns:
            Self for method chaining
        """
        if isinstance(condition, str):
            condition = MedicalCondition(condition, mesh_mapper=self.mesh_mapper)
        self.conditions.append(condition)
        return self
    
    def add_intervention(self, intervention: Union[str, 'MedicalIntervention']) -> 'MedicalQueryBuilder':
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
    
    def set_study_design(self, design: Union[str, StudyDesign]) -> 'MedicalQueryBuilder':
        """
        Add a study design to the query.
        
        Args:
            design: Study design string or object
        
        Returns:
            Self for method chaining
        """
        if isinstance(design, str):
            design = StudyDesign(design, mesh_mapper=self.mesh_mapper)
        self.study_design = design
        return self
    
    def set_population(self, population: Union[str, 'PopulationGroup']) -> 'MedicalQueryBuilder':
        """
        Add a population group to the query.
        
        Args:
            population: Population group string or object
        
        Returns:
            Self for method chaining
        """
        if isinstance(population, str):
            # Implement PopulationGroup if needed
            self.population = population
        else:
            self.population = population
        return self
    
    def set_date_range(self, date_range: 'DateRange') -> 'MedicalQueryBuilder':
        """
        Set a date range for the query.
        
        Args:
            date_range: DateRange object
        
        Returns:
            Self for method chaining
        """
        self.date_range = date_range
        return self
    
    def set_years(self, n: int) -> 'MedicalQueryBuilder':
        """
        Set the date range to the last N years.
        
        Args:
            n: Number of years
        
        Returns:
            Self for method chaining
        """
        date_range = DateRange()
        date_range.last_n_years(n)
        self.date_range = date_range
        return self
    
    def set_journal_filter(self, filter: 'JournalFilter') -> 'MedicalQueryBuilder':
        """
        Set a journal filter for the query.
        
        Args:
            filter: JournalFilter object
        
        Returns:
            Self for method chaining
        """
        self.journal_filter = filter
        return self
    
    def set_publication_type_filter(self, filter: 'PublicationTypeFilter') -> 'MedicalQueryBuilder':
        """
        Set a publication type filter for the query.
        
        Args:
            filter: PublicationTypeFilter object
        
        Returns:
            Self for method chaining
        """
        self.publication_type_filter = filter
        return self
    
    def set_language_filter(self, filter: 'LanguageFilter') -> 'MedicalQueryBuilder':
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
        language_filter = LanguageFilter(["english"])
        self.language_filter = language_filter
        return self
    
    def high_quality_evidence_only(self) -> 'MedicalQueryBuilder':
        """
        Filter to only include high-quality evidence.
        
        Returns:
            Self for method chaining
        """
        pub_type_filter = PublicationTypeFilter()
        pub_type_filter.high_quality_evidence_only()
        self.publication_type_filter = pub_type_filter
        return self
    
    def add_filter(self, filter_term: str) -> 'MedicalQueryBuilder':
        """
        Add a custom filter term to the query.
        
        Args:
            filter_term: Custom filter term
        
        Returns:
            Self for method chaining
        """
        self.custom_filters.append(filter_term)
        return self
    
    def humans_filter(self) -> 'MedicalQueryBuilder':
        """
        Filter to only include studies on humans.
        
        Returns:
            Self for method chaining
        """
        self.humans_only = True
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
        
        # Add conditions (P - Problem/Population)
        if self.conditions:
            condition_parts = []
            for condition in self.conditions:
                condition_parts.append(condition.build_query_term(use_mesh=use_mesh))
            if condition_parts:
                query_parts.append(" OR ".join(condition_parts))
        
        # Add population specifics if provided
        if self.population:
            if isinstance(self.population, str):
                query_parts.append(f'"{self.population}"[Title/Abstract]')
            else:
                query_parts.append(self.population.build_query_term(use_mesh=use_mesh))
        
        # Add interventions (I - Intervention)
        if self.interventions:
            intervention_parts = []
            for intervention in self.interventions:
                intervention_parts.append(intervention.build_query_term(use_mesh=use_mesh))
            if intervention_parts:
                query_parts.append(" OR ".join(intervention_parts))
        
        # Add outcomes (O - Outcome)
        if self.outcomes:
            outcome_parts = []
            for outcome in self.outcomes:
                outcome_parts.append(outcome.build_query_term(use_mesh=use_mesh))
            if outcome_parts:
                query_parts.append(" OR ".join(outcome_parts))
        
        # Add study design (S - Study Design) if available
        if self.study_design:
            query_parts.append(self.study_design.build_query_term(use_mesh=use_mesh))
        
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
            pub_type_term = self.publication_type_filter.build_query_term()
            if pub_type_term:
                query_parts.append(pub_type_term)
        
        # Add language filter
        if self.language_filter:
            language_term = self.language_filter.build_query_term()
            if language_term:
                query_parts.append(language_term)
        
        # Add human filter
        if self.humans_only:
            query_parts.append("humans[MeSH Terms]")
        
        # Add custom filters
        for filter_term in self.custom_filters:
            query_parts.append(filter_term)
        
        # Combine all parts with AND
        query = " AND ".join([f"({part})" for part in query_parts if part])
        
        return query
    
    def build_simple_query(self, use_mesh: bool = True) -> str:
        """
        Build a simple query combining main terms.
        
        Args:
            use_mesh: Whether to use MeSH terms if available
        
        Returns:
            PubMed query string
        """
        # For simple query we just combine all main terms
        all_terms = []
        
        # Add conditions
        for condition in self.conditions:
            all_terms.append(condition.build_query_term(use_mesh=use_mesh))
        
        # Add interventions
        for intervention in self.interventions:
            all_terms.append(intervention.build_query_term(use_mesh=use_mesh))
        
        # Add outcomes
        for outcome in self.outcomes:
            all_terms.append(outcome.build_query_term(use_mesh=use_mesh))
        
        # Combine main terms with OR for broader search
        main_query = " OR ".join([f"({term})" for term in all_terms if term])
        
        # Constraints are combined with AND
        constraints = []
        
        # Add study design if available
        if self.study_design:
            constraints.append(self.study_design.build_query_term(use_mesh=use_mesh))
        
        # Add date range
        if self.date_range:
            date_term = self.date_range.build_query_term()
            if date_term:
                constraints.append(date_term)
        
        # Add human filter
        if self.humans_only:
            constraints.append("humans[MeSH Terms]")
        
        # Combine main query with constraints
        if constraints:
            query = f"({main_query}) AND " + " AND ".join([f"({constraint})" for constraint in constraints if constraint])
        else:
            query = main_query
        
        return query
    
    def build(self, query_type: str = "pico", use_mesh: bool = True) -> str:
        """
        Build the query string.
        
        Args:
            query_type: Type of query to build ('pico' or 'simple')
            use_mesh: Whether to use MeSH terms if available
        
        Returns:
            PubMed query string
        """
        if query_type.lower() == "pico":
            return self.build_pico_query(use_mesh=use_mesh)
        else:
            return self.build_simple_query(use_mesh=use_mesh)
    
    @classmethod
    def from_template(cls, template_name: str) -> 'MedicalQueryBuilder':
        """
        Create a query builder from a template.
        
        Args:
            template_name: Name of the template
        
        Returns:
            MedicalQueryBuilder instance
        """
        builder = cls()
        
        # Apply template settings
        if template_name.lower() == "clinical_trial":
            builder.set_publication_type_filter(PublicationTypeFilter(["clinical trial"]))
            builder.set_years(5)
        elif template_name.lower() == "high_quality":
            builder.high_quality_evidence_only()
            builder.set_years(10)
        elif template_name.lower() == "recent":
            builder.set_years(2)
        elif template_name.lower() == "comprehensive":
            builder.set_years(20)
        
        return builder