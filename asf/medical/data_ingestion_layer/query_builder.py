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
from typing import List, Optional, Set
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
        Add a synonym for the condition.
        Args:
            synonym: Synonym term
        Returns:
            Self for method chaining
        Add a subtype of the condition.
        Args:
            subtype: Subtype term
        Returns:
            Self for method chaining
        Build a query term for this condition.
        Args:
            use_mesh: Whether to use MeSH terms if available
            include_synonyms: Whether to include synonyms
            include_subtypes: Whether to include subtypes
        Returns:
            Query term string
    def __init__(self, 
                 name: str, 
                 intervention_type: str = 'treatment',
                 mesh_mapper: Optional[MeshMapper] = None):
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
        Add a synonym for the outcome metric.
        Args:
            synonym: Synonym term
        Returns:
            Self for method chaining
        Add a related outcome metric.
        Args:
            metric: Related metric term
        Returns:
            Self for method chaining
        Build a query term for this outcome metric.
        Args:
            use_mesh: Whether to use MeSH terms if available
            include_synonyms: Whether to include synonyms
            include_related: Whether to include related metrics
        Returns:
            Query term string
    def __init__(self, 
                 design_type: str, 
                 mesh_mapper: Optional[MeshMapper] = None):
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
        Add an age group to the population.
        Args:
            age_group: Age group term
        Returns:
            Self for method chaining
        Add a demographic characteristic to the population.
        Args:
            demographic: Demographic term
        Returns:
            Self for method chaining
        Add a condition to the population.
        Args:
            condition: Condition term
        Returns:
            Self for method chaining
        Build a query term for this population group.
        Args:
            use_mesh: Whether to use MeSH terms if available
            include_age_groups: Whether to include age groups
            include_demographics: Whether to include demographics
            include_conditions: Whether to include conditions
        Returns:
            Query term string
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
        Add a journal to the filter.
        Args:
            journal_name: Journal name
        Returns:
            Self for method chaining
        Add a journal ISSN to the filter.
        Args:
            issn: Journal ISSN
        Returns:
            Self for method chaining
        Set a minimum impact factor for journals.
        Note: This can't be directly queried in PubMed,
        so it will need post-processing of results.
        Args:
            min_factor: Minimum impact factor
        Returns:
            Self for method chaining
        Filter to only include open access journals.
        Args:
            value: Whether to only include open access journals
        Returns:
            Self for method chaining
        Build a query term for this journal filter.
        Returns:
            Query term string
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
    High-level query builder for medical literature searches.
        Initialize the query builder.
        Args:
            mesh_mapper: MeshMapper instance for MeSH term mapping
        Add a medical condition to the query.
        Args:
            condition: Medical condition string or object
        Returns:
            Self for method chaining
        Add a medical intervention to the query.
        Args:
            intervention: Medical intervention string or object
        Returns:
            Self for method chaining
        Add an outcome metric to the query.
        Args:
            outcome: Outcome metric string or object
        Returns:
            Self for method chaining
        Add a study design to the query.
        Args:
            design: Study design string or object
        Returns:
            Self for method chaining
        Add a population group to the query.
        Args:
            population: Population group string or object
        Returns:
            Self for method chaining
        Set a date range for the query.
        Args:
            date_range: DateRange object
        Returns:
            Self for method chaining
        Set the date range to the last N years.
        Args:
            n: Number of years
        Returns:
            Self for method chaining
        Set a journal filter for the query.
        Args:
            filter: JournalFilter object
        Returns:
            Self for method chaining
        Set a publication type filter for the query.
        Args:
            filter: PublicationTypeFilter object
        Returns:
            Self for method chaining
        Set a language filter for the query.
        Args:
            filter: LanguageFilter object
        Returns:
            Self for method chaining
        Filter to only include English articles.
        Returns:
            Self for method chaining
        Filter to only include high-quality evidence.
        Returns:
            Self for method chaining
        Add a custom filter term to the query.
        Args:
            filter_term: Custom filter term
        Returns:
            Self for method chaining
        Filter to only include studies on humans.
        Returns:
            Self for method chaining
        Build a query following the PICO format
        (Population, Intervention, Comparison, Outcome).
        Args:
            use_mesh: Whether to use MeSH terms if available
        Returns:
            PubMed query string
        Build a simple query combining main terms.
        Args:
            use_mesh: Whether to use MeSH terms if available
        Returns:
            PubMed query string
        Create a query builder from a template.
        Args:
            template_name: Name of the template
        Returns:
            MedicalQueryBuilder instance
    Interface between the MedicalQueryBuilder and NCBIClient.
        Initialize the query interface.
        Args:
            ncbi_client: NCBIClient instance
        Create a new query builder.
        Returns:
            MedicalQueryBuilder instance
        Create a query builder from a template.
        Args:
            template_name: Name of the template
        Returns:
            MedicalQueryBuilder instance
        Execute the search using the current query builder.
        Args:
            max_results: Maximum number of results to return
            query_type: Type of query to build ('pico' or 'simple')
            use_mesh: Whether to use MeSH terms if available
        Returns:
            Dictionary with search results
        Execute the search and return fetched abstracts.
        Args:
            max_results: Maximum number of results to return
            query_type: Type of query to build ('pico' or 'simple')
            use_mesh: Whether to use MeSH terms if available
        Returns:
            List of fetched abstracts
        Explain the components of the built query.
        Args:
            query_type: Type of query to build ('pico' or 'simple')
            use_mesh: Whether to use MeSH terms if available
        Returns:
            Dictionary with query explanation