Metadata Extraction Module for Medical Research Synthesizer

This module enhances the NCBIClient with specialized functions to extract,
standardize, and enrich publication metadata from PubMed and other sources.

import os
import json
import datetime
import pandas as pd
import requests
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('metadata_extraction')

class PublicationMetadataExtractor:
    Extract and standardize metadata from scientific publications.
    Can be used to enhance publication data retrieved from NCBI.

    def __init__(self,
                 impact_factor_source: Optional[str] = None,
                     """
                     __init__ function.
                     
                     This function provides functionality for...
                     Args:
                         impact_factor_source: Description of impact_factor_source
                         crossref_email: Description of crossref_email
                         cache_dir: Description of cache_dir
                     """
                 crossref_email: Optional[str] = None,
                 cache_dir: Optional[str] = None):
        self.crossref_email = crossref_email

        self.cache_dir = cache_dir or os.path.join(os.path.expanduser("~"), ".pubmed_cache")
        os.makedirs(self.cache_dir, exist_ok=True)

        self.impact_factors = {}
        if impact_factor_source and os.path.exists(impact_factor_source):
            try:
                if impact_factor_source.endswith('.csv'):
                    df = pd.read_csv(impact_factor_source)
                    for _, row in df.iterrows():
                        if 'Journal' in row and 'Impact_Factor' in row:
                            self.impact_factors[row['Journal'].lower()] = float(row['Impact_Factor'])
                        if 'ISSN' in row and 'Impact_Factor' in row and pd.notna(row['ISSN']):
                            self.impact_factors[row['ISSN']] = float(row['Impact_Factor'])
                        if 'eISSN' in row and 'Impact_Factor' in row and pd.notna(row['eISSN']):
                            self.impact_factors[row['eISSN']] = float(row['Impact_Factor'])
                elif impact_factor_source.endswith('.json'):
                    with open(impact_factor_source, 'r') as f:
                        self.impact_factors = json.load(f)
                logger.info(f"Loaded impact factors for {len(self.impact_factors)} journals")
            except Exception as e:
                logger.error(f"Error loading impact factor data: {str(e)}")

        self.citation_cache_file = os.path.join(self.cache_dir, "citation_cache.json")
        self.metadata_cache_file = os.path.join(self.cache_dir, "metadata_cache.json")

        self.citation_cache = {}
        self.metadata_cache = {}

        self._load_caches()

    def _load_caches(self):
        """Load citation and metadata caches from disk if they exist.

    Args:
        # TODO: Add parameter descriptions

    Returns:
        # TODO: Add return description
    """
        try:
            if os.path.exists(self.citation_cache_file):
                with open(self.citation_cache_file, 'r') as f:
                    self.citation_cache = json.load(f)
                logger.info(f"Loaded citation cache with {len(self.citation_cache)} entries")

            if os.path.exists(self.metadata_cache_file):
                with open(self.metadata_cache_file, 'r') as f:
                    self.metadata_cache = json.load(f)
                logger.info(f"Loaded metadata cache with {len(self.metadata_cache)} entries")
        except Exception as e:
            logger.error(f"Error loading caches: {str(e)}")

    def _save_caches(self):
        """Save citation and metadata caches to disk.

    Args:
        # TODO: Add parameter descriptions

    Returns:
        # TODO: Add return description
    """
        try:
            with open(self.citation_cache_file, 'w') as f:
                json.dump(self.citation_cache, f)

            with open(self.metadata_cache_file, 'w') as f:
                json.dump(self.metadata_cache, f)
        except Exception as e:
            logger.error(f"Error saving caches: {str(e)}")

    def standardize_publication_date(self, pub_date: Optional[str]) -> Optional[Dict[str, int]]:
        """
        Standardize publication date to a common format.

        Args:
            pub_date: Publication date in various formats

        Returns:
            Dict with standardized 'year', 'month', 'day' if parseable, None otherwise
        """
        if not pub_date:
            return None

        try:
            if '-' in pub_date:
                parts = pub_date.split('-')
                result = {'year': int(parts[0])}
                if len(parts) > 1:
                    month_part = parts[1].strip()
                    month_map = {
                        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
                        'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
                    }

                    if month_part.lower() in month_map:
                        result['month'] = month_map[month_part.lower()]
                    else:
                        try:
                            result['month'] = int(month_part)
                        except ValueError:
                            pass

                if len(parts) > 2:
                    try:
                        result['day'] = int(parts[2].strip())
                    except ValueError:
                        pass

                return result

            elif len(pub_date.strip()) == 4 and pub_date.strip().isdigit():
                return {'year': int(pub_date.strip())}

            else:
                for fmt in ['%Y %b %d', '%Y-%m-%d', '%b %Y', '%Y %b', '%d %b %Y']:
                    try:
                        dt = datetime.datetime.strptime(pub_date, fmt)
                        result = {'year': dt.year}
                        if dt.month:
                            result['month'] = dt.month
                        if dt.day:
                            result['day'] = dt.day
                        return result
                    except ValueError:
                        continue

                import re
                year_match = re.search(r'\b(19|20)\d{2}\b', pub_date)
                if year_match:
                    return {'year': int(year_match.group(0))}

                return None
        except Exception as e:
            logger.error(f"Error standardizing date '{pub_date}': {str(e)}")
            return None

    def format_standardized_date(self, date_dict: Dict[str, int], format_type: str = 'iso') -> str:
        """
        Format a standardized date dictionary into a string.

        Args:
            date_dict: Dictionary with year, month, day
            format_type: 'iso' for YYYY-MM-DD, 'human' for more readable format

        Returns:
            Formatted date string
        """
        if not date_dict or 'year' not in date_dict:
            return "Unknown date"

        if format_type == 'iso':
            if 'month' in date_dict and 'day' in date_dict:
                return f"{date_dict['year']:04d}-{date_dict['month']:02d}-{date_dict['day']:02d}"
            elif 'month' in date_dict:
                return f"{date_dict['year']:04d}-{date_dict['month']:02d}"
            else:
                return f"{date_dict['year']:04d}"
        elif format_type == 'human':
            month_names = [
                'January', 'February', 'March', 'April', 'May', 'June',
                'July', 'August', 'September', 'October', 'November', 'December'
            ]

            if 'month' in date_dict and 'day' in date_dict:
                return f"{month_names[date_dict['month']-1]} {date_dict['day']}, {date_dict['year']}"
            elif 'month' in date_dict:
                return f"{month_names[date_dict['month']-1]} {date_dict['year']}"
            else:
                return f"{date_dict['year']}"
        else:
            return str(date_dict['year'])

    def get_impact_factor(self, journal_title: Optional[str] = None, issn: Optional[str] = None) -> Optional[float]:
        """
        Get the impact factor for a journal.

        Args:
            journal_title: Journal title
            issn: Journal ISSN

        Returns:
            Impact factor if available, None otherwise
        """
        if not self.impact_factors:
            return None

        if issn and issn in self.impact_factors:
            return self.impact_factors[issn]

        if journal_title:
            if journal_title.lower() in self.impact_factors:
                return self.impact_factors[journal_title.lower()]

            journal_lower = journal_title.lower()
            for known_journal in self.impact_factors:
                if isinstance(known_journal, str) and journal_lower in known_journal.lower() or known_journal.lower() in journal_lower:
                    return self.impact_factors[known_journal]

        return None

    def get_journal_quartile(self, impact_factor: float, field: str = 'medicine') -> Optional[int]:
        """
        Estimate the journal quartile (Q1-Q4) based on impact factor and field.
        This is a simplified approximation and may not match official quartiles.

        Args:
            impact_factor: Journal impact factor
            field: Research field (medicine, biology, etc.)

        Returns:
            Estimated quartile (1-4) or None if can't be determined
        """
        if not impact_factor:
            return None

        thresholds = {
            'medicine': [5.0, 3.0, 1.5],  # Q1: >5.0, Q2: >3.0, Q3: >1.5, Q4: ≤1.5
            'biology': [4.0, 2.5, 1.0],
            'physics': [3.5, 2.0, 1.0],
            'chemistry': [4.5, 3.0, 1.5],
            'default': [4.0, 2.5, 1.0]
        }

        field_thresholds = thresholds.get(field.lower(), thresholds['default'])

        if impact_factor > field_thresholds[0]:
            return 1  # Q1
        elif impact_factor > field_thresholds[1]:
            return 2  # Q2
        elif impact_factor > field_thresholds[2]:
            return 3  # Q3
        else:
            return 4  # Q4

    def get_citation_count(self, pmid: str, refresh: bool = False) -> Optional[int]:
        """
        Get the citation count for a publication.
        Uses cached data if available unless refresh is True.

        Args:
            pmid: PubMed ID
            refresh: Whether to refresh the cache

        Returns:
            Citation count if available, None otherwise
        """
        if not refresh and pmid in self.citation_cache:
            cache_time = self.citation_cache[pmid].get('timestamp', 0)
            if (datetime.datetime.now().timestamp() - cache_time) < 30 * 24 * 60 * 60:
                return self.citation_cache[pmid].get('count')

        citation_count = self._get_pmc_citations(pmid)

        if citation_count is None:
            citation_count = self._get_crossref_citations(pmid)

        if citation_count is not None:
            self.citation_cache[pmid] = {
                'count': citation_count,
                'timestamp': datetime.datetime.now().timestamp()
            }
            self._save_caches()

        return citation_count

    def _get_pmc_citations(self, pmid: str) -> Optional[int]:
        """
        Get citation count from PubMed Central.

        Args:
            pmid: PubMed ID

        Returns:
            Citation count if available, None otherwise
        """
        try:
            url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi?dbfrom=pubmed&db=pubmed&id={pmid}&linkname=pubmed_pubmed_citedin&retmode=json"
            response = requests.get(url)
            response.raise_for_status()

            data = response.json()

            for linkset in data.get('linksets', []):
                if 'linksetdbs' in linkset:
                    for linksetdb in linkset['linksetdbs']:
                        if linksetdb.get('linkname') == 'pubmed_pubmed_citedin':
                            return len(linksetdb.get('links', []))

            return 0  # No citations found
        except Exception as e:
            logger.error(f"Error getting PMC citations for {pmid}: {str(e)}")
            return None

    def _get_crossref_citations(self, pmid: str) -> Optional[int]:
        """
        Get citation count from CrossRef.

        Args:
            pmid: PubMed ID

        Returns:
            Citation count if available, None otherwise
        """
        try:
            doi = self._get_doi_from_pmid(pmid)
            if not doi:
                return None

            url = f"https://api.crossref.org/works/{doi}"
            headers = {}
            if self.crossref_email:
                headers['User-Agent'] = f"PublicationMetadataExtractor/1.0 (mailto:{self.crossref_email})"

            response = requests.get(url, headers=headers)
            response.raise_for_status()

            data = response.json()

            return data.get('message', {}).get('is-referenced-by-count')
        except Exception as e:
            logger.error(f"Error getting CrossRef citations for {pmid}: {str(e)}")
            return None

    def _get_doi_from_pmid(self, pmid: str) -> Optional[str]:
        """
        Get DOI for a PubMed ID.

        Args:
            pmid: PubMed ID

        Returns:
            DOI if available, None otherwise
        """
        try:
            url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pubmed&id={pmid}&retmode=json"
            response = requests.get(url)
            response.raise_for_status()

            data = response.json()

            article_ids = data.get('result', {}).get(pmid, {}).get('articleids', [])
            for article_id in article_ids:
                if article_id.get('idtype') == 'doi':
                    return article_id.get('value')

            return None
        except Exception as e:
            logger.error(f"Error getting DOI for {pmid}: {str(e)}")
            return None

    def extract_publication_type(self, pubmed_article: Dict[str, Any]) -> List[str]:
        """
        Extract the publication type (e.g., clinical trial, review, etc.).

        Args:
            pubmed_article: Dictionary containing PubMed article data

        Returns:
            List of publication types
        """
        pub_types = []

        if isinstance(pubmed_article, dict):
            if 'publication_types' in pubmed_article:
                return pubmed_article['publication_types']

            if 'mesh_terms' in pubmed_article:
                mesh_terms = pubmed_article['mesh_terms']
                pub_type_keywords = [
                    "Clinical Trial", "Review", "Meta-Analysis", "Randomized Controlled Trial",
                    "Practice Guideline", "Guideline", "Systematic Review", "Case Reports",
                    "Comparative Study", "Journal Article", "Observational Study", "Editorial"
                ]

                for term in mesh_terms:
                    if any(keyword.lower() in term.lower() for keyword in pub_type_keywords):
                        pub_types.append(term)

        if not pub_types:
            pub_types.append("Journal Article")

        return pub_types

    def extract_author_affiliations(self, pubmed_article: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Extract author affiliations from PubMed article data.

        Args:
            pubmed_article: Dictionary containing PubMed article data

        Returns:
            Dictionary mapping author names to lists of affiliations
        """
        affiliations = {}

        if 'authors' in pubmed_article and isinstance(pubmed_article['authors'], list):
            for author in pubmed_article['authors']:
                if isinstance(author, str):
                    affiliations[author] = []
                elif isinstance(author, dict) and 'name' in author:
                    affiliations[author['name']] = author.get('affiliations', [])

        return affiliations

    def categorize_journal(self, journal_title: str) -> List[str]:
        """
        Categorize a journal into research fields.

        Args:
            journal_title: Journal title

        Returns:
            List of research fields
        """
        medical_keywords = [
            'medicine', 'medical', 'health', 'clinical', 'hospital',
            'therapy', 'disease', 'patient', 'nursing', 'surgery'
        ]

        biology_keywords = [
            'biology', 'cell', 'molecular', 'gene', 'organism',
            'ecology', 'evolution', 'microbiology', 'bioscience'
        ]

        chemistry_keywords = [
            'chemistry', 'chemical', 'biochemistry', 'molecule',
            'compound', 'reaction', 'synthesis'
        ]

        categories = []
        journal_lower = journal_title.lower()

        if any(keyword in journal_lower for keyword in medical_keywords):
            categories.append('Medicine')

        if any(keyword in journal_lower for keyword in biology_keywords):
            categories.append('Biology')

        if any(keyword in journal_lower for keyword in chemistry_keywords):
            categories.append('Chemistry')

        if not categories:
            categories.append('Science')

        return categories

    def extract_funding_info(self, pubmed_article: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Extract funding information from PubMed article data.

        Args:
            pubmed_article: Dictionary containing PubMed article data

        Returns:
            List of funding information dictionaries
        """
        funding_info = []

        if 'funding' in pubmed_article:
            return pubmed_article['funding']

        if 'abstract' in pubmed_article and pubmed_article['abstract']:
            abstract = pubmed_article['abstract'].lower()
            funding_keywords = [
                'funded by', 'supported by', 'grant from',
                'financial support', 'funding was provided'
            ]

            for keyword in funding_keywords:
                if keyword in abstract:
                    sentences = abstract.split('.')
                    for sentence in sentences:
                        if keyword in sentence:
                            funding_info.append({
                                'source': 'abstract',
                                'description': sentence.strip()
                            })

        return funding_info

    def enrich_publication_metadata(self, pubmed_article: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enrich publication metadata with additional information.

        Args:
            pubmed_article: Dictionary containing PubMed article data

        Returns:
            Enriched publication metadata
        """
        pmid = pubmed_article.get('pmid')

        if pmid and pmid in self.metadata_cache:
            cache_time = self.metadata_cache[pmid].get('cache_timestamp', 0)
            if (datetime.datetime.now().timestamp() - cache_time) < 7 * 24 * 60 * 60:
                return self.metadata_cache[pmid]

        enriched = dict(pubmed_article)

        if 'publication_date' in pubmed_article:
            date_dict = self.standardize_publication_date(pubmed_article['publication_date'])
            if date_dict:
                enriched['standard_date'] = date_dict
                enriched['iso_date'] = self.format_standardized_date(date_dict, 'iso')
                enriched['human_date'] = self.format_standardized_date(date_dict, 'human')

        if 'journal' in pubmed_article:
            impact_factor = self.get_impact_factor(pubmed_article['journal'])
            enriched['impact_factor'] = impact_factor

            if impact_factor:
                quartile = self.get_journal_quartile(impact_factor, 'medicine')
                enriched['journal_quartile'] = quartile

        if pmid:
            enriched['citation_count'] = self.get_citation_count(pmid)

        enriched['publication_types'] = self.extract_publication_type(pubmed_article)

        enriched['author_affiliations'] = self.extract_author_affiliations(pubmed_article)

        if 'journal' in pubmed_article:
            enriched['journal_categories'] = self.categorize_journal(pubmed_article['journal'])

        enriched['funding_info'] = self.extract_funding_info(pubmed_article)

        authority_score = self._calculate_authority_score(enriched)
        enriched['authority_score'] = authority_score

        enriched['cache_timestamp'] = datetime.datetime.now().timestamp()

        if pmid:
            self.metadata_cache[pmid] = enriched
            self._save_caches()

        return enriched

    def _calculate_authority_score(self, publication: Dict[str, Any]) -> int:
        """
        Calculate an authority score (0-100) for a publication.

        Args:
            publication: Publication metadata

        Returns:
            Authority score (0-100)
        """
        score = 50  # Start with a neutral score

        impact_factor = publication.get('impact_factor')
        if impact_factor:
            if impact_factor > 10:
                score += 30
            elif impact_factor > 5:
                score += 20
            elif impact_factor > 3:
                score += 15
            elif impact_factor > 1:
                score += 10
            else:
                score += 5

        quartile = publication.get('journal_quartile')
        if quartile:
            if quartile == 1:
                score += 10
            elif quartile == 2:
                score += 7
            elif quartile == 3:
                score += 3
            else:  # Q4
                score += 1

        citations = publication.get('citation_count')
        if citations:
            if citations > 100:
                score += 20
            elif citations > 50:
                score += 15
            elif citations > 20:
                score += 10
            elif citations > 5:
                score += 5
            else:
                score += 2

        pub_types = publication.get('publication_types', [])
        pub_type_scores = {
            'systematic review': 20,
            'meta-analysis': 20,
            'randomized controlled trial': 18,
            'clinical trial': 15,
            'review': 12,
            'guideline': 15,
            'practice guideline': 15,
            'comparative study': 10,
            'journal article': 5,
            'case report': 3,
            'editorial': 2
        }

        max_pub_type_score = 0
        for pub_type in pub_types:
            for key, value in pub_type_scores.items():
                if key in pub_type.lower():
                    max_pub_type_score = max(max_pub_type_score, value)

        score += max_pub_type_score

        if 'standard_date' in publication and 'year' in publication['standard_date']:
            year = publication['standard_date']['year']
            current_year = datetime.datetime.now().year
            if year >= current_year - 1:
                score += 10  # Very recent (0-1 years)
            elif year >= current_year - 3:
                score += 7   # Recent (1-3 years)
            elif year >= current_year - 5:
                score += 5   # Somewhat recent (3-5 years)
            elif year >= current_year - 10:
                score += 2   # Older (5-10 years)
            else:
                score -= 5   # Very old (>10 years)

        return max(0, min(100, score))

    def batch_enrich_publications(self, pubmed_articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Enrich multiple publications with additional metadata.

        Args:
            pubmed_articles: List of publication metadata dictionaries

        Returns:
            List of enriched publication metadata dictionaries
        """
        return [self.enrich_publication_metadata(article) for article in pubmed_articles]

    def compare_publication_authority(self, pub1: Dict[str, Any], pub2: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare the authority of two publications.

        Args:
            pub1: First publication metadata
            pub2: Second publication metadata

        Returns:
            Dictionary with comparison results
        """
        if 'authority_score' not in pub1:
            pub1 = self.enrich_publication_metadata(pub1)
        if 'authority_score' not in pub2:
            pub2 = self.enrich_publication_metadata(pub2)

        authority_diff = pub1['authority_score'] - pub2['authority_score']

        higher_authority = "equal"
        if authority_diff > 0:
            higher_authority = "publication1"
        elif authority_diff < 0:
            higher_authority = "publication2"

        factor_comparison = {}

        for factor in ['impact_factor', 'journal_quartile', 'citation_count']:
            if factor in pub1 and factor in pub2 and pub1[factor] is not None and pub2[factor] is not None:
                factor_comparison[factor] = pub1[factor] - pub2[factor]

        pub1_types = set([pt.lower() for pt in pub1.get('publication_types', [])])
        pub2_types = set([pt.lower() for pt in pub2.get('publication_types', [])])

        evidence_hierarchy = [
            'meta-analysis', 'systematic review', 'randomized controlled trial',
            'clinical trial', 'cohort study', 'case-control study', 'case series',
            'case report', 'expert opinion'
        ]

        pub1_highest_evidence = None
        pub2_highest_evidence = None

        for evidence_type in evidence_hierarchy:
            if any(evidence_type in pt for pt in pub1_types) and not pub1_highest_evidence:
                pub1_highest_evidence = evidence_type
            if any(evidence_type in pt for pt in pub2_types) and not pub2_highest_evidence:
                pub2_highest_evidence = evidence_type

        factor_comparison['highest_evidence_level'] = {
            'publication1': pub1_highest_evidence,
            'publication2': pub2_highest_evidence
        }

        pub1_year = pub1.get('standard_date', {}).get('year')
        pub2_year = pub2.get('standard_date', {}).get('year')

        if pub1_year and pub2_year:
            factor_comparison['publication_year_difference'] = pub1_year - pub2_year

        return {
            'authority_difference': authority_diff,
            'higher_authority': higher_authority,
            'factor_comparison': factor_comparison,
            'publication1_score': pub1['authority_score'],
            'publication2_score': pub2['authority_score']
        }

    def find_contradictions(self, publications: List[Dict[str, Any]], threshold: float = 0.7, use_biomedlm: bool = True) -> List[Dict[str, Any]]:
        """
        Find potentially contradictory findings among publications.
        Uses BioMedLM for contradiction scoring if available, with fallback to keyword-based approach.

        Args:
            publications: List of publication metadata dictionaries
            threshold: Contradiction score threshold above which findings are considered contradictory
            use_biomedlm: Whether to use BioMedLM for contradiction scoring

        Returns:
            List of dictionaries describing contradictions
        """
        contradictions = []
        biomedlm_scorer = None

        if use_biomedlm:
            try:
                from asf.medical.models.biomedlm_wrapper import BioMedLMScorer
                biomedlm_scorer = BioMedLMScorer(use_negation_detection=True)
                logger.info("Using BioMedLM with negation detection for contradiction scoring")
            except Exception as e:
                logger.warning(f"Failed to initialize BioMedLM scorer: {e}. Falling back to keyword-based approach.")
                biomedlm_scorer = None

        contradiction_phrases = [
            'however', 'in contrast', 'conversely', 'on the contrary',
            'contrary to', 'conflicts with', 'disagrees with', 'disputes',
            'challenges', 'refutes', 'contradicts', 'inconsistent with'
        ]

        positive_findings = ['effective', 'beneficial', 'improvement', 'increased', 'enhanced']
        negative_findings = ['ineffective', 'harmful', 'no improvement', 'decreased', 'reduced', 'no effect']

        for i in range(len(publications)):
            for j in range(i+1, len(publications)):
                pub1 = publications[i]
                pub2 = publications[j]

                if not pub1.get('abstract') or not pub2.get('abstract'):
                    continue

                contradiction_found = False
                contradiction_score = 0.0
                detailed_scores = {}

                if biomedlm_scorer:
                    try:
                        abstract1 = pub1.get('abstract', '')
                        abstract2 = pub2.get('abstract', '')

                        contradiction_result = biomedlm_scorer.detect_contradiction(abstract1, abstract2)

                        contradiction_score = contradiction_result.get('contradiction_score', 0.0)
                        contradiction_found = contradiction_result.get('has_contradiction', False)
                        detailed_scores = contradiction_result

                        if contradiction_found:
                            logger.info(f"BioMedLM found contradiction between PMID {pub1.get('pmid')} and {pub2.get('pmid')} with score {contradiction_score}")
                            if 'contradiction_type' in contradiction_result:
                                logger.info(f"Contradiction type: {contradiction_result['contradiction_type']}")

                            try:
                                explanation = biomedlm_scorer.explain_contradiction(abstract1, abstract2)
                                if explanation and 'summary' in explanation:
                                    logger.info(f"Contradiction explanation: {explanation['summary']}")
                                    detailed_scores['explanation'] = explanation
                            except Exception as e:
                                logger.warning(f"Error generating contradiction explanation: {e}")
                    except Exception as e:
                        logger.warning(f"Error using BioMedLM for contradiction scoring: {e}. Falling back to keyword-based approach.")
                        biomedlm_scorer = None  # Disable for future iterations if it fails

                if not biomedlm_scorer and not contradiction_found:
                    for phrase in contradiction_phrases:
                        if phrase in pub2.get('abstract', '').lower() and pub1.get('title', '').lower() in pub2.get('abstract', '').lower():
                            contradiction_found = True
                            contradiction_score = 0.8  # Arbitrary score for keyword-based detection
                            break
                        if phrase in pub1.get('abstract', '').lower() and pub2.get('title', '').lower() in pub1.get('abstract', '').lower():
                            contradiction_found = True
                            contradiction_score = 0.8  # Arbitrary score for keyword-based detection
                            break

                    if not contradiction_found:
                        pub1_positive = any(term in pub1.get('abstract', '').lower() for term in positive_findings)
                        pub1_negative = any(term in pub1.get('abstract', '').lower() for term in negative_findings)
                        pub2_positive = any(term in pub2.get('abstract', '').lower() for term in positive_findings)
                        pub2_negative = any(term in pub2.get('abstract', '').lower() for term in negative_findings)

                        if (pub1_positive and pub2_negative) or (pub1_negative and pub2_positive):
                            contradiction_found = True
                            contradiction_score = 0.75  # Arbitrary score for opposing findings

                if contradiction_found:
                    authority_comparison = self.compare_publication_authority(pub1, pub2)

                    contradiction_entry = {
                        'publication1': {
                            'pmid': pub1.get('pmid'),
                            'title': pub1.get('title'),
                            'abstract_snippet': pub1.get('abstract', '')[:200] + '...' if pub1.get('abstract') else None,
                            'authority_score': pub1.get('authority_score')
                        },
                        'publication2': {
                            'pmid': pub2.get('pmid'),
                            'title': pub2.get('title'),
                            'abstract_snippet': pub2.get('abstract', '')[:200] + '...' if pub2.get('abstract') else None,
                            'authority_score': pub2.get('authority_score')
                        },
                        'authority_comparison': authority_comparison,
                        'contradiction_score': contradiction_score,
                        'confidence': 'high' if contradiction_score > 0.8 else ('medium' if contradiction_score > 0.6 else 'low')
                    }

                    if detailed_scores:
                        contradiction_entry['detailed_scores'] = detailed_scores

                        if 'explanation' in detailed_scores:
                            contradiction_entry['explanation'] = detailed_scores['explanation']

                    contradiction_entry['detection_method'] = 'biomedlm' if biomedlm_scorer else 'keyword'

                    contradictions.append(contradiction_entry)

        return contradictions

def load_impact_factors_from_jcr(jcr_file_path: str) -> Dict[str, float]:
    """
    Load impact factors from Journal Citation Reports (JCR) file.

    Args:
        jcr_file_path: Path to JCR file (CSV format)

    Returns:
        Dictionary mapping journal names and ISSNs to impact factors
    """
    impact_factors = {}

    try:
        df = pd.read_csv(jcr_file_path)

        journal_col = next((col for col in df.columns if 'journal' in col.lower()), None)
        issn_col = next((col for col in df.columns if 'issn' in col.lower()), None)
        impact_col = next((col for col in df.columns if 'impact' in col.lower() and 'factor' in col.lower()), None)

        if not journal_col or not impact_col:
            logger.error(f"Could not identify journal or impact factor columns in {jcr_file_path}")
            return impact_factors

        for _, row in df.iterrows():
            journal_name = row[journal_col]
            impact_factor = row[impact_col]

            if pd.notna(journal_name) and pd.notna(impact_factor):
                impact_factors[journal_name.lower()] = float(impact_factor)

                if issn_col and pd.notna(row[issn_col]):
                    impact_factors[row[issn_col]] = float(impact_factor)

        logger.info(f"Loaded {len(impact_factors)} impact factors from {jcr_file_path}")

    except Exception as e:
        logger.error(f"Error loading impact factors from {jcr_file_path}: {str(e)}")

    return impact_factors

if __name__ == "__main__":
    extractor = PublicationMetadataExtractor(
        impact_factor_source="journal_impact_factors.csv",
        crossref_email="your_email@example.com"
    )

    publication = {
        "pmid": "34735427",
        "title": "Efficacy and Safety of COVID-19 Vaccines",
        "abstract": "This systematic review evaluates the efficacy and safety of COVID-19 vaccines...",
        "authors": ["Smith J", "Johnson A", "Williams B"],
        "publication_date": "2021-10-15",
        "journal": "New England Journal of Medicine"
    }

    enriched = extractor.enrich_publication_metadata(publication)

    print(f"Title: {enriched['title']}")
    print(f"Publication Date: {enriched.get('human_date')}")
    print(f"Impact Factor: {enriched.get('impact_factor')}")
    print(f"Journal Quartile: {enriched.get('journal_quartile')}")
    print(f"Publication Types: {', '.join(enriched.get('publication_types', []))}")
    print(f"Authority Score: {enriched.get('authority_score')}/100")