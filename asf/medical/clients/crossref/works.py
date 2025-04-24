"""
CrossRef Works-specific functionality for the Medical Research Synthesizer.
This module provides specialized functions for working with scholarly works in CrossRef.
"""
import logging
from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime
import itertools

logger = logging.getLogger(__name__)

class WorksService:
    """
    Service class for expanded functionality related to CrossRef works.
    """
    
    def __init__(self, client):
        """
        Initialize the WorksService.
        
        Args:
            client: The parent CrossRefClient instance
        """
        self.client = client
    
    def get_coauthors(self, doi: str) -> List[Dict]:
        """
        Get the coauthors of a work identified by DOI.
        
        Args:
            doi: DOI of the work
            
        Returns:
            List[Dict]: List of authors with their metadata
        """
        work = self.client.get_work_by_doi(doi)
        if not work or 'author' not in work:
            return []
        
        return work['author']
    
    def get_cited_by(self, doi: str, max_results: int = 100) -> List[Dict]:
        """
        Get works that cite the specified DOI.
        
        Args:
            doi: DOI of the work
            max_results: Maximum number of results to return
            
        Returns:
            List[Dict]: List of works citing the specified DOI
        """
        filter_params = {"references": doi}
        return self.client.search_works(
            query=None, 
            filter=filter_params,
            max_results=max_results,
            sort="relevance"
        )
    
    def get_author_publications(self, author_name: str, max_results: int = 100) -> List[Dict]:
        """
        Get publications by a specific author.
        
        Args:
            author_name: Name of the author
            max_results: Maximum number of results to return
            
        Returns:
            List[Dict]: List of publications by the author
        """
        query = f"author:\"{author_name}\""
        return self.client.search_works(
            query=query,
            max_results=max_results,
            sort="relevance"
        )
    
    def find_related_works(self, doi: str, max_results: int = 10) -> List[Dict]:
        """
        Find works related to the specified DOI based on shared references.
        
        Args:
            doi: DOI of the work
            max_results: Maximum number of results to return
            
        Returns:
            List[Dict]: List of related works
        """
        # Get the references of the specified work
        work_references = self.client.get_references(doi)
        if not work_references:
            return []
        
        # Extract DOIs from references (if available)
        ref_dois = []
        for ref in work_references:
            if 'DOI' in ref:
                ref_dois.append(ref['DOI'])
        
        if not ref_dois:
            return []
        
        # Find works that cite the same references
        related_works = []
        for ref_doi in ref_dois[:5]:  # Limit to first 5 references to avoid too many API calls
            filter_params = {"references": ref_doi}
            citing_works = self.client.search_works(
                query=None,
                filter=filter_params,
                max_results=20
            )
            related_works.extend(citing_works)
            
            # Don't exceed max_results
            if len(related_works) >= max_results:
                break
        
        # Deduplicate and remove the original work
        unique_works = []
        seen_dois = set([doi])  # Include original DOI to filter it out
        
        for work in related_works:
            work_doi = work.get('DOI')
            if work_doi and work_doi not in seen_dois:
                seen_dois.add(work_doi)
                unique_works.append(work)
                
                if len(unique_works) >= max_results:
                    break
        
        return unique_works
    
    def get_publication_timeline(self, query: str, start_year: int, end_year: int = None) -> Dict[int, int]:
        """
        Get a timeline of publications matching a query by year.
        
        Args:
            query: Search query
            start_year: Starting year for the timeline
            end_year: Ending year for the timeline (defaults to current year)
            
        Returns:
            Dict[int, int]: Dictionary mapping years to publication counts
        """
        if end_year is None:
            end_year = datetime.now().year
        
        timeline = {year: 0 for year in range(start_year, end_year + 1)}
        
        for year in range(start_year, end_year + 1):
            filter_params = {
                "from-pub-date": f"{year}-01-01",
                "until-pub-date": f"{year}-12-31"
            }
            
            # First get the count only
            results = self.client.search_works(
                query=query,
                filter=filter_params,
                max_results=0
            )
            
            # The crossrefapi package might not expose the total count directly,
            # so we need to check how to access it based on the returned structure
            if hasattr(results, 'total_results'):
                count = results.total_results
            elif isinstance(results, dict) and 'message' in results and 'total-results' in results['message']:
                count = results['message']['total-results']
            else:
                # If we can't get the count directly, estimate based on the length of results
                temp_results = self.client.search_works(
                    query=query,
                    filter=filter_params,
                    max_results=1000
                )
                count = len(temp_results)
            
            timeline[year] = count
        
        return timeline
    
    def find_collaboration_network(self, author_name: str, depth: int = 1, max_coauthors: int = 10) -> Dict:
        """
        Build a collaboration network for an author up to specified depth.
        
        Args:
            author_name: Name of the seed author
            depth: Depth of collaboration network to explore (1 = direct coauthors only)
            max_coauthors: Maximum number of coauthors to include per author
            
        Returns:
            Dict: Collaboration network structure
        """
        # Initialize network
        network = {
            "nodes": [],
            "links": []
        }
        
        # Keep track of authors we've processed
        processed_authors = set()
        authors_to_process = [(author_name, 0)]  # (name, depth)
        author_ids = {}  # Maps author names to IDs in the network
        
        current_id = 0
        
        while authors_to_process:
            current_author, current_depth = authors_to_process.pop(0)
            
            if current_author in processed_authors or current_depth > depth:
                continue
                
            processed_authors.add(current_author)
            
            if current_author not in author_ids:
                author_ids[current_author] = current_id
                network["nodes"].append({
                    "id": current_id,
                    "name": current_author
                })
                current_id += 1
            
            # Get publications by this author
            publications = self.get_author_publications(current_author, max_results=10)
            
            # Extract coauthors
            coauthors = set()
            for pub in publications:
                if 'author' in pub:
                    for author in pub['author']:
                        if 'given' in author and 'family' in author:
                            coauthor_name = f"{author['given']} {author['family']}"
                            if coauthor_name != current_author:
                                coauthors.add(coauthor_name)
            
            # Add top coauthors to the network
            for coauthor in list(coauthors)[:max_coauthors]:
                if coauthor not in author_ids:
                    author_ids[coauthor] = current_id
                    network["nodes"].append({
                        "id": current_id,
                        "name": coauthor
                    })
                    current_id += 1
                
                # Add link between author and coauthor
                network["links"].append({
                    "source": author_ids[current_author],
                    "target": author_ids[coauthor]
                })
                
                # Add coauthor to processing queue if we haven't reached max depth
                if current_depth < depth:
                    authors_to_process.append((coauthor, current_depth + 1))
        
        return network