"""
CrossRef Journals-specific functionality for the Medical Research Synthesizer.
This module provides specialized functions for working with journal metadata in CrossRef.
"""
import logging
from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

class JournalsService:
    """
    Service class for expanded functionality related to CrossRef journals.
    """
    
    def __init__(self, client):
        """
        Initialize the JournalsService.
        
        Args:
            client: The parent CrossRefClient instance
        """
        self.client = client
    
    def get_recent_articles(self, issn: str, max_results: int = 20) -> List[Dict]:
        """
        Get recent articles published in a specific journal.
        
        Args:
            issn: ISSN of the journal
            max_results: Maximum number of results to return
            
        Returns:
            List[Dict]: List of recent articles
        """
        filter_params = {"issn": issn}
        return self.client.search_works(
            query=None,
            filter=filter_params,
            max_results=max_results,
            sort="published-online",
            order="desc"
        )
    
    def get_journal_activity(self, issn: str, years: int = 5) -> Dict[str, Any]:
        """
        Get publication activity statistics for a journal over the past N years.
        
        Args:
            issn: ISSN of the journal
            years: Number of years to analyze
            
        Returns:
            Dict: Journal activity statistics
        """
        current_year = datetime.now().year
        start_year = current_year - years
        
        # Get journal metadata
        journal = self.client.get_journal_by_issn(issn)
        if not journal:
            return {"error": f"Journal with ISSN {issn} not found"}
        
        # Initialize results structure
        activity = {
            "journal": journal.get("title", "Unknown Journal"),
            "issn": issn,
            "yearly_counts": {},
            "total_publications": 0,
            "analysis_period": f"{start_year}-{current_year}"
        }
        
        # Get publication counts by year
        for year in range(start_year, current_year + 1):
            filter_params = {
                "issn": issn,
                "from-pub-date": f"{year}-01-01",
                "until-pub-date": f"{year}-12-31"
            }
            
            # Use works API to get publications for this year
            works = self.client.search_works(
                query=None,
                filter=filter_params,
                max_results=0  # We only need the count
            )
            
            # Try to get the count from the API response
            if hasattr(works, 'total_results'):
                count = works.total_results
            elif isinstance(works, dict) and 'message' in works and 'total-results' in works['message']:
                count = works['message']['total-results']
            else:
                # Fall back to estimating count by fetching actual results
                temp_works = self.client.search_works(
                    query=None,
                    filter=filter_params,
                    max_results=1000
                )
                count = len(temp_works)
                
            activity["yearly_counts"][str(year)] = count
            activity["total_publications"] += count
        
        # Calculate average publications per year
        if years > 0:
            activity["avg_publications_per_year"] = activity["total_publications"] / years
        else:
            activity["avg_publications_per_year"] = 0
        
        return activity
    
    def compare_journals(self, issns: List[str]) -> Dict[str, Any]:
        """
        Compare multiple journals based on publication metrics.
        
        Args:
            issns: List of journal ISSNs to compare
            
        Returns:
            Dict: Comparison data
        """
        comparison = {
            "journals": [],
            "comparison_date": datetime.now().isoformat()
        }
        
        for issn in issns:
            # Get basic journal data
            journal = self.client.get_journal_by_issn(issn)
            if not journal:
                logger.warning(f"Journal with ISSN {issn} not found")
                continue
            
            # Get publication activity
            activity = self.get_journal_activity(issn, years=3)
            
            # Add to comparison
            journal_data = {
                "title": journal.get("title", "Unknown Journal"),
                "issn": issn,
                "publisher": journal.get("publisher", "Unknown Publisher"),
                "total_publications": activity.get("total_publications", 0),
                "avg_publications_per_year": activity.get("avg_publications_per_year", 0),
                "yearly_counts": activity.get("yearly_counts", {})
            }
            
            comparison["journals"].append(journal_data)
        
        return comparison
    
    def find_similar_journals(self, issn: str, max_results: int = 5) -> List[Dict]:
        """
        Find journals similar to the specified journal based on common authors and subjects.
        
        Args:
            issn: ISSN of the reference journal
            max_results: Maximum number of similar journals to return
            
        Returns:
            List[Dict]: List of similar journals
        """
        # Get the reference journal
        journal = self.client.get_journal_by_issn(issn)
        if not journal:
            logger.error(f"Journal with ISSN {issn} not found")
            return []
        
        # Get recent articles from this journal
        articles = self.get_recent_articles(issn, max_results=50)
        
        # Extract authors and subjects
        authors = set()
        for article in articles:
            if 'author' in article:
                for author in article['author']:
                    if 'given' in author and 'family' in author:
                        author_name = f"{author['given']} {author['family']}"
                        authors.add(author_name)
        
        # Find journals with the same authors
        similar_journals = {}
        
        for author in list(authors)[:10]:  # Limit to 10 authors to avoid too many API calls
            filter_params = {"has-journal-metadata": "true"}
            author_articles = self.client.search_works(
                query=f"author:\"{author}\"",
                filter=filter_params,
                max_results=50
            )
            
            # Extract journals from these articles
            for article in author_articles:
                if 'ISSN' in article and isinstance(article['ISSN'], list):
                    for journal_issn in article['ISSN']:
                        if journal_issn != issn:
                            if journal_issn not in similar_journals:
                                similar_journals[journal_issn] = {'count': 0, 'data': None}
                            similar_journals[journal_issn]['count'] += 1
        
        # Sort journals by count and get journal metadata
        sorted_journals = sorted(similar_journals.items(), key=lambda x: x[1]['count'], reverse=True)
        
        result = []
        for journal_issn, info in sorted_journals[:max_results]:
            journal_data = self.client.get_journal_by_issn(journal_issn)
            if journal_data:
                journal_data['common_authors'] = info['count']
                result.append(journal_data)
        
        return result