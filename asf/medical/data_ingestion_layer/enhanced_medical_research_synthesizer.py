"""
Enhanced Medical Research Synthesizer

This module integrates the NCBIClient with advanced features:
1. Metadata Extraction - for enhanced publication analysis
2. Incremental Updates - for maintaining fresh knowledge bases
3. Query Builder - for easy construction of complex medical queries

Usage example for contradictory findings analysis.
"""

from ncbi_api import NCBIClient
from metadata_extraction import PublicationMetadataExtractor
from incremental_updates import IncrementalUpdateManager, NCBIIncrementalClient
from query_builder import (
    MedicalQueryBuilder, NCBIQueryInterface, 
    MedicalCondition, MedicalIntervention, OutcomeMetric, StudyDesign
)

import os
import json
import datetime
import logging
from typing import Dict, List, Optional, Any, Set
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('medical_research_synthesizer')

class EnhancedMedicalResearchSynthesizer:
    """
    Integrates all enhanced NCBI client features for medical research synthesis.
    """
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 email: Optional[str] = None,
                 storage_dir: Optional[str] = None,
                 impact_factor_source: Optional[str] = None):
        """
        Initialize the enhanced medical research synthesizer.
        
        Args:
            api_key: NCBI API key
            email: User email for API identification
            storage_dir: Directory for storing data
            impact_factor_source: Path to journal impact factor data
        """
        # Set up storage directory
        self.storage_dir = storage_dir or os.path.join(os.path.expanduser("~"), ".medical_research")
        os.makedirs(self.storage_dir, exist_ok=True)
        
        # Initialize base NCBI client
        self.ncbi_client = NCBIClient(api_key=api_key, email=email)
        
        # Initialize enhanced components
        self.metadata_extractor = PublicationMetadataExtractor(impact_factor_source=impact_factor_source)
        self.update_manager = IncrementalUpdateManager(storage_dir=self.storage_dir)
        self.incremental_client = NCBIIncrementalClient(self.ncbi_client, self.update_manager)
        self.query_interface = NCBIQueryInterface(self.ncbi_client)
        
        # Create knowledge base directory
        self.kb_dir = os.path.join(self.storage_dir, "knowledge_bases")
        os.makedirs(self.kb_dir, exist_ok=True)
    
    def create_query(self) -> MedicalQueryBuilder:
        """
        Create a new medical query builder.
        
        Returns:
            MedicalQueryBuilder instance
        """
        return self.query_interface.new_query()
    
    def create_query_from_template(self, template_name: str) -> MedicalQueryBuilder:
        """
        Create a query builder from a template.
        
        Args:
            template_name: Name of the template
            
        Returns:
            MedicalQueryBuilder instance
        """
        return self.query_interface.from_template(template_name)
    
    def search_and_enrich(self, 
                         query_builder: Optional[MedicalQueryBuilder] = None,
                         query: Optional[str] = None,
                         max_results: int = 20) -> List[Dict[str, Any]]:
        """
        Search PubMed and enrich the results with metadata.
        
        Args:
            query_builder: MedicalQueryBuilder instance (optional)
            query: Raw query string (optional, used if query_builder not provided)
            max_results: Maximum number of results to return
            
        Returns:
            List of enriched publication data
        """
        # Get the query
        if query_builder:
            # Use the provided query builder
            self.query_interface.query_builder = query_builder
            search_results = self.query_interface.search(max_results=max_results)
        elif query:
            # Use the provided raw query
            search_results = self.ncbi_client.search_pubmed(query, max_results=max_results)
            
            if not search_results or 'esearchresult' not in search_results:
                return []
                
            # Extract count and IDs
            id_list = search_results['esearchresult'].get('idlist', [])
            
            # Fetch abstracts
            abstracts = self.ncbi_client.fetch_pubmed_abstracts(id_list=id_list)
            
            if not abstracts:
                return []
                
            search_results = {'results': abstracts}
        else:
            raise ValueError("Either query_builder or query must be provided")
        
        # Enrich the results with metadata
        enriched_results = []
        for article in search_results['results']:
            enriched = self.metadata_extractor.enrich_publication_metadata(article)
            enriched_results.append(enriched)
        
        return enriched_results
    
    def incremental_search_and_enrich(self,
                                     query: str,
                                     max_results: int = 100,
                                     days_overlap: int = 1) -> List[Dict[str, Any]]:
        """
        Perform an incremental search and enrich the results.
        
        Args:
            query: Query string
            max_results: Maximum number of results to return
            days_overlap: Number of days to overlap with previous query
            
        Returns:
            List of enriched new publications
        """
        # Perform incremental search
        search_result = self.incremental_client.search_pubmed_incremental(
            query, 
            max_results=max_results,
            days_overlap=days_overlap,
            return_only_new=True
        )
        
        # If no new results, return empty list
        if not search_result['results']:
            return []
        
        # Enrich results with metadata
        enriched_results = []
        for article in search_result['results']:
            enriched = self.metadata_extractor.enrich_publication_metadata(article)
            enriched_results.append(enriched)
        
        return enriched_results
    
    def search_and_analyze_contradictions(self,
                                         query: str,
                                         max_results: int = 100) -> Dict[str, Any]:
        """
        Search for literature and analyze potential contradictions.
        
        Args:
            query: Query string
            max_results: Maximum number of results to return
            
        Returns:
            Dictionary with contradiction analysis
        """
        # Search for literature
        search_results = self.ncbi_client.search_pubmed(query, max_results=max_results)
        
        if not search_results or 'esearchresult' not in search_results:
            return {'contradictions': [], 'total_articles': 0}
            
        # Extract IDs and fetch abstracts
        id_list = search_results['esearchresult'].get('idlist', [])
        abstracts = self.ncbi_client.fetch_pubmed_abstracts(id_list=id_list)
        
        if not abstracts:
            return {'contradictions': [], 'total_articles': 0}
        
        # Enrich with metadata
        enriched_articles = []
        for article in abstracts:
            enriched = self.metadata_extractor.enrich_publication_metadata(article)
            enriched_articles.append(enriched)
        
        # Find contradictions
        contradictions = self.metadata_extractor.find_contradictions(enriched_articles)
        
        # Analyze contradictions by topic
        contradiction_analysis = self._analyze_contradictions_by_topic(contradictions, enriched_articles)
        
        return {
            'contradictions': contradictions,
            'by_topic': contradiction_analysis,
            'total_articles': len(enriched_articles),
            'num_contradictions': len(contradictions)
        }
    
    def _analyze_contradictions_by_topic(self, 
                                       contradictions: List[Dict[str, Any]],
                                       articles: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Analyze contradictions by topic.
        
        Args:
            contradictions: List of contradiction dictionaries
            articles: List of enriched articles
            
        Returns:
            Dictionary mapping topics to lists of contradictions
        """
        # Create a dictionary of articles by PMID for easy lookup
        articles_by_pmid = {article.get('pmid'): article for article in articles}
        
        # Group contradictions by topic
        topics = defaultdict(list)
        
        for contradiction in contradictions:
            # Get the full articles
            pmid1 = contradiction['publication1']['pmid']
            pmid2 = contradiction['publication2']['pmid']
            
            article1 = articles_by_pmid.get(pmid1)
            article2 = articles_by_pmid.get(pmid2)
            
            if not article1 or not article2:
                continue
            
            # Extract MeSH terms
            mesh_terms1 = set(article1.get('mesh_terms', []))
            mesh_terms2 = set(article2.get('mesh_terms', []))
            
            # Find common MeSH terms
            common_terms = mesh_terms1.intersection(mesh_terms2)
            
            # Assign to topics based on common MeSH terms
            if common_terms:
                for term in common_terms:
                    topics[term].append(contradiction)
            else:
                # If no common MeSH terms, use a generic topic
                topics['general'].append(contradiction)
        
        return dict(topics)
    
    def create_and_update_knowledge_base(self,
                                        name: str,
                                        query: str,
                                        schedule: str = 'weekly',
                                        max_results: int = 100) -> Dict[str, Any]:
        """
        Create and schedule updates for a knowledge base.
        
        Args:
            name: Knowledge base name
            query: Query string
            schedule: Update frequency ('daily', 'weekly', 'monthly')
            max_results: Maximum results per update
            
        Returns:
            Dictionary with knowledge base information
        """
        # Create knowledge base file
        kb_file = os.path.join(self.kb_dir, f"{name}.json")
        
        # Perform initial search
        search_result = self.incremental_client.search_pubmed_incremental(
            query, 
            max_results=max_results,
            return_only_new=True
        )
        
        # Enrich and save results
        enriched_results = []
        for article in search_result['results']:
            enriched = self.metadata_extractor.enrich_publication_metadata(article)
            enriched_results.append(enriched)
        
        # Save to knowledge base
        with open(kb_file, 'w') as f:
            json.dump(enriched_results, f, indent=2)
        
        # Schedule regular updates
        update_info = self.incremental_client.schedule_regular_updates(
            query,
            kb_file,
            schedule=schedule,
            max_results=max_results
        )
        
        return {
            'name': name,
            'query': query,
            'kb_file': kb_file,
            'initial_results': len(enriched_results),
            'update_schedule': schedule,
            'update_info': update_info,
            'created_date': datetime.datetime.now().isoformat()
        }
    
    def get_knowledge_base(self, name: str) -> List[Dict[str, Any]]:
        """
        Get articles from a knowledge base.
        
        Args:
            name: Knowledge base name
            
        Returns:
            List of articles in the knowledge base
        """
        kb_file = os.path.join(self.kb_dir, f"{name}.json")
        
        if not os.path.exists(kb_file):
            logger.error(f"Knowledge base '{name}' not found")
            return []
        
        try:
            with open(kb_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading knowledge base: {str(e)}")
            return []
    
    def search_cap_contradictory_treatments(self) -> Dict[str, Any]:
        """
        Specialized search for contradictory findings on CAP treatments.
        
        Returns:
            Dictionary with analysis
        """
        # Create query for CAP treatment studies
        query_builder = self.create_query()
        query_builder.add_condition(
            MedicalCondition("community-acquired pneumonia")
            .add_synonym("CAP")
        )
        query_builder.add_intervention(
            MedicalIntervention("antibiotic therapy")
            .add_alternative("beta-lactam")
            .add_alternative("macrolide")
            .add_alternative("fluoroquinolone")
            .add_alternative("doxycycline")
        )
        query_builder.add_study_design(
            StudyDesign("randomized controlled trial")
            .add_related_design("clinical trial")
        )
        query_builder.last_n_years(10)
        
        # Execute the search and enrich results
        enriched_results = self.search_and_enrich(query_builder=query_builder, max_results=50)
        
        # Find contradictions
        contradictions = self.metadata_extractor.find_contradictions(enriched_results)
        
        # Group by intervention type
        intervention_contradictions = defaultdict(list)
        
        for contradiction in contradictions:
            # Extract intervention terms from abstracts
            abstract1 = contradiction['publication1']['abstract_snippet']
            abstract2 = contradiction['publication2']['abstract_snippet']
            
            # Check for key antibiotics
            antibiotics = ['beta-lactam', 'macrolide', 'fluoroquinolone', 'doxycycline', 
                         'azithromycin', 'levofloxacin', 'amoxicillin', 'ceftriaxone']
            
            found_antibiotics = set()
            for antibiotic in antibiotics:
                if (abstract1 and antibiotic.lower() in abstract1.lower()) or \
                   (abstract2 and antibiotic.lower() in abstract2.lower()):
                    found_antibiotics.add(antibiotic)
            
            if found_antibiotics:
                for antibiotic in found_antibiotics:
                    intervention_contradictions[antibiotic].append(contradiction)
            else:
                intervention_contradictions['unspecified'].append(contradiction)
        
        # Compare authorities of contradictory studies
        authority_analysis = {}
        for intervention, contras in intervention_contradictions.items():
            higher_authority = {'publication1': 0, 'publication2': 0, 'equal': 0}
            
            for contra in contras:
                comparison = contra['authority_comparison']
                higher_authority[comparison['higher_authority']] += 1
            
            authority_analysis[intervention] = higher_authority
        
        return {
            'total_articles': len(enriched_results),
            'num_contradictions': len(contradictions),
            'contradictions_by_intervention': dict(intervention_contradictions),
            'authority_analysis': authority_analysis
        }
    
    def generate_evidence_comparison_table(self, contradictions: List[Dict[str, Any]]) -> str:
        """
        Generate a markdown table comparing contradictory evidence.
        
        Args:
            contradictions: List of contradiction dictionaries
            
        Returns:
            Markdown table as string
        """
        if not contradictions:
            return "No contradictions found."
        
        table = "| Finding 1 | Finding 2 | Authority Comparison | Confidence |\n"
        table += "|-----------|-----------|---------------------|------------|\n"
        
        for i, contradiction in enumerate(contradictions[:10]):  # Limit to 10 for readability
            pub1 = contradiction['publication1']
            pub2 = contradiction['publication2']
            authority = contradiction['authority_comparison']
            confidence = contradiction.get('confidence', 'medium')
            
            higher = authority['higher_authority']
            diff = abs(authority['authority_difference'])
            
            if higher == 'publication1':
                authority_text = f"Publication 1 stronger (+{diff:.1f})"
            elif higher == 'publication2':
                authority_text = f"Publication 2 stronger (+{diff:.1f})"
            else:
                authority_text = "Equal authority"
            
            table += f"| **{pub1['title']}** (Score: {pub1['authority_score']}) | "
            table += f"**{pub2['title']}** (Score: {pub2['authority_score']}) | "
            table += f"{authority_text} | {confidence.capitalize()} |\n"
        
        return table
    
    def cap_duration_vs_agent_analysis(self) -> Dict[str, Any]:
        """
        Specialized analysis comparing treatment duration vs agent for CAP.
        
        Returns:
            Dictionary with analysis
        """
        # Create query for treatment duration
        duration_query = self.create_query()
        duration_query.add_condition(
            MedicalCondition("community-acquired pneumonia")
            .add_synonym("CAP")
        )
        duration_query.add_intervention(
            MedicalIntervention("antibiotic duration")
            .add_alternative("treatment duration")
            .add_alternative("short course")
            .add_alternative("long course")
        )
        duration_query.last_n_years(10)
        
        # Create query for treatment agent choice
        agent_query = self.create_query()
        agent_query.add_condition(
            MedicalCondition("community-acquired pneumonia")
            .add_synonym("CAP")
        )
        agent_query.add_intervention(
            MedicalIntervention("antibiotic choice")
            .add_alternative("macrolide")
            .add_alternative("fluoroquinolone")
            .add_alternative("beta-lactam")
        )
        agent_query.last_n_years(10)
        
        # Execute the searches and enrich results
        duration_results = self.search_and_enrich(query_builder=duration_query, max_results=30)
        agent_results = self.search_and_enrich(query_builder=agent_query, max_results=30)
        
        # Find contradictions in each dataset
        duration_contradictions = self.metadata_extractor.find_contradictions(duration_results)
        agent_contradictions = self.metadata_extractor.find_contradictions(agent_results)
        
        # Generate comparison tables
        duration_table = self.generate_evidence_comparison_table(duration_contradictions)
        agent_table = self.generate_evidence_comparison_table(agent_contradictions)
        
        # Calculate consensus strength based on authority scores
        def calculate_consensus(articles, keyword):
            if not articles:
                return {"strength": "unknown", "score": 0}
                
            relevant_articles = [a for a in articles if keyword.lower() in a.get('abstract', '').lower()]
            if not relevant_articles:
                return {"strength": "unknown", "score": 0}
                
            # Higher scores = stronger evidence
            avg_authority = sum(a.get('authority_score', 50) for a in relevant_articles) / len(relevant_articles)
            contradictions_ratio = len([c for c in duration_contradictions 
                                     if keyword.lower() in c['publication1'].get('abstract_snippet', '').lower() or
                                        keyword.lower() in c['publication2'].get('abstract_snippet', '').lower()]) / len(relevant_articles) if relevant_articles else 0
            
            consensus_score = avg_authority * (1 - contradictions_ratio)
            
            if consensus_score > 70:
                strength = "strong"
            elif consensus_score > 50:
                strength = "moderate"
            elif consensus_score > 30:
                strength = "weak"
            else:
                strength = "very weak"
                
            return {"strength": strength, "score": consensus_score}
        
        # Calculate consensus for different durations and agents
        duration_consensus = {
            "short_course": calculate_consensus(duration_results, "short course"),
            "long_course": calculate_consensus(duration_results, "long course")
        }
        
        agent_consensus = {
            "macrolide": calculate_consensus(agent_results, "macrolide"),
            "fluoroquinolone": calculate_consensus(agent_results, "fluoroquinolone"),
            "beta_lactam": calculate_consensus(agent_results, "beta-lactam")
        }
        
        return {
            "duration_articles": len(duration_results),
            "duration_contradictions": len(duration_contradictions),
            "duration_table": duration_table,
            "duration_consensus": duration_consensus,
            
            "agent_articles": len(agent_results),
            "agent_contradictions": len(agent_contradictions),
            "agent_table": agent_table,
            "agent_consensus": agent_consensus
        }


# Example usage
if __name__ == "__main__":
    # Initialize the synthesizer
    synthesizer = EnhancedMedicalResearchSynthesizer(
        email="your_email@example.com",
        impact_factor_source="journal_impact_factors.csv"
    )
    
    print("Analyzing contradictory findings on Community-Acquired Pneumonia treatments...")
    
    # Analyze contradictions in CAP treatment literature
    analysis = synthesizer.search_cap_contradictory_treatments()
    
    print(f"\nFound {analysis['num_contradictions']} contradictions in {analysis['total_articles']} articles")
    
    # Print contradictions by intervention
    for intervention, contradictions in analysis['contradictions_by_intervention'].items():
        if contradictions:
            print(f"\n{intervention.capitalize()} treatment contradictions: {len(contradictions)}")
            
            # Print authority comparison
            authority = analysis['authority_analysis'][intervention]
            print(f"  Higher authority: Publication 1: {authority['publication1']}, " 
                  f"Publication 2: {authority['publication2']}, "
                  f"Equal: {authority['equal']}")
    
    print("\nPerforming detailed analysis of treatment duration vs. agent choice...")
    
    # Compare treatment duration vs. agent choice
    cap_analysis = synthesizer.cap_duration_vs_agent_analysis()
    
    # Print results
    print(f"\nTreatment Duration Analysis ({cap_analysis['duration_articles']} articles, "
          f"{cap_analysis['duration_contradictions']} contradictions)")
    
    print("\nConsensus on treatment duration:")
    for duration, consensus in cap_analysis['duration_consensus'].items():
        print(f"  {duration.replace('_', ' ').title()}: {consensus['strength'].capitalize()} "
              f"consensus (score: {consensus['score']:.1f})")
    
    print(f"\nAntibiotic Agent Analysis ({cap_analysis['agent_articles']} articles, "
          f"{cap_analysis['agent_contradictions']} contradictions)")
    
    print("\nConsensus on antibiotic choice:")
    for agent, consensus in cap_analysis['agent_consensus'].items():
        print(f"  {agent.replace('_', ' ').title()}: {consensus['strength'].capitalize()} "
              f"consensus (score: {consensus['score']:.1f})")
    
    print("\nCreating a regularly updating knowledge base for CAP treatment...")
    
    # Create a knowledge base with scheduled updates
    kb_info = synthesizer.create_and_update_knowledge_base(
        name="cap_treatment",
        query="community acquired pneumonia[Title/Abstract] AND treatment[Title/Abstract]",
        schedule="weekly",
        max_results=50
    )
    
    print(f"Created knowledge base '{kb_info['name']}' with {kb_info['initial_results']} initial articles")
    print(f"Updates scheduled: {kb_info['update_schedule']}")
    print(f"Knowledge base file: {kb_info['kb_file']}")