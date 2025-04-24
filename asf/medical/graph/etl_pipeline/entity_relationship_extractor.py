"""
Entity Relationship Extractor

This module extracts relationships between entities in the biomedical knowledge graph.
It identifies relationships between articles, clinical trials, concepts, and other entities
based on shared attributes, mentions, and explicit connections.
"""

import logging
from typing import Dict, List, Any, Optional, Set, Tuple
import itertools

logger = logging.getLogger("biomedical_etl.transformers.entity_relationship")

class EntityRelationshipExtractor:
    """
    Extractor for relationships between entities in the biomedical knowledge graph.
    
    This class provides methods for identifying and extracting relationships between
    articles, clinical trials, concepts, and other entities based on shared attributes,
    mentions, and explicit connections.
    """
    
    def __init__(self):
        """Initialize the entity relationship extractor."""
        logger.info("Initialized Entity Relationship Extractor")
    
    def extract_relationships(self, nodes: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        Extract relationships between entities.
        
        Args:
            nodes: Dictionary of node lists by type
            
        Returns:
            List of relationship dictionaries
        """
        logger.info("Extracting relationships between entities")
        relationships = []
        
        # Extract article-author relationships
        if "articles" in nodes:
            author_rels = self._extract_article_author_relationships(nodes["articles"])
            relationships.extend(author_rels)
            logger.info(f"Extracted {len(author_rels)} article-author relationships")
        
        # Extract article-mesh relationships
        if "articles" in nodes:
            mesh_rels = self._extract_article_mesh_relationships(nodes["articles"])
            relationships.extend(mesh_rels)
            logger.info(f"Extracted {len(mesh_rels)} article-mesh relationships")
        
        # Extract clinical trial-condition relationships
        if "clinical_trials" in nodes:
            condition_rels = self._extract_trial_condition_relationships(nodes["clinical_trials"])
            relationships.extend(condition_rels)
            logger.info(f"Extracted {len(condition_rels)} trial-condition relationships")
        
        # Extract clinical trial-intervention relationships
        if "clinical_trials" in nodes:
            intervention_rels = self._extract_trial_intervention_relationships(nodes["clinical_trials"])
            relationships.extend(intervention_rels)
            logger.info(f"Extracted {len(intervention_rels)} trial-intervention relationships")
        
        # Extract clinical trial-location relationships
        if "clinical_trials" in nodes:
            location_rels = self._extract_trial_location_relationships(nodes["clinical_trials"])
            relationships.extend(location_rels)
            logger.info(f"Extracted {len(location_rels)} trial-location relationships")
        
        # Extract clinical trial-sponsor relationships
        if "clinical_trials" in nodes:
            sponsor_rels = self._extract_trial_sponsor_relationships(nodes["clinical_trials"])
            relationships.extend(sponsor_rels)
            logger.info(f"Extracted {len(sponsor_rels)} trial-sponsor relationships")
        
        # Extract MeSH term hierarchical relationships
        if "mesh_terms" in nodes:
            mesh_hierarchy_rels = self._extract_mesh_hierarchy_relationships(nodes["mesh_terms"])
            relationships.extend(mesh_hierarchy_rels)
            logger.info(f"Extracted {len(mesh_hierarchy_rels)} MeSH hierarchy relationships")
        
        # Extract concept-concept relationships (e.g., IS_A, TREATS, etc.)
        if "mesh_terms" in nodes:
            concept_rels = self._extract_concept_relationships(nodes["mesh_terms"])
            relationships.extend(concept_rels)
            logger.info(f"Extracted {len(concept_rels)} concept-concept relationships")
        
        # Extract relationships between articles based on shared characteristics
        if "articles" in nodes:
            article_similarity_rels = self._extract_article_similarity_relationships(nodes["articles"])
            relationships.extend(article_similarity_rels)
            logger.info(f"Extracted {len(article_similarity_rels)} article similarity relationships")
        
        # Extract relationships between articles and clinical trials
        if "articles" in nodes and "clinical_trials" in nodes:
            article_trial_rels = self._extract_article_trial_relationships(
                nodes["articles"], nodes["clinical_trials"]
            )
            relationships.extend(article_trial_rels)
            logger.info(f"Extracted {len(article_trial_rels)} article-trial relationships")
        
        logger.info(f"Extracted {len(relationships)} total relationships")
        return relationships
    
    def _extract_article_author_relationships(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract relationships between articles and authors.
        
        Args:
            articles: List of article dictionaries
            
        Returns:
            List of article-author relationship dictionaries
        """
        relationships = []
        
        for article in articles:
            pmid = article.get("pmid")
            if not pmid:
                continue
            
            authors = article.get("authors", [])
            for author in authors:
                author_name = author.get("name")
                if author_name:
                    relationships.append({
                        "source_type": "Author",
                        "source_id": author_name,  # Using name as ID for simplicity
                        "target_type": "Article",
                        "target_id": pmid,
                        "type": "AUTHORED",
                        "properties": {}
                    })
        
        return relationships
    
    def _extract_article_mesh_relationships(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract relationships between articles and MeSH terms.
        
        Args:
            articles: List of article dictionaries
            
        Returns:
            List of article-mesh relationship dictionaries
        """
        relationships = []
        
        for article in articles:
            pmid = article.get("pmid")
            if not pmid:
                continue
            
            mesh_terms = article.get("mesh_terms", [])
            for mesh_term in mesh_terms:
                ui = mesh_term.get("ui")
                is_major = mesh_term.get("major", False)
                
                if ui:
                    properties = {"is_major": is_major}
                    
                    relationships.append({
                        "source_type": "Article",
                        "source_id": pmid,
                        "target_type": "MeshTerm",
                        "target_id": ui,
                        "type": "INDEXED_BY",
                        "properties": properties
                    })
        
        return relationships
    
    def _extract_trial_condition_relationships(self, trials: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract relationships between clinical trials and conditions.
        
        Args:
            trials: List of clinical trial dictionaries
            
        Returns:
            List of trial-condition relationship dictionaries
        """
        relationships = []
        
        for trial in trials:
            nct_id = trial.get("nct_id")
            if not nct_id:
                continue
            
            conditions = trial.get("conditions", [])
            for condition in conditions:
                condition_name = condition.get("name")
                
                if condition_name:
                    # Use name as ID for simplicity
                    # In a production system, we would map to standard condition IDs
                    relationships.append({
                        "source_type": "ClinicalTrial",
                        "source_id": nct_id,
                        "target_type": "Condition",
                        "target_id": condition_name,
                        "type": "STUDIES",
                        "properties": {}
                    })
        
        return relationships
    
    def _extract_trial_intervention_relationships(self, trials: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract relationships between clinical trials and interventions.
        
        Args:
            trials: List of clinical trial dictionaries
            
        Returns:
            List of trial-intervention relationship dictionaries
        """
        relationships = []
        
        for trial in trials:
            nct_id = trial.get("nct_id")
            if not nct_id:
                continue
            
            interventions = trial.get("interventions", [])
            for intervention in interventions:
                intervention_name = intervention.get("name")
                intervention_type = intervention.get("type")
                
                if intervention_name:
                    properties = {"type": intervention_type} if intervention_type else {}
                    
                    # Use name as ID for simplicity
                    # In a production system, we would map to standard intervention IDs
                    relationships.append({
                        "source_type": "ClinicalTrial",
                        "source_id": nct_id,
                        "target_type": "Intervention",
                        "target_id": intervention_name,
                        "type": "USES",
                        "properties": properties
                    })
        
        return relationships
    
    def _extract_trial_location_relationships(self, trials: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract relationships between clinical trials and locations.
        
        Args:
            trials: List of clinical trial dictionaries
            
        Returns:
            List of trial-location relationship dictionaries
        """
        relationships = []
        
        for trial in trials:
            nct_id = trial.get("nct_id")
            if not nct_id:
                continue
            
            locations = trial.get("locations", [])
            for location in locations:
                facility_name = location.get("name")
                city = location.get("city")
                country = location.get("country")
                
                if facility_name and country:
                    # Create a composite ID for the facility
                    facility_id = f"{facility_name}|{city}|{country}".replace(" ", "_")
                    
                    relationships.append({
                        "source_type": "Location",
                        "source_id": facility_id,
                        "target_type": "ClinicalTrial",
                        "target_id": nct_id,
                        "type": "CONDUCTED",
                        "properties": {}
                    })
        
        return relationships
    
    def _extract_trial_sponsor_relationships(self, trials: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract relationships between clinical trials and sponsors.
        
        Args:
            trials: List of clinical trial dictionaries
            
        Returns:
            List of trial-sponsor relationship dictionaries
        """
        relationships = []
        
        for trial in trials:
            nct_id = trial.get("nct_id")
            if not nct_id:
                continue
            
            sponsor = trial.get("lead_sponsor")
            if sponsor:
                relationships.append({
                    "source_type": "Organization",
                    "source_id": sponsor,  # Using name as ID for simplicity
                    "target_type": "ClinicalTrial",
                    "target_id": nct_id,
                    "type": "SPONSORS",
                    "properties": {}
                })
        
        return relationships
    
    def _extract_mesh_hierarchy_relationships(self, mesh_terms: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract hierarchical relationships between MeSH terms.
        
        Args:
            mesh_terms: List of MeSH term dictionaries
            
        Returns:
            List of MeSH hierarchy relationship dictionaries
        """
        relationships = []
        
        # This is a simplified implementation
        # In a real system, we would extract the hierarchical relationships from the MeSH RDF
        
        # Create a map of tree numbers to term IDs
        tree_to_term = {}
        for term in mesh_terms:
            term_id = term.get("id")
            tree_numbers = term.get("tree_numbers", [])
            
            for tree_number in tree_numbers:
                tree_to_term[tree_number] = term_id
        
        # Extract parent-child relationships based on tree numbers
        for tree_number, term_id in tree_to_term.items():
            # Get the parent tree number by removing the last segment
            if "." in tree_number:
                parent_tree = tree_number.rsplit(".", 1)[0]
                if parent_tree in tree_to_term:
                    parent_id = tree_to_term[parent_tree]
                    
                    relationships.append({
                        "source_type": "MeshTerm",
                        "source_id": term_id,
                        "target_type": "MeshTerm",
                        "target_id": parent_id,
                        "type": "IS_A",
                        "properties": {
                            "child_tree": tree_number,
                            "parent_tree": parent_tree
                        }
                    })
        
        return relationships
    
    def _extract_concept_relationships(self, mesh_terms: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract relationships between medical concepts.
        
        Args:
            mesh_terms: List of MeSH term dictionaries
            
        Returns:
            List of concept relationship dictionaries
        """
        # This is a simplified implementation
        # In a real system, we would extract concept relationships from multiple sources
        
        # For this demo, we'll return an empty list
        return []
    
    def _extract_article_similarity_relationships(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract similarity relationships between articles.
        
        Args:
            articles: List of article dictionaries
            
        Returns:
            List of article similarity relationship dictionaries
        """
        relationships = []
        
        # This is a simplified implementation that creates SIMILAR_TO relationships
        # based on shared MeSH terms
        
        # Group articles by MeSH terms
        mesh_to_articles = {}
        for article in articles:
            pmid = article.get("pmid")
            if not pmid:
                continue
            
            mesh_terms = article.get("mesh_terms", [])
            for mesh_term in mesh_terms:
                ui = mesh_term.get("ui")
                if ui:
                    if ui not in mesh_to_articles:
                        mesh_to_articles[ui] = []
                    mesh_to_articles[ui].append(pmid)
        
        # Create SIMILAR_TO relationships for articles sharing MeSH terms
        processed_pairs = set()
        
        for ui, article_ids in mesh_to_articles.items():
            if len(article_ids) < 2:
                continue
            
            # For each pair of articles that share this MeSH term
            for pmid1, pmid2 in itertools.combinations(article_ids, 2):
                # Ensure we don't create duplicate relationships
                pair_key = tuple(sorted([pmid1, pmid2]))
                if pair_key in processed_pairs:
                    continue
                
                processed_pairs.add(pair_key)
                
                relationships.append({
                    "source_type": "Article",
                    "source_id": pmid1,
                    "target_type": "Article",
                    "target_id": pmid2,
                    "type": "SIMILAR_TO",
                    "properties": {
                        "shared_mesh": ui
                    }
                })
        
        return relationships
    
    def _extract_article_trial_relationships(
        self,
        articles: List[Dict[str, Any]],
        trials: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Extract relationships between articles and clinical trials.
        
        Args:
            articles: List of article dictionaries
            trials: List of clinical trial dictionaries
            
        Returns:
            List of article-trial relationship dictionaries
        """
        relationships = []
        
        # This is a simplified implementation that creates REFERENCES relationships
        # based on matching NCT IDs in article text
        
        # Extract NCT IDs from articles
        for article in articles:
            pmid = article.get("pmid")
            if not pmid:
                continue
            
            # Search for NCT IDs in abstract
            abstract = article.get("abstract", "")
            
            # Simple regex pattern to find NCT IDs
            import re
            nct_pattern = r"NCT\d{8}"
            
            nct_ids = re.findall(nct_pattern, abstract)
            for nct_id in nct_ids:
                # Check if this NCT ID corresponds to a trial in our data
                for trial in trials:
                    if trial.get("nct_id") == nct_id:
                        relationships.append({
                            "source_type": "Article",
                            "source_id": pmid,
                            "target_type": "ClinicalTrial",
                            "target_id": nct_id,
                            "type": "REFERENCES",
                            "properties": {}
                        })
        
        return relationships