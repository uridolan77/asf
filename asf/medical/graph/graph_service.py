"""
Graph service for the Medical Research Synthesizer.

This module provides a service for interacting with the graph database.
"""

import logging
from typing import Dict, List, Optional, Any, Union, Tuple

from asf.medical.graph.memgraph_client import MemgraphClient
from asf.medical.graph.neo4j_client import Neo4jClient
from asf.medical.core.config import settings

# Set up logging
logger = logging.getLogger(__name__)

class GraphService:
    """
    Service for interacting with the graph database.
    
    This service provides a unified interface for interacting with different graph databases.
    """
    
    _instance = None
    
    def __new__(cls):
        """
        Create a singleton instance of the graph service.
        
        Returns:
            GraphService: The singleton instance
        """
        if cls._instance is None:
            cls._instance = super(GraphService, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the graph service."""
        self.graph_db_type = settings.GRAPH_DB_TYPE.lower()
        self.client = None
        
        logger.info(f"Graph service initialized with graph_db_type={self.graph_db_type}")
    
    def get_client(self):
        """
        Get the graph database client.
        
        Returns:
            The graph database client
            
        Raises:
            ValueError: If the graph database type is not supported
        """
        if self.client is None:
            if self.graph_db_type == "memgraph":
                logger.info("Using Memgraph client")
                self.client = MemgraphClient()
            elif self.graph_db_type == "neo4j":
                logger.info("Using Neo4j client")
                self.client = Neo4jClient()
            else:
                raise ValueError(f"Unsupported graph database type: {self.graph_db_type}")
        
        return self.client
    
    def connect(self) -> bool:
        """
        Connect to the graph database.
        
        Returns:
            True if connection is successful, False otherwise
        """
        client = self.get_client()
        return client.connect()
    
    def disconnect(self) -> None:
        """Disconnect from the graph database."""
        if self.client:
            self.client.disconnect()
    
    def execute_query(self, query: str, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Execute a Cypher query.
        
        Args:
            query: Cypher query
            params: Query parameters
            
        Returns:
            Query results
            
        Raises:
            Exception: If the query fails
        """
        client = self.get_client()
        return client.execute_query(query, params)
    
    def create_article_node(self, article: Dict[str, Any]) -> str:
        """
        Create an article node in the graph.
        
        Args:
            article: Article data
            
        Returns:
            Article ID
            
        Raises:
            Exception: If the query fails
        """
        client = self.get_client()
        return client.create_article_node(article)
    
    def create_author_relationship(self, pmid: str, author: str) -> None:
        """
        Create an author node and relationship to an article.
        
        Args:
            pmid: Article PMID
            author: Author name
            
        Raises:
            Exception: If the query fails
        """
        client = self.get_client()
        client.create_author_relationship(pmid, author)
    
    def create_concept_node(self, concept: Dict[str, Any]) -> str:
        """
        Create a concept node in the graph.
        
        Args:
            concept: Concept data
            
        Returns:
            Concept ID
            
        Raises:
            Exception: If the query fails
        """
        client = self.get_client()
        return client.create_concept_node(concept)
    
    def create_article_concept_relationship(
        self,
        pmid: str,
        cui: str,
        relationship_type: str = "MENTIONS",
        properties: Dict[str, Any] = None
    ) -> None:
        """
        Create a relationship between an article and a concept.
        
        Args:
            pmid: Article PMID
            cui: Concept CUI
            relationship_type: Relationship type
            properties: Relationship properties
            
        Raises:
            Exception: If the query fails
        """
        client = self.get_client()
        client.create_article_concept_relationship(pmid, cui, relationship_type, properties)
    
    def create_concept_concept_relationship(
        self,
        cui1: str,
        cui2: str,
        relationship_type: str,
        properties: Dict[str, Any] = None
    ) -> None:
        """
        Create a relationship between two concepts.
        
        Args:
            cui1: First concept CUI
            cui2: Second concept CUI
            relationship_type: Relationship type
            properties: Relationship properties
            
        Raises:
            Exception: If the query fails
        """
        client = self.get_client()
        client.create_concept_concept_relationship(cui1, cui2, relationship_type, properties)
    
    def create_contradiction_relationship(
        self,
        pmid1: str,
        pmid2: str,
        contradiction_score: float,
        confidence: str,
        topic: Optional[str] = None,
        explanation: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Create a contradiction relationship between two articles.
        
        Args:
            pmid1: First article PMID
            pmid2: Second article PMID
            contradiction_score: Contradiction score
            confidence: Confidence level
            topic: Contradiction topic
            explanation: Contradiction explanation
            
        Raises:
            Exception: If the query fails
        """
        client = self.get_client()
        client.create_contradiction_relationship(
            pmid1, pmid2, contradiction_score, confidence, topic, explanation
        )
    
    def get_article(self, pmid: str) -> Optional[Dict[str, Any]]:
        """
        Get an article from the graph.
        
        Args:
            pmid: Article PMID
            
        Returns:
            Article data or None if not found
            
        Raises:
            Exception: If the query fails
        """
        client = self.get_client()
        return client.get_article(pmid)
    
    def get_concept(self, cui: str) -> Optional[Dict[str, Any]]:
        """
        Get a concept from the graph.
        
        Args:
            cui: Concept CUI
            
        Returns:
            Concept data or None if not found
            
        Raises:
            Exception: If the query fails
        """
        client = self.get_client()
        return client.get_concept(cui)
    
    def get_article_concepts(self, pmid: str) -> List[Dict[str, Any]]:
        """
        Get concepts mentioned in an article.
        
        Args:
            pmid: Article PMID
            
        Returns:
            List of concepts
            
        Raises:
            Exception: If the query fails
        """
        client = self.get_client()
        return client.get_article_concepts(pmid)
    
    def get_concept_articles(self, cui: str) -> List[Dict[str, Any]]:
        """
        Get articles that mention a concept.
        
        Args:
            cui: Concept CUI
            
        Returns:
            List of articles
            
        Raises:
            Exception: If the query fails
        """
        client = self.get_client()
        return client.get_concept_articles(cui)
    
    def get_contradictions(self, pmid: str = None) -> List[Dict[str, Any]]:
        """
        Get contradictions in the graph.
        
        Args:
            pmid: Article PMID (optional)
            
        Returns:
            List of contradictions
            
        Raises:
            Exception: If the query fails
        """
        client = self.get_client()
        return client.get_contradictions(pmid)
    
    def get_related_concepts(self, cui: str, relationship_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get concepts related to a concept.
        
        Args:
            cui: Concept CUI
            relationship_type: Relationship type (optional)
            
        Returns:
            List of related concepts
            
        Raises:
            Exception: If the query fails
        """
        client = self.get_client()
        return client.get_related_concepts(cui, relationship_type)
    
    def build_knowledge_graph(self, articles: List[Dict[str, Any]], concepts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Build a knowledge graph from articles and concepts.
        
        Args:
            articles: List of articles
            concepts: List of concepts
            
        Returns:
            Knowledge graph statistics
            
        Raises:
            Exception: If the query fails
        """
        logger.info(f"Building knowledge graph with {len(articles)} articles and {len(concepts)} concepts")
        
        # Connect to the graph database
        self.connect()
        
        # Create article nodes
        article_ids = []
        for article in articles:
            article_id = self.create_article_node(article)
            article_ids.append(article_id)
        
        # Create concept nodes
        concept_ids = []
        for concept in concepts:
            concept_id = self.create_concept_node(concept)
            concept_ids.append(concept_id)
        
        # Create article-concept relationships
        relationship_count = 0
        for article in articles:
            pmid = article.get("pmid", "")
            
            # Get concepts mentioned in the article
            if "concepts" in article:
                for concept in article["concepts"]:
                    cui = concept.get("ui", "")
                    frequency = concept.get("frequency", 1)
                    
                    # Create relationship
                    self.create_article_concept_relationship(
                        pmid, cui, "MENTIONS", {"frequency": frequency}
                    )
                    relationship_count += 1
        
        # Create concept-concept relationships
        for concept in concepts:
            cui = concept.get("ui", "")
            
            # Get related concepts
            if "related_concepts" in concept:
                for related_concept in concept["related_concepts"]:
                    related_cui = related_concept.get("ui", "")
                    relationship_type = related_concept.get("relationship_type", "RELATED_TO")
                    
                    # Create relationship
                    self.create_concept_concept_relationship(
                        cui, related_cui, relationship_type
                    )
                    relationship_count += 1
        
        logger.info(f"Knowledge graph built with {len(article_ids)} articles, {len(concept_ids)} concepts, and {relationship_count} relationships")
        
        # Return statistics
        return {
            "article_count": len(article_ids),
            "concept_count": len(concept_ids),
            "relationship_count": relationship_count
        }
