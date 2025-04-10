"""
Memgraph client for the Medical Research Synthesizer.

This module provides a client for interacting with the Memgraph database.
"""

import logging
import os
from typing import Dict, List, Optional, Any, Union, Tuple
import pymemgraph

from asf.medical.core.config import settings

# Set up logging
logger = logging.getLogger(__name__)

class MemgraphClient:
    """
    Client for interacting with the Memgraph database.
    
    This client provides methods for storing and querying medical knowledge graphs.
    """
    
    _instance = None
    
    def __new__(cls):
        """
        Create a singleton instance of the Memgraph client.
        
        Returns:
            MemgraphClient: The singleton instance
        """
        if cls._instance is None:
            cls._instance = super(MemgraphClient, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the Memgraph client."""
        self.host = settings.MEMGRAPH_HOST
        self.port = settings.MEMGRAPH_PORT
        self.connection = None
        
        logger.info(f"Memgraph client initialized with host={self.host}, port={self.port}")
    
    def connect(self):
        """
        Connect to the Memgraph database.
        
        Returns:
            bool: True if connection is successful, False otherwise
        """
        try:
            logger.info(f"Connecting to Memgraph at {self.host}:{self.port}")
            self.connection = pymemgraph.connect(host=self.host, port=self.port)
            logger.info("Connected to Memgraph")
            return True
        except Exception as e:
            logger.error(f"Error connecting to Memgraph: {str(e)}")
            return False
    
    def disconnect(self):
        """Disconnect from the Memgraph database."""
        if self.connection:
            self.connection.close()
            self.connection = None
            logger.info("Disconnected from Memgraph")
    
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
        if not self.connection:
            if not self.connect():
                raise Exception("Not connected to Memgraph")
        
        try:
            logger.debug(f"Executing query: {query}")
            cursor = self.connection.cursor()
            cursor.execute(query, params or {})
            
            # Get column names
            columns = [column[0] for column in cursor.description] if cursor.description else []
            
            # Get results
            results = []
            for row in cursor:
                result = {}
                for i, value in enumerate(row):
                    result[columns[i]] = value
                results.append(result)
            
            cursor.close()
            
            logger.debug(f"Query returned {len(results)} results")
            
            return results
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            raise
    
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
        # Extract article properties
        pmid = article.get("pmid", "")
        title = article.get("title", "")
        abstract = article.get("abstract", "")
        journal = article.get("journal", "")
        publication_date = article.get("publication_date", "")
        authors = article.get("authors", [])
        
        # Create query
        query = """
        CREATE (a:Article {
            pmid: $pmid,
            title: $title,
            abstract: $abstract,
            journal: $journal,
            publication_date: $publication_date
        })
        RETURN a.pmid AS id
        """
        
        # Execute query
        params = {
            "pmid": pmid,
            "title": title,
            "abstract": abstract,
            "journal": journal,
            "publication_date": publication_date
        }
        
        result = self.execute_query(query, params)
        
        # Create author nodes and relationships
        for author in authors:
            self.create_author_relationship(pmid, author)
        
        return result[0]["id"] if result else pmid
    
    def create_author_relationship(self, pmid: str, author: str) -> None:
        """
        Create an author node and relationship to an article.
        
        Args:
            pmid: Article PMID
            author: Author name
            
        Raises:
            Exception: If the query fails
        """
        # Create query
        query = """
        MATCH (a:Article {pmid: $pmid})
        MERGE (au:Author {name: $author})
        MERGE (au)-[:AUTHORED]->(a)
        """
        
        # Execute query
        params = {
            "pmid": pmid,
            "author": author
        }
        
        self.execute_query(query, params)
    
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
        # Extract concept properties
        cui = concept.get("ui", "")
        name = concept.get("name", "")
        semantic_types = concept.get("semantic_types", [])
        
        # Create query
        query = """
        CREATE (c:Concept {
            cui: $cui,
            name: $name,
            semantic_types: $semantic_types
        })
        RETURN c.cui AS id
        """
        
        # Execute query
        params = {
            "cui": cui,
            "name": name,
            "semantic_types": semantic_types
        }
        
        result = self.execute_query(query, params)
        
        return result[0]["id"] if result else cui
    
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
        # Create query
        query = f"""
        MATCH (a:Article {{pmid: $pmid}})
        MATCH (c:Concept {{cui: $cui}})
        MERGE (a)-[r:{relationship_type}]->(c)
        """
        
        # Add properties if provided
        if properties:
            property_set = ", ".join([f"r.{key} = ${key}" for key in properties.keys()])
            query += f" SET {property_set}"
        
        # Execute query
        params = {
            "pmid": pmid,
            "cui": cui,
            **properties or {}
        }
        
        self.execute_query(query, params)
    
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
        # Create query
        query = f"""
        MATCH (c1:Concept {{cui: $cui1}})
        MATCH (c2:Concept {{cui: $cui2}})
        MERGE (c1)-[r:{relationship_type}]->(c2)
        """
        
        # Add properties if provided
        if properties:
            property_set = ", ".join([f"r.{key} = ${key}" for key in properties.keys()])
            query += f" SET {property_set}"
        
        # Execute query
        params = {
            "cui1": cui1,
            "cui2": cui2,
            **properties or {}
        }
        
        self.execute_query(query, params)
    
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
        # Create query
        query = """
        MATCH (a1:Article {pmid: $pmid1})
        MATCH (a2:Article {pmid: $pmid2})
        MERGE (a1)-[r:CONTRADICTS]->(a2)
        SET r.contradiction_score = $contradiction_score,
            r.confidence = $confidence
        """
        
        # Add topic if provided
        if topic:
            query += ", r.topic = $topic"
        
        # Add explanation if provided
        if explanation:
            query += ", r.explanation = $explanation"
        
        # Execute query
        params = {
            "pmid1": pmid1,
            "pmid2": pmid2,
            "contradiction_score": contradiction_score,
            "confidence": confidence,
            "topic": topic,
            "explanation": explanation
        }
        
        self.execute_query(query, params)
    
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
        # Create query
        query = """
        MATCH (a:Article {pmid: $pmid})
        OPTIONAL MATCH (au:Author)-[:AUTHORED]->(a)
        RETURN a.pmid AS pmid,
               a.title AS title,
               a.abstract AS abstract,
               a.journal AS journal,
               a.publication_date AS publication_date,
               collect(au.name) AS authors
        """
        
        # Execute query
        params = {
            "pmid": pmid
        }
        
        result = self.execute_query(query, params)
        
        return result[0] if result else None
    
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
        # Create query
        query = """
        MATCH (c:Concept {cui: $cui})
        RETURN c.cui AS cui,
               c.name AS name,
               c.semantic_types AS semantic_types
        """
        
        # Execute query
        params = {
            "cui": cui
        }
        
        result = self.execute_query(query, params)
        
        return result[0] if result else None
    
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
        # Create query
        query = """
        MATCH (a:Article {pmid: $pmid})-[r:MENTIONS]->(c:Concept)
        RETURN c.cui AS cui,
               c.name AS name,
               c.semantic_types AS semantic_types,
               r.frequency AS frequency
        """
        
        # Execute query
        params = {
            "pmid": pmid
        }
        
        return self.execute_query(query, params)
    
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
        # Create query
        query = """
        MATCH (a:Article)-[r:MENTIONS]->(c:Concept {cui: $cui})
        RETURN a.pmid AS pmid,
               a.title AS title,
               a.journal AS journal,
               a.publication_date AS publication_date,
               r.frequency AS frequency
        """
        
        # Execute query
        params = {
            "cui": cui
        }
        
        return self.execute_query(query, params)
    
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
        # Create query
        if pmid:
            query = """
            MATCH (a1:Article {pmid: $pmid})-[r:CONTRADICTS]->(a2:Article)
            RETURN a1.pmid AS pmid1,
                   a1.title AS title1,
                   a2.pmid AS pmid2,
                   a2.title AS title2,
                   r.contradiction_score AS contradiction_score,
                   r.confidence AS confidence,
                   r.topic AS topic
            UNION
            MATCH (a1:Article)-[r:CONTRADICTS]->(a2:Article {pmid: $pmid})
            RETURN a1.pmid AS pmid1,
                   a1.title AS title1,
                   a2.pmid AS pmid2,
                   a2.title AS title2,
                   r.contradiction_score AS contradiction_score,
                   r.confidence AS confidence,
                   r.topic AS topic
            """
            
            # Execute query
            params = {
                "pmid": pmid
            }
        else:
            query = """
            MATCH (a1:Article)-[r:CONTRADICTS]->(a2:Article)
            RETURN a1.pmid AS pmid1,
                   a1.title AS title1,
                   a2.pmid AS pmid2,
                   a2.title AS title2,
                   r.contradiction_score AS contradiction_score,
                   r.confidence AS confidence,
                   r.topic AS topic
            """
            
            # Execute query
            params = {}
        
        return self.execute_query(query, params)
    
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
        # Create query
        if relationship_type:
            query = f"""
            MATCH (c1:Concept {{cui: $cui}})-[r:{relationship_type}]->(c2:Concept)
            RETURN c2.cui AS cui,
                   c2.name AS name,
                   c2.semantic_types AS semantic_types,
                   type(r) AS relationship_type
            """
        else:
            query = """
            MATCH (c1:Concept {cui: $cui})-[r]->(c2:Concept)
            RETURN c2.cui AS cui,
                   c2.name AS name,
                   c2.semantic_types AS semantic_types,
                   type(r) AS relationship_type
            """
        
        # Execute query
        params = {
            "cui": cui
        }
        
        return self.execute_query(query, params)
