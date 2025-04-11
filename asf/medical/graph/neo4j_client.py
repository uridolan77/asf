"""
Neo4j client for the Medical Research Synthesizer.

This module provides a client for interacting with the Neo4j database.
"""

import logging
import time
from typing import Dict, List, Optional, Any
import numpy as np
from neo4j import GraphDatabase, exceptions as neo4j_exceptions

from asf.medical.core.config import settings

# Set up logging
logger = logging.getLogger(__name__)

class Neo4jClient:
    """
    Client for interacting with the Neo4j database.

    This client provides methods for storing and querying medical knowledge graphs.
    """

    _instance = None

    def __new__(cls):
        """
        Create a singleton instance of the Neo4j client.

        Returns:
            Neo4jClient: The singleton instance
        """
        if cls._instance is None:
            cls._instance = super(Neo4jClient, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize the Neo4j client."""
        self.uri = settings.NEO4J_URI
        self.user = settings.NEO4J_USER
        self.password = settings.NEO4J_PASSWORD.get_secret_value()
        self.driver = None

        logger.info(f"Neo4j client initialized with uri={self.uri}, user={self.user}")

    def connect(self):
        """
        Connect to the Neo4j database.

        Returns:
            bool: True if connection is successful, False otherwise
        """
        try:
            logger.info(f"Connecting to Neo4j at {self.uri}")
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            logger.info("Connected to Neo4j")
            return True
        except Exception as e:
            logger.error(f"Error connecting to Neo4j: {str(e)}")
            return False

    def disconnect(self):
        """Disconnect from the Neo4j database."""
        if self.driver:
            self.driver.close()
            self.driver = None
            logger.info("Disconnected from Neo4j")

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
        if not self.driver:
            if not self.connect():
                raise Exception("Not connected to Neo4j")

        try:
            logger.debug(f"Executing query: {query}")
            with self.driver.session() as session:
                result = session.run(query, params or {})
                records = list(result)

                # Convert records to dictionaries
                results = []
                for record in records:
                    results.append(dict(record))

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
        }

        # Add properties if provided
        if properties:
            params.update(properties)

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
        }

        # Add properties if provided
        if properties:
            params.update(properties)

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

    def vector_search(self, embedding: np.ndarray, max_results: int = 20, max_retries: int = 3, retry_delay: float = 1.0) -> List[Dict[str, Any]]:
        """
        Search for articles with similar embeddings.

        Args:
            embedding: Query embedding as a numpy array
            max_results: Maximum number of results to return
            max_retries: Maximum number of retries on failure
            retry_delay: Delay between retries in seconds

        Returns:
            List of articles with similarity scores

        Raises:
            ValueError: If the embedding is invalid
            ConnectionError: If the database connection fails after retries
            RuntimeError: If the query execution fails after retries
        """
        if not isinstance(embedding, np.ndarray):
            raise ValueError("Embedding must be a numpy array")

        if max_results < 1:
            raise ValueError("max_results must be at least 1")

        logger.info(f"Neo4j vector search with max_results={max_results}")

        # Connect to the database with retry
        retry_count = 0
        connected = False

        while retry_count < max_retries and not connected:
            try:
                connected = self.connect()
                if not connected:
                    raise ConnectionError("Failed to connect to Neo4j database")
            except Exception as e:
                retry_count += 1
                if retry_count >= max_retries:
                    logger.error(f"Failed to connect to Neo4j database after {max_retries} retries: {str(e)}")
                    raise ConnectionError(f"Failed to connect to Neo4j database: {str(e)}") from e
                logger.warning(f"Connection attempt {retry_count} failed, retrying in {retry_delay} seconds: {str(e)}")
                time.sleep(retry_delay)

        # Convert embedding to string for Cypher query
        try:
            embedding_str = str(embedding.tolist())
        except Exception as e:
            logger.error(f"Failed to convert embedding to string: {str(e)}")
            raise ValueError(f"Invalid embedding format: {str(e)}") from e

        # Query for articles with similar embeddings
        # This assumes that articles have an 'embedding' property
        query = """
        MATCH (a:Article)
        WHERE a.embedding IS NOT NULL
        WITH a, gds.similarity.cosine(a.embedding, $embedding) AS similarity
        WHERE similarity > 0.5  -- Add a similarity threshold
        ORDER BY similarity DESC
        LIMIT $max_results
        RETURN a.pmid AS pmid, a.title AS title, a.abstract AS abstract, a.authors AS authors,
               a.publication_date AS publication_date, a.journal AS journal, similarity
        """

        params = {
            "embedding": embedding_str,
            "max_results": max_results
        }

        # Execute query with retry
        retry_count = 0
        while retry_count < max_retries:
            try:
                results = self.execute_query(query, params)
                logger.info(f"Neo4j vector search found {len(results)} results")
                return results
            except neo4j_exceptions.ServiceUnavailable as e:
                retry_count += 1
                if retry_count >= max_retries:
                    logger.error(f"Neo4j service unavailable after {max_retries} retries: {str(e)}")
                    raise ConnectionError(f"Neo4j service unavailable: {str(e)}") from e
                logger.warning(f"Neo4j service unavailable, retry {retry_count} in {retry_delay} seconds: {str(e)}")
                time.sleep(retry_delay)
            except neo4j_exceptions.ClientError as e:
                logger.error(f"Neo4j client error: {str(e)}")
                raise ValueError(f"Invalid query or parameters: {str(e)}") from e
            except Exception as e:
                logger.error(f"Error performing Neo4j vector search: {str(e)}")
                raise RuntimeError(f"Failed to execute Neo4j vector search: {str(e)}") from e

        # This should never be reached due to the exception handling above
        return []