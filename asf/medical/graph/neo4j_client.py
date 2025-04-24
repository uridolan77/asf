"""
Neo4j client for the Medical Research Synthesizer.

This module provides a client for interacting with the Neo4j graph database,
a popular and mature graph database. The client implements methods for storing
and querying medical knowledge graphs, including articles, concepts, authors,
and their relationships.

The client follows the singleton pattern to ensure a single connection
to the database is maintained throughout the application. It handles
connection management, query execution, and error handling.

Features:
- Connection management with Neo4j database
- Cypher query execution with parameter binding
- Methods for creating and querying graph entities (articles, concepts, etc.)
- Vector similarity search for semantic retrieval
- Error handling and logging
"""
import logging
from typing import Dict, List, Any
from neo4j import GraphDatabase
from medical.core.config import settings
logger = logging.getLogger(__name__)
class Neo4jClient:
    """
    Client for interacting with the Neo4j database.

    This client provides methods for storing and querying medical knowledge graphs
    in Neo4j. It implements the singleton pattern to ensure a single connection
    is maintained throughout the application.

    The client supports operations such as:
    - Creating and retrieving articles, authors, and concepts
    - Establishing relationships between entities
    - Querying for contradictions between articles
    - Performing vector similarity search for semantic retrieval

    All methods include proper error handling and logging to ensure robustness
    in a production environment.
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
        """
        Initialize the Neo4j client.

        This method initializes the client with connection parameters from the
        application settings. It sets up the URI, username, and password for the Neo4j
        database connection but does not establish the connection until
        the connect() method is called.

        The connection parameters are retrieved from the application settings:
        - NEO4J_URI: URI for the Neo4j server (e.g., bolt://localhost:7687)
        - NEO4J_USER: Username for authentication
        - NEO4J_PASSWORD: Password for authentication
        """
        self.uri = settings.NEO4J_URI
        self.user = settings.NEO4J_USER
        self.password = settings.NEO4J_PASSWORD.get_secret_value()
        self.driver = None
        logger.info(f"Neo4j client initialized with uri={self.uri}, user={self.user}")
    def connect(self) -> bool:
        """
        Connect to the Neo4j database.

        This method establishes a connection to the Neo4j database using the
        URI, username, and password specified during initialization. If a connection
        already exists, it is closed before establishing a new one.

        The method handles connection errors and logs appropriate messages.

        Returns:
            bool: True if connection is successful, False otherwise
        """
        if self.driver:
            self.driver.close()
            self.driver = None
            logger.info("Disconnected from Neo4j")

        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            # Test the connection
            with self.driver.session() as session:
                session.run("RETURN 1")
            logger.info(f"Connected to Neo4j at {self.uri}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {str(e)}")
            self.driver = None
            return False
    def execute_query(self, query: str, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Execute a Cypher query against the Neo4j database.

        This method executes a Cypher query with optional parameters against the
        Neo4j database. It handles connection management, query execution,
        and result processing.

        Args:
            query: The Cypher query to execute
            params: Optional dictionary of query parameters

        Returns:
            List of dictionaries containing the query results

        Raises:
            ConnectionError: If connection to the database fails
            RuntimeError: If the query execution fails
        """
        if not self.driver:
            connected = self.connect()
            if not connected:
                raise ConnectionError("Failed to connect to Neo4j database")

        try:
            with self.driver.session() as session:
                result = session.run(query, params or {})
                return [dict(record) for record in result]
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            raise RuntimeError(f"Error executing query: {str(e)}") from e

    def create_article(self, article: Dict[str, Any]) -> str:
        """
        Create an article node in the graph database.

        This method creates a new article node in the Neo4j database with the
        provided article data. The article data should include at least a PMID,
        title, and abstract.

        Args:
            article: Dictionary containing article data (PMID, title, abstract, etc.)

        Returns:
            The PMID of the created article

        Raises:
            ValueError: If required article data is missing
            RuntimeError: If the article creation fails
        """
        required_fields = ["pmid", "title", "abstract"]
        for field in required_fields:
            if field not in article:
                raise ValueError(f"Missing required article field: {field}")

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

        result = self.execute_query(query, article)
        if not result:
            raise RuntimeError("Failed to create article")

        return result[0]["id"]

    def create_author(self, pmid: str, author: str) -> None:
        """
        Create an author node and relationship to an article.

        This method creates a new author node in the Neo4j database and establishes
        an AUTHORED relationship between the author and the specified article.
        If the author already exists, it will reuse the existing node.

        Args:
            pmid: Article PMID
            author: Author name

        Raises:
            ValueError: If PMID or author is empty
            RuntimeError: If the author creation fails
        """
        if not pmid:
            raise ValueError("PMID cannot be empty")
        if not author:
            raise ValueError("Author name cannot be empty")

        query = """
        MATCH (a:Article {pmid: $pmid})
        MERGE (au:Author {name: $author})
        MERGE (au)-[:AUTHORED]->(a)
        """

        self.execute_query(query, {"pmid": pmid, "author": author})

    def create_concept(self, concept: Dict[str, Any]) -> str:
        """
        Create a concept node in the graph database.

        This method creates a new concept node in the Neo4j database with the
        provided concept data. The concept data should include at least a CUI (Concept
        Unique Identifier), name, and semantic types.

        Args:
            concept: Dictionary containing concept data (CUI, name, semantic_types, etc.)

        Returns:
            The CUI of the created concept

        Raises:
            ValueError: If required concept data is missing
            RuntimeError: If the concept creation fails
        """
        required_fields = ["cui", "name", "semantic_types"]
        for field in required_fields:
            if field not in concept:
                raise ValueError(f"Missing required concept field: {field}")

        query = """
        CREATE (c:Concept {
            cui: $cui,
            name: $name,
            semantic_types: $semantic_types
        })
        RETURN c.cui AS id
        """

        result = self.execute_query(query, concept)
        if not result:
            raise RuntimeError("Failed to create concept")

        return result[0]["id"]

    def create_article_concept_relationship(self, pmid: str, cui: str, relationship_type: str = "MENTIONS", properties: Dict[str, Any] = None) -> None:
        """
        Create a relationship between an article and a concept.

        This method creates a relationship between an article and a concept in the
        Neo4j database. The relationship type can be specified, with MENTIONS being
        the default. Additional properties can be provided for the relationship.

        Args:
            pmid: Article PMID
            cui: Concept CUI
            relationship_type: Relationship type (default: MENTIONS)
            properties: Optional dictionary of relationship properties

        Raises:
            ValueError: If PMID or CUI is empty
            RuntimeError: If the relationship creation fails
        """
        if not pmid:
            raise ValueError("PMID cannot be empty")
        if not cui:
            raise ValueError("CUI cannot be empty")

        query = f"""
        MATCH (a:Article {{pmid: $pmid}})
        MATCH (c:Concept {{cui: $cui}})
        MERGE (a)-[r:{relationship_type}]->(c)
        """

        params = {"pmid": pmid, "cui": cui}
        if properties:
            query += """
            SET r += $properties
            """
            params["properties"] = properties

        self.execute_query(query, params)

    def create_concept_relationship(self, cui1: str, cui2: str, relationship_type: str, properties: Dict[str, Any] = None) -> None:
        """
        Create a relationship between two concepts.

        This method creates a relationship between two concepts in the Neo4j database.
        The relationship type must be specified. Additional properties can be provided
        for the relationship.

        Args:
            cui1: First concept CUI
            cui2: Second concept CUI
            relationship_type: Relationship type
            properties: Optional dictionary of relationship properties

        Raises:
            ValueError: If CUI1, CUI2, or relationship_type is empty
            RuntimeError: If the relationship creation fails
        """
        if not cui1:
            raise ValueError("First concept CUI cannot be empty")
        if not cui2:
            raise ValueError("Second concept CUI cannot be empty")
        if not relationship_type:
            raise ValueError("Relationship type cannot be empty")

        query = f"""
        MATCH (c1:Concept {{cui: $cui1}})
        MATCH (c2:Concept {{cui: $cui2}})
        MERGE (c1)-[r:{relationship_type}]->(c2)
        """

        params = {"cui1": cui1, "cui2": cui2}
        if properties:
            query += """
            SET r += $properties
            """
            params["properties"] = properties

        self.execute_query(query, params)

    def create_contradiction(self, pmid1: str, pmid2: str, contradiction_score: float, confidence: float, topic: str = None, explanation: str = None) -> None:
        """
        Create a contradiction relationship between two articles.

        This method creates a CONTRADICTS relationship between two articles in the
        Neo4j database. The relationship includes properties such as contradiction score,
        confidence level, topic, and explanation.

        Args:
            pmid1: First article PMID
            pmid2: Second article PMID
            contradiction_score: Contradiction score (0.0 to 1.0)
            confidence: Confidence level (0.0 to 1.0)
            topic: Optional contradiction topic
            explanation: Optional contradiction explanation

        Raises:
            ValueError: If PMIDs are empty or scores are invalid
            RuntimeError: If the contradiction creation fails
        """
        if not pmid1:
            raise ValueError("First article PMID cannot be empty")
        if not pmid2:
            raise ValueError("Second article PMID cannot be empty")
        if not 0 <= contradiction_score <= 1:
            raise ValueError("Contradiction score must be between 0 and 1")
        if not 0 <= confidence <= 1:
            raise ValueError("Confidence level must be between 0 and 1")

        query = """
        MATCH (a1:Article {pmid: $pmid1})
        MATCH (a2:Article {pmid: $pmid2})
        MERGE (a1)-[r:CONTRADICTS]->(a2)
        SET r.contradiction_score = $contradiction_score,
            r.confidence = $confidence
        """

        params = {
            "pmid1": pmid1,
            "pmid2": pmid2,
            "contradiction_score": contradiction_score,
            "confidence": confidence
        }

        if topic:
            query += """,
            r.topic = $topic
            """
            params["topic"] = topic

        if explanation:
            query += """,
            r.explanation = $explanation
            """
            params["explanation"] = explanation

        self.execute_query(query, params)

    def get_article(self, pmid: str) -> Dict[str, Any]:
        """
        Get an article from the graph database.

        This method retrieves an article from the Neo4j database by its PMID.
        It also retrieves the authors of the article.

        Args:
            pmid: Article PMID

        Returns:
            Dictionary containing article data or None if not found

        Raises:
            ValueError: If PMID is empty
            RuntimeError: If the query fails
        """
        if not pmid:
            raise ValueError("PMID cannot be empty")

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

        result = self.execute_query(query, {"pmid": pmid})
        return result[0] if result else None

    def get_concept(self, cui: str) -> Dict[str, Any]:
        """
        Get a concept from the graph database.

        This method retrieves a concept from the Neo4j database by its CUI
        (Concept Unique Identifier).

        Args:
            cui: Concept CUI

        Returns:
            Dictionary containing concept data or None if not found

        Raises:
            ValueError: If CUI is empty
            RuntimeError: If the query fails
        """
        if not cui:
            raise ValueError("CUI cannot be empty")

        query = """
        MATCH (c:Concept {cui: $cui})
        RETURN c.cui AS cui,
               c.name AS name,
               c.semantic_types AS semantic_types
        """

        result = self.execute_query(query, {"cui": cui})
        return result[0] if result else None

    def get_concepts_by_article(self, pmid: str) -> List[Dict[str, Any]]:
        """
        Get concepts mentioned in an article.

        This method retrieves all concepts that are mentioned in a specific article
        from the Neo4j database. It includes the frequency of mentions if available.

        Args:
            pmid: Article PMID

        Returns:
            List of dictionaries containing concept data

        Raises:
            ValueError: If PMID is empty
            RuntimeError: If the query fails
        """
        if not pmid:
            raise ValueError("PMID cannot be empty")

        query = """
        MATCH (a:Article {pmid: $pmid})-[r:MENTIONS]->(c:Concept)
        RETURN c.cui AS cui,
               c.name AS name,
               c.semantic_types AS semantic_types,
               r.frequency AS frequency
        """

        return self.execute_query(query, {"pmid": pmid})

    def get_articles_by_concept(self, cui: str, max_results: int = 100) -> List[Dict[str, Any]]:
        """
        Get articles that mention a concept.

        This method retrieves all articles that mention a specific concept from the
        Neo4j database. It includes the frequency of mentions if available.

        Args:
            cui: Concept CUI
            max_results: Maximum number of results to return (default: 100)

        Returns:
            List of dictionaries containing article data

        Raises:
            ValueError: If CUI is empty
            RuntimeError: If the query fails
        """
        if not cui:
            raise ValueError("CUI cannot be empty")

        query = """
        MATCH (a:Article)-[r:MENTIONS]->(c:Concept {cui: $cui})
        RETURN a.pmid AS pmid,
               a.title AS title,
               a.journal AS journal,
               a.publication_date AS publication_date,
               r.frequency AS frequency
        ORDER BY r.frequency DESC
        LIMIT $max_results
        """

        return self.execute_query(query, {"cui": cui, "max_results": max_results})

    def get_contradictions(self, pmid: str = None, max_results: int = 100) -> List[Dict[str, Any]]:
        """
        Get contradictions in the graph database.

        This method retrieves contradictions between articles from the Neo4j database.
        If a PMID is provided, it retrieves contradictions involving that article.
        Otherwise, it retrieves all contradictions.

        Args:
            pmid: Article PMID (optional)
            max_results: Maximum number of results to return (default: 100)

        Returns:
            List of dictionaries containing contradiction data

        Raises:
            RuntimeError: If the query fails
        """
        if pmid:
            # Get contradictions involving a specific article
            query = """
            MATCH (a1:Article {pmid: $pmid})-[r:CONTRADICTS]->(a2:Article)
            RETURN a1.pmid AS pmid1,
                   a1.title AS title1,
                   a2.pmid AS pmid2,
                   a2.title AS title2,
                   r.contradiction_score AS contradiction_score,
                   r.confidence AS confidence,
                   r.topic AS topic,
                   r.explanation AS explanation
            UNION
            MATCH (a1:Article)-[r:CONTRADICTS]->(a2:Article {pmid: $pmid})
            RETURN a1.pmid AS pmid1,
                   a1.title AS title1,
                   a2.pmid AS pmid2,
                   a2.title AS title2,
                   r.contradiction_score AS contradiction_score,
                   r.confidence AS confidence,
                   r.topic AS topic,
                   r.explanation AS explanation
            ORDER BY contradiction_score DESC
            LIMIT $max_results
            """
            return self.execute_query(query, {"pmid": pmid, "max_results": max_results})
        else:
            # Get all contradictions
            query = """
            MATCH (a1:Article)-[r:CONTRADICTS]->(a2:Article)
            RETURN a1.pmid AS pmid1,
                   a1.title AS title1,
                   a2.pmid AS pmid2,
                   a2.title AS title2,
                   r.contradiction_score AS contradiction_score,
                   r.confidence AS confidence,
                   r.topic AS topic,
                   r.explanation AS explanation
            ORDER BY contradiction_score DESC
            LIMIT $max_results
            """
            return self.execute_query(query, {"max_results": max_results})

    def get_related_concepts(self, cui: str, relationship_type: str = None, max_results: int = 100) -> List[Dict[str, Any]]:
        """
        Get concepts related to a concept.

        This method retrieves concepts that are related to a specific concept from the
        Neo4j database. If a relationship type is provided, it retrieves only concepts
        with that relationship. Otherwise, it retrieves all related concepts.

        Args:
            cui: Concept CUI
            relationship_type: Relationship type (optional)
            max_results: Maximum number of results to return (default: 100)

        Returns:
            List of dictionaries containing related concept data

        Raises:
            ValueError: If CUI is empty
            RuntimeError: If the query fails
        """
        if not cui:
            raise ValueError("CUI cannot be empty")

        if relationship_type:
            # Get concepts with a specific relationship type
            query = f"""
            MATCH (c1:Concept {{cui: $cui}})-[r:{relationship_type}]->(c2:Concept)
            RETURN c2.cui AS cui,
                   c2.name AS name,
                   c2.semantic_types AS semantic_types,
                   type(r) AS relationship_type
            ORDER BY c2.name
            LIMIT $max_results
            """
        else:
            # Get all related concepts
            query = """
            MATCH (c1:Concept {cui: $cui})-[r]->(c2:Concept)
            RETURN c2.cui AS cui,
                   c2.name AS name,
                   c2.semantic_types AS semantic_types,
                   type(r) AS relationship_type
            ORDER BY c2.name
            LIMIT $max_results
            """

        return self.execute_query(query, {"cui": cui, "max_results": max_results})

    def search_articles_by_embedding(self, embedding: List[float], max_results: int = 20, similarity_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Search for articles with similar embeddings.

        This method searches for articles with embeddings similar to the provided embedding
        using cosine similarity. It requires the Neo4j Graph Data Science (GDS) library.

        Args:
            embedding: Query embedding as a list of floats
            max_results: Maximum number of results to return (default: 20)
            similarity_threshold: Minimum similarity score (default: 0.5)

        Returns:
            List of dictionaries containing article data with similarity scores

        Raises:
            ValueError: If the embedding is invalid
            RuntimeError: If the query execution fails
        """
        if not embedding or not isinstance(embedding, list):
            raise ValueError("Embedding must be a non-empty list of floats")

        query = """
        MATCH (a:Article)
        WHERE a.embedding IS NOT NULL
        WITH a, gds.similarity.cosine(a.embedding, $embedding) AS similarity
        WHERE similarity > $similarity_threshold
        ORDER BY similarity DESC
        LIMIT $max_results
        RETURN a.pmid AS pmid,
               a.title AS title,
               a.abstract AS abstract,
               a.journal AS journal,
               a.publication_date AS publication_date,
               similarity
        """

        params = {
            "embedding": embedding,
            "max_results": max_results,
            "similarity_threshold": similarity_threshold
        }

        return self.execute_query(query, params)