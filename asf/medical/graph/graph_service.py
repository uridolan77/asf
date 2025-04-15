"""
Graph service for the Medical Research Synthesizer.

This module provides a unified service for interacting with different graph
database backends (Neo4j, Memgraph). It abstracts away the specific details
of each database implementation, providing a consistent interface for storing
and querying medical knowledge graphs.

The GraphService follows the singleton pattern to ensure a single instance
is used throughout the application. It dynamically loads the appropriate
client based on the configured graph database type in the application settings.

Features:
- Dynamic client loading based on configuration
- Unified interface for different graph database backends
- Connection management and query execution
- Methods for creating and querying graph entities (articles, concepts, etc.)
- Knowledge graph building and analysis
"""
import logging
from typing import Dict, List, Any
from medical.core.config import settings
logger = logging.getLogger(__name__)
class GraphService:
    """
    Service for interacting with the graph database.

    This service provides a unified interface for interacting with different graph
    databases (Neo4j, Memgraph). It abstracts away the specific details of each
    database implementation, allowing the application to work with any supported
    graph database without changing the code that uses this service.

    The service dynamically loads the appropriate client based on the configured
    graph database type in the application settings. It implements the singleton
    pattern to ensure a single instance is used throughout the application.

    Supported graph database types:
    - neo4j: Neo4j graph database
    - memgraph: Memgraph graph database
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
        """
        Initialize the graph service.

        This method initializes the service with the graph database type specified
        in the application settings. It does not establish a connection to the
        database until the connect() method is called.

        The graph database type is determined by the GRAPH_DB_TYPE setting in the
        application configuration. Supported types are 'neo4j' and 'memgraph'.

        The client is lazily loaded when needed, allowing the service to be
        initialized without immediately connecting to the database.
        """
        self.graph_db_type = settings.GRAPH_DB_TYPE.lower()
        self.client = None
        logger.info(f"Graph service initialized with graph_db_type={self.graph_db_type}")
    def get_client(self):
        """
        Get the graph database client.

        This method returns the appropriate graph database client based on the
        configured graph database type. If the client has not been initialized yet,
        it dynamically imports and instantiates the appropriate client class.

        Supported graph database types:
        - neo4j: Uses Neo4jClient
        - memgraph: Uses MemgraphClient

        Returns:
            The graph database client instance

        Raises:
            ValueError: If the graph database type is not supported
        """
        if self.client is None:
            if self.graph_db_type == "neo4j":
                from medical.graph.neo4j_client import Neo4jClient
                self.client = Neo4jClient()
                logger.info("Initialized Neo4j client")
            elif self.graph_db_type == "memgraph":
                from medical.graph.memgraph_client import MemgraphClient
                self.client = MemgraphClient()
                logger.info("Initialized Memgraph client")
            else:
                raise ValueError(f"Unsupported graph database type: {self.graph_db_type}")
        return self.client
    def execute_query(self, query: str, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Execute a Cypher query against the graph database.

        This method executes a Cypher query with optional parameters against the
        graph database using the appropriate client. It ensures the client is
        initialized and connected before executing the query.

        Args:
            query: The Cypher query to execute
            params: Optional dictionary of query parameters

        Returns:
            List of dictionaries containing the query results

        Raises:
            Exception: If the query execution fails
        """
        client = self.get_client()
        return client.execute_query(query, params)

    def connect(self) -> bool:
        """
        Connect to the graph database.

        This method establishes a connection to the graph database using the
        appropriate client. If a client already exists, it is disconnected
        before establishing a new connection.

        Returns:
            True if connection is successful, False otherwise
        """
        client = self.get_client()
        return client.connect()

    def disconnect(self) -> None:
        """
        Disconnect from the graph database.

        This method disconnects from the graph database if a client exists.
        It is safe to call this method even if no connection has been established.

        Returns:
            None
        """
        if self.client:
            self.client.disconnect()
            logger.info("Disconnected from graph database")

    def create_article(self, article: Dict[str, Any]) -> str:
        """
        Create an article node in the graph database.

        This method creates a new article node in the graph database with the
        provided article data. The article data should include at least a PMID,
        title, and abstract.

        Args:
            article: Dictionary containing article data (PMID, title, abstract, etc.)

        Returns:
            The PMID of the created article

        Raises:
            Exception: If the article creation fails
        """
        client = self.get_client()
        return client.create_article(article)