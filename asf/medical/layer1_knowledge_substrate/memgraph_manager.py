"""
Memgraph Manager
This module provides a manager for Memgraph database operations,
serving as an alternative to Neo4j in the ChronoGnosisLayer.
"""
import logging
import asyncio
from typing import List
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("memgraph-manager")
RETRY_BACKOFF_SECONDS = 1
MAX_RETRIES = 3
class MemgraphConfig:
    """Configuration for Memgraph connection."""
    def __init__(
        self,
        host: str = "localhost",
        port: int = 7687,
        username: str = "",
        password: str = "",
        encrypted: bool = False
    ):
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.encrypted = encrypted
class MemgraphManager:
    """
    Manager for Memgraph database operations.
    Provides an interface similar to Neo4j's AsyncGraphDatabase.
        Initialize the Memgraph manager.
        Args:
            config: Memgraph configuration
        try:
            import mgclient
            loop = asyncio.get_event_loop()
            self.connection = await loop.run_in_executor(
                None,
                lambda: mgclient.connect(
                    host=self.config.host,
                    port=self.config.port,
                    username=self.config.username,
                    password=self.config.password,
                    encrypted=self.config.encrypted
                )
            )
            self.connection.autocommit = True
            self.cursor = await loop.run_in_executor(
                None, lambda: self.connection.cursor()
            )
            logger.info(f"Connected to Memgraph at {self.config.host}:{self.config.port}")
        except ImportError:
            logger.error("Failed to import mgclient. Please install it: pip install mgclient")
            raise ImportError("Please install mgclient: pip install mgclient")
        except Exception as e:
            logger.error(f"Failed to connect to Memgraph: {e}")
            raise RuntimeError(f"Failed to connect to Memgraph: {e}")
    async def close(self):
        Run a Cypher query on Memgraph.
        Args:
            query: Cypher query
            params: Query parameters
        Returns:
            List of results as dictionaries
        Fetch a subgraph around an entity.
        Args:
            entity_id: ID of the central entity
            hops: Number of hops to traverse
        Returns:
            Subgraph data in Neo4j-compatible format
        MATCH path = (source:Entity {{id: $entity_id}})-[*1..{hops}]-(target:Entity)
        RETURN source, target, relationships(path) as rels
                MATCH (n:Entity {{id: $entity_id}})
                CALL mg.alpha.get_subgraph(n, {hops})
                YIELD nodes, relationships
                RETURN nodes, relationships
        Get all entity IDs from the database.
        Returns:
            List of entity IDs
        Create a new entity in the database.
        Args:
            entity_data: Entity data
        Returns:
            ID of the created entity
        CREATE (n:Entity)
        SET n.id = $id, {properties}
        RETURN n.id AS id
        Create a relationship between two entities.
        Args:
            source_id: ID of the source entity
            target_id: ID of the target entity
            rel_type: Type of relationship
            properties: Relationship properties
        Returns:
            True if successful
        MATCH (a:Entity {{id: $source_id}}), (b:Entity {{id: $target_id}})
        CREATE (a)-[r:{rel_type}{prop_string}]->(b)
        RETURN r