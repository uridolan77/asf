"""
Memgraph Manager

This module provides a manager for Memgraph database operations,
serving as an alternative to Neo4j in the ChronoGnosisLayer.
"""

import logging
import asyncio
from typing import Dict, List, Optional, Tuple, Union, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("memgraph-manager")

# Constants
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
    """

    def __init__(self, config: MemgraphConfig):
        """
        Initialize the Memgraph manager.

        Args:
            config: Memgraph configuration
        """
        self.config = config
        self.connection = None
        self.cursor = None

    async def connect(self):
        """Establish connection to Memgraph."""
        try:
            import mgclient

            # mgclient doesn't support async operations natively,
            # so we'll use asyncio to run it in a thread pool
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

            # Set autocommit to True for simplicity
            self.connection.autocommit = True

            # Create cursor
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
        """Close connection to Memgraph."""
        if self.connection:
            try:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self.connection.close)
                logger.info("Closed connection to Memgraph")
            except Exception as e:
                logger.error(f"Error closing Memgraph connection: {e}")

    async def run_query(self, query: str, params: Dict = None) -> List[Dict]:
        """
        Run a Cypher query on Memgraph.

        Args:
            query: Cypher query
            params: Query parameters

        Returns:
            List of results as dictionaries
        """
        if params is None:
            params = {}

        retry_count = 0
        while retry_count < MAX_RETRIES:
            try:
                loop = asyncio.get_event_loop()

                # Execute query
                await loop.run_in_executor(
                    None, lambda: self.cursor.execute(query, params)
                )

                # Fetch results
                results = await loop.run_in_executor(
                    None, self.cursor.fetchall
                )

                # Convert results to dictionaries
                columns = [column[0] for column in self.cursor.description]
                return [dict(zip(columns, row)) for row in results]
            except Exception as e:
                retry_count += 1
                logger.warning(
                    f"Memgraph query failed (attempt {retry_count}/{MAX_RETRIES}): {e}"
                )
                if retry_count >= MAX_RETRIES:
                    logger.error(f"Memgraph query failed after {MAX_RETRIES} attempts: {e}")
                    raise RuntimeError(f"Memgraph query failed: {e}")
                await asyncio.sleep(RETRY_BACKOFF_SECONDS * (2 ** retry_count))

    async def fetch_subgraph(self, entity_id: str, hops: int = 2) -> List[Dict]:
        """
        Fetch a subgraph around an entity.

        Args:
            entity_id: ID of the central entity
            hops: Number of hops to traverse

        Returns:
            Subgraph data in Neo4j-compatible format
        """
        # First, try using the path-based approach for Neo4j compatibility
        query = f"""
        MATCH path = (source:Entity {{id: $entity_id}})-[*1..{hops}]-(target:Entity)
        RETURN source, target, relationships(path) as rels
        """

        try:
            result = await self.run_query(query, {"entity_id": entity_id})

            # Format result to match Neo4j format expected by ChronoGnosisLayer
            formatted_result = []
            for record in result:
                formatted_record = {
                    "source_node": record["source"],
                    "target_nodes": [record["target"]],
                    "rels": record["rels"]
                }
                formatted_result.append(formatted_record)

            return formatted_result
        except Exception as e:
            # Fall back to the subgraph procedure if the path-based approach fails
            logger.warning(f"Path-based subgraph query failed for entity {entity_id}: {e}. Falling back to subgraph procedure.")

            try:
                # Use Memgraph's subgraph procedure
                fallback_query = f"""
                MATCH (n:Entity {{id: $entity_id}})
                CALL mg.alpha.get_subgraph(n, {hops})
                YIELD nodes, relationships
                RETURN nodes, relationships
                """

                result = await self.run_query(fallback_query, {"entity_id": entity_id})

                # Convert to Neo4j-compatible format
                if result and len(result) > 0:
                    nodes = result[0].get("nodes", [])
                    relationships = result[0].get("relationships", [])

                    # Find the source node
                    source_node = None
                    for node in nodes:
                        if node.get("id") == entity_id:
                            source_node = node
                            break

                    if not source_node:
                        return []

                    # Create formatted result
                    formatted_result = []
                    for rel in relationships:
                        source_id = rel.get("source")
                        target_id = rel.get("target")

                        # Find target node
                        target_node = None
                        for node in nodes:
                            if node.get("id") == target_id:
                                target_node = node
                                break

                        if target_node:
                            formatted_record = {
                                "source_node": source_node,
                                "target_nodes": [target_node],
                                "rels": [rel]
                            }
                            formatted_result.append(formatted_record)

                    return formatted_result

                return []
            except Exception as e:
                logger.error(f"Error fetching subgraph for entity {entity_id}: {e}")
                return []

    async def get_all_entity_ids(self) -> List[str]:
        """
        Get all entity IDs from the database.

        Returns:
            List of entity IDs
        """
        query = "MATCH (n:Entity) RETURN n.id AS id"

        try:
            result = await self.run_query(query)
            return [row["id"] for row in result]
        except Exception as e:
            logger.error(f"Error getting all entity IDs: {e}")
            raise RuntimeError(f"Failed to get all entity IDs: {e}")

    async def create_entity(self, entity_data: Dict) -> str:
        """
        Create a new entity in the database.

        Args:
            entity_data: Entity data

        Returns:
            ID of the created entity
        """
        # Extract ID from entity data or generate a new one
        entity_id = entity_data.get("id")
        if not entity_id:
            import uuid
            entity_id = str(uuid.uuid4())
            entity_data["id"] = entity_id

        # Create properties string for Cypher query
        properties = ", ".join([f"n.{key} = ${key}" for key in entity_data.keys()])

        query = f"""
        CREATE (n:Entity)
        SET n.id = $id, {properties}
        RETURN n.id AS id
        """

        try:
            result = await self.run_query(query, entity_data)
            return result[0]["id"]
        except Exception as e:
            logger.error(f"Error creating entity: {e}")
            raise RuntimeError(f"Failed to create entity: {e}")

    async def create_relationship(
        self, source_id: str, target_id: str, rel_type: str, properties: Dict = None
    ) -> bool:
        """
        Create a relationship between two entities.

        Args:
            source_id: ID of the source entity
            target_id: ID of the target entity
            rel_type: Type of relationship
            properties: Relationship properties

        Returns:
            True if successful
        """
        if properties is None:
            properties = {}

        # Combine all parameters
        params = {
            "source_id": source_id,
            "target_id": target_id,
            **properties
        }

        # Create properties string for Cypher query
        prop_string = ""
        if properties:
            prop_list = [f"{key}: ${key}" for key in properties.keys()]
            prop_string = f" {{ {', '.join(prop_list)} }}"

        query = f"""
        MATCH (a:Entity {{id: $source_id}}), (b:Entity {{id: $target_id}})
        CREATE (a)-[r:{rel_type}{prop_string}]->(b)
        RETURN r
        """

        try:
            await self.run_query(query, params)
            return True
        except Exception as e:
            logger.error(f"Error creating relationship: {e}")
            raise RuntimeError(f"Failed to create relationship: {e}")
