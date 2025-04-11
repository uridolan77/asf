import asyncio
from typing import Dict, Any, Optional, List, Tuple, Union
from neo4j import AsyncGraphDatabase, AsyncResult, AsyncTransaction
from abc import ABC, abstractmethod

class DatabaseDriver(ABC):
    """Abstract base class for database drivers."""
    @abstractmethod
    async def add_node(self, label: str, properties: Dict[str, Any]) -> None:
        pass

    @abstractmethod
    async def get_node(self, label: str, key: str, value: Any) -> Optional[Dict[str, Any]]:
        pass

    @abstractmethod
    async def update_node(self, label: str, key: str, value: Any, properties: Dict[str, Any]) -> None:
        pass

    @abstractmethod
    async def delete_node(self, label: str, key: str, value: Any) -> None:
        pass

    @abstractmethod
    async def add_relationship(self, start_node_id: str, end_node_id: str, relationship_type: str,
                               properties: Dict[str, Any], start_node_label: str, end_node_label: str) -> None:
        pass

    @abstractmethod
    async def get_relationship(self, start_node_id: str, end_node_id: str, relationship_type: str, start_node_label: str, end_node_label: str) -> Any:
        pass

    @abstractmethod
    async def get_relationships(self, node_id: str, relationship_type: str, node_label: str, direction: str = "OUTGOING") -> List[Any]:
        pass
    
    @abstractmethod
    async def delete_relationship(self, start_node_id: str, end_node_id: str, relationship_type: str, start_node_label: str, end_node_label: str) -> None:
        pass

    @abstractmethod
    async def get_all_nodes(self, label: str) -> List[Dict[str, Any]]:
      pass

    @abstractmethod
    async def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Any]:
        pass


class Neo4jDBDriver(DatabaseDriver):
    def __init__(self, uri, user, password):
        self.driver = AsyncGraphDatabase.driver(uri, auth=(user, password))

    async def close(self):
        await self.driver.close()

    async def add_node(self, label: str, properties: Dict[str, Any]) -> None:
        async with self.driver.session() as session:
            query = f"CREATE (n:{label} $props)"
            await session.run(query, props=properties)

    async def get_node(self, label: str, key: str, value: Any) -> Optional[Dict[str, Any]]:
        async with self.driver.session() as session:
            query = f"MATCH (n:{label} {{{key}: $value}}) RETURN n"
            result = await session.run(query, value=value)
            record = await result.single()
            return record["n"]._properties if record else None

    async def update_node(self, label: str, key: str, value: Any, properties: Dict[str, Any]) -> None:
        async with self.driver.session() as session:
            query = f"MATCH (n:{label} {{{key}: $value}}) SET n += $props"
            await session.run(query, value=value, props=properties)

    async def delete_node(self, label: str, key: str, value: Any) -> None:
        async with self.driver.session() as session:
            query = f"MATCH (n:{label} {{{key}: $value}}) DETACH DELETE n"
            await session.run(query, value=value)
    async def add_relationship(self, start_node_id: str, end_node_id: str, relationship_type: str,
                               properties: Dict[str, Any], start_node_label: str, end_node_label: str) -> None:
      async with self.driver.session() as session:
        query = (
            f"MATCH (a:{start_node_label}), (b:{end_node_label}) "
            f"WHERE a.{'symbol_id' if start_node_label == 'Symbol' else 'name'} = $start_node_id AND b.{'symbol_id' if end_node_label == 'Symbol' else 'name'} = $end_node_id "
            f"CREATE (a)-[r:{relationship_type}]->(b) "
            f"SET r += $props"
        )
        await session.run(query, start_node_id=start_node_id, end_node_id=end_node_id, props=properties)

    async def get_relationship(self, start_node_id: str, end_node_id: str, relationship_type: str, start_node_label: str, end_node_label: str) -> Any:
         async with self.driver.session() as session:
            query = (
                f"MATCH (a:{start_node_label})-[r:{relationship_type}]->(b:{end_node_label}) "
                f"WHERE a.{'symbol_id' if start_node_label == 'Symbol' else 'name'} = $start_node_id AND b.{'symbol_id' if end_node_label == 'Symbol' else 'name'} = $end_node_id "
                f"RETURN r"
            )
            result = await session.run(query, start_node_id=start_node_id, end_node_id=end_node_id)
            record = await result.single()
            return record["r"] if record else None

    async def get_relationships(self, node_id: str, relationship_type: str, node_label: str, direction: str = "OUTGOING") -> List[Any]:
        async with self.driver.session() as session:
            if direction == "OUTGOING":
                query = (
                    f"MATCH (a:{node_label})-[r:{relationship_type}]->(b) "
                    f"WHERE a.{'symbol_id' if node_label == 'Symbol' else 'name'} = $node_id "
                    f"RETURN r, b"  # Return the relationship AND the end node
                )
            elif direction == "INCOMING":
                query = (
                    f"MATCH (a:{node_label})<-[r:{relationship_type}]-(b) "
                    f"WHERE a.{'symbol_id' if node_label == 'Symbol' else 'name'} = $node_id "
                    f"RETURN r, b"  # Return the relationship AND the start node (which is 'b' in this case)
                )
            else:  # BOTH
                query = (
                    f"MATCH (a:{node_label})-[r:{relationship_type}]-(b) "
                    f"WHERE a.{'symbol_id' if node_label == 'Symbol' else 'name'} = $node_id "
                    f"RETURN r, b"
                )

            result = await session.run(query, node_id=node_id)
            records = await result.data()  # Use .data() to get all records as dictionaries
            return records

    async def delete_relationship(self, start_node_id: str, end_node_id: str, relationship_type: str, start_node_label: str, end_node_label: str) -> None:
        async with self.driver.session() as session:
            query = (
                f"MATCH (a:{start_node_label})-[r:{relationship_type}]->(b:{end_node_label}) "
                f"WHERE a.{'symbol_id' if start_node_label == 'Symbol' else 'name'} = $start_node_id AND b.{'symbol_id' if end_node_label == 'Symbol' else 'name'} = $end_node_id "
                f"DELETE r"
            )
            await session.run(query, start_node_id=start_node_id, end_node_id=end_node_id)

    async def get_all_nodes(self, label: str) -> List[Dict[str, Any]]:
        async with self.driver.session() as session:
            query = f"MATCH (n:{label}) RETURN n"
            result = await session.run(query)
            records = await result.data()
            return [record['n'] for record in records] #Extract

    async def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Any]:
        async with self.driver.session() as session:
            result = await session.run(query, params)
            return await result.data()


def create_db_driver(db_type: str = "neo4j", uri: str = "bolt://localhost:7687", user: str = "neo4j", password: str = "password") -> DatabaseDriver:
    """Factory function to create database drivers."""
    if db_type == "neo4j":
        return Neo4jDBDriver(uri, user, password)
    else:
        raise ValueError(f"Unsupported database type: {db_type}")