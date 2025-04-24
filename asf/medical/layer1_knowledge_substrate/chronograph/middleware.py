"""ChronoGraph Middleware Layer Module.

This module implements the middleware layer for the ChronoGraph knowledge system,
providing a high-level interface for temporal knowledge graph operations.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import torch

from asf.medical.layer1_knowledge_substrate.chronograph_gnosis_layer import ChronoGnosisLayer
from asf.medical.layer1_knowledge_substrate.chronograph.config import Config, KAFKA_TOPIC, CACHE_TTL_SECONDS
from asf.medical.layer1_knowledge_substrate.chronograph.exceptions import (
    ChronoSecurityError,
    ChronoQueryError,
    ChronoIngestionError,
)
from asf.medical.layer1_knowledge_substrate.chronograph.managers import (
    DatabaseManager,
    CacheManager,
    SecurityManager,
    KafkaManager,
    MetricsManager,
)

# Configure logging
logger = logging.getLogger("chronograph")


class ChronographMiddleware:
    """The main middleware class.
    
    This class serves as the primary interface to the ChronoGraph knowledge system,
    providing access to temporal knowledge graph embeddings, trend analysis, and
    entity management. It integrates with the ChronoGnosisLayer for advanced
    temporal-aware knowledge representation.
    """
    def __init__(self, config: Config, database_manager=None, kafka_manager=None, gnosis_layer=None):
        self.config = config
        if database_manager is None:  # allow for dependency injection
            self.database = DatabaseManager(self.config)
        else:
            self.database = database_manager
            
        self.cache = CacheManager(self.config)
        self.security = SecurityManager(self.config)
        
        if kafka_manager is None:
            self.kafka = KafkaManager(self.config)
        else:
            self.kafka = kafka_manager
            
        self.metrics = MetricsManager(self.config)
        
        # Initialize ChronoGnosisLayer
        if gnosis_layer is None:
            self.gnosis = ChronoGnosisLayer(self.config.gnosis)
        else:
            self.gnosis = gnosis_layer
            
        # Embedding cache TTL (1 hour by default)
        self.embedding_cache_ttl = 3600
        
    async def startup(self):
        """Initialize all components of the middleware."""
        # Start database connections
        await self.database.connect()
        
        # Start ChronoGnosisLayer
        await self.gnosis.startup()
        
        # Start other components
        await self.kafka.connect()
        await self.cache.connect()
        
        # Start metrics server
        self.metrics.start()
        
        logger.info("Chronograph Middleware started successfully.")
        
    async def shutdown(self):
        """Shutdown all components of the middleware."""
        await self.security.stop_key_rotation()
        await self.kafka.close()
        await self.cache.close()
        await self.database.close()
        await self.gnosis.shutdown()
        logger.info("Chronograph Middleware shut down.")
        
    async def ingest_data(self, data_points: List[Dict], token: str = None):
        """Ingest data points into the chronograph system.
        
        Args:
            data_points: List of data points to ingest
            token: Optional security token for authentication
            
        Returns:
            List of entity IDs created
            
        Raises:
            ChronoSecurityError: If authentication fails
            ChronoIngestionError: If ingestion fails
        """
        # Validate token if provided
        if token:
            try:
                self.security.validate_token(token)
            except ChronoSecurityError as e:
                raise ChronoSecurityError(f"Authentication failed: {e}")
        
        # Track metrics
        with self.metrics.ingest_latency_summary.time():
            try:
                entity_ids = []
                for data_point in data_points:
                    entity_id = await self.database.create_entity(data_point)
                    entity_ids.append(entity_id)
                    # Invalidate cache for this entity
                    await self.cache.invalidate(f"entity:{entity_id}")
                    
                # Send to Kafka for async processing
                await self.kafka.send_message(KAFKA_TOPIC, {
                    "action": "ingest",
                    "entity_ids": entity_ids,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
                
                return entity_ids
            except Exception as e:
                logger.error(f"Error ingesting data: {e}")
                raise ChronoIngestionError(f"Failed to ingest data: {e}")
    
    async def get_entity(self, entity_id: str, include_history: bool = False) -> Optional[Dict]:
        """Get entity data by ID.
        
        Args:
            entity_id: ID of the entity to retrieve
            include_history: Whether to include historical data
            
        Returns:
            Entity data or None if not found
            
        Raises:
            ChronoQueryError: If query fails
        """
        # Try to get from cache first
        cache_key = f"entity:{entity_id}:{include_history}"
        cached_result = await self.cache.get(cache_key)
        
        if cached_result:
            self.metrics.cache_hits_counter.inc()
            return cached_result
            
        self.metrics.cache_misses_counter.inc()
        
        # Query database
        with self.metrics.query_latency_summary.time():
            try:
                # Get current entity data
                query = "MATCH (e:Entity {id: $id}) RETURN e"
                result = await self.database.execute_neo4j_query(query, {"id": entity_id})
                
                if not result:
                    return None
                    
                entity_data = result[0]["e"]
                
                # Add historical data if requested
                if include_history:
                    history_query = """
                        SELECT timestamp, data
                        FROM entities
                        WHERE id = $1
                        ORDER BY timestamp DESC
                    """
                    history = await self.database.execute_timescale_query(history_query, entity_id)
                    entity_data["history"] = history
                
                # Cache the result
                await self.cache.set(cache_key, entity_data, ttl=CACHE_TTL_SECONDS)
                
                return entity_data
            except Exception as e:
                logger.error(f"Error retrieving entity {entity_id}: {e}")
                raise ChronoQueryError(f"Failed to retrieve entity: {e}")
    
    async def generate_embeddings(self, entity_ids: List[str], metadata: Optional[Dict[str, Dict[str, float]]] = None) -> Dict[str, Dict[str, torch.Tensor]]:
        """Generate embeddings for entities using the ChronoGnosisLayer.
        
        Args:
            entity_ids: List of entity IDs to generate embeddings for
            metadata: Optional dictionary mapping entity IDs to metadata dictionaries
                (impact factor, citation count, design score)
                
        Returns:
            Dictionary mapping entity IDs to embeddings in different spaces
            (euclidean, hyperbolic, fused)
            
        Raises:
            ChronoQueryError: If embedding generation fails
        """
        # Try to get from cache first
        cache_hits = []
        cache_misses = []
        results = {}
        
        # Check cache for each entity
        for entity_id in entity_ids:
            cache_key = f"embedding:{entity_id}"
            cached_result = await self.cache.get(cache_key)
            
            if cached_result:
                cache_hits.append(entity_id)
                results[entity_id] = cached_result
            else:
                cache_misses.append(entity_id)
        
        # Update metrics
        if cache_hits:
            self.metrics.cache_hits_counter.inc(len(cache_hits))
        if cache_misses:
            self.metrics.cache_misses_counter.inc(len(cache_misses))
        
        # Generate embeddings for cache misses
        if cache_misses:
            with self.metrics.embedding_latency_summary.time():
                try:
                    # Generate embeddings using ChronoGnosisLayer
                    new_embeddings = await self.gnosis.generate_embeddings(cache_misses, metadata)
                    
                    # Cache the results
                    for entity_id, embedding in new_embeddings.items():
                        cache_key = f"embedding:{entity_id}"
                        await self.cache.set(cache_key, embedding, ttl=self.embedding_cache_ttl)
                        results[entity_id] = embedding
                        
                except Exception as e:
                    logger.error(f"Error generating embeddings: {e}")
                    raise ChronoQueryError(f"Failed to generate embeddings: {e}")
        
        return results
    
    async def analyze_temporal_trends(self, entity_ids: List[str], time_window: int = 30) -> Dict[str, List[Dict[str, Any]]]:
        """Analyze temporal trends for entities using the ChronoGnosisLayer.
        
        Args:
            entity_ids: List of entity IDs to analyze
            time_window: Time window in days for trend analysis
            
        Returns:
            Dictionary mapping entity IDs to lists of emerging trends
            
        Raises:
            ChronoQueryError: If trend analysis fails
        """
        # Try to get from cache first
        cache_key = f"trends:{','.join(entity_ids)}:{time_window}"
        cached_result = await self.cache.get(cache_key)
        
        if cached_result:
            self.metrics.cache_hits_counter.inc()
            return cached_result
            
        self.metrics.cache_misses_counter.inc()
        
        # Analyze trends
        with self.metrics.trend_analysis_latency_summary.time():
            try:
                # Analyze trends using ChronoGnosisLayer
                trends = await self.gnosis.analyze_temporal_trends(entity_ids, time_window)
                
                # Cache the results (shorter TTL for trends)
                await self.cache.set(cache_key, trends, ttl=min(3600, self.embedding_cache_ttl // 2))
                
                return trends
            except Exception as e:
                logger.error(f"Error analyzing temporal trends: {e}")
                raise ChronoQueryError(f"Failed to analyze temporal trends: {e}")
    
    async def get_neighbors(self, entity_id: str, max_distance: int = 1) -> List[Tuple[str, str, float]]:
        """Get neighboring entities in the knowledge graph.
        
        Args:
            entity_id: ID of the entity to get neighbors for
            max_distance: Maximum distance to traverse
            
        Returns:
            List of tuples (entity_id, relationship_type, confidence)
            
        Raises:
            ChronoQueryError: If query fails
        """
        # Try to get from cache first
        cache_key = f"neighbors:{entity_id}:{max_distance}"
        cached_result = await self.cache.get(cache_key)
        
        if cached_result:
            self.metrics.cache_hits_counter.inc()
            return cached_result
            
        self.metrics.cache_misses_counter.inc()
        
        # Query database
        with self.metrics.query_latency_summary.time():
            try:
                query = f"""
                    MATCH (e:Entity {{id: $id}})-[r:RELATES_TO*1..{max_distance}]-(n:Entity)
                    RETURN n.id as neighbor_id, type(r) as rel_type, r.confidence as confidence
                    LIMIT 100
                """
                result = await self.database.execute_neo4j_query(query, {"id": entity_id})
                
                neighbors = [(r["neighbor_id"], r["rel_type"], r["confidence"]) for r in result]
                
                # Cache the result
                await self.cache.set(cache_key, neighbors, ttl=CACHE_TTL_SECONDS)
                
                return neighbors
            except Exception as e:
                logger.error(f"Error retrieving neighbors for entity {entity_id}: {e}")
                raise ChronoQueryError(f"Failed to retrieve neighbors: {e}")
    
    def _generate_cache_key(self, query: str, params: Dict, as_of: Optional[datetime] = None) -> str:
        """Generate a cache key for a query.
        
        Args:
            query: The query string
            params: Query parameters
            as_of: Optional timestamp for historical queries
            
        Returns:
            Cache key string
        """
        # Create a deterministic representation of the parameters
        param_str = json.dumps(params, sort_keys=True) if params else ""
        as_of_str = as_of.isoformat() if as_of else ""
        
        # Combine and hash
        key_base = f"{query}:{param_str}:{as_of_str}"
        return f"query:{hash(key_base)}"
