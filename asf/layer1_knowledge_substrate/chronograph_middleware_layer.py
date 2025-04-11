import asyncio
import json
import logging
import os
import time
import uuid
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import aiokafka
import asyncpg
import zstandard
from aioredis import Redis
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from jose import ExpiredSignatureError, JWTError, jwk, jwt
from neo4j import AsyncGraphDatabase
from pydantic import BaseModel, BaseSettings, Field, validator
from prometheus_client import (
    Counter,
    Gauge,
    Summary,
    start_http_server,
)

RETRY_BACKOFF_SECONDS = 1
MAX_RETRIES = 3
KAFKA_TOPIC = "chrono-ingest"
CACHE_TTL_SECONDS = 60
TOKEN_ALGORITHM = "RS256"

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("chronograph-v2")


class KafkaConfig(BaseModel):
    bootstrap_servers: str = Field("localhost:9092", env="KAFKA_SERVERS")


class Neo4jConfig(BaseModel):
    uri: str = Field("bolt://localhost:7687", env="NEO4J_URI")
    user: str = Field("neo4j", env="NEO4J_USER")
    password: str = Field("password", env="NEO4J_PASSWORD")


class TimescaleConfig(BaseModel):
    dbname: str = Field("chronograph", env="TIMESCALE_DBNAME")
    user: str = Field("tsadmin", env="TIMESCALE_USER")
    password: str = Field("secret", env="TIMESCALE_PASSWORD")
    host: str = Field("localhost", env="TIMESCALE_HOST")
    port: int = Field(5432, env="TIMESCALE_PORT")
    schema: str = Field("public", env="TIMESCALE_SCHEMA")


class RedisConfig(BaseModel):
    host: str = Field("localhost", env="REDIS_HOST")
    port: int = Field(6379, env="REDIS_PORT")
    db: int = Field(0, env="REDIS_DB")


class SecurityConfig(BaseModel):
    private_key_pem: str = Field(..., env="PRIVATE_KEY_PEM")  # Required!
    public_key_pem: str = Field(..., env="PUBLIC_KEY_PEM")  # Required!
    key_rotation_interval_hours: int = Field(1, env="KEY_ROTATION_INTERVAL_HOURS")

    @validator("private_key_pem", "public_key_pem", pre=True)
    def load_key_from_env(cls, value):
        if value.startswith("file:"):  # Load from file
            file_path = value[5:]
            try:
                with open(file_path, "r") as f:
                    return f.read()
            except FileNotFoundError:
                raise ValueError(f"Key file not found: {file_path}")
        return value  # Assume it's the key itself


class MetricsConfig(BaseModel):
    port: int = Field(9100, env="METRICS_PORT")


class Config(BaseSettings):
    kafka: KafkaConfig = KafkaConfig()
    neo4j: Neo4jConfig = Neo4jConfig()
    timescale: TimescaleConfig = TimescaleConfig()
    redis: RedisConfig = RedisConfig()
    security: SecurityConfig
    metrics: MetricsConfig = MetricsConfig()

    class Config:
        env_prefix = ""  # No prefix for environment variables


class ChronoError(Exception):
    """Base Chronograph exception."""

    pass


class ChronoIngestionError(ChronoError):
    """Data ingestion failure."""

    pass


class ChronoQueryError(ChronoError):
    """Query processing failure."""

    pass


class ChronoSecurityError(ChronoError):
    """Security validation failure."""

    pass


class ChronoDatabaseError(ChronoError):
    """Database interaction error."""

    pass


class ChronoCacheError(ChronoError):
    """Cache interaction error."""

    pass


class ChronoKafkaError(ChronoError):
    """Kafka interaction error."""

    pass


# --- Database Manager ---
class DatabaseManager:
    def __init__(self, config: Config):
        self.timescale_config = config.timescale
        self.neo4j_config = config.neo4j
        self.timescale_pool = None
        self.neo4j_driver = None

    async def connect(self):
        """Establish database connections."""
        self.timescale_pool = await asyncpg.create_pool(
            database=self.timescale_config.dbname,
            user=self.timescale_config.user,
            password=self.timescale_config.password,
            host=self.timescale_config.host,
            port=self.timescale_config.port,
            min_size=1,  # Minimum connections in the pool
            max_size=10,  # Maximum connections in the pool
        )
        self.neo4j_driver = AsyncGraphDatabase.driver(
            self.neo4j_config.uri,
            auth=(self.neo4j_config.user, self.neo4j_config.password),
        )
        await self.neo4j_driver.verify_connectivity()
        logger.info("Database connections established.")

    async def close(self):
        retry_count = 0
        while retry_count < MAX_RETRIES:
            try:
                async with self.timescale_pool.acquire() as conn:
                    async with conn.transaction():
                        if as_of:
                            await conn.execute(
                                f"SET TRANSACTION SNAPSHOT '{as_of.isoformat()}'"
                            )
                        result = await conn.fetch(query, *args)
                        return result
            except (asyncpg.PostgresError, OSError) as e:
                retry_count += 1
                logger.warning(
                    f"TimescaleDB query failed (attempt {retry_count}/{MAX_RETRIES}): {e}"
                )
                if retry_count >= MAX_RETRIES:
                    raise ChronoDatabaseError(f"TimescaleDB query failed: {e}")
                await asyncio.sleep(RETRY_BACKOFF_SECONDS * (2**retry_count))

    async def execute_neo4j_query(self, query: str, params: Dict = None) -> List:
        entity_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()

        neo4j_query = """
            CREATE (e:Entity {id: $id, labels: $labels, properties: $properties})
            RETURN e
            INSERT INTO entities (id, timestamp, data)
            VALUES ($1, $2, $3)

    def __init__(self, max_size: int = 1024):
        self.cache = OrderedDict()
        self.max_size = max_size

    async def get(self, key: str) -> Optional[Any]:
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)  # Remove least recently used

    async def invalidate(self, key: str):

    def __init__(self, config: Config):
        self.redis_config = config.redis
        self.lru_cache = LRUCache()
        self.redis = None  # Will be initialized in connect()

    async def connect(self):
        if self.redis:
            await self.redis.close()
        logger.info("Redis connection closed.")

    async def get(self, key: str) -> Optional[Any]:
        await self.lru_cache.set(key, value)
        if self.redis:
            try:
                if isinstance(value, (dict, list)):
                    value_str = json.dumps(value)  # Serialize to JSON
                else:
                    value_str = str(value)
                await self.redis.setex(key, ttl, value_str)
            except Exception as e:
                    logger.error(f"Redis SET error: {e}")

    async def invalidate(self, key: str):

        def __init__(self, config: Config):
            self.security_config = config.security
            self.current_key_id = "key-1"  # Start with a default key ID
            self.keys = {
                self.current_key_id: jwk.construct(
                    {"kty": "oct", "k": self.security_config.private_key_pem},
                    algorithm=TOKEN_ALGORITHM,
                )  # Initial key
            }
            self.public_keys = {
                self.current_key_id: jwk.construct(
                    {"kty": "oct", "k": self.security_config.public_key_pem},
                    algorithm=TOKEN_ALGORITHM,
                )
            }
            self.rotation_task = None

        async def start_key_rotation(self):
            while True:
                await asyncio.sleep(
                    self.security_config.key_rotation_interval_hours * 3600
                )
                new_key_id = f"key-{len(self.keys) + 1}"
                logger.info(f"Rotating keys. New key ID: {new_key_id}")

                private_key = rsa.generate_private_key(
                    public_exponent=65537, key_size=2048
                )
                private_pem = private_key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.PKCS8,
                    encryption_algorithm=serialization.NoEncryption(),
                )
                self.keys[new_key_id] = jwk.construct(
                    {"kty": "oct", "k": private_pem.decode()}, algorithm=TOKEN_ALGORITHM
                )

                public_key = private_key.public_key()
                public_pem = public_key.public_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PublicFormat.SubjectPublicKeyInfo,
                )
                self.public_keys[new_key_id] = jwk.construct(
                    {"kty": "oct", "k": public_pem.decode()}, algorithm=TOKEN_ALGORITHM
                )

                self.current_key_id = new_key_id
                if len(self.keys) > 2:
                    old_key_id = f"key-{len(self.keys) - 2}"
                    del self.keys[old_key_id]
                    del self.public_keys[old_key_id]

        def generate_token(self, payload: Dict, expiry_delta: timedelta = timedelta(hours=1)) -> str:
            """Generate a JWT."""
            to_encode = payload.copy()
            now = datetime.utcnow()
            expire = now + expiry_delta
            to_encode.update({"iat": now, "exp": expire, "kid": self.current_key_id})
            encoded_jwt = jwt.encode(
                to_encode,
                self.keys[self.current_key_id].to_dict(),
                algorithm=TOKEN_ALGORITHM,
                headers={"kid": self.current_key_id},
            )
            return encoded_jwt

        def validate_token(self, token: str) -> Dict:
            """Validate a JWT and return the payload."""
            try:
                header = jwt.get_unverified_header(token)
                kid = header.get("kid")
                if not kid or kid not in self.public_keys:
                    raise ChronoSecurityError("Invalid token: Key ID not found")

                payload = jwt.decode(
                    token,
                    self.public_keys[kid].to_dict(),
                    algorithms=[TOKEN_ALGORITHM],
                )
                return payload
            except ExpiredSignatureError:
                raise ChronoSecurityError("Token has expired")
            except JWTError as e:
                raise ChronoSecurityError(f"Invalid token: {e}")

        def generate_refresh_token(self, user_id: str) -> str:
            """Generates a refresh token (basic example)."""
            payload = {"sub": user_id, "type": "refresh"}
            return self.generate_token(payload, expiry_delta=timedelta(days=7))  # Longer expiry

        def refresh_access_token(self, refresh_token: str) -> Tuple[str, str]:
            """Refreshes an access token using a refresh token (basic example)."""

            try:
                payload = self.validate_token(refresh_token)
                if payload.get("type") != "refresh":
                    raise ChronoSecurityError("Invalid refresh token")

                user_id = payload.get("sub")
                if not user_id:
                    raise ChronoSecurityError("Invalid refresh token payload")
                new_access_token = self.generate_token({"sub": user_id})
                new_refresh_token = self.generate_refresh_token(
                    user_id
                )  # Optionally issue a new refresh token
                return new_access_token, new_refresh_token

            except ChronoSecurityError as e:
                raise ChronoSecurityError(f"Refresh token validation failed:{e}")

    class KafkaManager:
        """Manages Kafka producer and consumer."""

        def __init__(self, config: Config):
            self.kafka_config = config.kafka
            self.producer = None
            self.consumer = None
            self.zstd_compressor = zstandard.ZstdCompressor(level=3)
            self.zstd_decompressor = zstandard.ZstdDecompressor()

        async def connect(self):
            """Establish Kafka connections."""
            self.producer = aiokafka.AIOKafkaProducer(
                bootstrap_servers=self.kafka_config.bootstrap_servers,
                value_serializer=self._serialize_message,
                compression_type="zstd",
                acks="all",  # Wait for all in-sync replicas to ack
                retries=MAX_RETRIES,  # Enable retries
                retry_backoff_ms=int(RETRY_BACKOFF_SECONDS * 1000),
            )
            await self.producer.start()

            self.consumer = aiokafka.AIOKafkaConsumer(
                KAFKA_TOPIC,
                bootstrap_servers=self.kafka_config.bootstrap_servers,
                value_deserializer=self._deserialize_message,
                auto_offset_reset="earliest",  # Start from the beginning if no offset
                enable_auto_commit=True,  # Commit offsets automatically
                group_id="chronograph-consumer-group",  # Consumer group ID
            )
            await self.consumer.start()

            logger.info("Kafka connections established.")

        async def close(self):
            if isinstance(message, dict):
                message_str = json.dumps(message)
            else:
                message_str = str(message)
            return self.zstd_compressor.compress(message_str.encode("utf-8"))

        def _deserialize_message(self, message: bytes) -> Any:
            """Deserialize a message from bytes using Zstandard."""
            try:
                decompressed = self.zstd_decompressor.decompress(message)
                return json.loads(decompressed.decode("utf-8"))
            except (zstandard.ZstdError, json.JSONDecodeError) as e:
                logger.error(f"Failed to deserialize Kafka message: {e}")
                return None  # Or raise an exception, depending on your error handling

        async def send_message(self, topic: str, message: Any):
            """Send a message to Kafka with retry logic."""
            retry_count = 0
            while retry_count < MAX_RETRIES:
                try:
                    await self.producer.send_and_wait(topic, message)
                    return  # Success, exit the loop
                except aiokafka.errors.KafkaError as e:
                    retry_count += 1
                    logger.warning(
                        f"Kafka send failed (attempt {retry_count}/{MAX_RETRIES}): {e}"
                    )
                    if retry_count >= MAX_RETRIES:
                        raise ChronoKafkaError(f"Kafka send failed: {e}")
                    await asyncio.sleep(RETRY_BACKOFF_SECONDS * (2**retry_count))

        async def consume_messages(self):

        def __init__(self, config: Config):
            self.config = config
            self.ingested_counter = Counter(
                "chronograph_ingested_total", "Total number of data points ingested"
            )
            self.cache_hits_counter = Counter(
                "chronograph_cache_hits_total", "Total number of cache hits"
            )
            self.cache_misses_counter = Counter(
                "chronograph_cache_misses_total", "Total number of cache misses"
            )
            self.query_latency_summary = Summary(
                "chronograph_query_latency_seconds", "Summary of query processing times"
            )
            self.system_load_gauge = Gauge(
                "chronograph_system_load", "System load average"
            )  # Example gauge

        def start_http_server(self):
            """Start the Prometheus metrics HTTP server."""
            start_http_server(self.config.metrics.port)
            logger.info(f"Prometheus metrics server started on port {self.config.metrics.port}")

        def update_system_load(self):
            try:
                load1, load5, load15 = os.getloadavg()
                self.system_load_gauge.set(load1)
            except OSError:
                pass


    # --- Mock Database and Kafka Managers ---
    class MockDatabaseManager:
        """Mocks database interactions for testing"""

        def __init__(self, config: Config):
            self.entities = {}
            self.timeseries_data = {}

        async def connect(self):
            logger.info("Mock database connected.")

        async def close(self):
            logger.info("Mock database closed.")

        async def execute_timescale_query(
            self, query: str, *args, as_of: Optional[datetime] = None
        ) -> List[Dict]:
            entity_id = args[0]
            if "INSERT" in query:
                self.timeseries_data.setdefault(entity_id, []).append(args)
                return []  # Insert doesn't typically return data
            elif "SELECT" in query:
                data = self.timeseries_data.get(entity_id, [])
                if as_of:
                    filtered_data = [
                        entry for entry in data if entry[1] <= as_of.isoformat()
                    ]
                    return (
                        filtered_data
                    )  # adjust depending on the specific select query structure
                return data

            return []

        async def execute_neo4j_query(self, query: str, params: Dict = None) -> List:
            if "CREATE" in query:
                entity_id = params["id"]
                self.entities[entity_id] = params
                return [{"e": params}]
            elif "MATCH" in query:  # basic match
                return [
                    {"e": entity} for entity in self.entities.values() if entity["id"] == entity_id
                ]
            return []

        async def create_entity(self, entity_data: Dict) -> str:
            entity_id = str(uuid.uuid4())
            await self.execute_neo4j_query(
                "", {"id": entity_id, "labels": [], "properties": entity_data}
            )  # simplified
            await self.execute_timescale_query(
                "", entity_id, datetime.utcnow().isoformat(), json.dumps(entity_data)
            )
            return entity_id


    class MockKafkaManager:
        """Mocks Kafka interactions for testing"""

        def __init__(self, config: Config):
            self.messages = []

        async def connect(self):
            logger.info("Mock Kafka connected")

        async def close(self):
            logger.info("Mock Kafka Closed")

        def _serialize_message(self, message: Any) -> bytes:
            return json.dumps(message).encode("utf-8")

        def _deserialize_message(self, message: bytes) -> Any:
            return json.loads(message.decode("utf-8"))

        async def send_message(self, topic: str, message: Any):
            self.messages.append((topic, message))
            logger.info(f"Mock Kafka Received Message on topic {topic}")

        async def consume_messages(self):
            while True:
                if self.messages:
                    topic, message = self.messages.pop(0)
                    logger.info("Mock Kafka Consumed Message: %s, %s", topic, message)
                await asyncio.sleep(1)
# --- Chronograph Middleware ---
class ChronographMiddleware:
    """The main middleware class."""

    def __init__(self, config: Config, database_manager=None, kafka_manager=None):
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

    async def startup(self):
        await self.security.stop_key_rotation()
        await self.kafka.close()
        await self.cache.close()
        await self.database.close()
        logger.info("Chronograph Middleware shut down.")

    async def ingest_data(self, data_points: List[Dict], token: str):

        try:
            self.security.validate_token(token)  # Validate token
        except ChronoSecurityError as e:
            raise ChronoSecurityError(f"Authentication failed: {e}")

        cache_key = self._generate_cache_key(query, params, as_of)
        cached_result = await self.cache.get(cache_key)
        if cached_result:
            self.metrics.cache_hits_counter.inc()
            return cached_result
        self.metrics.cache_misses_counter.inc()

        with self.metrics.query_latency_summary.time():
            if "neo4j" in query.lower():
                result = await self.database.execute_neo4j_query(query, params)
            elif "timescale" in query.lower() or "select" in query.lower():
                result = await self.database.execute_timescale_query(query, *params.values(), as_of=as_of)
            else:
                raise ChronoQueryError("Unsupported query type")

        await self.cache.set(cache_key, result)
        return result

    def _generate_cache_key(
        self, query: str, params: Dict, as_of: Optional[datetime] = None
    ) -> str:
        try:
            self.security.validate_token(token)  # Validate token
        except ChronoSecurityError as e:
            raise ChronoSecurityError(f"Authentication failed: {e}")

        entity_id = await self.database.create_entity(entity_data)
        await self.cache.invalidate(f"entity:{entity_id}")
        return entity_id


    async def main():