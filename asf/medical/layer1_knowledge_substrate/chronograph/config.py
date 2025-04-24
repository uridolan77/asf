"""Configuration for the ChronoGraph middleware layer.

This module defines configuration classes and constants used throughout the
ChronoGraph middleware to configure various components and behaviors.
"""

from typing import List

# Import GnosisConfig from the gnosis layer
from asf.medical.layer1_knowledge_substrate.chronograph_gnosis_layer import GnosisConfig

# Constants
KAFKA_TOPIC = "chronograph_events"
CACHE_TTL_SECONDS = 3600
TOKEN_ALGORITHM = "RS256"


class DatabaseConfig:
    """Database configuration for the ChronoGraph middleware.

    This class holds configuration parameters for connecting to various databases
    used by the ChronoGraph middleware, including Neo4j, TimescaleDB, and Memgraph.
    """
    def __init__(self):
        # Neo4j configuration
        self.neo4j_uri = "bolt://localhost:7687"  # URI for Neo4j connection
        self.neo4j_user = "neo4j"  # Neo4j username
        self.neo4j_password = "password"  # Neo4j password

        # TimescaleDB configuration
        self.timescale_uri = "postgresql://postgres:password@localhost:5432/chronograph"  # URI for TimescaleDB

        # Memgraph configuration
        self.memgraph_uri = "bolt://localhost:7687"  # URI for Memgraph connection
        self.memgraph_user = "memgraph"  # Memgraph username
        self.memgraph_password = "memgraph"  # Memgraph password


class CacheConfig:
    """Cache configuration for the ChronoGraph middleware.

    This class holds configuration parameters for the caching system used by
    the ChronoGraph middleware, including Redis connection details and TTL settings.
    """
    def __init__(self):
        self.redis_uri = "redis://localhost:6379/0"  # URI for Redis connection
        self.ttl = CACHE_TTL_SECONDS  # Default TTL for cached items in seconds


class SecurityConfig:
    """Security configuration for the ChronoGraph middleware.

    This class holds configuration parameters for security features of the
    ChronoGraph middleware, including JWT token settings and key management.
    """
    def __init__(self):
        self.token_algorithm = TOKEN_ALGORITHM  # Algorithm used for JWT tokens
        self.private_key_path = "keys/private.pem"  # Path to private key for signing tokens
        self.public_key_path = "keys/public.pem"  # Path to public key for verifying tokens
        self.key_rotation_interval_hours = 24  # Interval for key rotation in hours


class KafkaConfig:
    """Kafka configuration for the ChronoGraph middleware.

    This class holds configuration parameters for Kafka messaging system used by
    the ChronoGraph middleware for asynchronous processing and event handling.
    """
    def __init__(self):
        self.bootstrap_servers: List[str] = ["localhost:9092"]  # List of Kafka bootstrap servers
        self.topic = KAFKA_TOPIC  # Default Kafka topic for ChronoGraph events
        self.group_id = "chronograph-middleware"  # Consumer group ID for Kafka


class MetricsConfig:
    """Metrics configuration for the ChronoGraph middleware.

    This class holds configuration parameters for metrics collection and reporting
    in the ChronoGraph middleware, including Prometheus settings.
    """
    def __init__(self):
        self.port = 8000  # Port for the metrics server
        self.namespace = "chronograph"  # Namespace for metrics


class Config:
    """Main configuration class for the ChronoGraph middleware.

    This class serves as a container for all configuration components of the
    ChronoGraph middleware, providing a single point of access to all settings.
    It includes database, cache, security, Kafka, metrics, and gnosis configurations.
    """
    def __init__(self):
        self.database = DatabaseConfig()  # Database configuration
        self.cache = CacheConfig()  # Cache configuration
        self.security = SecurityConfig()  # Security configuration
        self.kafka = KafkaConfig()  # Kafka configuration
        self.metrics = MetricsConfig()  # Metrics configuration
        self.gnosis = GnosisConfig()  # ChronoGnosisLayer configuration

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'Config':
        """Create a Config instance from a dictionary.

        This method allows creating a configuration object from a dictionary,
        which is useful for loading configuration from JSON or YAML files.

        Args:
            config_dict: Dictionary containing configuration values

        Returns:
            A Config instance with values from the dictionary
        """
        config = cls()

        # Database config
        if 'database' in config_dict:
            db_config = config_dict['database']
            if 'neo4j_uri' in db_config:
                config.database.neo4j_uri = db_config['neo4j_uri']
            if 'neo4j_user' in db_config:
                config.database.neo4j_user = db_config['neo4j_user']
            if 'neo4j_password' in db_config:
                config.database.neo4j_password = db_config['neo4j_password']
            if 'timescale_uri' in db_config:
                config.database.timescale_uri = db_config['timescale_uri']
            if 'memgraph_uri' in db_config:
                config.database.memgraph_uri = db_config['memgraph_uri']
            if 'memgraph_user' in db_config:
                config.database.memgraph_user = db_config['memgraph_user']
            if 'memgraph_password' in db_config:
                config.database.memgraph_password = db_config['memgraph_password']

        # Cache config
        if 'cache' in config_dict:
            cache_config = config_dict['cache']
            if 'redis_uri' in cache_config:
                config.cache.redis_uri = cache_config['redis_uri']
            if 'ttl' in cache_config:
                config.cache.ttl = cache_config['ttl']

        # Security config
        if 'security' in config_dict:
            security_config = config_dict['security']
            if 'token_algorithm' in security_config:
                config.security.token_algorithm = security_config['token_algorithm']
            if 'private_key_path' in security_config:
                config.security.private_key_path = security_config['private_key_path']
            if 'public_key_path' in security_config:
                config.security.public_key_path = security_config['public_key_path']
            if 'key_rotation_interval_hours' in security_config:
                config.security.key_rotation_interval_hours = security_config['key_rotation_interval_hours']

        # Kafka config
        if 'kafka' in config_dict:
            kafka_config = config_dict['kafka']
            if 'bootstrap_servers' in kafka_config:
                config.kafka.bootstrap_servers = kafka_config['bootstrap_servers']
            if 'topic' in kafka_config:
                config.kafka.topic = kafka_config['topic']
            if 'group_id' in kafka_config:
                config.kafka.group_id = kafka_config['group_id']

        # Metrics config
        if 'metrics' in config_dict:
            metrics_config = config_dict['metrics']
            if 'port' in metrics_config:
                config.metrics.port = metrics_config['port']
            if 'namespace' in metrics_config:
                config.metrics.namespace = metrics_config['namespace']

        # Gnosis config
        if 'gnosis' in config_dict:
            # This would need to be implemented based on the GnosisConfig class
            pass

        return config

    def to_dict(self) -> dict:
        """Convert the Config instance to a dictionary.

        This method allows serializing the configuration object to a dictionary,
        which is useful for saving configuration to JSON or YAML files.

        Returns:
            A dictionary representation of the configuration
        """
        return {
            'database': {
                'neo4j_uri': self.database.neo4j_uri,
                'neo4j_user': self.database.neo4j_user,
                'neo4j_password': self.database.neo4j_password,
                'timescale_uri': self.database.timescale_uri,
                'memgraph_uri': self.database.memgraph_uri,
                'memgraph_user': self.database.memgraph_user,
                'memgraph_password': self.database.memgraph_password,
            },
            'cache': {
                'redis_uri': self.cache.redis_uri,
                'ttl': self.cache.ttl,
            },
            'security': {
                'token_algorithm': self.security.token_algorithm,
                'private_key_path': self.security.private_key_path,
                'public_key_path': self.security.public_key_path,
                'key_rotation_interval_hours': self.security.key_rotation_interval_hours,
            },
            'kafka': {
                'bootstrap_servers': self.database.neo4j_uri,
                'topic': self.kafka.topic,
                'group_id': self.kafka.group_id,
            },
            'metrics': {
                'port': self.metrics.port,
                'namespace': self.metrics.namespace,
            },
            # Gnosis config would need to be implemented based on the GnosisConfig class
        }
