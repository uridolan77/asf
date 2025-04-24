from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import registry

# Create mapper registry with namespace
mapper_registry = registry()

# Create Base with explicit class registry to avoid conflicts
Base = declarative_base(
    metadata=mapper_registry.metadata,
    class_registry=dict()
)

# Set module path to ensure unique registration
Base.__module__ = "asf.bollm.backend.models"
