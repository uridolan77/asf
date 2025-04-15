"""
Settings module for the Medical Research Synthesizer.

This module provides configuration settings for the Medical Research Synthesizer.
"""
import os
from typing import Dict, Any, Optional, List
from pydantic import BaseSettings, Field

class Settings(BaseSettings):
    """
    Settings for the Medical Research Synthesizer.
    
    This class uses Pydantic's BaseSettings to load configuration from environment variables.
    """
    # Environment
    ENVIRONMENT: str = Field("development", env="ENVIRONMENT")
    
    # Logging
    LOG_LEVEL: str = Field("info", env="LOG_LEVEL")
    LOG_FORMAT: str = Field("json", env="LOG_FORMAT")
    
    # Cache
    CACHE_ENABLED: bool = Field(True, env="CACHE_ENABLED")
    CACHE_TTL: int = Field(3600, env="CACHE_TTL")  # Default TTL in seconds
    
    # API Keys
    PUBMED_API_KEY: Optional[str] = Field(None, env="PUBMED_API_KEY")
    CLINICAL_TRIALS_API_KEY: Optional[str] = Field(None, env="CLINICAL_TRIALS_API_KEY")
    UMLS_API_KEY: Optional[str] = Field(None, env="UMLS_API_KEY")
    
    # Database
    DATABASE_URL: str = Field("sqlite:///./medical_research.db", env="DATABASE_URL")
    
    # Storage
    STORAGE_PATH: str = Field("./storage", env="STORAGE_PATH")
    
    # Search
    MAX_SEARCH_RESULTS: int = Field(100, env="MAX_SEARCH_RESULTS")
    
    # RAG
    EMBEDDING_MODEL: str = Field("all-MiniLM-L6-v2", env="EMBEDDING_MODEL")
    VECTOR_DB_PATH: str = Field("./vector_db", env="VECTOR_DB_PATH")
    
    class Config:
        """Pydantic config"""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True

# Create a global settings instance
settings = Settings()

# Export the settings instance
__all__ = ["settings"]
