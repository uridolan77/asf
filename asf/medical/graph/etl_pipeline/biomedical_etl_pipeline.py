"""
Biomedical Evidence Graph ETL Pipeline

This module orchestrates the extraction, transformation, and loading of biomedical data
from various sources (PubMed, ClinicalTrials.gov, MeSH) into a Neo4j graph database.
It also computes and stores biomedical sentence embeddings to enable semantic search.

The pipeline follows a modular architecture with the following components:
1. Extractors: Extract data from various sources
2. Transformers: Transform data into a graph-compatible format
3. Embedders: Compute embeddings for text data
4. Loaders: Load data into Neo4j
5. Orchestrator: Coordinate the ETL process

This architecture supports both full pipeline runs and incremental updates.
"""

import os
import logging
import asyncio
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pathlib import Path
import traceback
import json

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("biomedical_etl.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("biomedical_etl")

class EtlConfig:
    """Configuration for the ETL pipeline."""
    
    def __init__(
        self,
        sources: List[str] = None,
        batch_size: int = 100,
        max_articles: int = 10000,
        max_trials: int = 5000,
        embedding_model: str = "BioLinkBERT-base",
        cache_dir: str = "./etl_cache",
        neo4j_uri: Optional[str] = None,
        neo4j_user: Optional[str] = None,
        neo4j_password: Optional[str] = None,
        pubmed_email: Optional[str] = None,
        pubmed_api_key: Optional[str] = None,
        ct_gov_api_key: Optional[str] = None,
        incremental: bool = True,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        search_terms: List[str] = None
    ):
        """
        Initialize ETL configuration.
        
        Args:
            sources: List of data sources to process (e.g., "pubmed", "clinicaltrials", "mesh")
            batch_size: Number of records to process in each batch
            max_articles: Maximum number of articles to process
            max_trials: Maximum number of trials to process
            embedding_model: Name of the model to use for computing embeddings
            cache_dir: Directory to store cache files
            neo4j_uri: URI for Neo4j connection
            neo4j_user: Username for Neo4j connection
            neo4j_password: Password for Neo4j connection
            pubmed_email: Email for PubMed API access
            pubmed_api_key: API key for PubMed
            ct_gov_api_key: API key for ClinicalTrials.gov
            incremental: Whether to run in incremental mode
            start_date: Start date for data extraction (format: YYYY-MM-DD)
            end_date: End date for data extraction (format: YYYY-MM-DD)
            search_terms: List of search terms to use for extracting data
        """
        self.sources = sources or ["pubmed", "clinicaltrials", "mesh"]
        self.batch_size = batch_size
        self.max_articles = max_articles
        self.max_trials = max_trials
        self.embedding_model = embedding_model
        self.cache_dir = cache_dir
        
        # Set up environment-based configuration or use provided values
        self.neo4j_uri = neo4j_uri or os.environ.get("NEO4J_URI", "bolt://localhost:7687")
        self.neo4j_user = neo4j_user or os.environ.get("NEO4J_USER", "neo4j")
        self.neo4j_password = neo4j_password or os.environ.get("NEO4J_PASSWORD")
        
        self.pubmed_email = pubmed_email or os.environ.get("PUBMED_EMAIL")
        self.pubmed_api_key = pubmed_api_key or os.environ.get("PUBMED_API_KEY")
        self.ct_gov_api_key = ct_gov_api_key or os.environ.get("CT_GOV_API_KEY")
        
        # Set up date ranges
        self.incremental = incremental
        self.start_date = start_date
        self.end_date = end_date or datetime.now().strftime("%Y-%m-%d")
        
        # If incremental and no start_date, default to 30 days ago
        if self.incremental and not self.start_date:
            self.start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        
        # Search terms for extracting data
        self.search_terms = search_terms or [
            "diabetes", "cancer", "covid-19", "hypertension", 
            "alzheimer", "asthma", "obesity", "heart failure"
        ]
        
        # Create cache directory
        os.makedirs(self.cache_dir, exist_ok=True)

class BiomedicalEtlPipeline:
    """
    Main orchestrator for the biomedical ETL pipeline.
    
    This class coordinates the extraction, transformation, loading,
    and embedding computation processes for biomedical data.
    """
    
    def __init__(self, config: EtlConfig):
        """
        Initialize the ETL pipeline.
        
        Args:
            config: Configuration for the ETL pipeline
        """
        self.config = config
        self.extractors = {}
        self.transformers = {}
        self.loaders = {}
        self.embedders = {}
        
        # Initialize components
        self._init_extractors()
        self._init_transformers()
        self._init_loaders()
        self._init_embedders()
        
        logger.info(f"Initialized Biomedical ETL Pipeline with sources: {config.sources}")
    
    def _init_extractors(self):
        """Initialize data source extractors."""
        if "pubmed" in self.config.sources:
            from extractors.pubmed_extractor import PubmedExtractor
            self.extractors["pubmed"] = PubmedExtractor(
                email=self.config.pubmed_email,
                api_key=self.config.pubmed_api_key,
                batch_size=self.config.batch_size,
                max_articles=self.config.max_articles,
                cache_dir=os.path.join(self.config.cache_dir, "pubmed")
            )
            
        if "clinicaltrials" in self.config.sources:
            from extractors.clinicaltrials_extractor import ClinicalTrialsExtractor
            self.extractors["clinicaltrials"] = ClinicalTrialsExtractor(
                api_key=self.config.ct_gov_api_key,
                batch_size=self.config.batch_size,
                max_trials=self.config.max_trials,
                cache_dir=os.path.join(self.config.cache_dir, "clinicaltrials")
            )
            
        if "mesh" in self.config.sources:
            from extractors.mesh_extractor import MeshExtractor
            self.extractors["mesh"] = MeshExtractor(
                cache_dir=os.path.join(self.config.cache_dir, "mesh")
            )
    
    def _init_transformers(self):
        """Initialize data transformers."""
        from transformers.article_transformer import ArticleTransformer
        from transformers.clinical_trial_transformer import ClinicalTrialTransformer
        from transformers.mesh_transformer import MeshTransformer
        from transformers.entity_relationship_extractor import EntityRelationshipExtractor
        
        self.transformers["article"] = ArticleTransformer()
        self.transformers["clinical_trial"] = ClinicalTrialTransformer()
        self.transformers["mesh"] = MeshTransformer()
        self.transformers["relationships"] = EntityRelationshipExtractor()
    
    def _init_loaders(self):
        """Initialize data loaders."""
        from loaders.neo4j_loader import Neo4jLoader
        
        self.loaders["neo4j"] = Neo4jLoader(
            uri=self.config.neo4j_uri,
            user=self.config.neo4j_user,
            password=self.config.neo4j_password,
            batch_size=self.config.batch_size
        )
    
    def _init_embedders(self):
        """Initialize text embedders."""
        from embedders.bio_sentence_embedder import BioSentenceEmbedder
        
        self.embedders["sentence"] = BioSentenceEmbedder(
            model_name=self.config.embedding_model,
            cache_dir=os.path.join(self.config.cache_dir, "embeddings")
        )
    
    async def run(self):
        """Run the full ETL pipeline."""
        try:
            logger.info("Starting Biomedical ETL Pipeline")
            
            # Step 1: Extract data
            extracted_data = await self._extract_data()
            
            # Step 2: Transform data
            transformed_data = self._transform_data(extracted_data)
            
            # Step 3: Compute embeddings
            embedded_data = await self._compute_embeddings(transformed_data)
            
            # Step 4: Load data into Neo4j
            await self._load_data(embedded_data)
            
            logger.info("Biomedical ETL Pipeline completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"ETL Pipeline failed: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    
    async def _extract_data(self) -> Dict[str, Any]:
        """
        Extract data from all configured sources.
        
        Returns:
            Dictionary of extracted data by source
        """
        logger.info("Extracting data from sources")
        extracted_data = {}
        
        for source, extractor in self.extractors.items():
            logger.info(f"Extracting data from {source}")
            
            try:
                if source == "pubmed":
                    extracted_data[source] = await extractor.extract(
                        search_terms=self.config.search_terms,
                        start_date=self.config.start_date,
                        end_date=self.config.end_date
                    )
                elif source == "clinicaltrials":
                    extracted_data[source] = await extractor.extract(
                        search_terms=self.config.search_terms,
                        start_date=self.config.start_date,
                        end_date=self.config.end_date
                    )
                elif source == "mesh":
                    extracted_data[source] = await extractor.extract()
            
            except Exception as e:
                logger.error(f"Error extracting from {source}: {str(e)}")
                logger.error(traceback.format_exc())
        
        logger.info(f"Extracted data from {len(extracted_data)} sources")
        return extracted_data
    
    def _transform_data(self, extracted_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform extracted data into a graph-compatible format.
        
        Args:
            extracted_data: Dictionary of extracted data by source
            
        Returns:
            Dictionary of transformed data
        """
        logger.info("Transforming extracted data")
        transformed_data = {
            "nodes": {},
            "relationships": []
        }
        
        try:
            # Transform PubMed articles
            if "pubmed" in extracted_data:
                articles = self.transformers["article"].transform(extracted_data["pubmed"])
                transformed_data["nodes"]["articles"] = articles
            
            # Transform ClinicalTrials.gov trials
            if "clinicaltrials" in extracted_data:
                trials = self.transformers["clinical_trial"].transform(extracted_data["clinicaltrials"])
                transformed_data["nodes"]["clinical_trials"] = trials
            
            # Transform MeSH terms
            if "mesh" in extracted_data:
                mesh_terms = self.transformers["mesh"].transform(extracted_data["mesh"])
                transformed_data["nodes"]["mesh_terms"] = mesh_terms
            
            # Extract relationships between entities
            if "relationships" in self.transformers:
                relationships = self.transformers["relationships"].extract_relationships(transformed_data["nodes"])
                transformed_data["relationships"] = relationships
        
        except Exception as e:
            logger.error(f"Error transforming data: {str(e)}")
            logger.error(traceback.format_exc())
        
        logger.info("Data transformation completed")
        return transformed_data
    
    async def _compute_embeddings(self, transformed_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute embeddings for text data.
        
        Args:
            transformed_data: Dictionary of transformed data
            
        Returns:
            Dictionary of transformed data with embeddings
        """
        logger.info("Computing text embeddings")
        
        try:
            if "sentence" in self.embedders:
                # Compute embeddings for article abstracts
                if "articles" in transformed_data["nodes"]:
                    articles = transformed_data["nodes"]["articles"]
                    transformed_data["nodes"]["articles"] = await self.embedders["sentence"].compute_article_embeddings(articles)
                
                # Compute embeddings for clinical trial descriptions
                if "clinical_trials" in transformed_data["nodes"]:
                    trials = transformed_data["nodes"]["clinical_trials"]
                    transformed_data["nodes"]["clinical_trials"] = await self.embedders["sentence"].compute_trial_embeddings(trials)
        
        except Exception as e:
            logger.error(f"Error computing embeddings: {str(e)}")
            logger.error(traceback.format_exc())
        
        logger.info("Embedding computation completed")
        return transformed_data
    
    async def _load_data(self, embedded_data: Dict[str, Any]) -> None:
        """
        Load transformed and embedded data into Neo4j.
        
        Args:
            embedded_data: Dictionary of transformed data with embeddings
        """
        logger.info("Loading data into Neo4j")
        
        try:
            neo4j_loader = self.loaders.get("neo4j")
            if neo4j_loader:
                # Connect to Neo4j
                connected = await neo4j_loader.connect()
                if not connected:
                    logger.error("Failed to connect to Neo4j")
                    return
                
                # Initialize database schema
                await neo4j_loader.init_schema()
                
                # Load nodes
                for node_type, nodes in embedded_data["nodes"].items():
                    logger.info(f"Loading {len(nodes)} {node_type} into Neo4j")
                    await neo4j_loader.load_nodes(node_type, nodes)
                
                # Load relationships
                if "relationships" in embedded_data:
                    logger.info(f"Loading {len(embedded_data['relationships'])} relationships into Neo4j")
                    await neo4j_loader.load_relationships(embedded_data["relationships"])
                
                # Create indexes for embedding vectors
                await neo4j_loader.create_vector_indexes()
                
                # Disconnect from Neo4j
                await neo4j_loader.disconnect()
        
        except Exception as e:
            logger.error(f"Error loading data into Neo4j: {str(e)}")
            logger.error(traceback.format_exc())
        
        logger.info("Data loading completed")

async def run_pipeline(config: Optional[EtlConfig] = None):
    """
    Run the biomedical ETL pipeline with the given configuration.
    
    Args:
        config: EtlConfig instance, or None to use default configuration
    """
    if config is None:
        config = EtlConfig()
    
    pipeline = BiomedicalEtlPipeline(config)
    return await pipeline.run()

if __name__ == "__main__":
    # Run the pipeline with default configuration
    asyncio.run(run_pipeline())