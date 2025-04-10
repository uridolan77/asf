import os
import json
import time
import threading
import logging
from typing import Dict, List, Optional, Any, Set, Union, Callable
from datetime import datetime

# Import necessary libraries for GraphRAG implementation
import pandas as pd
import numpy as np
from dataclasses import dataclass, field

# For text processing and NER
import re
import spacy
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# For embeddings and vector store
import faiss
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('medical_graphrag')

class MedicalEntityExtractor:
    """
    Extract medical entities (conditions, treatments, medications, etc.)
    from text using named entity recognition.
    """
    
    def __init__(self, model_name: str = "emilyalsentzer/Bio_ClinicalBERT"):
        """
        Initialize the entity extractor with a medical/clinical model.
        
        Args:
            model_name: Name of the pre-trained model to use
        """
        self.model_name = model_name
        
        # Load spaCy model for basic NER
        try:
            self.nlp = spacy.load("en_core_sci_md")
            logger.info("Loaded spaCy model for scientific/medical entity extraction")
        except:
            # Fallback to standard model
            self.nlp = spacy.load("en_core_web_sm")
            logger.warning("Scientific model not available, using standard spaCy model")
            
        # Initialize BERT-based NER model for medical entities
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        self.ner_pipeline = pipeline(
            "ner", 
            model=self.model, 
            tokenizer=self.tokenizer,
            aggregation_strategy="simple"
        )
        
        logger.info(f"Initialized medical entity extractor with model: {model_name}")
    
    def extract_entities_from_text(self, text: str) -> Dict[str, List[str]]:
        """
        Extract medical entities from text.
        
        Args:
            text: Input text to extract entities from
            
        Returns:
            Dictionary mapping entity types to lists of entities
        """
        if not text:
            return {}
            
        # Process with spaCy
        doc = self.nlp(text)
        
        # Extract entities from spaCy
        entities = {
            "CONDITION": [],
            "MEDICATION": [],
            "PROCEDURE": [],
            "ANATOMY": [],
            "TIME": [],
            "MEASUREMENT": [],
            "PERSON": [],
            "ORGANIZATION": [],
            "LOCATION": [],
        }
        
        # Map spaCy entity types to our categories
        entity_mapping = {
            "DISEASE": "CONDITION",
            "CHEMICAL": "MEDICATION",
            "DRUG": "MEDICATION",
            "ORG": "ORGANIZATION",
            "GPE": "LOCATION",
            "LOC": "LOCATION",
            "PERSON": "PERSON",
            "DATE": "TIME",
            "TIME": "TIME",
        }
        
        # Extract entities from spaCy
        for ent in doc.ents:
            mapped_type = entity_mapping.get(ent.label_, ent.label_)
            if mapped_type in entities:
                entities[mapped_type].append(ent.text)
        
        # Use BERT-based model for more specific medical entity recognition
        try:
            bert_entities = self.ner_pipeline(text)
            for entity in bert_entities:
                entity_text = entity['word']
                entity_type = entity['entity_group']
                
                # Map BERT entity types to our categories
                if "DISEASE" in entity_type or "PROBLEM" in entity_type:
                    entities["CONDITION"].append(entity_text)
                elif "TREATMENT" in entity_type or "PROCEDURE" in entity_type:
                    entities["PROCEDURE"].append(entity_text)
                elif "DRUG" in entity_type or "CHEMICAL" in entity_type:
                    entities["MEDICATION"].append(entity_text)
                elif "BODY" in entity_type or "ANAT" in entity_type:
                    entities["ANATOMY"].append(entity_text)
        except Exception as e:
            logger.error(f"Error in BERT-based NER: {str(e)}")
        
        # Remove duplicates and sort
        for entity_type in entities:
            entities[entity_type] = sorted(list(set(entities[entity_type])))
        
        return entities

class TextChunker:
    """
    Handles intelligent chunking of text documents with various strategies.
    """
    
    def __init__(self, 
                chunk_size: int = 1000, 
                chunk_overlap: int = 200,
                strategy: str = "recursive"):
        """
        Initialize the text chunker.
        
        Args:
            chunk_size: Target size of chunks
            chunk_overlap: Overlap between chunks
            strategy: Chunking strategy ('recursive', 'sentence', 'paragraph')
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strategy = strategy
        
        # Initialize the chunker based on strategy
        if strategy == "recursive":
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
        else:
            # Default to recursive if strategy not recognized
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            
        logger.info(f"Initialized text chunker with strategy: {strategy}")
    
    def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Split text into chunks with metadata.
        
        Args:
            text: Text to chunk
            metadata: Metadata to include with each chunk
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        if not text:
            return []
            
        # Split the text into chunks
        chunks = self.text_splitter.create_documents([text], [metadata or {}])
        
        # Convert to dictionary format
        result = []
        for i, chunk in enumerate(chunks):
            chunk_dict = {
                'content': chunk.page_content,
                'metadata': chunk.metadata.copy()
            }
            
            # Add chunk number to metadata
            chunk_dict['metadata']['chunk_num'] = i
            chunk_dict['metadata']['chunk_total'] = len(chunks)
            
            result.append(chunk_dict)
            
        return result

class VectorStore:
    """
    Vector store for document embeddings with metadata for graph construction.
    """
    
    def __init__(self, embedding_dim: int = 1536):
        """
        Initialize the vector store.
        
        Args:
            embedding_dim: Dimension of the embeddings
        """
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.documents = []
        self.embeddings_model = OpenAIEmbeddings()
        self.last_updated = datetime.now()
        
        # For tracking document sources
        self.document_sources = {}  # Maps source_id to document_ids
        
        logger.info(f"Initialized vector store with embedding dimension: {embedding_dim}")
    
    def add_document(self, 
                    document: Dict[str, Any], 
                    embedding: Optional[List[float]] = None,
                    source_id: Optional[str] = None) -> int:
        """
        Add a document to the vector store.
        
        Args:
            document: Document with metadata
            embedding: Pre-computed embedding (optional)
            source_id: Identifier for the document source (for tracking updates)
            
        Returns:
            Index of the added document
        """
        # Generate embedding if not provided
        if embedding is None:
            text = document.get('content', '')
            embedding = self.embeddings_model.embed_query(text)
        
        # Add to FAISS index
        faiss_embedding = np.array([embedding]).astype('float32')
        self.index.add(faiss_embedding)
        
        # Store document
        doc_id = len(self.documents)
        document['id'] = doc_id
        self.documents.append(document)
        
        # Track document source if provided
        if source_id:
            if source_id not in self.document_sources:
                self.document_sources[source_id] = []
            self.document_sources[source_id].append(doc_id)
        
        # Update last modified timestamp
        self.last_updated = datetime.now()
        
        return doc_id
        
    def update_document(self, doc_id: int, document: Dict[str, Any]) -> bool:
        """
        Update an existing document.
        
        Args:
            doc_id: Document ID to update
            document: New document content and metadata
            
        Returns:
            Success flag
        """
        if doc_id < 0 or doc_id >= len(self.documents):
            logger.error(f"Cannot update document {doc_id}: ID out of range")
            return False
            
        # Update content
        old_doc = self.documents[doc_id]
        document['id'] = doc_id
        self.documents[doc_id] = document
        
        # Update embedding if content changed
        if old_doc.get('content') != document.get('content'):
            # Remove old embedding and add new one
            # Note: FAISS doesn't support direct updates, so we need to rebuild the index
            # This is a simplified approach - in production, you might want to batch updates
            embeddings = []
            for i, doc in enumerate(self.documents):
                if i == doc_id:
                    # Generate new embedding for the updated document
                    embedding = self.embeddings_model.embed_query(document.get('content', ''))
                else:
                    # Use the existing embedding
                    dist, _ = self.index.search(np.array([[0] * self.embedding_dim]).astype('float32'), 1)
                    embedding = [0] * self.embedding_dim  # Placeholder - in production you'd store embeddings
            
            # Rebuild index
            self.index = faiss.IndexFlatL2(self.embedding_dim)
            for embedding in embeddings:
                self.index.add(np.array([embedding]).astype('float32'))
        
        # Update last modified timestamp
        self.last_updated = datetime.now()
        
        return True
    
    def search(self, 
              query: str, 
              k: int = 5, 
              query_embedding: Optional[List[float]] = None) -> List[Dict[str, Any]]:
        """
        Search for documents by vector similarity.
        
        Args:
            query: Query string
            k: Number of results to return
            query_embedding: Pre-computed query embedding (optional)
            
        Returns:
            List of matching documents with distances
        """
        # Generate embedding if not provided
        if query_embedding is None:
            query_embedding = self.embeddings_model.embed_query(query)
        
        # Search in FAISS index
        faiss_query = np.array([query_embedding]).astype('float32')
        distances, indices = self.index.search(faiss_query, k)
        
        # Return results
        results = []
        for i, doc_idx in enumerate(indices[0]):
            if doc_idx >= 0 and doc_idx < len(self.documents):
                doc = self.documents[doc_idx].copy()
                doc['distance'] = float(distances[0][i])
                results.append(doc)
        
        return results

class QueryRewriter:
    """
    Rewrites user queries to improve semantic search retrieval.
    """
    
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        """
        Initialize the query rewriter.
        
        Args:
            model_name: LLM to use for query rewriting
        """
        from langchain_openai import ChatOpenAI
        from langchain.prompts import ChatPromptTemplate
        
        self.model_name = model_name
        self.llm = ChatOpenAI(model_name=model_name, temperature=0.1)
        
        # Create prompt template for query rewriting
        self.prompt_template = ChatPromptTemplate.from_template(
            """You are an AI assistant specializing in medical query rewriting. 
            Your task is to rewrite the user's query to improve retrieval of relevant medical information.
            
            Original query: {query}
            
            Rewrite this query to:
            1. Expand medical acronyms and abbreviations
            2. Include synonyms for medical terms
            3. Make implicit medical concepts explicit
            4. Add relevant medical context
            
            Return ONLY the rewritten query without explanations or additional text."""
        )
        
        self.chain = self.prompt_template | self.llm
        
        logger.info(f"Initialized query rewriter with model: {model_name}")
    
    def rewrite_query(self, query: str) -> str:
        """
        Rewrite a user query to improve retrieval.
        
        Args:
            query: Original user query
            
        Returns:
            Rewritten query
        """
        try:
            rewritten_query = self.chain.invoke({"query": query}).content.strip()
            logger.info(f"Rewrote query: '{query}' → '{rewritten_query}'")
            return rewritten_query
        except Exception as e:
            logger.error(f"Error rewriting query: {str(e)}")
            return query  # Fall back to original query on error

class GraphRetriever:
    """
    Graph-based retrieval system that extends vector search with
    entity-relationship traversal.
    """
    
    def __init__(self, vector_store: VectorStore):
        """
        Initialize the graph retriever.
        
        Args:
            vector_store: Vector store instance
        """
        self.vector_store = vector_store
        self.entity_to_docs = {}  # Maps entities to document IDs
        self.doc_to_entities = {}  # Maps document IDs to entities
        self.query_rewriter = QueryRewriter()
        
        logger.info("Initialized graph retriever")
    
    def add_document(self, 
                   document: Dict[str, Any],
                   source_id: Optional[str] = None) -> int:
        """
        Add a document to the graph retriever.
        
        Args:
            document: Document with metadata and entities
            source_id: Identifier for the document source (for tracking updates)
            
        Returns:
            Document ID
        """
        # Add to vector store
        doc_id = self.vector_store.add_document(document, source_id=source_id)
        
        # Extract entities for the graph
        entities = {}
        for entity_type, entity_list in document.get('entities', {}).items():
            for entity in entity_list:
                entity_key = f"{entity_type}:{entity}"
                
                # Update entity to docs mapping
                if entity_key not in self.entity_to_docs:
                    self.entity_to_docs[entity_key] = set()
                self.entity_to_docs[entity_key].add(doc_id)
                
                # Update doc to entities mapping
                if doc_id not in self.doc_to_entities:
                    self.doc_to_entities[doc_id] = set()
                self.doc_to_entities[doc_id].add(entity_key)
        
        return doc_id
    
    def simple_search(self, 
                     query: str, 
                     k: int = 5,
                     rewrite_query: bool = False) -> List[Dict[str, Any]]:
        """
        Perform a simple vector search.
        
        Args:
            query: Query string
            k: Number of results to return
            rewrite_query: Whether to rewrite the query for better retrieval
            
        Returns:
            List of matching documents
        """
        if rewrite_query:
            query = self.query_rewriter.rewrite_query(query)
            
        return self.vector_store.search(query, k)
    
    def graph_search(self, 
                    query: str, 
                    k: int = 5, 
                    depth: int = 1, 
                    expansion_factor: int = 2,
                    rewrite_query: bool = False) -> List[Dict[str, Any]]:
        """
        Perform a graph-based search.
        
        Args:
            query: Query string
            k: Number of initial results from vector search
            depth: Graph traversal depth
            expansion_factor: Number of related documents to retrieve per entity
            rewrite_query: Whether to rewrite the query for better retrieval
            
        Returns:
            List of matching documents
        """
        # Rewrite the query if requested
        if rewrite_query:
            query = self.query_rewriter.rewrite_query(query)
        
        # First, get initial results from vector search
        initial_results = self.vector_store.search(query, k)
        
        # Keep track of seen documents to avoid duplicates
        seen_docs = set()
        for doc in initial_results:
            seen_docs.add(doc['id'])
        
        all_results = list(initial_results)
        
        # Perform graph traversal
        current_docs = initial_results
        
        for d in range(depth):
            related_docs = []
            
            # For each current document
            for doc in current_docs:
                doc_id = doc['id']
                
                # Get the entities in this document
                if doc_id not in self.doc_to_entities:
                    continue
                
                doc_entities = self.doc_to_entities[doc_id]
                
                # For each entity, get related documents
                for entity_key in doc_entities:
                    if entity_key not in self.entity_to_docs:
                        continue
                    
                    related_doc_ids = self.entity_to_docs[entity_key]
                    
                    # Add related documents if not seen before
                    for related_id in related_doc_ids:
                        if related_id in seen_docs:
                            continue
                        
                        seen_docs.add(related_id)
                        related_doc = self.vector_store.documents[related_id].copy()
                        
                        # Add connection information
                        entity_type, entity_value = entity_key.split(':', 1)
                        related_doc['connection'] = {
                            'type': 'entity',
                            'entity_type': entity_type,
                            'entity_value': entity_value,
                            'connection_depth': d + 1,
                            'from_document_id': doc_id
                        }
                        
                        related_docs.append(related_doc)
                        
                        # Limit the number of related documents per entity at higher depths
                        max_docs_at_depth = expansion_factor // (d + 1) if d > 0 else expansion_factor
                        if len(related_docs) >= max_docs_at_depth:
                            break
            
            # Update for next iteration
            current_docs = related_docs
            all_results.extend(related_docs)
            
            # Break if no more related documents found
            if not related_docs:
                break
        
        # Sort by relevance score first, then by connection depth
        all_results.sort(key=lambda x: (x.get('distance', 0), x.get('connection', {}).get('connection_depth', 0)))
        
        return all_results
    
    def get_graph_visualization_data(self, document_ids: List[int]) -> Dict[str, Any]:
        """
        Generate data for visualizing the graph of documents and entities.
        
        Args:
            document_ids: List of document IDs to include
            
        Returns:
            Dictionary with nodes and edges for visualization
        """
        nodes = []
        edges = []
        
        # Create document nodes
        for doc_id in document_ids:
            if doc_id >= len(self.vector_store.documents):
                continue
                
            doc = self.vector_store.documents[doc_id]
            nodes.append({
                "id": f"doc_{doc_id}",
                "label": doc.get('title', f"Document {doc_id}"),
                "type": "document"
            })
        
        # Create entity nodes and edges
        for doc_id in document_ids:
            if doc_id not in self.doc_to_entities:
                continue
                
            for entity_key in self.doc_to_entities[doc_id]:
                entity_type, entity_text = entity_key.split(':', 1)
                
                entity_id = f"entity_{entity_key.replace(' ', '_')}"
                
                # Add entity node if not already added
                if not any(node["id"] == entity_id for node in nodes):
                    nodes.append({
                        "id": entity_id,
                        "label": entity_text,
                        "type": entity_type
                    })
                
                # Add edge
                edges.append({
                    "source": f"doc_{doc_id}",
                    "target": entity_id,
                    "label": entity_type
                })
        
        return {
            "nodes": nodes,
            "edges": edges
        }

class AutoIndexer:
    """
    Manages background indexing of data sources and automatic updates.
    """
    
    def __init__(self, 
                indexing_func: Callable, 
                interval_seconds: int = 3600,  # Default to hourly checks
                data_source_id: Optional[str] = None):
        """
        Initialize the auto-indexer.
        
        Args:
            indexing_func: Function to call for indexing
            interval_seconds: How often to check for updates (in seconds)
            data_source_id: Identifier for the data source
        """
        self.indexing_func = indexing_func
        self.interval_seconds = interval_seconds
        self.data_source_id = data_source_id
        self.last_run = None
        self.is_running = False
        self.thread = None
        self.stop_event = threading.Event()
        
        logger.info(f"Initialized auto-indexer for data source: {data_source_id}")
    
    def start(self):
        """Start the background indexing thread."""
        if self.is_running:
            logger.warning("Auto-indexer is already running")
            return
            
        self.stop_event.clear()
        self.is_running = True
        self.thread = threading.Thread(target=self._indexing_loop)
        self.thread.daemon = True
        self.thread.start()
        
        logger.info("Started auto-indexer background thread")
    
    def stop(self):
        """Stop the background indexing thread."""
        if not self.is_running:
            logger.warning("Auto-indexer is not running")
            return
            
        self.stop_event.set()
        self.thread.join(timeout=30)
        self.is_running = False
        
        logger.info("Stopped auto-indexer background thread")
    
    def _indexing_loop(self):
        """Background thread for periodic indexing."""
        logger.info(f"Starting indexing loop for data source: {self.data_source_id}")
        
        while not self.stop_event.is_set():
            try:
                # Run the indexing function
                self.indexing_func()
                self.last_run = datetime.now()
                logger.info(f"Completed indexing run for data source: {self.data_source_id}")
            except Exception as e:
                logger.error(f"Error during indexing: {str(e)}")
            
            # Sleep until next indexing interval or until stopped
            for _ in range(self.interval_seconds):
                if self.stop_event.is_set():
                    break
                time.sleep(1)
    
    def force_run(self):
        """Force an immediate indexing run."""
        try:
            logger.info(f"Forcing indexing run for data source: {self.data_source_id}")
            self.indexing_func()
            self.last_run = datetime.now()
            logger.info(f"Completed forced indexing run for data source: {self.data_source_id}")
            return True
        except Exception as e:
            logger.error(f"Error during forced indexing run: {str(e)}")
            return False

class ClinicalTrialsGraphRAG:
    """
    GraphRAG application for clinical trials data.
    """
    
    def __init__(self, api_key: Optional[str] = None, auto_index: bool = True):
        """
        Initialize the clinical trials GraphRAG application.
        
        Args:
            api_key: OpenAI API key for embeddings
            auto_index: Whether to enable automatic indexing
        """
        # Set API key if provided
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        
        # Initialize components
        self.entity_extractor = MedicalEntityExtractor()
        self.vector_store = VectorStore()
        self.graph_retriever = GraphRetriever(self.vector_store)
        self.text_chunker = TextChunker(chunk_size=1000, chunk_overlap=200)
        
        # Track registered data sources
        self.data_sources = {}
        self.auto_indexers = {}
        self.auto_index_enabled = auto_index
        
        # LLM for response generation
        from langchain_openai import ChatOpenAI
        self.llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
        
        logger.info("Initialized Clinical Trials GraphRAG application")
    
 
    def ingest_trial(self, 
                    trial_data: Dict[str, Any], 
                    update_existing: bool = True) -> int:
        """
        Ingest a clinical trial into the GraphRAG system.
        Args:
            trial_data: Clinical trial data
            update_existing: Whether to update existing trial data
            
        Returns:
            Document ID
        """
        # Check if we already have this trial
        nct_id = trial_data.get('nct_id', '')
        source_id = f"trial:{nct_id}" if nct_id else None
        
        # Extract text content
        content_parts = []
        
        if 'brief_title' in trial_data:
            content_parts.append(f"Title: {trial_data['brief_title']}")
        
        if 'official_title' in trial_data:
            content_parts.append(f"Official Title: {trial_data['official_title']}")
        
        if 'brief_summary' in trial_data:
            content_parts.append(f"Summary: {trial_data['brief_summary']}")
        
        if 'detailed_description' in trial_data:
            content_parts.append(f"Description: {trial_data['detailed_description']}")
        
        if 'eligibility' in trial_data:
            content_parts.append(f"Eligibility: {trial_data['eligibility']}")
        
        if 'condition' in trial_data:
            conditions = trial_data['condition'] if isinstance(trial_data['condition'], list) else [trial_data['condition']]
            content_parts.append(f"Conditions: {', '.join(conditions)}")
        
        if 'intervention' in trial_data:
            interventions = trial_data['intervention'] if isinstance(trial_data['intervention'], list) else [trial_data['intervention']]
            content_parts.append(f"Interventions: {', '.join(interventions)}")
        
        # Combine content
        content = "\n\n".join(content_parts)
        
        # Split content into chunks
        chunks = self.text_chunker.chunk_text(content, {
            'nct_id': nct_id,
            'title': trial_data.get('brief_title', ''),
            'source_id': source_id,
            'type': 'clinical_trial'
        })
        
        # Extract entities and add each chunk
        doc_ids = []
        for chunk in chunks:
            # Extract entities for this chunk
            entities = self.entity_extractor.extract_entities_from_text(chunk['content'])
            
            # Explicitly add conditions and interventions as entities
            if 'condition' in trial_data:
                conditions = trial_data['condition'] if isinstance(trial_data['condition'], list) else [trial_data['condition']]
                if "CONDITION" not in entities:
                    entities["CONDITION"] = []
                entities["CONDITION"].extend(conditions)
            
            if 'intervention' in trial_data:
                interventions = trial_data['intervention'] if isinstance(trial_data['intervention'], list) else [trial_data['intervention']]
                if "MEDICATION" not in entities:
                    entities["MEDICATION"] = []
                entities["MEDICATION"].extend(interventions)
            
            # Remove duplicates
            for entity_type in entities:
                entities[entity_type] = list(set(entities[entity_type]))
            
            # Create document
            document = {
                'title': trial_data.get('brief_title', ''),
                'content': chunk['content'],
                'entities': entities,
                'metadata': {
                    'nct_id': nct_id,
                    'status': trial_data.get('status', ''),
                    'phase': trial_data.get('phase', ''),
                    'study_type': trial_data.get('study_type', ''),
                    'locations': trial_data.get('location', []),
                    'chunk_num': chunk['metadata'].get('chunk_num', 0),
                    'chunk_total': chunk['metadata'].get('chunk_total', 1)
                }
            }
            
            # Add to graph retriever
            doc_id = self.graph_retriever.add_document(document, source_id=source_id)
            doc_ids.append(doc_id)
        
        # Return the first document ID as a reference
        return doc_ids[0] if doc_ids else -1
    
    def register_data_source(self, 
                           source_id: str, 
                           source_type: str,
                           indexing_func: Callable,
                           indexing_interval: int = 3600) -> bool:
        """
        Register a data source for automatic indexing.
        
        Args:
            source_id: Unique identifier for the data source
            source_type: Type of data source (e.g., 'api', 'file', 'database')
            indexing_func: Function to call for indexing
            indexing_interval: How often to check for updates (in seconds)
            
        Returns:
            Success flag
        """
        # Register the data source
        self.data_sources[source_id] = {
            'type': source_type,
            'last_indexed': None
        }
        
        # Create auto-indexer if enabled
        if self.auto_index_enabled:
            indexer = AutoIndexer(
                indexing_func=indexing_func,
                interval_seconds=indexing_interval,
                data_source_id=source_id
            )
            
            self.auto_indexers[source_id] = indexer
            indexer.start()
            
            logger.info(f"Registered data source '{source_id}' with auto-indexing")
        else:
            logger.info(f"Registered data source '{source_id}' without auto-indexing")
        
        return True
    
    def ingest_from_api(self, 
                       api_client, 
                       query: str, 
                       max_results: int = 100, 
                       source_id: Optional[str] = None,
                       register_for_updates: bool = True,
                       update_interval: int = 86400) -> List[int]:
        """
        Ingest clinical trials data from the API.
        
        Args:
            api_client: Clinical trials API client
            query: Search query
            max_results: Maximum number of results to ingest
            source_id: Identifier for the data source
            register_for_updates: Whether to register for automatic updates
            update_interval: How often to check for updates (in seconds)
            
        Returns:
            List of document IDs
        """
        # Generate source ID if not provided
        if not source_id:
            source_id = f"api_query:{query.replace(' ', '_')[:50]}"
        
        # Define indexing function for this data source
        def index_from_api():
            # Search for trials
            search_results = api_client.searchStudies(query, max_results=max_results)
            
            # Extract study IDs
            study_ids = []
            if search_results and 'studies' in search_results:
                for study in search_results['studies']:
                    nct_id = None
                    if 'protocolSection' in study and 'identificationModule' in study['protocolSection']:
                        nct_id = study['protocolSection']['identificationModule'].get('nctId')
                    if nct_id:
                        study_ids.append(nct_id)
            
            # Update last indexed timestamp
            if source_id in self.data_sources:
                self.data_sources[source_id]['last_indexed'] = datetime.now()
            
            # Fetch details for each study
            doc_ids = []
            for nct_id in study_ids:
                try:
                    # Get study details
                    study = api_client.getStudy(nct_id)
                    
                    # Extract relevant information
                    protocol = study.get('protocolSection', {})
                    
                    trial_data = {
                        'nct_id': nct_id,
                        'brief_title': protocol.get('identificationModule', {}).get('briefTitle', ''),
                        'official_title': protocol.get('identificationModule', {}).get('officialTitle', ''),
                        'status': protocol.get('statusModule', {}).get('overallStatus', ''),
                        'phase': protocol.get('designModule', {}).get('phases', []),
                        'study_type': protocol.get('designModule', {}).get('studyType', ''),
                        'brief_summary': protocol.get('descriptionModule', {}).get('briefSummary', ''),
                        'detailed_description': protocol.get('descriptionModule', {}).get('detailedDescription', ''),
                        'condition': protocol.get('conditionsModule', {}).get('conditions', []),
                        'intervention': [i.get('name', '') for i in protocol.get('armsInterventionsModule', {}).get('interventions', [])],
                        'eligibility': protocol.get('eligibilityModule', {}).get('eligibilityCriteria', ''),
                        'location': [
                            {
                                'facility': loc.get('facility', ''),
                                'city': loc.get('city', ''),
                                'state': loc.get('state', ''),
                                'country': loc.get('country', '')
                            }
                            for loc in protocol.get('contactsLocationsModule', {}).get('locations', [])
                        ]
                    }
                    
                    # Ingest the trial
                    doc_id = self.ingest_trial(trial_data)
                    doc_ids.append(doc_id)
                    
                    logger.info(f"Ingested trial {nct_id} as document {doc_id}")
                    
                except Exception as e:
                    logger.error(f"Error ingesting trial {nct_id}: {str(e)}")
            
            return doc_ids
        
        # Register the data source if requested
        if register_for_updates:
            self.register_data_source(
                source_id=source_id,
                source_type='api',
                indexing_func=index_from_api,
                indexing_interval=update_interval
            )
        
        # Run initial indexing
        return index_from_api()
    
    def search(self, 
              query: str, 
              use_graph: bool = True, 
              k: int = 5,
              rewrite_query: bool = True) -> List[Dict[str, Any]]:
        """
        Search for clinical trials.
        
        Args:
            query: Search query
            use_graph: Whether to use graph-based retrieval
            k: Number of initial results to retrieve
            rewrite_query: Whether to rewrite the query for better results
            
        Returns:
            List of matching documents
        """
        if use_graph:
            results = self.graph_retriever.graph_search(query, k=k, rewrite_query=rewrite_query)
        else:
            results = self.graph_retriever.simple_search(query, k=k, rewrite_query=rewrite_query)
        
        return results
    
    def generate_response(self, query: str, use_graph: bool = True, k: int = 5) -> Dict[str, Any]:
        """
        Generate a response to a query using RAG.
        
        Args:
            query: User query
            use_graph: Whether to use graph-based retrieval
            k: Number of initial results to retrieve
            
        Returns:
            Dictionary with response and supporting documents
        """
        # Search for relevant documents
        results = self.search(query, use_graph=use_graph, k=k)
        
        if not results:
            return {
                'response': "I don't have enough information to answer that question.",
                'sources': []
            }
        
        # Extract content from retrieved documents
        contexts = []
        sources = []
        
        for result in results:
            # Add content
            contexts.append(result.get('content', ''))
            
            # Add source info
            sources.append({
                'id': result.get('id'),
                'title': result.get('title', ''),
                'nct_id': result.get('metadata', {}).get('nct_id', ''),
                'relevance_score': 1.0 - result.get('distance', 0) if 'distance' in result else 1.0,
                'connection': result.get('connection')
            })
        
        # Combine contexts
        context_text = "\n\n".join(contexts)
        
        # Generate response
        from langchain.prompts import ChatPromptTemplate
        
        prompt = ChatPromptTemplate.from_template(
            """You are a helpful medical research assistant. Answer the user's question based on the context provided.
            
            Context:
            {context}
            
            Question: {query}
            
            Provide a concise, informative answer based only on the provided context. If the context doesn't contain 
            the information needed to answer the question, say "I don't have enough information to answer that question."
            """
        )
        
        chain = prompt | self.llm
        
        try:
            response = chain.invoke({
                "context": context_text,
                "query": query
            }).content
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            response = "I'm sorry, I encountered an error while generating a response."
        
        return {
            'response': response,
            'sources': sources
        }
    
    def visualize_graph(self, document_ids: List[int]) -> Dict[str, Any]:
        """
        Generate data for visualizing the graph.
        
        Args:
            document_ids: List of document IDs
            
        Returns:
            Graph visualization data
        """
        return self.graph_retriever.get_graph_visualization_data(document_ids)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the GraphRAG system.
        
        Returns:
            Dictionary of statistics
        """
        # Count entities
        entity_counts = {}
        for entity_type in set(key.split(':')[0] for key in self.graph_retriever.entity_to_docs.keys()):
            entity_counts[entity_type] = len([k for k in self.graph_retriever.entity_to_docs.keys() if k.startswith(f"{entity_type}:")])
        
        # Count documents by type
        doc_type_counts = {}
        for doc in self.vector_store.documents:
            doc_type = doc.get('metadata', {}).get('type', 'unknown')
            if doc_type not in doc_type_counts:
                doc_type_counts[doc_type] = 0
            doc_type_counts[doc_type] += 1
        
        # Get data source info
        data_sources_info = {}
        for source_id, source_info in self.data_sources.items():
            data_sources_info[source_id] = {
                'type': source_info.get('type'),
                'last_indexed': source_info.get('last_indexed').isoformat() if source_info.get('last_indexed') else None,
                'auto_indexing': source_id in self.auto_indexers
            }
        
        return {
            'total_documents': len(self.vector_store.documents),
            'total_entities': len(self.graph_retriever.entity_to_docs),
            'entity_counts': entity_counts,
            'document_types': doc_type_counts,
            'data_sources': data_sources_info,
            'last_updated': self.vector_store.last_updated.isoformat()
        }

# Example usage
def main():
    # Initialize the GraphRAG application
    api_key = os.getenv("OPENAI_API_KEY")
    app = ClinicalTrialsGraphRAG(api_key=api_key)
    
    # Ingest some sample clinical trials
    sample_trial = {
        'nct_id': 'NCT12345678',
        'brief_title': 'Efficacy of Remdesivir for Treatment of COVID-19',
        'official_title': 'A Randomized Controlled Trial of Remdesivir for the Treatment of Severe COVID-19 Pneumonia',
        'brief_summary': 'This study evaluates the efficacy and safety of remdesivir in hospitalized adults diagnosed with COVID-19 pneumonia.',
        'detailed_description': 'This is a randomized, double-blind, placebo-controlled trial to evaluate the safety and efficacy of remdesivir in hospitalized adults diagnosed with severe COVID-19. Participants will receive either remdesivir or placebo for 10 days, and will be followed for 28 days to assess outcomes.',
        'condition': ['COVID-19', 'Severe Pneumonia', 'SARS-CoV-2 Infection'],
        'intervention': ['Remdesivir', 'Placebo'],
        'eligibility': 'Inclusion Criteria: - Adults aged 18 years or older - Confirmed SARS-CoV-2 infection - Hospitalized with severe pneumonia - Oxygen saturation ≤94% on room air Exclusion Criteria: - Pregnant or breastfeeding - Anticipated transfer to another hospital - ALT or AST > 5 times the upper limit of normal',
        'status': 'Recruiting',
        'phase': ['Phase 3'],
        'study_type': 'Interventional',
        'location': [
            {'facility': 'University Hospital', 'city': 'New York', 'state': 'NY', 'country': 'United States'},
            {'facility': 'Medical Center', 'city': 'Los Angeles', 'state': 'CA', 'country': 'United States'}
        ]
    }
    
    doc_id = app.ingest_trial(sample_trial)
    print(f"Ingested sample trial as document {doc_id}")
    
    # Test search
    query = "COVID-19 treatment with antiviral medications"
    results = app.search(query, use_graph=True, k=3)
    
    print(f"\nSearch results for: '{query}'")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result.get('title')} ({result.get('metadata', {}).get('nct_id', '')})")
        if 'connection' in result:
            print(f"   Connection: {result['connection']}")
        print(f"   Entities: {', '.join([f'{k}: {len(v)}' for k, v in result.get('entities', {}).items() if v])}")
    
    # Generate a response
    response = app.generate_response(query)
    print(f"\nGenerated response: {response['response']}")
    print(f"Based on {len(response['sources'])} sources")
    
    # Get system statistics
    stats = app.get_statistics()
    print(f"\nSystem statistics: {stats}")

if __name__ == "__main__":
    main()