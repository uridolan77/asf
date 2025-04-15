#!/usr/bin/env python3
"""
SNOMED CT Client

A comprehensive client for interacting with SNOMED CT clinical terminology.
This client supports multiple access methods:
1. UMLS API (default for US users)
2. SNOMED CT Snowstorm API (for direct access)
3. Local SNOMED CT files (for offline or high-performance usage)

Features:
- Concept search and retrieval
- Hierarchical navigation
- Relationship traversal
- Subsumption testing
- Expression constraints
- Caching for performance optimization
"""

import os
import json
import time
import hashlib
import requests
from typing import Dict, List, Optional, Union, Set, Tuple
from functools import lru_cache
from datetime import datetime, timedelta
import logging
import sqlite3
import re
from pathlib import Path
import pandas as pd
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("snomedct")

# Constants
# SNOMED CT concept model attribute types
IS_A_RELATIONSHIP = "116680003"  # |is a| relationship type
FINDING_SITE = "363698007"       # |Finding site| attribute
ASSOCIATED_WITH = "47429007"     # |Associated with| attribute
CAUSATIVE_AGENT = "246075003"    # |Causative agent| attribute
PATHOLOGICAL_PROCESS = "370135005"  # |Pathological process| attribute
METHOD = "260686004"             # |Method| attribute
PROCEDURE_SITE = "363704007"     # |Procedure site| attribute

# Common high-level SNOMED CT concepts
CLINICAL_FINDING = "404684003"   # |Clinical finding|
PROCEDURE = "71388002"           # |Procedure|
BODY_STRUCTURE = "123037004"     # |Body structure|
ORGANISM = "410607006"           # |Organism|
SUBSTANCE = "105590001"          # |Substance|
PHARMACEUTICAL_PRODUCT = "373873005"  # |Pharmaceutical / biologic product|
SITUATION = "243796009"          # |Situation with explicit context|
EVENT = "272379006"              # |Event|

# Cache settings
DEFAULT_CACHE_TTL = 86400  # 24 hours in seconds
DEFAULT_CACHE_SIZE = 1000  # Number of items to cache in memory


class SnomedCache:
    """Cache for SNOMED CT concept data to improve performance."""
    
    def __init__(self, cache_dir: Optional[str] = None, ttl: int = DEFAULT_CACHE_TTL, 
                 memory_size: int = DEFAULT_CACHE_SIZE):
        """
        Initialize the SNOMED CT cache.
        
        Args:
            cache_dir: Directory to store persistent cache. If None, only memory cache is used.
            ttl: Time-to-live for cached items in seconds. Default is 24 hours.
            memory_size: Number of items to cache in memory.
        """
        self.ttl = ttl
        self.memory_size = memory_size
        self.memory_cache = {}
        self.disk_cache_enabled = cache_dir is not None
        
        if self.disk_cache_enabled:
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize SQLite cache if disk cache is enabled
            self.db_path = self.cache_dir / "snomed_cache.db"
            self._init_db()
    
    def _init_db(self) -> None:
        """Initialize the SQLite database for persistent caching."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create cache table if it doesn't exist
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS snomed_cache (
            key TEXT PRIMARY KEY,
            data TEXT NOT NULL,
            created_at TIMESTAMP NOT NULL
        )
        ''')
        
        # Create index on timestamp for cleanup
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_created_at ON snomed_cache (created_at)')
        
        conn.commit()
        conn.close()
    
    def _generate_key(self, prefix: str, *args) -> str:
        """Generate a unique cache key based on the function arguments."""
        key_str = prefix + ":" + json.dumps(args, sort_keys=True)
        return hashlib.md5(key_str.encode('utf-8')).hexdigest()
    
    def get(self, prefix: str, *args) -> Optional[Dict]:
        """
        Get an item from the cache.
        
        Args:
            prefix: A string prefix to categorize the cache entry
            *args: Arguments to generate the unique key
            
        Returns:
            The cached data or None if not found or expired
        """
        key = self._generate_key(prefix, *args)
        
        # Check memory cache first
        if key in self.memory_cache:
            entry = self.memory_cache[key]
            if time.time() - entry["timestamp"] < self.ttl:
                return entry["data"]
            else:
                # Remove expired item from memory cache
                del self.memory_cache[key]
        
        # If not in memory and disk cache is enabled, check disk
        if self.disk_cache_enabled:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get entry and check if it's expired
            cursor.execute(
                "SELECT data, created_at FROM snomed_cache WHERE key = ?", 
                (key,)
            )
            result = cursor.fetchone()
            
            if result:
                data_json, created_at = result
                created_timestamp = datetime.fromisoformat(created_at).timestamp()
                
                if time.time() - created_timestamp < self.ttl:
                    data = json.loads(data_json)
                    
                    # Add to memory cache for faster access next time
                    self._add_to_memory_cache(key, data)
                    
                    conn.close()
                    return data
                else:
                    # Remove expired entry
                    cursor.execute("DELETE FROM snomed_cache WHERE key = ?", (key,))
                    conn.commit()
            
            conn.close()
        
        return None
    
    def _add_to_memory_cache(self, key: str, data: Dict) -> None:
        """Add an item to the memory cache, managing cache size."""
        if len(self.memory_cache) >= self.memory_size:
            # Simple strategy: remove oldest item
            oldest_key = min(self.memory_cache.keys(), 
                            key=lambda k: self.memory_cache[k]["timestamp"])
            del self.memory_cache[oldest_key]
        
        self.memory_cache[key] = {
            "data": data,
            "timestamp": time.time()
        }
    
    def set(self, prefix: str, data: Dict, *args) -> None:
        """
        Store an item in the cache.
        
        Args:
            prefix: A string prefix to categorize the cache entry
            data: The data to cache
            *args: Arguments to generate the unique key
        """
        key = self._generate_key(prefix, *args)
        
        # Add to memory cache
        self._add_to_memory_cache(key, data)
        
        # If disk cache is enabled, store there too
        if self.disk_cache_enabled:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Store the data with the current timestamp
            now = datetime.now().isoformat()
            cursor.execute(
                "INSERT OR REPLACE INTO snomed_cache (key, data, created_at) VALUES (?, ?, ?)",
                (key, json.dumps(data), now)
            )
            
            conn.commit()
            conn.close()
    
    def clear_expired(self) -> int:
        """
        Clear expired items from the cache.
        
        Returns:
            Number of items removed
        """
        # Clear expired items from memory cache
        now = time.time()
        expired_keys = [k for k, v in self.memory_cache.items() 
                       if now - v["timestamp"] >= self.ttl]
        
        for key in expired_keys:
            del self.memory_cache[key]
        
        memory_cleared = len(expired_keys)
        
        # Clear expired items from disk cache if enabled
        disk_cleared = 0
        if self.disk_cache_enabled:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Calculate the expiration timestamp
            expiration_time = (datetime.now() - timedelta(seconds=self.ttl)).isoformat()
            
            # Delete expired entries
            cursor.execute(
                "DELETE FROM snomed_cache WHERE created_at < ?", 
                (expiration_time,)
            )
            
            disk_cleared = cursor.rowcount
            conn.commit()
            conn.close()
        
        return memory_cleared + disk_cleared
    
    def clear_all(self) -> int:
        """
        Clear all items from the cache.
        
        Returns:
            Number of items removed
        """
        memory_cleared = len(self.memory_cache)
        self.memory_cache = {}
        
        disk_cleared = 0
        if self.disk_cache_enabled:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("DELETE FROM snomed_cache")
            disk_cleared = cursor.rowcount
            
            conn.commit()
            conn.close()
        
        return memory_cleared + disk_cleared


class UMLSAuthenticator:
    """Handles authentication with the UMLS API."""
    
    def __init__(self, api_key: str, cache_dir: Optional[str] = None):
        """
        Initialize the UMLS authenticator.
        
        Args:
            api_key: UMLS API key
            cache_dir: Directory to cache authentication tokens
        """
        self.api_key = api_key
        self.token = None
        self.token_expiry = None
        
        if cache_dir:
            self.token_cache_path = Path(cache_dir) / "umls_token.json"
            self._load_cached_token()
        else:
            self.token_cache_path = None
    
    def _load_cached_token(self) -> None:
        """Load authentication token from cache if available and not expired."""
        if not self.token_cache_path or not self.token_cache_path.exists():
            return
        
        try:
            with open(self.token_cache_path, 'r') as f:
                token_data = json.load(f)
                
            if 'token' in token_data and 'expiry' in token_data:
                expiry = datetime.fromisoformat(token_data['expiry'])
                
                # Add a safety margin of 5 minutes
                if expiry > datetime.now() + timedelta(minutes=5):
                    self.token = token_data['token']
                    self.token_expiry = expiry
                    logger.info("Loaded valid UMLS token from cache")
        except Exception as e:
            logger.warning(f"Failed to load cached UMLS token: {e}")
    
    def _save_token_to_cache(self) -> None:
        """Save authentication token to cache."""
        if not self.token_cache_path or not self.token or not self.token_expiry:
            return
        
        try:
            token_data = {
                'token': self.token,
                'expiry': self.token_expiry.isoformat()
            }
            
            self.token_cache_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.token_cache_path, 'w') as f:
                json.dump(token_data, f)
                
            logger.info("Saved UMLS token to cache")
        except Exception as e:
            logger.warning(f"Failed to save UMLS token to cache: {e}")
    
    def get_token(self) -> str:
        """
        Get a valid authentication token for UMLS API.
        
        Returns:
            The authentication token
        """
        # Return existing token if it's valid
        if self.token and self.token_expiry and self.token_expiry > datetime.now():
            return self.token
        
        # Otherwise, request a new token
        auth_endpoint = "https://utslogin.nlm.nih.gov/cas/v1/api-key"
        
        try:
            response = requests.post(
                auth_endpoint,
                data={"apikey": self.api_key}
            )
            response.raise_for_status()
            
            # Extract token from response
            self.token = response.text
            
            # Tokens are typically valid for 8 hours
            self.token_expiry = datetime.now() + timedelta(hours=7, minutes=50)
            
            self._save_token_to_cache()
            
            logger.info("Successfully obtained new UMLS token")
            return self.token
            
        except Exception as e:
            logger.error(f"Failed to authenticate with UMLS: {e}")
            raise RuntimeError(f"UMLS authentication failed: {e}")


class SnomedClient:
    """
    Client for interacting with SNOMED CT terminology.
    
    This class provides a unified interface to access SNOMED CT concepts and relationships
    using either the UMLS API, direct SNOMED CT API, or local files.
    """
    
    def __init__(self, access_mode: str = "umls", 
                 api_key: Optional[str] = None,
                 api_url: Optional[str] = None,
                 cache_dir: Optional[str] = None,
                 edition: str = "US",
                 version: Optional[str] = None,
                 local_data_path: Optional[str] = None):
        """
        Initialize the SNOMED CT client.
        
        Args:
            access_mode: How to access SNOMED CT. Options: "umls", "api", "local"
            api_key: API key for authentication (required for "umls" and may be required for "api")
            api_url: Base URL for API access (required for "api" mode)
            cache_dir: Directory to store cache data
            edition: SNOMED CT edition to use (e.g., "US", "INT")
            version: Specific version of SNOMED CT to use
            local_data_path: Path to local SNOMED CT files (required for "local" mode)
        """
        self.access_mode = access_mode.lower()
        self.edition = edition
        self.version = version
        
        # Set up caching
        cache_ttl = int(os.environ.get("SNOMED_CACHE_TTL", DEFAULT_CACHE_TTL))
        self.cache = SnomedCache(cache_dir=cache_dir, ttl=cache_ttl)
        
        # Configure client based on access mode
        if self.access_mode == "umls":
            if not api_key:
                raise ValueError("API key is required for UMLS access mode")
            
            self.umls_auth = UMLSAuthenticator(api_key, cache_dir=cache_dir)
            self.api_base_url = "https://uts-ws.nlm.nih.gov/rest"
            
        elif self.access_mode == "api":
            if not api_url:
                raise ValueError("API URL is required for direct API access mode")
            
            self.api_base_url = api_url.rstrip('/')
            self.api_key = api_key  # May be None if the API doesn't require authentication
            
        elif self.access_mode == "local":
            if not local_data_path:
                raise ValueError("Local data path is required for local access mode")
            
            self.local_data_path = Path(local_data_path)
            if not self.local_data_path.exists():
                raise ValueError(f"Local data path does not exist: {local_data_path}")
            
            # Initialize local data access (will be loaded on demand)
            self._concepts_df = None
            self._relationships_df = None
            self._descriptions_df = None
            
        else:
            raise ValueError(f"Invalid access mode: {access_mode}. Must be 'umls', 'api', or 'local'.")
    
    def _make_umls_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """
        Make an authenticated request to the UMLS API.
        
        Args:
            endpoint: The API endpoint to call
            params: Optional query parameters
            
        Returns:
            The JSON response as a dictionary
        """
        if params is None:
            params = {}
        
        # Add authentication token
        token = self.umls_auth.get_token()
        params['ticket'] = token
        
        # Make the request
        url = f"{self.api_base_url}/{endpoint}"
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"UMLS API request failed: {e}")
            
            # If the error is due to an expired token, try once more with a new token
            if response.status_code == 401:
                logger.info("Token may have expired, requesting a new one")
                
                # Force token refresh
                self.umls_auth.token = None
                token = self.umls_auth.get_token()
                params['ticket'] = token
                
                response = requests.get(url, params=params)
                response.raise_for_status()
                return response.json()
            
            raise RuntimeError(f"UMLS API request failed: {e}") from e
    
    def _make_snowstorm_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """
        Make a request to the SNOMED CT Snowstorm API.
        
        Args:
            endpoint: The API endpoint to call
            params: Optional query parameters
            
        Returns:
            The JSON response as a dictionary
        """
        if params is None:
            params = {}
        
        # Add API key if provided
        headers = {}
        if self.api_key:
            headers['Authorization'] = f"Bearer {self.api_key}"
        
        # Make the request
        url = f"{self.api_base_url}/{endpoint}"
        
        try:
            response = requests.get(url, params=params, headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"SNOMED CT API request failed: {e}")
            raise RuntimeError(f"SNOMED CT API request failed: {e}") from e
    
    def _load_local_data(self) -> None:
        """Load SNOMED CT data from local files."""
        # Only load if not already loaded
        if self._concepts_df is not None:
            return
        
        logger.info("Loading SNOMED CT data from local files...")
        
        try:
            # Load concepts
            concepts_path = self.local_data_path / "sct2_Concept_Full.txt"
            if not concepts_path.exists():
                # Try alternative filename patterns
                potential_paths = list(self.local_data_path.glob("*Concept*.txt"))
                if potential_paths:
                    concepts_path = potential_paths[0]
                else:
                    raise FileNotFoundError(f"Could not find concepts file in {self.local_data_path}")
            
            logger.info(f"Loading concepts from {concepts_path}")
            self._concepts_df = pd.read_csv(
                concepts_path, 
                sep='\t', 
                dtype=str,
                usecols=['id', 'effectiveTime', 'active', 'moduleId', 'definitionStatusId']
            )
            self._concepts_df['active'] = self._concepts_df['active'].astype(bool)
            
            # Load relationships
            relationships_path = self.local_data_path / "sct2_Relationship_Full.txt"
            if not relationships_path.exists():
                potential_paths = list(self.local_data_path.glob("*Relationship*.txt"))
                if potential_paths:
                    relationships_path = potential_paths[0]
                else:
                    raise FileNotFoundError(f"Could not find relationships file in {self.local_data_path}")
            
            logger.info(f"Loading relationships from {relationships_path}")
            self._relationships_df = pd.read_csv(
                relationships_path, 
                sep='\t', 
                dtype=str,
                usecols=['id', 'effectiveTime', 'active', 'sourceId', 'destinationId', 'relationshipGroup', 'typeId']
            )
            self._relationships_df['active'] = self._relationships_df['active'].astype(bool)
            
            # Load descriptions
            descriptions_path = self.local_data_path / "sct2_Description_Full.txt"
            if not descriptions_path.exists():
                potential_paths = list(self.local_data_path.glob("*Description*.txt"))
                if potential_paths:
                    descriptions_path = potential_paths[0]
                else:
                    raise FileNotFoundError(f"Could not find descriptions file in {self.local_data_path}")
            
            logger.info(f"Loading descriptions from {descriptions_path}")
            self._descriptions_df = pd.read_csv(
                descriptions_path, 
                sep='\t', 
                dtype=str,
                usecols=['id', 'effectiveTime', 'active', 'conceptId', 'term', 'typeId']
            )
            self._descriptions_df['active'] = self._descriptions_df['active'].astype(bool)
            
            logger.info("Successfully loaded SNOMED CT data from local files")
            
        except Exception as e:
            logger.error(f"Failed to load local SNOMED CT data: {e}")
            raise RuntimeError(f"Failed to load local SNOMED CT data: {e}") from e
    
    def get_concept(self, concept_id: str) -> Dict:
        """
        Get a SNOMED CT concept by its identifier.
        
        Args:
            concept_id: The SNOMED CT concept identifier
            
        Returns:
            A dictionary containing concept details
        """
        # Check cache first
        cached_result = self.cache.get("concept", concept_id)
        if cached_result:
            return cached_result
        
        try:
            if self.access_mode == "umls":
                # UMLS access mode
                result = self._get_concept_umls(concept_id)
            elif self.access_mode == "api":
                # Direct API access mode
                result = self._get_concept_api(concept_id)
            else:
                # Local data access mode
                result = self._get_concept_local(concept_id)
            
            # Cache the result
            self.cache.set("concept", result, concept_id)
            
            return result
        except Exception as e:
            logger.error(f"Failed to get concept {concept_id}: {e}")
            raise
    
    def _get_concept_umls(self, concept_id: str) -> Dict:
        """Get a concept using UMLS API."""
        # First, try to find the concept in UMLS
        umls_version = self.version or "current"
        source = f"SNOMEDCT_{self.edition}" if self.edition else "SNOMEDCT"
        
        # Construct the API call
        endpoint = f"content/{umls_version}/source/{source}/{concept_id}"
        
        response = self._make_umls_request(endpoint)
        
        # Extract and format the concept data
        result = {
            "conceptId": concept_id,
            "active": response.get("active", True),  # UMLS might not expose this directly
            "definitionStatus": response.get("definitionStatusId", ""),
            "moduleId": response.get("moduleId", ""),
            "effectiveTime": response.get("effectiveTime", ""),
            "fsn": "",
            "preferredTerm": "",
            "terms": []
        }
        
        # Get the fully specified name and other terms
        atoms_endpoint = f"content/{umls_version}/source/{source}/{concept_id}/atoms"
        atoms_response = self._make_umls_request(atoms_endpoint)
        
        if "result" in atoms_response:
            terms_data = {}
            
            for atom in atoms_response["result"]:
                term_type = atom.get("termType")
                term = atom.get("name", "")
                
                if term:
                    if term_type == "FN":  # Fully Specified Name
                        result["fsn"] = term
                    elif term_type == "PT":  # Preferred Term
                        result["preferredTerm"] = term
                    
                    terms_data[term] = {
                        "term": term,
                        "type": term_type,
                        "lang": atom.get("language", "en")
                    }
            
            # Convert terms dictionary to list
            result["terms"] = list(terms_data.values())
        
        # Get parent and children relationships
        relations_endpoint = f"content/{umls_version}/source/{source}/{concept_id}/relations"
        relations_response = self._make_umls_request(relations_endpoint)
        
        parents = []
        children = []
        relationships = []
        
        if "result" in relations_response:
            for relation in relations_response["result"]:
                rel_type = relation.get("relationLabel")
                related_id = relation.get("relatedId")
                
                if rel_type and related_id:
                    if rel_type == "PAR":  # Parent
                        parents.append(related_id)
                    elif rel_type == "CHD":  # Child
                        children.append(related_id)
                    
                    relationships.append({
                        "type": rel_type,
                        "typeId": relation.get("relationshipGroup", ""),
                        "destinationId": related_id,
                        "active": True  # Assume active
                    })
        
        result["parents"] = parents
        result["children"] = children
        result["relationships"] = relationships
        
        return result
    
    def _get_concept_api(self, concept_id: str) -> Dict:
        """Get a concept using direct SNOMED CT API."""
        # Construct the API call for Snowstorm API
        endpoint = f"concepts/{concept_id}"
        
        params = {}
        if self.edition:
            params["edition"] = self.edition
        if self.version:
            params["version"] = self.version
        
        response = self._make_snowstorm_request(endpoint, params)
        
        # Extract and format the concept data
        result = {
            "conceptId": concept_id,
            "active": response.get("active", False),
            "definitionStatus": response.get("definitionStatusId", ""),
            "moduleId": response.get("moduleId", ""),
            "effectiveTime": response.get("effectiveTime", ""),
            "fsn": "",
            "preferredTerm": "",
            "terms": [],
            "parents": [],
            "children": [],
            "relationships": []
        }
        
        # Extract FSN and terms
        fsn = response.get("fsn", {})
        if fsn:
            result["fsn"] = fsn.get("term", "")
        
        pt = response.get("pt", {})
        if pt:
            result["preferredTerm"] = pt.get("term", "")
        
        # Extract descriptions/terms
        if "descriptions" in response:
            for desc in response["descriptions"]:
                if desc.get("active", False):
                    result["terms"].append({
                        "term": desc.get("term", ""),
                        "type": desc.get("type", ""),
                        "lang": desc.get("lang", "en")
                    })
        
        # Extract relationships
        if "relationships" in response:
            for rel in response["relationships"]:
                if rel.get("active", False):
                    type_id = rel.get("typeId", "")
                    target_id = rel.get("destinationId", "")
                    
                    # Add relationship
                    result["relationships"].append({
                        "type": rel.get("type", {}).get("fsn", {}).get("term", ""),
                        "typeId": type_id,
                        "destinationId": target_id,
                        "active": rel.get("active", False)
                    })
                    
                    # Track parents (is-a relationships)
                    if type_id == IS_A_RELATIONSHIP:
                        result["parents"].append(target_id)
        
        # Get children (requires separate API call)
        children_endpoint = f"concepts/{concept_id}/children"
        children_response = self._make_snowstorm_request(children_endpoint, params)
        
        if "items" in children_response:
            for child in children_response["items"]:
                result["children"].append(child.get("conceptId", ""))
        
        return result
    
    def _get_concept_local(self, concept_id: str) -> Dict:
        """Get a concept using local SNOMED CT data."""
        # Load data if not already loaded
        self._load_local_data()
        
        # Query the concept
        concept_row = self._concepts_df[self._concepts_df['id'] == concept_id]
        
        if concept_row.empty:
            raise ValueError(f"Concept {concept_id} not found in local data")
        
        # Extract concept data
        row = concept_row.iloc[0]
        
        result = {
            "conceptId": concept_id,
            "active": bool(row['active']),
            "definitionStatus": row['definitionStatusId'],
            "moduleId": row['moduleId'],
            "effectiveTime": row['effectiveTime'],
            "fsn": "",
            "preferredTerm": "",
            "terms": [],
            "parents": [],
            "children": [],
            "relationships": []
        }
        
        # Get descriptions
        descriptions = self._descriptions_df[
            (self._descriptions_df['conceptId'] == concept_id) & 
            (self._descriptions_df['active'] == True)
        ]
        
        for _, desc in descriptions.iterrows():
            term = desc['term']
            type_id = desc['typeId']
            
            result["terms"].append({
                "term": term,
                "type": type_id,
                "lang": "en"  # Default to English
            })
            
            # Set FSN and preferred term
            if type_id == "900000000000003001":  # FSN
                result["fsn"] = term
            elif type_id == "900000000000013009":  # Synonym
                # In absence of better data, use first synonym as preferred term
                if not result["preferredTerm"]:
                    result["preferredTerm"] = term
        
        # Get relationships
        relationships = self._relationships_df[
            (self._relationships_df['sourceId'] == concept_id) & 
            (self._relationships_df['active'] == True)
        ]
        
        for _, rel in relationships.iterrows():
            type_id = rel['typeId']
            destination_id = rel['destinationId']
            
            # Add to relationships
            result["relationships"].append({
                "type": type_id,  # We don't have the human-readable type here
                "typeId": type_id,
                "destinationId": destination_id,
                "active": True
            })
            
            # Track parents (is-a relationships)
            if type_id == IS_A_RELATIONSHIP:
                result["parents"].append(destination_id)
        
        # Get children (concepts that have this concept as parent)
        children = self._relationships_df[
            (self._relationships_df['destinationId'] == concept_id) & 
            (self._relationships_df['typeId'] == IS_A_RELATIONSHIP) &
            (self._relationships_df['active'] == True)
        ]
        
        result["children"] = children['sourceId'].tolist()
        
        return result
    
    def search(self, term: str, max_results: int = 20, active_only: bool = True) -> List[Dict]:
        """
        Search for SNOMED CT concepts.
        
        Args:
            term: The search term
            max_results: Maximum number of results to return
            active_only: Whether to return only active concepts
            
        Returns:
            A list of concept dictionaries
        """
        # Check cache first
        cached_result = self.cache.get("search", term, max_results, active_only)
        if cached_result:
            return cached_result
        
        try:
            if self.access_mode == "umls":
                # UMLS access mode
                result = self._search_umls(term, max_results, active_only)
            elif self.access_mode == "api":
                # Direct API access mode
                result = self._search_api(term, max_results, active_only)
            else:
                # Local data access mode
                result = self._search_local(term, max_results, active_only)
            
            # Cache the result
            self.cache.set("search", result, term, max_results, active_only)
            
            return result
        except Exception as e:
            logger.error(f"Failed to search for '{term}': {e}")
            raise
    
    def _search_umls(self, term: str, max_results: int, active_only: bool) -> List[Dict]:
        """Search for concepts using UMLS API."""
        umls_version = self.version or "current"
        source = f"SNOMEDCT_{self.edition}" if self.edition else "SNOMEDCT"
        
        # Construct the API call
        endpoint = f"search/{umls_version}"
        
        params = {
            "string": term,
            "sabs": source,
            "pageSize": max_results,
            "pageNumber": 1,
            "returnIdType": "code"
        }
        
        response = self._make_umls_request(endpoint, params)
        
        results = []
        
        if "result" in response and "results" in response["result"]:
            for item in response["result"]["results"]:
                concept_id = item.get("ui", "")
                
                if not concept_id:
                    continue
                
                # Check if concept is active (need extra call)
                if active_only:
                    try:
                        concept = self.get_concept(concept_id)
                        if not concept.get("active", False):
                            continue
                    except:
                        # Skip if we can't determine active status
                        continue
                
                concept_data = {
                    "conceptId": concept_id,
                    "fsn": item.get("name", ""),
                    "preferredTerm": item.get("name", ""),  # Use the same as FSN initially
                    "active": True,  # Assume active if we got here
                    "score": item.get("score", 0)
                }
                
                results.append(concept_data)
        
        return results
    
    def _search_api(self, term: str, max_results: int, active_only: bool) -> List[Dict]:
        """Search for concepts using direct SNOMED CT API."""
        # Construct the API call for Snowstorm API
        endpoint = "concepts"
        
        params = {
            "term": term,
            "limit": max_results,
            "active": "true" if active_only else "false"
        }
        
        if self.edition:
            params["edition"] = self.edition
        if self.version:
            params["version"] = self.version
        
        response = self._make_snowstorm_request(endpoint, params)
        
        results = []
        
        if "items" in response:
            for item in response["items"]:
                concept_id = item.get("conceptId", "")
                
                if not concept_id:
                    continue
                
                fsn = item.get("fsn", {}).get("term", "")
                pt = item.get("pt", {}).get("term", "")
                
                concept_data = {
                    "conceptId": concept_id,
                    "fsn": fsn,
                    "preferredTerm": pt or fsn,
                    "active": item.get("active", False),
                    "score": item.get("score", 0)
                }
                
                results.append(concept_data)
        
        return results
    
    def _search_local(self, term: str, max_results: int, active_only: bool) -> List[Dict]:
        """Search for concepts using local SNOMED CT data."""
        # Load data if not already loaded
        self._load_local_data()
        
        # Search in descriptions
        term_lower = term.lower()
        
        # Filter active concepts if requested
        filtered_concepts = self._concepts_df
        if active_only:
            filtered_concepts = filtered_concepts[filtered_concepts['active'] == True]
        
        # Get list of valid concept IDs
        valid_ids = set(filtered_concepts['id'])
        
        # Filter descriptions
        filtered_descriptions = self._descriptions_df
        if active_only:
            filtered_descriptions = filtered_descriptions[filtered_descriptions['active'] == True]
        
        # Match term in descriptions and filter for valid concepts
        matches = filtered_descriptions[
            (filtered_descriptions['term'].str.lower().str.contains(term_lower)) &
            (filtered_descriptions['conceptId'].isin(valid_ids))
        ]
        
        # Score based on exact match and term length
        scored_matches = []
        for _, desc in matches.iterrows():
            concept_id = desc['conceptId']
            match_term = desc['term']
            
            # Calculate a simple score
            # Exact matches get higher score, shorter terms get higher score
            exact_match = match_term.lower() == term_lower
            score = 100 if exact_match else 50
            score -= min(len(match_term) / 10, 30)  # Reduce score for longer terms
            
            scored_matches.append({
                "conceptId": concept_id,
                "term": match_term,
                "typeId": desc['typeId'],
                "score": score
            })
        
        # Sort by score
        scored_matches.sort(key=lambda x: x["score"], reverse=True)
        
        # Take top matches
        top_matches = scored_matches[:max_results]
        
        # Group by concept ID and take highest scoring match for each
        concept_best_matches = {}
        for match in top_matches:
            concept_id = match["conceptId"]
            
            if concept_id not in concept_best_matches or match["score"] > concept_best_matches[concept_id]["score"]:
                concept_best_matches[concept_id] = match
        
        # Format results
        results = []
        for concept_id, match in concept_best_matches.items():
            # Get concept details
            try:
                concept = self.get_concept(concept_id)
                results.append({
                    "conceptId": concept_id,
                    "fsn": concept.get("fsn", match["term"]),
                    "preferredTerm": concept.get("preferredTerm", match["term"]),
                    "active": True,  # Must be active based on our filtering
                    "score": match["score"]
                })
            except Exception as e:
                logger.warning(f"Error getting details for concept {concept_id}: {e}")
                
                # Include basic info even if we couldn't get full details
                results.append({
                    "conceptId": concept_id,
                    "fsn": match["term"],
                    "preferredTerm": match["term"],
                    "active": True,
                    "score": match["score"]
                })
        
        # Sort final results by score
        results.sort(key=lambda x: x["score"], reverse=True)
        
        return results[:max_results]
    
    def get_parents(self, concept_id: str, direct_only: bool = True) -> List[Dict]:
        """
        Get parent concepts for a given concept.
        
        Args:
            concept_id: The SNOMED CT concept identifier
            direct_only: If True, return only direct parents; if False, return all ancestors
            
        Returns:
            A list of concept dictionaries
        """
        # For direct parents, just get the concept and extract parents
        if direct_only:
            concept = self.get_concept(concept_id)
            parent_ids = concept.get("parents", [])
            
            # Get details for each parent
            parents = []
            for parent_id in parent_ids:
                try:
                    parent = self.get_concept(parent_id)
                    parents.append(parent)
                except Exception as e:
                    logger.warning(f"Error getting parent {parent_id}: {e}")
            
            return parents
        
        # For all ancestors, we need to traverse the hierarchy
        # Check cache first
        cached_result = self.cache.get("ancestors", concept_id)
        if cached_result:
            return cached_result
        
        try:
            if self.access_mode == "umls":
                # UMLS access mode
                result = self._get_ancestors_umls(concept_id)
            elif self.access_mode == "api":
                # Direct API access mode
                result = self._get_ancestors_api(concept_id)
            else:
                # Local data access mode
                result = self._get_ancestors_local(concept_id)
            
            # Cache the result
            self.cache.set("ancestors", result, concept_id)
            
            return result
        except Exception as e:
            logger.error(f"Failed to get ancestors for {concept_id}: {e}")
            raise
    
    def _get_ancestors_umls(self, concept_id: str) -> List[Dict]:
        """Get all ancestors using UMLS API."""
        umls_version = self.version or "current"
        source = f"SNOMEDCT_{self.edition}" if self.edition else "SNOMEDCT"
        
        # UMLS doesn't have a direct ancestor tree endpoint, so we'll traverse manually
        ancestors = []
        to_process = [concept_id]
        processed = set()
        
        while to_process:
            current_id = to_process.pop(0)
            
            if current_id in processed:
                continue
            
            processed.add(current_id)
            
            try:
                concept = self.get_concept(current_id)
                
                # Skip the original concept in the results
                if current_id != concept_id:
                    ancestors.append(concept)
                
                # Add parents to processing queue
                for parent_id in concept.get("parents", []):
                    if parent_id not in processed:
                        to_process.append(parent_id)
            except Exception as e:
                logger.warning(f"Error processing ancestor {current_id}: {e}")
        
        return ancestors
    
    def _get_ancestors_api(self, concept_id: str) -> List[Dict]:
        """Get all ancestors using direct SNOMED CT API."""
        # Snowstorm API has a direct endpoint for ancestors
        endpoint = f"concepts/{concept_id}/ancestors"
        
        params = {}
        if self.edition:
            params["edition"] = self.edition
        if self.version:
            params["version"] = self.version
        
        response = self._make_snowstorm_request(endpoint, params)
        
        ancestors = []
        
        if "items" in response:
            for item in response["items"]:
                concept_id = item.get("conceptId", "")
                
                if not concept_id:
                    continue
                
                try:
                    # Get full concept details
                    ancestor = self.get_concept(concept_id)
                    ancestors.append(ancestor)
                except Exception as e:
                    logger.warning(f"Error getting ancestor {concept_id}: {e}")
                    
                    # Include basic info even if we couldn't get full details
                    fsn = item.get("fsn", {}).get("term", "")
                    pt = item.get("pt", {}).get("term", "")
                    
                    ancestors.append({
                        "conceptId": concept_id,
                        "fsn": fsn,
                        "preferredTerm": pt or fsn,
                        "active": item.get("active", False)
                    })
        
        return ancestors
    
    def _get_ancestors_local(self, concept_id: str) -> List[Dict]:
        """Get all ancestors using local SNOMED CT data."""
        # Load data if not already loaded
        self._load_local_data()
        
        ancestors = []
        to_process = [concept_id]
        processed = set()
        
        while to_process:
            current_id = to_process.pop(0)
            
            if current_id in processed:
                continue
            
            processed.add(current_id)
            
            try:
                # Skip the original concept in the results
                if current_id != concept_id:
                    ancestor = self.get_concept(current_id)
                    ancestors.append(ancestor)
                
                # Get relationships where this concept is the source and type is IS_A
                parent_rels = self._relationships_df[
                    (self._relationships_df['sourceId'] == current_id) & 
                    (self._relationships_df['typeId'] == IS_A_RELATIONSHIP) &
                    (self._relationships_df['active'] == True)
                ]
                
                # Add parent IDs to processing queue
                for _, rel in parent_rels.iterrows():
                    parent_id = rel['destinationId']
                    if parent_id not in processed:
                        to_process.append(parent_id)
            except Exception as e:
                logger.warning(f"Error processing ancestor {current_id}: {e}")
        
        return ancestors
    
    def get_children(self, concept_id: str, direct_only: bool = True) -> List[Dict]:
        """
        Get child concepts for a given concept.
        
        Args:
            concept_id: The SNOMED CT concept identifier
            direct_only: If True, return only direct children; if False, return all descendants
            
        Returns:
            A list of concept dictionaries
        """
        # For direct children, just get the concept and extract children
        if direct_only:
            concept = self.get_concept(concept_id)
            child_ids = concept.get("children", [])
            
            # Get details for each child
            children = []
            for child_id in child_ids:
                try:
                    child = self.get_concept(child_id)
                    children.append(child)
                except Exception as e:
                    logger.warning(f"Error getting child {child_id}: {e}")
            
            return children
        
        # For all descendants, we need to traverse the hierarchy
        # Check cache first
        cached_result = self.cache.get("descendants", concept_id)
        if cached_result:
            return cached_result
        
        try:
            if self.access_mode == "umls":
                # UMLS access mode
                result = self._get_descendants_umls(concept_id)
            elif self.access_mode == "api":
                # Direct API access mode
                result = self._get_descendants_api(concept_id)
            else:
                # Local data access mode
                result = self._get_descendants_local(concept_id)
            
            # Cache the result
            self.cache.set("descendants", result, concept_id)
            
            return result
        except Exception as e:
            logger.error(f"Failed to get descendants for {concept_id}: {e}")
            raise
    
    def _get_descendants_umls(self, concept_id: str) -> List[Dict]:
        """Get all descendants using UMLS API."""
        umls_version = self.version or "current"
        source = f"SNOMEDCT_{self.edition}" if self.edition else "SNOMEDCT"
        
        # UMLS doesn't have a direct descendant tree endpoint, so we'll traverse manually
        descendants = []
        to_process = [concept_id]
        processed = set()
        
        while to_process:
            current_id = to_process.pop(0)
            
            if current_id in processed:
                continue
            
            processed.add(current_id)
            
            try:
                concept = self.get_concept(current_id)
                
                # Skip the original concept in the results
                if current_id != concept_id:
                    descendants.append(concept)
                
                # Add children to processing queue
                for child_id in concept.get("children", []):
                    if child_id not in processed:
                        to_process.append(child_id)
            except Exception as e:
                logger.warning(f"Error processing descendant {current_id}: {e}")
        
        return descendants
    
    def _get_descendants_api(self, concept_id: str) -> List[Dict]:
        """Get all descendants using direct SNOMED CT API."""
        # Snowstorm API has a direct endpoint for descendants
        endpoint = f"concepts/{concept_id}/descendants"
        
        params = {}
        if self.edition:
            params["edition"] = self.edition
        if self.version:
            params["version"] = self.version
        
        response = self._make_snowstorm_request(endpoint, params)
        
        descendants = []
        
        if "items" in response:
            for item in response["items"]:
                concept_id = item.get("conceptId", "")
                
                if not concept_id:
                    continue
                
                try:
                    # Get full concept details
                    descendant = self.get_concept(concept_id)
                    descendants.append(descendant)
                except Exception as e:
                    logger.warning(f"Error getting descendant {concept_id}: {e}")
                    
                    # Include basic info even if we couldn't get full details
                    fsn = item.get("fsn", {}).get("term", "")
                    pt = item.get("pt", {}).get("term", "")
                    
                    descendants.append({
                        "conceptId": concept_id,
                        "fsn": fsn,
                        "preferredTerm": pt or fsn,
                        "active": item.get("active", False)
                    })
        
        return descendants
    
    def _get_descendants_local(self, concept_id: str) -> List[Dict]:
        """Get all descendants using local SNOMED CT data."""
        # Load data if not already loaded
        self._load_local_data()
        
        descendants = []
        to_process = [concept_id]
        processed = set()
        
        while to_process:
            current_id = to_process.pop(0)
            
            if current_id in processed:
                continue
            
            processed.add(current_id)
            
            try:
                # Skip the original concept in the results
                if current_id != concept_id:
                    descendant = self.get_concept(current_id)
                    descendants.append(descendant)
                
                # Get relationships where this concept is the target and type is IS_A
                child_rels = self._relationships_df[
                    (self._relationships_df['destinationId'] == current_id) & 
                    (self._relationships_df['typeId'] == IS_A_RELATIONSHIP) &
                    (self._relationships_df['active'] == True)
                ]
                
                # Add child IDs to processing queue
                for _, rel in child_rels.iterrows():
                    child_id = rel['sourceId']
                    if child_id not in processed:
                        to_process.append(child_id)
            except Exception as e:
                logger.warning(f"Error processing descendant {current_id}: {e}")
        
        return descendants
    
    def get_relationships(self, concept_id: str, relationship_type: Optional[str] = None) -> List[Dict]:
        """
        Get relationships for a given concept.
        
        Args:
            concept_id: The SNOMED CT concept identifier
            relationship_type: Optional relationship type ID to filter by
            
        Returns:
            A list of relationship dictionaries
        """
        concept = self.get_concept(concept_id)
        relationships = concept.get("relationships", [])
        
        # Filter by relationship type if specified
        if relationship_type:
            relationships = [rel for rel in relationships if rel.get("typeId") == relationship_type]
        
        return relationships
    
    def is_a(self, concept_id: str, potential_parent_id: str) -> bool:
        """
        Check if a concept is a subtype of another concept.
        
        Args:
            concept_id: The concept to check
            potential_parent_id: The potential parent concept
            
        Returns:
            True if the concept is a subtype of the potential parent, False otherwise
        """
        # Direct check
        concept = self.get_concept(concept_id)
        if potential_parent_id in concept.get("parents", []):
            return True
        
        # Recursive check for indirect ancestry
        ancestors = self.get_parents(concept_id, direct_only=False)
        ancestor_ids = [a.get("conceptId") for a in ancestors]
        
        return potential_parent_id in ancestor_ids
    
    def get_subtypes(self, concept_id: str, include_self: bool = False) -> List[Dict]:
        """
        Get all subtypes of a concept (equivalent to descendants).
        
        Args:
            concept_id: The SNOMED CT concept identifier
            include_self: Whether to include the concept itself in the results
            
        Returns:
            A list of concept dictionaries
        """
        result = self.get_children(concept_id, direct_only=False)
        
        if include_self:
            result = [self.get_concept(concept_id)] + result
        
        return result
    
    def get_supertypes(self, concept_id: str, include_self: bool = False) -> List[Dict]:
        """
        Get all supertypes of a concept (equivalent to ancestors).
        
        Args:
            concept_id: The SNOMED CT concept identifier
            include_self: Whether to include the concept itself in the results
            
        Returns:
            A list of concept dictionaries
        """
        result = self.get_parents(concept_id, direct_only=False)
        
        if include_self:
            result = [self.get_concept(concept_id)] + result
        
        return result
    
    def evaluate_ecl(self, expression: str, max_results: int = 200) -> List[Dict]:
        """
        Evaluate an Expression Constraint Language (ECL) expression.
        
        Args:
            expression: The ECL expression
            max_results: Maximum number of results to return
            
        Returns:
            A list of concept dictionaries
        """
        # Check cache first
        cached_result = self.cache.get("ecl", expression, max_results)
        if cached_result:
            return cached_result
        
        try:
            if self.access_mode == "api":
                # Only direct API supports ECL
                result = self._evaluate_ecl_api(expression, max_results)
            else:
                # For UMLS and local modes, implement a basic ECL parser
                result = self._evaluate_ecl_basic(expression, max_results)
            
            # Cache the result
            self.cache.set("ecl", result, expression, max_results)
            
            return result
        except Exception as e:
            logger.error(f"Failed to evaluate ECL '{expression}': {e}")
            raise
    
    def _evaluate_ecl_api(self, expression: str, max_results: int) -> List[Dict]:
        """Evaluate ECL using direct SNOMED CT API."""
        # Snowstorm API has a direct endpoint for ECL
        endpoint = "concepts/ecl"
        
        params = {
            "ecl": expression,
            "limit": max_results
        }
        
        if self.edition:
            params["edition"] = self.edition
        if self.version:
            params["version"] = self.version
        
        response = self._make_snowstorm_request(endpoint, params)
        
        results = []
        
        if "items" in response:
            for item in response["items"]:
                concept_id = item.get("conceptId", "")
                
                if not concept_id:
                    continue
                
                try:
                    # Get full concept details
                    concept = self.get_concept(concept_id)
                    results.append(concept)
                except Exception as e:
                    logger.warning(f"Error getting ECL result {concept_id}: {e}")
                    
                    # Include basic info even if we couldn't get full details
                    fsn = item.get("fsn", {}).get("term", "")
                    pt = item.get("pt", {}).get("term", "")
                    
                    results.append({
                        "conceptId": concept_id,
                        "fsn": fsn,
                        "preferredTerm": pt or fsn,
                        "active": item.get("active", False)
                    })
        
        return results
    
    def _evaluate_ecl_basic(self, expression: str, max_results: int) -> List[Dict]:
        """
        Basic implementation of ECL evaluation for UMLS and local modes.
        
        Supports only simple expressions like:
        - <123456789 (descendants of)
        - >123456789 (ancestors of)
        - <<123456789 (self or descendants)
        - >>123456789 (self or ancestors)
        - 123456789 (exact match)
        - (123456789 OR 987654321) (disjunction)
        """
        expression = expression.strip()
        
        # Handle simple descendant/ancestor expressions
        if expression.startswith("<<"):
            # Self or descendants
            concept_id = expression[2:].strip()
            return self.get_subtypes(concept_id, include_self=True)[:max_results]
        
        elif expression.startswith("<"):
            # Descendants
            concept_id = expression[1:].strip()
            return self.get_subtypes(concept_id, include_self=False)[:max_results]
        
        elif expression.startswith(">>"):
            # Self or ancestors
            concept_id = expression[2:].strip()
            return self.get_supertypes(concept_id, include_self=True)[:max_results]
        
        elif expression.startswith(">"):
            # Ancestors
            concept_id = expression[1:].strip()
            return self.get_supertypes(concept_id, include_self=False)[:max_results]
        
        # Handle direct concept reference
        elif expression.isdigit():
            concept = self.get_concept(expression)
            return [concept]
        
        # Handle basic disjunction (OR)
        elif "OR" in expression.upper():
            parts = expression.upper().split("OR")
            result = []
            
            for part in parts:
                part_result = self._evaluate_ecl_basic(part.strip(), max_results)
                result.extend(part_result)
                
                if len(result) >= max_results:
                    return result[:max_results]
            
            return result
        
        # Handle parentheses
        elif expression.startswith("(") and expression.endswith(")"):
            return self._evaluate_ecl_basic(expression[1:-1].strip(), max_results)
        
        else:
            raise ValueError(f"Unsupported ECL expression: {expression}")
    
    def get_concept_attributes(self, concept_id: str, attribute_type: Optional[str] = None) -> List[Dict]:
        """
        Get attribute relationships for a concept.
        
        Args:
            concept_id: The SNOMED CT concept identifier
            attribute_type: Optional attribute type ID to filter by
            
        Returns:
            A list of attribute dictionaries
        """
        # Get all relationships
        relationships = self.get_relationships(concept_id)
        
        # Filter out IS_A relationships to get only attributes
        attributes = [rel for rel in relationships if rel.get("typeId") != IS_A_RELATIONSHIP]
        
        # Filter by attribute type if specified
        if attribute_type:
            attributes = [attr for attr in attributes if attr.get("typeId") == attribute_type]
        
        return attributes
    
    def get_all_snomed_versions(self) -> List[str]:
        """
        Get a list of available SNOMED CT versions.
        
        Returns:
            A list of version identifiers
        """
        if self.access_mode == "api":
            # Only Snowstorm API supports this
            endpoint = "versions"
            
            try:
                response = self._make_snowstorm_request(endpoint)
                
                versions = []
                if "items" in response:
                    for item in response["items"]:
                        versions.append(item.get("version"))
                
                return versions
            except Exception as e:
                logger.error(f"Failed to get SNOMED CT versions: {e}")
                return []
        else:
            # Not supported in UMLS or local modes
            logger.warning("Getting all SNOMED CT versions is not supported in this access mode")
            return []
    
    def get_snomed_editions(self) -> List[str]:
        """
        Get a list of available SNOMED CT editions.
        
        Returns:
            A list of edition identifiers
        """
        if self.access_mode == "api":
            # Only Snowstorm API supports this
            endpoint = "codesystems"
            
            try:
                response = self._make_snowstorm_request(endpoint)
                
                editions = []
                if "items" in response:
                    for item in response["items"]:
                        editions.append(item.get("shortName"))
                
                return editions
            except Exception as e:
                logger.error(f"Failed to get SNOMED CT editions: {e}")
                return []
        else:
            # Not supported in UMLS or local modes
            logger.warning("Getting SNOMED CT editions is not supported in this access mode")
            return []


class SnomedSubsumptionTester:
    """Test subsumption relationships between SNOMED CT concepts."""
    
    def __init__(self, client: SnomedClient):
        """
        Initialize the subsumption tester.
        
        Args:
            client: A SnomedClient instance
        """
        self.client = client
    
    def is_subsumed_by(self, concept_a: str, concept_b: str) -> bool:
        """
        Test if concept A is subsumed by concept B (A is a kind of B).
        
        Args:
            concept_a: The first concept ID
            concept_b: The second concept ID
            
        Returns:
            True if concept A is subsumed by concept B, False otherwise
        """
        return self.client.is_a(concept_a, concept_b)
    
    def get_subsumption_status(self, concept_a: str, concept_b: str) -> str:
        """
        Get the subsumption relationship between two concepts.
        
        Args:
            concept_a: The first concept ID
            concept_b: The second concept ID
            
        Returns:
            One of: "EQUIVALENT", "A_SUBSUMES_B", "B_SUBSUMES_A", "DISJOINT"
        """
        a_subsumes_b = self.is_subsumed_by(concept_b, concept_a)
        b_subsumes_a = self.is_subsumed_by(concept_a, concept_b)
        
        if a_subsumes_b and b_subsumes_a:
            return "EQUIVALENT"
        elif a_subsumes_b:
            return "A_SUBSUMES_B"
        elif b_subsumes_a:
            return "B_SUBSUMES_A"
        else:
            return "DISJOINT"


class SnomedExpressionConverter:
    """Convert between different SNOMED CT expression formats."""
    
    def __init__(self, client: SnomedClient):
        """
        Initialize the expression converter.
        
        Args:
            client: A SnomedClient instance
        """
        self.client = client
    
    def to_human_readable(self, expression: str) -> str:
        """
        Convert a SNOMED CT expression to human-readable form.
        
        Args:
            expression: SNOMED CT expression in standard form
            
        Returns:
            Human-readable version of the expression
        """
        # Basic implementation for simple expressions
        try:
            # Handle focus concept
            parts = expression.split(":")
            focus_part = parts[0].strip()
            
            # Extract concept IDs
            concept_id_pattern = r'(\d+)'
            concept_ids = re.findall(concept_id_pattern, focus_part)
            
            if not concept_ids:
                return expression
            
            # Get focus concept
            focus_concept = self.client.get_concept(concept_ids[0])
            result = focus_concept.get("preferredTerm", "")
            
            # Handle attributes if present
            if len(parts) > 1 and parts[1].strip():
                attributes_part = parts[1].strip()
                attribute_pattern = r'(\d+)\s*=\s*(\d+)'
                attribute_matches = re.findall(attribute_pattern, attributes_part)
                
                attribute_texts = []
                for attr_type_id, attr_value_id in attribute_matches:
                    try:
                        attr_type = self.client.get_concept(attr_type_id)
                        attr_value = self.client.get_concept(attr_value_id)
                        
                        attr_text = f"{attr_type.get('preferredTerm', attr_type_id)} = {attr_value.get('preferredTerm', attr_value_id)}"
                        attribute_texts.append(attr_text)
                    except Exception:
                        # If we can't resolve the concept, use the ID
                        attribute_texts.append(f"{attr_type_id} = {attr_value_id}")
                
                if attribute_texts:
                    result += ": " + ", ".join(attribute_texts)
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to convert expression to human readable: {e}")
            return expression
    
    def to_scg(self, expression: str) -> str:
        """
        Convert a SNOMED CT expression to SNOMED Compositional Grammar (SCG) format.
        
        Args:
            expression: SNOMED CT expression
            
        Returns:
            SCG version of the expression
        """
        # This is a placeholder for a more complex implementation
        # The actual implementation would depend on the input format
        return expression


class SnomedReferenceSetManager:
    """Manage and query SNOMED CT reference sets."""
    
    def __init__(self, client: SnomedClient):
        """
        Initialize the reference set manager.
        
        Args:
            client: A SnomedClient instance
        """
        self.client = client
    
    def get_reference_set_members(self, refset_id: str, 
                                 max_results: int = 100) -> List[Dict]:
        """
        Get members of a reference set.
        
        Args:
            refset_id: The reference set identifier
            max_results: Maximum number of results to return
            
        Returns:
            A list of member dictionaries
        """
        # Check if API access is available
        if self.client.access_mode == "api":
            return self._get_refset_members_api(refset_id, max_results)
        else:
            return self._get_refset_members_basic(refset_id, max_results)
    
    def _get_refset_members_api(self, refset_id: str, max_results: int) -> List[Dict]:
        """Get reference set members using the SNOMED CT API."""
        # Snowstorm API endpoint for reference set members
        endpoint = f"members"
        
        params = {
            "referenceSet": refset_id,
            "limit": max_results
        }
        
        if self.client.edition:
            params["edition"] = self.client.edition
        if self.client.version:
            params["version"] = self.client.version
        
        try:
            response = self.client._make_snowstorm_request(endpoint, params)
            
            members = []
            if "items" in response:
                for item in response["items"]:
                    member = {
                        "id": item.get("memberId", ""),
                        "referencedComponentId": item.get("referencedComponentId", ""),
                        "active": item.get("active", False),
                        "effectiveTime": item.get("effectiveTime", "")
                    }
                    
                    # Add any additional fields
                    for key, value in item.items():
                        if key not in member:
                            member[key] = value
                    
                    members.append(member)
            
            return members
        except Exception as e:
            logger.error(f"Failed to get reference set members: {e}")
            return []
    
    def _get_refset_members_basic(self, refset_id: str, max_results: int) -> List[Dict]:
        """Basic implementation for local file mode."""
        # Load local data if available
        if hasattr(self.client, '_load_local_data'):
            self.client._load_local_data()
        
        members = []
        
        # Check if we have a simple_refset_members DataFrame
        if hasattr(self.client, '_simple_refset_df') and self.client._simple_refset_df is not None:
            refset_members = self.client._simple_refset_df[
                (self.client._simple_refset_df['refsetId'] == refset_id) &
                (self.client._simple_refset_df['active'] == True)
            ]
            
            for _, member in refset_members.iterrows():
                if len(members) >= max_results:
                    break
                
                members.append({
                    "id": member.get('id', ''),
                    "referencedComponentId": member.get('referencedComponentId', ''),
                    "active": True,
                    "effectiveTime": member.get('effectiveTime', '')
                })
        
        return members


# Example usage
def example_usage():
    """Demonstrate the usage of the SNOMED CT client."""
    # Initialize the client with UMLS access
    client = SnomedClient(
        access_mode="umls",
        api_key="your-umls-api-key-here",
        cache_dir="./snomed_cache"
    )
    
    # Search for a concept
    search_results = client.search("myocardial infarction", max_results=5)
    print(f"Found {len(search_results)} concepts for 'myocardial infarction':")
    for result in search_results:
        print(f"  {result['conceptId']}: {result['preferredTerm']}")
    
    # Get a specific concept
    concept = client.get_concept("22298006")  # Myocardial infarction
    print(f"\nDetails for concept {concept['conceptId']}:")
    print(f"  FSN: {concept['fsn']}")
    print(f"  Preferred Term: {concept['preferredTerm']}")
    print(f"  Active: {concept['active']}")
    
    # Get parents
    parents = client.get_parents("22298006")
    print(f"\nDirect parents of Myocardial infarction:")
    for parent in parents:
        print(f"  {parent['conceptId']}: {parent['preferredTerm']}")
    
    # Get children
    children = client.get_children("22298006")
    print(f"\nDirect children of Myocardial infarction:")
    for child in children:
        print(f"  {child['conceptId']}: {child['preferredTerm']}")
    
    # Test subsumption
    subsumption = SnomedSubsumptionTester(client)
    is_subsumed = subsumption.is_subsumed_by("22298006", "27624003")  # Is MI a kind of disease?
    print(f"\nIs Myocardial infarction a kind of Disease? {is_subsumed}")


if __name__ == "__main__":
    example_usage()