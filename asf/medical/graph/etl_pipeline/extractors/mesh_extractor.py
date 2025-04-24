"""
MeSH Data Extractor

This module extracts Medical Subject Headings (MeSH) data from the NLM API
and/or MeSH RDF downloads. It provides structured access to MeSH terms,
hierarchies, and relationships.
"""

import os
import logging
import asyncio
import json
import aiohttp
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import pandas as pd
import gzip
import shutil
import tempfile

logger = logging.getLogger("biomedical_etl.extractors.mesh")

class MeshExtractor:
    """
    Extractor for Medical Subject Headings (MeSH) data.
    
    This class provides methods for extracting MeSH terms, hierarchies,
    and relationships. It supports both the NLM API and direct file downloads.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_dir: str = "./cache/mesh",
        mesh_year: Optional[str] = None,
        use_local_files: bool = True
    ):
        """
        Initialize the MeSH extractor.
        
        Args:
            api_key: API key for the NLM API (optional)
            cache_dir: Directory to store cache files
            mesh_year: Year of MeSH to extract (default: latest)
            use_local_files: Whether to use local files if available
        """
        self.api_key = api_key
        self.cache_dir = cache_dir
        self.mesh_year = mesh_year or "2023"  # Default to 2023 if not specified
        self.use_local_files = use_local_files
        
        # Base URL for NLM API
        self.base_url = "https://id.nlm.nih.gov/mesh"
        
        # URL for MeSH RDF downloads
        self.download_url = f"https://nlmpubs.nlm.nih.gov/projects/mesh/rdf/mesh{self.mesh_year}.nt.gz"
        
        # Create cache directory
        os.makedirs(self.cache_dir, exist_ok=True)
        
        logger.info(f"Initialized MeSH extractor for year {self.mesh_year}")
    
    async def extract(self) -> Dict[str, Any]:
        """
        Extract MeSH data.
        
        Returns:
            Dictionary containing MeSH terms, hierarchies, and relationships
        """
        logger.info(f"Extracting MeSH data for year {self.mesh_year}")
        
        # First check if we have cached data
        cache_path = Path(self.cache_dir) / f"mesh_{self.mesh_year}.json"
        
        if cache_path.exists():
            logger.info(f"Loading cached MeSH data from {cache_path}")
            return self._load_from_cache(cache_path)
        
        # Try to use local files if enabled
        if self.use_local_files:
            local_data = await self._extract_from_local_files()
            if local_data:
                self._save_to_cache(local_data, cache_path)
                return local_data
        
        # Fall back to API extraction
        api_data = await self._extract_from_api()
        if api_data:
            self._save_to_cache(api_data, cache_path)
            return api_data
        
        # If both methods fail, return an empty result
        logger.error("Failed to extract MeSH data from both local files and API")
        return {"terms": [], "hierarchies": [], "relationships": []}
    
    async def _extract_from_local_files(self) -> Optional[Dict[str, Any]]:
        """
        Extract MeSH data from local RDF files.
        
        Returns:
            Dictionary containing MeSH data or None if extraction fails
        """
        logger.info("Extracting MeSH data from local files")
        
        # Check if we need to download the RDF file
        local_rdf_path = Path(self.cache_dir) / f"mesh{self.mesh_year}.nt"
        
        if not local_rdf_path.exists():
            try:
                await self._download_mesh_rdf()
            except Exception as e:
                logger.error(f"Failed to download MeSH RDF: {str(e)}")
                return None
        
        if not local_rdf_path.exists():
            logger.error(f"MeSH RDF file not found at {local_rdf_path}")
            return None
        
        try:
            # Parse the RDF file
            mesh_data = self._parse_mesh_rdf(local_rdf_path)
            logger.info(f"Extracted {len(mesh_data['terms'])} MeSH terms from local files")
            return mesh_data
        except Exception as e:
            logger.error(f"Error parsing MeSH RDF: {str(e)}")
            return None
    
    async def _download_mesh_rdf(self) -> bool:
        """
        Download the MeSH RDF file.
        
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Downloading MeSH RDF for year {self.mesh_year}")
        
        download_path = Path(self.cache_dir) / f"mesh{self.mesh_year}.nt.gz"
        extracted_path = Path(self.cache_dir) / f"mesh{self.mesh_year}.nt"
        
        # Download the gzipped file
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.download_url) as response:
                    response.raise_for_status()
                    with open(download_path, 'wb') as f:
                        while True:
                            chunk = await response.content.read(8192)
                            if not chunk:
                                break
                            f.write(chunk)
            
            # Extract the gzipped file
            with gzip.open(download_path, 'rb') as f_in:
                with open(extracted_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            logger.info(f"Successfully downloaded and extracted MeSH RDF to {extracted_path}")
            return True
        except Exception as e:
            logger.error(f"Error downloading or extracting MeSH RDF: {str(e)}")
            return False
    
    def _parse_mesh_rdf(self, rdf_path: Path) -> Dict[str, Any]:
        """
        Parse the MeSH RDF file.
        
        Args:
            rdf_path: Path to the RDF file
            
        Returns:
            Dictionary containing MeSH terms, hierarchies, and relationships
        """
        logger.info(f"Parsing MeSH RDF file at {rdf_path}")
        
        # In a real implementation, we would use a proper RDF parser
        # This is a simplified implementation that extracts basic information
        terms = []
        hierarchies = []
        relationships = []
        
        # Prefixes for MeSH URIs
        descriptor_prefix = "http://id.nlm.nih.gov/mesh/D"
        qualifier_prefix = "http://id.nlm.nih.gov/mesh/Q"
        
        # Maps for storing term information
        term_map = {}
        
        # Read the file line by line to avoid loading the entire file into memory
        with open(rdf_path, 'r', encoding='utf-8') as f:
            # Process the file in chunks
            line_count = 0
            
            for line in f:
                line_count += 1
                
                if line_count % 100000 == 0:
                    logger.info(f"Processed {line_count} lines")
                
                # Parse the N-Triples line
                try:
                    parts = line.strip().split(' ', 2)
                    if len(parts) != 3:
                        continue
                    
                    subject = parts[0].strip('<>')
                    predicate = parts[1].strip('<>')
                    object_part = parts[2]
                    
                    # Extract MeSH Descriptors
                    if subject.startswith(descriptor_prefix) or subject.startswith(qualifier_prefix):
                        term_id = subject.split('/')[-1]
                        
                        # Extract preferred label
                        if predicate == "http://www.w3.org/2000/01/rdf-schema#label":
                            label = object_part.strip(' ."')
                            
                            if term_id not in term_map:
                                term_map[term_id] = {
                                    "id": term_id,
                                    "label": label,
                                    "tree_numbers": []
                                }
                            else:
                                term_map[term_id]["label"] = label
                        
                        # Extract tree numbers
                        elif predicate == "http://id.nlm.nih.gov/mesh/vocab#treeNumber":
                            tree_number = object_part.strip(' ."')
                            
                            if term_id not in term_map:
                                term_map[term_id] = {
                                    "id": term_id,
                                    "label": "",
                                    "tree_numbers": [tree_number]
                                }
                            else:
                                term_map[term_id]["tree_numbers"].append(tree_number)
                        
                        # Extract broader/narrower relationships
                        elif predicate == "http://id.nlm.nih.gov/mesh/vocab#broaderDescriptor":
                            broader_id = object_part.strip(' .<>').split('/')[-1]
                            
                            hierarchies.append({
                                "child_id": term_id,
                                "parent_id": broader_id,
                                "type": "broader"
                            })
                        
                        # Extract other relationships
                        elif predicate == "http://id.nlm.nih.gov/mesh/vocab#preferredConcept":
                            concept_id = object_part.strip(' .<>').split('/')[-1]
                            
                            relationships.append({
                                "source_id": term_id,
                                "target_id": concept_id,
                                "type": "preferredConcept"
                            })
                        
                        elif predicate == "http://id.nlm.nih.gov/mesh/vocab#pharmacologicalAction":
                            action_id = object_part.strip(' .<>').split('/')[-1]
                            
                            relationships.append({
                                "source_id": term_id,
                                "target_id": action_id,
                                "type": "pharmacologicalAction"
                            })
                
                except Exception as e:
                    logger.warning(f"Error parsing line {line_count}: {str(e)}")
        
        # Convert term map to list
        terms = list(term_map.values())
        
        logger.info(f"Parsed {len(terms)} terms, {len(hierarchies)} hierarchies, and {len(relationships)} relationships")
        
        return {
            "terms": terms,
            "hierarchies": hierarchies,
            "relationships": relationships
        }
    
    async def _extract_from_api(self) -> Dict[str, Any]:
        """
        Extract MeSH data from the NLM API.
        
        Returns:
            Dictionary containing MeSH data
        """
        logger.info("Extracting MeSH data from NLM API")
        
        # This is a placeholder for the actual API extraction
        # In a real implementation, we would use the NLM API to extract MeSH data
        
        async with aiohttp.ClientSession() as session:
            # Get top-level tree numbers
            top_level_trees = await self._get_top_level_trees(session)
            
            # Extract terms for each top-level tree
            terms = []
            hierarchies = []
            relationships = []
            
            for tree in top_level_trees[:5]:  # Limit to 5 trees for demonstration
                tree_terms, tree_hierarchies = await self._get_tree_terms(session, tree)
                terms.extend(tree_terms)
                hierarchies.extend(tree_hierarchies)
            
            return {
                "terms": terms,
                "hierarchies": hierarchies,
                "relationships": relationships
            }
    
    async def _get_top_level_trees(self, session: aiohttp.ClientSession) -> List[str]:
        """
        Get top-level MeSH tree numbers.
        
        Args:
            session: aiohttp ClientSession
            
        Returns:
            List of top-level tree numbers
        """
        url = f"{self.base_url}/trees"
        
        headers = {}
        if self.api_key:
            headers["api_key"] = self.api_key
        
        params = {
            "year": self.mesh_year,
            "format": "json"
        }
        
        async with session.get(url, headers=headers, params=params) as response:
            response.raise_for_status()
            data = await response.json()
            
            # Extract tree numbers from the response
            trees = []
            for tree in data.get("trees", []):
                tree_number = tree.get("treeNumber", "")
                if tree_number:
                    trees.append(tree_number)
            
            return trees
    
    async def _get_tree_terms(
        self,
        session: aiohttp.ClientSession,
        tree_number: str
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Get terms for a MeSH tree.
        
        Args:
            session: aiohttp ClientSession
            tree_number: MeSH tree number
            
        Returns:
            Tuple of (terms, hierarchies)
        """
        url = f"{self.base_url}/tree/{tree_number}"
        
        headers = {}
        if self.api_key:
            headers["api_key"] = self.api_key
        
        params = {
            "year": self.mesh_year,
            "format": "json"
        }
        
        async with session.get(url, headers=headers, params=params) as response:
            response.raise_for_status()
            data = await response.json()
            
            # Extract terms and hierarchies from the response
            terms = []
            hierarchies = []
            
            # Process the current term
            current_term = {
                "id": data.get("descriptor", {}).get("ui", ""),
                "label": data.get("descriptor", {}).get("name", ""),
                "tree_numbers": [tree_number]
            }
            
            if current_term["id"]:
                terms.append(current_term)
            
            # Process children
            for child in data.get("children", []):
                child_id = child.get("descriptor", {}).get("ui", "")
                child_label = child.get("descriptor", {}).get("name", "")
                child_tree = child.get("treeNumber", "")
                
                if child_id and child_tree:
                    terms.append({
                        "id": child_id,
                        "label": child_label,
                        "tree_numbers": [child_tree]
                    })
                    
                    hierarchies.append({
                        "child_id": child_id,
                        "parent_id": current_term["id"],
                        "type": "broader"
                    })
            
            return terms, hierarchies
    
    def _save_to_cache(self, data: Dict[str, Any], cache_path: Path) -> None:
        """
        Save MeSH data to cache.
        
        Args:
            data: MeSH data dictionary
            cache_path: Path to save the cache file
        """
        with open(cache_path, 'w') as f:
            json.dump(data, f)
    
    def _load_from_cache(self, cache_path: Path) -> Dict[str, Any]:
        """
        Load MeSH data from cache.
        
        Args:
            cache_path: Path to the cache file
            
        Returns:
            MeSH data dictionary
        """
        with open(cache_path, 'r') as f:
            return json.load(f)