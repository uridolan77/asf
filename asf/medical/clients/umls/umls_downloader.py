"""
UMLS downloader for the Medical Research Synthesizer.

This module provides functionality for downloading UMLS terminology files,
including RxNorm, SNOMED CT, and full UMLS releases.
"""
import os
import json
import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import httpx

logger = logging.getLogger(__name__)

class UMLSDownloader:
    """
    Client for downloading UMLS terminology files.
    
    This client provides methods for listing available releases and 
    downloading RxNorm, SNOMED CT, and UMLS terminology files.
    """
    def __init__(
        self,
        api_key: str,
        releases_base_url: str = "https://uts-ws.nlm.nih.gov/releases",
        download_base_url: str = "https://uts-ws.nlm.nih.gov/download",
        download_dir: Optional[str] = None
    ):
        """
        Initialize a new UMLS downloader.
        
        Args:
            api_key: The UMLS API key
            releases_base_url: Base URL for the releases API
            download_base_url: Base URL for the download API
            download_dir: Directory to save downloaded files (default: current directory)
        """
        self.api_key = api_key
        self.releases_base_url = releases_base_url
        self.download_base_url = download_base_url
        self.download_dir = download_dir or os.getcwd()
        self.client = httpx.AsyncClient(timeout=120.0)  # Longer timeout for downloads
    
    async def close(self):
        """
        Close the HTTP client.
        
        Should be called when the client is no longer needed.
        """
        await self.client.aclose()
    
    async def get_releases(
        self, 
        release_type: Optional[str] = None,
        current_only: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Get available releases.
        
        Args:
            release_type: Type of release to get (e.g., "rxnorm-weekly-updates")
            current_only: Whether to get only the current release
            
        Returns:
            List of releases
            
        Raises:
            httpx.HTTPError: If the request fails
        """
        params = {}
        
        if release_type:
            params['releaseType'] = release_type
        
        if current_only:
            params['current'] = 'true'
        
        try:
            response = await self.client.get(self.releases_base_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            return data.get('result', [])
        except httpx.HTTPError as e:
            logger.error(f"Failed to get releases: {str(e)}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode releases response: {str(e)}")
            return []
    
    async def download_release(
        self, 
        download_url: str, 
        output_path: Optional[str] = None,
        chunk_size: int = 1024 * 1024  # 1 MB chunks
    ) -> str:
        """
        Download a release file.
        
        Args:
            download_url: URL of the file to download
            output_path: Path to save the file to (default: derived from URL)
            chunk_size: Chunk size for streaming download
            
        Returns:
            Path to the downloaded file
            
        Raises:
            httpx.HTTPError: If the request fails
            IOError: If there's an error writing to the file
        """
        # Add API key to download URL
        params = {
            'url': download_url,
            'apiKey': self.api_key
        }
        
        # Set up output path
        if not output_path:
            # Extract filename from URL
            filename = os.path.basename(download_url)
            output_path = os.path.join(self.download_dir, filename)
        
        # Ensure download directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        try:
            # Stream the download
            async with self.client.stream('GET', self.download_base_url, params=params) as response:
                response.raise_for_status()
                
                file_size = int(response.headers.get('content-length', 0))
                downloaded = 0
                
                logger.info(f"Downloading file: {output_path} ({file_size / (1024*1024):.2f} MB)")
                
                # Create the file and stream the content
                with open(output_path, 'wb') as f:
                    async for chunk in response.aiter_bytes(chunk_size):
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        # Log progress
                        progress = (downloaded / file_size) * 100 if file_size > 0 else 0
                        logger.info(f"Download progress: {progress:.1f}% ({downloaded / (1024*1024):.2f} MB)")
                
                logger.info(f"Download complete: {output_path}")
                return output_path
        except httpx.HTTPError as e:
            logger.error(f"Failed to download file: {str(e)}")
            raise
        except IOError as e:
            logger.error(f"Failed to write to file: {str(e)}")
            raise
    
    async def get_latest_release_url(
        self, 
        release_type: str
    ) -> Optional[str]:
        """
        Get the URL for the latest release of a specific type.
        
        Args:
            release_type: Type of release to get URL for
            
        Returns:
            URL for the latest release, or None if not found
        """
        releases = await self.get_releases(release_type=release_type, current_only=True)
        
        if not releases:
            return None
        
        # Get the first (most recent) release
        latest_release = releases[0]
        return latest_release.get('downloadURL')
    
    async def download_latest(
        self, 
        release_type: str, 
        output_path: Optional[str] = None
    ) -> Optional[str]:
        """
        Download the latest release of a specific type.
        
        Args:
            release_type: Type of release to download (e.g., "rxnorm-weekly-updates")
            output_path: Path to save the file to (default: derived from URL)
            
        Returns:
            Path to the downloaded file, or None if download failed
        """
        url = await self.get_latest_release_url(release_type)
        if not url:
            logger.error(f"No releases found for type: {release_type}")
            return None
        
        try:
            return await self.download_release(url, output_path)
        except (httpx.HTTPError, IOError) as e:
            logger.error(f"Failed to download latest {release_type}: {str(e)}")
            return None


# RxNorm release types
class RxNormReleaseType:
    FULL_MONTHLY = "rxnorm-full-monthly-release"
    WEEKLY_UPDATES = "rxnorm-weekly-updates"
    PRESCRIBABLE_MONTHLY = "rxnorm-prescribable-content-monthly-release"
    PRESCRIBABLE_WEEKLY = "rxnorm-prescribable-content-weekly-updates"
    RXNAV_BOX = "rxnav-in-a-box"


# SNOMED CT release types
class SnomedCTReleaseType:
    US_EDITION = "snomed-ct-us-edition"
    US_TRANSITIVE_CLOSURE = "snomed-ct-us-edition-transitive-closure-resources"
    INTERNATIONAL = "snomed-ct-international-edition"
    CORE_PROBLEM_LIST = "snomed-ct-core-problem-list-subset"
    ICD10_CM_MAPPING = "snomed-ct-to-icd-10-cm-mapping-resources"
    SPANISH = "snomed-ct-spanish-edition"


# UMLS release types
class UMLSReleaseType:
    METATHESAURUS_FULL = "umls-metathesaurus-full-subset"
    METATHESAURUS_MRCONSO = "umls-metathesaurus-mrconso-file"
    FULL_RELEASE = "umls-full-release"