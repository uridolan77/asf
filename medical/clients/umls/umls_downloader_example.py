"""
Example script demonstrating the UMLSDownloader functionality.

This script shows how to use the UMLS downloader to list available releases
and download terminology files.

To run this example:
1. Ensure you have a valid UMLS API key
2. Set the API key in your environment variables as UMLS_API_KEY
3. Run this script
"""
import os
import asyncio
import logging
from dotenv import load_dotenv
from pathlib import Path

from .umls_downloader import (
    UMLSDownloader,
    RxNormReleaseType,
    SnomedCTReleaseType,
    UMLSReleaseType
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

async def list_releases_example(downloader: UMLSDownloader) -> None:
    """
    Demonstrate listing available releases.
    """
    # List all available RxNorm weekly updates
    logger.info("=== Listing RxNorm Weekly Updates ===")
    releases = await downloader.get_releases(
        release_type=RxNormReleaseType.WEEKLY_UPDATES,
        current_only=False
    )
    
    logger.info(f"Found {len(releases)} RxNorm Weekly Update releases")
    for idx, release in enumerate(releases[:3], 1):  # Show only first 3
        logger.info(f"{idx}. {release.get('name')}")
        logger.info(f"   Release date: {release.get('releaseDate')}")
        logger.info(f"   Download URL: {release.get('downloadURL')}")
    
    logger.info("\n")
    
    # List current SNOMED CT US Edition
    logger.info("=== Listing Current SNOMED CT US Edition ===")
    releases = await downloader.get_releases(
        release_type=SnomedCTReleaseType.US_EDITION,
        current_only=True
    )
    
    if releases:
        release = releases[0]
        logger.info(f"Current release: {release.get('name')}")
        logger.info(f"Release date: {release.get('releaseDate')}")
        logger.info(f"Download URL: {release.get('downloadURL')}")
    else:
        logger.info("No current release found")
    
    logger.info("\n")
    
    # List all available UMLS releases
    logger.info("=== Listing UMLS Full Releases ===")
    releases = await downloader.get_releases(
        release_type=UMLSReleaseType.FULL_RELEASE,
        current_only=False
    )
    
    logger.info(f"Found {len(releases)} UMLS Full Release releases")
    for idx, release in enumerate(releases[:3], 1):  # Show only first 3
        logger.info(f"{idx}. {release.get('name')}")
        logger.info(f"   Release date: {release.get('releaseDate')}")
        logger.info(f"   Download URL: {release.get('downloadURL')}")
    
    logger.info("\n")

async def download_example(downloader: UMLSDownloader) -> None:
    """
    Demonstrate downloading releases.
    
    Note: This function is commented out by default since it could
    download large files. Uncomment sections as needed.
    """
    # Create a temporary directory for downloads
    download_dir = Path("./umls_downloads")
    download_dir.mkdir(exist_ok=True)
    
    logger.info(f"Downloads will be saved to: {download_dir.absolute()}")
    
    # Example: Download latest RxNorm weekly update
    # Note: Commented out to prevent inadvertent downloads
    """
    logger.info("=== Downloading Latest RxNorm Weekly Update ===")
    output_path = await downloader.download_latest(
        release_type=RxNormReleaseType.WEEKLY_UPDATES,
        output_path=str(download_dir / "rxnorm_weekly_latest.zip")
    )
    
    if output_path:
        logger.info(f"Downloaded to: {output_path}")
    else:
        logger.error("Download failed")
    """
    
    # Example: Download latest SNOMED CT CORE Problem List
    # Note: Commented out to prevent inadvertent downloads
    """
    logger.info("=== Downloading Latest SNOMED CT CORE Problem List ===")
    output_path = await downloader.download_latest(
        release_type=SnomedCTReleaseType.CORE_PROBLEM_LIST,
        output_path=str(download_dir / "snomed_ct_core_problem_list_latest.zip")
    )
    
    if output_path:
        logger.info(f"Downloaded to: {output_path}")
    else:
        logger.error("Download failed")
    """
    
    logger.info("Download examples are commented out by default.")
    logger.info("Uncomment them in the script if you want to test downloads.")
    
    logger.info("\n")

async def main() -> None:
    """
    Main function to demonstrate the UMLS downloader capabilities.
    """
    # Get the API key from environment variables
    api_key = os.environ.get("UMLS_API_KEY")
    if not api_key:
        logger.error("UMLS_API_KEY environment variable not set")
        return
    
    # Create the UMLS downloader
    # By default, files will be downloaded to the current directory
    downloader = UMLSDownloader(api_key=api_key)
    
    try:
        # Run the examples
        await list_releases_example(downloader)
        await download_example(downloader)
        
        logger.info("All examples completed successfully")
    
    except Exception as e:
        logger.error(f"Error running examples: {e}")
    
    finally:
        # Make sure to close the client when done
        await downloader.close()

if __name__ == "__main__":
    asyncio.run(main())