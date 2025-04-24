"""
Example script for searching medical literature using the Medical Research Synthesizer API client.
This script demonstrates how to use the Medical Research Synthesizer API client to search for medical literature.
"""
import logging
from dotenv import load_dotenv
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
async def main():