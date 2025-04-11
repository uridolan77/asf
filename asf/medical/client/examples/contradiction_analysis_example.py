"""
Example script for analyzing contradictions using the Medical Research Synthesizer API client.

This script demonstrates how to use the Medical Research Synthesizer API client to analyze contradictions in medical literature.
"""

import os
import asyncio
import logging
from dotenv import load_dotenv

from asf.medical.client.api_client import MedicalResearchSynthesizerClient

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():