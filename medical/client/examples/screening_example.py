"""
Example script for screening articles using the Medical Research Synthesizer API client.
This script demonstrates how to use the Medical Research Synthesizer API client to screen articles according to PRISMA guidelines and assess risk of bias.
"""
import logging
from dotenv import load_dotenv
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
async def main():