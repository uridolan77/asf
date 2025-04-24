"""
Example script demonstrating the UMLSClient functionality.

This script shows how to use the UMLS client to search for concepts,
retrieve concept details, and perform other operations.

To run this example:
1. Ensure you have a valid UMLS API key
2. Set the API key in your environment variables as UMLS_API_KEY
3. Run this script
"""
import os
import asyncio
import logging
from dotenv import load_dotenv
from typing import Dict, Any, List

from .umls_client import UMLSClient

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

async def search_example(client: UMLSClient) -> None:
    """
    Demonstrate searching for UMLS concepts.
    """
    logger.info("=== Searching for 'diabetes' ===")
    results = await client.search("diabetes", search_type="words", max_results=5)
    
    logger.info(f"Found {len(results)} results")
    for idx, result in enumerate(results, 1):
        logger.info(f"{idx}. {result.get('name')} (CUI: {result.get('ui')})")
    
    logger.info("\n")

async def concept_example(client: UMLSClient) -> None:
    """
    Demonstrate retrieving concept details.
    """
    # C0011849 is the CUI for Diabetes Mellitus
    concept_id = "C0011849"
    
    logger.info(f"=== Getting details for concept {concept_id} ===")
    concept = await client.get_concept(concept_id)
    
    if concept:
        logger.info(f"Name: {concept.get('name')}")
        logger.info(f"UI: {concept.get('ui')}")
        logger.info(f"Source: {concept.get('rootSource')}")
        logger.info(f"Definition count: {concept.get('definitionCount', 0)}")
        logger.info(f"Atom count: {concept.get('atomCount', 0)}")
        logger.info(f"Relation count: {concept.get('relationCount', 0)}")
    else:
        logger.info(f"Could not find concept {concept_id}")
    
    logger.info("\n")

async def definitions_example(client: UMLSClient) -> None:
    """
    Demonstrate retrieving concept definitions.
    """
    # C0011849 is the CUI for Diabetes Mellitus
    concept_id = "C0011849"
    
    logger.info(f"=== Getting definitions for concept {concept_id} ===")
    definitions = await client.get_concept_definitions(concept_id)
    
    logger.info(f"Found {len(definitions)} definitions")
    for idx, definition in enumerate(definitions[:3], 1):  # Just show first 3
        logger.info(f"{idx}. Source: {definition.get('rootSource')}")
        logger.info(f"   Definition: {definition.get('value')[:100]}...")
    
    logger.info("\n")

async def semantic_types_example(client: UMLSClient) -> None:
    """
    Demonstrate retrieving concept semantic types.
    """
    # C0011849 is the CUI for Diabetes Mellitus
    concept_id = "C0011849"
    
    logger.info(f"=== Getting semantic types for concept {concept_id} ===")
    sem_types = await client.get_semantic_types(concept_id)
    
    logger.info(f"Found {len(sem_types)} semantic types")
    for idx, sem_type in enumerate(sem_types, 1):
        logger.info(f"{idx}. {sem_type.get('name')} (Type ID: {sem_type.get('ui')})")
    
    logger.info("\n")

async def relations_example(client: UMLSClient) -> None:
    """
    Demonstrate retrieving concept relations.
    """
    # C0011849 is the CUI for Diabetes Mellitus
    concept_id = "C0011849"
    
    logger.info(f"=== Getting relations for concept {concept_id} ===")
    relations = await client.get_concept_relations(concept_id)
    
    logger.info(f"Found {len(relations)} relations")
    for idx, relation in enumerate(relations[:5], 1):  # Just show first 5
        logger.info(f"{idx}. {relation.get('relationLabel')}: {relation.get('relatedId')}")
    
    logger.info("\n")

async def crosswalk_example(client: UMLSClient) -> None:
    """
    Demonstrate crosswalking between vocabularies.
    """
    source_vocab = "ICD10CM"
    source_code = "E11.9"  # Type 2 diabetes without complications
    target_vocab = "SNOMEDCT_US"
    
    logger.info(f"=== Crosswalking {source_vocab} code '{source_code}' to {target_vocab} ===")
    mappings = await client.crosswalk(source_vocab, source_code, target_vocab)
    
    logger.info(f"Found {len(mappings)} mappings")
    for idx, mapping in enumerate(mappings, 1):
        logger.info(f"{idx}. {mapping.get('name')} (Code: {mapping.get('ui')})")
    
    logger.info("\n")

async def contradiction_example(client: UMLSClient) -> None:
    """
    Demonstrate finding potential contradictions between terms.
    """
    term1 = "aspirin therapy"
    term2 = "peptic ulcer"
    
    logger.info(f"=== Checking for contradictions between '{term1}' and '{term2}' ===")
    result = await client.find_contradictions(term1, term2)
    
    if result.get('contradiction_found'):
        logger.info("Potential contradiction found:")
        logger.info(f"Type: {result.get('contradiction_type')}")
        logger.info(f"Explanation: {result.get('explanation')}")
        logger.info(f"Confidence: {result.get('confidence')}")
    else:
        logger.info(f"No direct contradiction found: {result.get('explanation')}")
    
    logger.info("\n")

async def extract_concepts_example(client: UMLSClient) -> None:
    """
    Demonstrate extracting medical concepts from free text.
    """
    text = "Patients with type 2 diabetes often develop hypertension and may require ACE inhibitors."
    
    logger.info(f"=== Extracting medical concepts from text: '{text}' ===")
    concepts = await client.extract_medical_concepts(text, max_results=3)
    
    logger.info(f"Found {len(concepts)} concepts")
    for idx, concept in enumerate(concepts, 1):
        logger.info(f"{idx}. '{concept.get('matched_text')}' => {concept.get('name')} (CUI: {concept.get('ui')})")
    
    logger.info("\n")

async def main() -> None:
    """
    Main function to demonstrate the UMLS client capabilities.
    """
    # Get the API key from environment variables
    api_key = os.environ.get("UMLS_API_KEY")
    if not api_key:
        logger.error("UMLS_API_KEY environment variable not set")
        return
    
    # Create the UMLS client
    client = UMLSClient(api_key=api_key)
    
    try:
        # Run the examples
        await search_example(client)
        await concept_example(client)
        await definitions_example(client)
        await semantic_types_example(client)
        await relations_example(client)
        await crosswalk_example(client)
        await contradiction_example(client)
        await extract_concepts_example(client)
        
        logger.info("All examples completed successfully")
    
    except Exception as e:
        logger.error(f"Error running examples: {e}")
    
    finally:
        # Make sure to close the client when done
        await client.close()

if __name__ == "__main__":
    asyncio.run(main())