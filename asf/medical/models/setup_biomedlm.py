"""
BioMedLM Setup Script

This script downloads and sets up the BioMedLM model for contradiction detection.
"""

import os
import logging
import argparse
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("biomedlm-setup")

def setup_biomedlm(model_name: str = "microsoft/BioMedLM", cache_dir: str = None, force_download: bool = False):
    """
    Download and set up the BioMedLM model.
    
    Args:
        model_name: Name of the BioMedLM model to use
        cache_dir: Directory to cache the model
        force_download: Whether to force download even if the model is already cached
    
    Returns:
        Tuple of (tokenizer, model)
    """
    logger.info(f"Setting up BioMedLM model: {model_name}")
    
    # Create cache directory if it doesn't exist
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        logger.info(f"Using cache directory: {cache_dir}")
    
    # Check if model is already downloaded
    if cache_dir and not force_download:
        model_path = Path(cache_dir) / model_name.split("/")[-1]
        if model_path.exists():
            logger.info(f"Model already exists at {model_path}. Loading from cache.")
            try:
                tokenizer = AutoTokenizer.from_pretrained(str(model_path))
                model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
                logger.info("Successfully loaded model from cache.")
                return tokenizer, model
            except Exception as e:
                logger.warning(f"Failed to load model from cache: {e}. Downloading from source.")
    
    # Download model and tokenizer
    try:
        logger.info(f"Downloading tokenizer for {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        
        logger.info(f"Downloading model for {model_name}...")
        model = AutoModelForSequenceClassification.from_pretrained(model_name, cache_dir=cache_dir)
        
        logger.info("Successfully downloaded model and tokenizer.")
        
        # Save model and tokenizer to cache directory if specified
        if cache_dir:
            model_path = Path(cache_dir) / model_name.split("/")[-1]
            logger.info(f"Saving model to {model_path}...")
            tokenizer.save_pretrained(str(model_path))
            model.save_pretrained(str(model_path))
            logger.info("Successfully saved model and tokenizer to cache directory.")
        
        return tokenizer, model
    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        raise RuntimeError(f"Failed to download model: {e}")

def test_biomedlm(tokenizer, model):
    """
    Test the BioMedLM model with a simple contradiction example.
    
    Args:
        tokenizer: BioMedLM tokenizer
        model: BioMedLM model
    """
    logger.info("Testing BioMedLM model...")
    
    # Example claims
    claim1 = "Aspirin is effective for treating headaches."
    claim2 = "Aspirin has no effect on headache symptoms."
    
    # Prepare input
    inputs = tokenizer(
        claim1, claim2, return_tensors="pt", padding=True, truncation=True, max_length=512
    )
    
    # Get model prediction
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get contradiction scores
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1)
    
    # Assuming binary classification (contradiction vs. non-contradiction)
    contradiction_score = probabilities[0, 1].item()
    agreement_score = probabilities[0, 0].item()
    
    logger.info(f"Claim 1: {claim1}")
    logger.info(f"Claim 2: {claim2}")
    logger.info(f"Contradiction score: {contradiction_score:.4f}")
    logger.info(f"Agreement score: {agreement_score:.4f}")
    
    if contradiction_score > agreement_score:
        logger.info("Result: Claims are contradictory")
    else:
        logger.info("Result: Claims are not contradictory")
    
    logger.info("BioMedLM test completed.")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Set up BioMedLM model")
    parser.add_argument("--model", type=str, default="microsoft/BioMedLM", help="Model name")
    parser.add_argument("--cache-dir", type=str, default="./models", help="Cache directory")
    parser.add_argument("--force-download", action="store_true", help="Force download")
    parser.add_argument("--test", action="store_true", help="Test the model after setup")
    
    args = parser.parse_args()
    
    try:
        tokenizer, model = setup_biomedlm(args.model, args.cache_dir, args.force_download)
        
        if args.test:
            test_biomedlm(tokenizer, model)
        
        logger.info("BioMedLM setup completed successfully.")
    except Exception as e:
        logger.error(f"BioMedLM setup failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
