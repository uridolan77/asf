"""
Migration Script for Enhanced Medical Research Synthesizer

This script helps migrate from the old MedicalResearchSynthesizer to the
enhanced version with improved performance, modularity, and features.

Usage:
    python migrate_to_enhanced.py --input-dir /path/to/input --output-dir /path/to/output

The script will:
1. Convert cached results to the new format
2. Update configuration files
3. Provide guidance on code changes needed
"""

import os
import sys
import argparse
import json
import shutil
import logging
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))


def convert_cache_format(old_cache_dir: str, new_cache_dir: str) -> bool:
    """
    Convert cache from old format to new format.
    
    Args:
        old_cache_dir: Old cache directory
        new_cache_dir: New cache directory
        
    Returns:
        True if conversion was successful, False otherwise
    """
    try:
        # Create new cache directory
        os.makedirs(new_cache_dir, exist_ok=True)
        
        # Check if old cache directory exists
        if not os.path.exists(old_cache_dir):
            logger.warning(f"Old cache directory {old_cache_dir} does not exist")
            return False
        
        # Check if old cache database exists
        old_db_path = os.path.join(old_cache_dir, "cache.db")
        if not os.path.exists(old_db_path):
            logger.warning(f"Old cache database {old_db_path} does not exist")
            return False
        
        # Import sqlite3 here to avoid dependency if not needed
        import sqlite3
        
        # Connect to old cache database
        old_conn = sqlite3.connect(old_db_path)
        old_cursor = old_conn.cursor()
        
        # Get all cache entries
        old_cursor.execute("SELECT hash, component, timestamp, size_bytes, filename FROM cache_entries")
        entries = old_cursor.fetchall()
        
        if not entries:
            logger.warning("No cache entries found in old cache")
            old_conn.close()
            return False
        
        logger.info(f"Found {len(entries)} cache entries in old cache")
        
        # Create new cache database
        new_db_path = os.path.join(new_cache_dir, "cache.db")
        new_conn = sqlite3.connect(new_db_path)
        new_cursor = new_conn.cursor()
        
        # Create cache table in new database
        new_cursor.execute('''
        CREATE TABLE IF NOT EXISTS cache_entries (
            hash TEXT PRIMARY KEY,
            stage TEXT,
            timestamp REAL,
            size_bytes INTEGER,
            filename TEXT
        )
        ''')
        
        # Copy cache entries
        for hash_val, component, timestamp, size_bytes, filename in entries:
            # Map old component names to new stage names
            stage = component
            if component == "full_document":
                stage = "document_processing"
            elif component == "entity_extraction":
                stage = "entity_extraction"
            elif component == "relation_extraction":
                stage = "relation_extraction"
            elif component == "summarization":
                stage = "summarization"
            
            # Copy cache file
            old_file_path = os.path.join(old_cache_dir, filename)
            new_file_path = os.path.join(new_cache_dir, filename)
            
            if os.path.exists(old_file_path):
                shutil.copy2(old_file_path, new_file_path)
                
                # Add entry to new database
                new_cursor.execute(
                    "INSERT INTO cache_entries VALUES (?, ?, ?, ?, ?)",
                    (hash_val, stage, timestamp, size_bytes, filename)
                )
        
        # Commit changes and close connections
        new_conn.commit()
        new_conn.close()
        old_conn.close()
        
        logger.info(f"Successfully converted cache from {old_cache_dir} to {new_cache_dir}")
        return True
    
    except Exception as e:
        logger.error(f"Error converting cache: {str(e)}")
        return False


def update_config_file(config_path: str, backup: bool = True) -> bool:
    """
    Update configuration file to use enhanced synthesizer.
    
    Args:
        config_path: Path to configuration file
        backup: Whether to create a backup of the original file
        
    Returns:
        True if update was successful, False otherwise
    """
    try:
        # Check if config file exists
        if not os.path.exists(config_path):
            logger.warning(f"Config file {config_path} does not exist")
            return False
        
        # Create backup if requested
        if backup:
            backup_path = f"{config_path}.bak"
            shutil.copy2(config_path, backup_path)
            logger.info(f"Created backup of config file at {backup_path}")
        
        # Load config file
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Update config to use enhanced synthesizer
        if "synthesizer" in config:
            # Add new fields for enhanced synthesizer
            config["synthesizer"]["use_enhanced"] = True
            
            # Add pipeline optimizer settings if not present
            if "pipeline_optimizer" not in config["synthesizer"]:
                config["synthesizer"]["pipeline_optimizer"] = {
                    "use_cache": True,
                    "cache_dir": config["synthesizer"].get("cache_dir", "cache"),
                    "max_workers": 4
                }
            
            # Add batch processor settings if not present
            if "batch_processor" not in config["synthesizer"]:
                config["synthesizer"]["batch_processor"] = {
                    "batch_size": 4,
                    "max_workers": 4
                }
            
            # Add model manager settings if not present
            if "model_manager" not in config["synthesizer"]:
                config["synthesizer"]["model_manager"] = {
                    "model_dir": "models",
                    "auto_retrain": False
                }
        
        # Save updated config
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Successfully updated config file {config_path}")
        return True
    
    except Exception as e:
        logger.error(f"Error updating config file: {str(e)}")
        return False


def generate_code_migration_guide() -> str:
    """
    Generate a guide for migrating code to use the enhanced synthesizer.
    
    Returns:
        Migration guide as a string
    """
    guide = """
# Migration Guide for Enhanced Medical Research Synthesizer

## Overview

The Enhanced Medical Research Synthesizer provides improved performance, modularity, and features
compared to the original implementation. This guide will help you migrate your code to use the
enhanced version.

## Import Changes

Replace:
```python
from medical.ml.document_processing import MedicalResearchSynthesizer
```

With:
```python
from medical.ml.document_processing import EnhancedMedicalResearchSynthesizer as MedicalResearchSynthesizer
```

Or use both side by side during migration:
```python
from medical.ml.document_processing import MedicalResearchSynthesizer
from medical.ml.document_processing import EnhancedMedicalResearchSynthesizer
```

## Initialization Changes

The enhanced synthesizer supports additional parameters:

```python
synthesizer = EnhancedMedicalResearchSynthesizer(
    document_processor_args={...},
    entity_extractor_args={...},
    relation_extractor_args={...},
    summarizer_args={...},
    device="cuda",
    use_cache=True,
    cache_dir="cache",
    cache_size_mb=1000,
    model_dir="models"  # New parameter for model management
)
```

## New Methods

The enhanced synthesizer provides new methods:

### Streaming Processing

```python
doc_structure, metrics = synthesizer.process_streaming(
    text_or_path,
    is_pdf=True,
    streaming_callback=lambda stage, result: print(f"Stage {stage} complete"),
    progress_callback=lambda stage, progress: print(f"Stage {stage}: {progress*100}%")
)
```

### Batch Processing

```python
metrics = synthesizer.process_batch(
    file_list=["doc1.pdf", "doc2.pdf"],
    output_dir="output",
    batch_size=4,
    all_pdfs=True,
    save_results=True,
    progress_callback=lambda progress: print(f"Progress: {progress}")
)
```

### Result Management

```python
# Save results in multiple formats
saved_files = synthesizer.save_results(
    doc_structure,
    output_dir="output",
    formats=["json", "csv", "pickle", "graphml", "txt", "html"]
)
```

### Model Updates

```python
# Update models with new labeled data
update_results = synthesizer.update_models(
    labeled_data={
        "entities": [...],
        "relations": [...],
        "summaries": [...]
    },
    learning_rate=1e-5,
    batch_size=4,
    epochs=1
)
```

## Resource Management

The enhanced synthesizer requires explicit cleanup:

```python
# Close synthesizer and release resources
synthesizer.close()
```

Best practice is to use a context manager or try-finally block:

```python
try:
    synthesizer = EnhancedMedicalResearchSynthesizer(...)
    # Use synthesizer
finally:
    synthesizer.close()
```

## API Integration

If you're using the synthesizer in an API, update your endpoints to use the enhanced version:

```python
@router.post("/process")
async def process_document(request: ProcessRequest):
    synthesizer = EnhancedMedicalResearchSynthesizer(...)
    try:
        doc_structure, metrics = synthesizer.process(
            request.text_or_path,
            is_pdf=request.is_pdf
        )
        return {
            "results": doc_structure.to_dict(),
            "metrics": metrics
        }
    finally:
        synthesizer.close()
```

## Performance Considerations

The enhanced synthesizer provides better performance through:

1. Parallel processing of multiple documents
2. Caching of intermediate results
3. Incremental processing for faster updates
4. Streaming results for better user experience

To maximize performance:

1. Use batch processing for multiple documents
2. Enable caching with sufficient cache size
3. Use streaming processing for better user experience
4. Close the synthesizer when done to release resources
"""
    
    return guide


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Migrate to Enhanced Medical Research Synthesizer")
    parser.add_argument("--old-cache-dir", default="cache", help="Old cache directory")
    parser.add_argument("--new-cache-dir", default="enhanced_cache", help="New cache directory")
    parser.add_argument("--config-path", help="Path to configuration file")
    parser.add_argument("--guide-output", default="migration_guide.md", help="Output path for migration guide")
    
    args = parser.parse_args()
    
    # Convert cache format
    if args.old_cache_dir and args.new_cache_dir:
        logger.info(f"Converting cache from {args.old_cache_dir} to {args.new_cache_dir}")
        convert_cache_format(args.old_cache_dir, args.new_cache_dir)
    
    # Update config file
    if args.config_path:
        logger.info(f"Updating config file {args.config_path}")
        update_config_file(args.config_path)
    
    # Generate migration guide
    guide = generate_code_migration_guide()
    
    # Save migration guide
    with open(args.guide_output, 'w') as f:
        f.write(guide)
    
    logger.info(f"Migration guide saved to {args.guide_output}")
    
    print("\nMigration complete!")
    print(f"1. Cache converted from {args.old_cache_dir} to {args.new_cache_dir}")
    if args.config_path:
        print(f"2. Config file updated at {args.config_path}")
    print(f"3. Migration guide saved to {args.guide_output}")
    print("\nPlease review the migration guide for code changes needed.")


if __name__ == "__main__":
    main()
