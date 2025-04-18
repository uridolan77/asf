#!/usr/bin/env python3
JSON Files Consolidator

This script finds all JSON files in a specified directory and 
combines them into a single output file.

import os
import json
import argparse
import glob
from pathlib import Path

def consolidate_json_files(input_dir, output_file, output_format="json", merge_key=None, text_field=None):
    """
    Consolidate all JSON files in the input directory into a single output file.
    
    Args:
        input_dir: Directory containing JSON files
        output_file: Path to the output file
        output_format: Format of the output file ("json" or "txt")
        merge_key: If specified, attempt to extract this key from each JSON file
        text_field: If specified and output_format is "txt", extract this field for text output
    """
    print(f"Scanning for JSON files in: {input_dir}")
    
    # Find all JSON files
    json_files = glob.glob(os.path.join(input_dir, "**/*.json"), recursive=True)
    
    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return
    
    print(f"Found {len(json_files)} JSON files")
    
    # Data structure to hold all content
    all_data = []
    skipped_files = []
    
    # Process each file
    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # If merge_key is specified, try to extract that key's value
            if merge_key and merge_key in data:
                all_data.append(data[merge_key])
            else:
                all_data.append(data)
                
            print(f"Processed: {file_path}")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            skipped_files.append(file_path)
    
    # Save the consolidated data
    if output_format.lower() == "json":
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_data, f, indent=2)
        print(f"Consolidated JSON saved to: {output_file}")
    
    elif output_format.lower() == "txt":
        # For text output, we need to extract the relevant text field
        if not text_field:
            print("Warning: No text field specified for text output. Using default serialization.")
            with open(output_file, 'w', encoding='utf-8') as f:
                for item in all_data:
                    f.write(str(item))
                    f.write("\n\n---\n\n")
        else:
            with open(output_file, 'w', encoding='utf-8') as f:
                for item in all_data:
                    # Try to extract the specified text field
                    if isinstance(item, dict) and text_field in item:
                        f.write(item[text_field])
                    elif isinstance(item, list):
                        # For lists, try to extract text field from each item
                        for subitem in item:
                            if isinstance(subitem, dict) and text_field in subitem:
                                f.write(subitem[text_field])
                                f.write("\n\n")
                    else:
                        # Fallback to string representation
                        f.write(str(item))
                    
                    f.write("\n\n---\n\n")
        print(f"Consolidated text saved to: {output_file}")
    
    # Print summary
    print(f"\nSummary:")
    print(f"  Processed: {len(json_files) - len(skipped_files)} files")
    print(f"  Skipped: {len(skipped_files)} files")
    
    if skipped_files:
        print("\nSkipped files:")
        for file in skipped_files:
            print(f"  - {file}")

def main():
    """
    main function.
    
    This function provides functionality for..."""
    parser = argparse.ArgumentParser(description="Consolidate JSON files into a single file")
    parser.add_argument("input_dir", help="Directory containing JSON files")
    parser.add_argument("--output", "-o", default="consolidated_output.json", 
                       help="Output file path (default: consolidated_output.json)")
    parser.add_argument("--format", "-f", choices=["json", "txt"], default="json",
                       help="Output format (json or txt, default: json)")
    parser.add_argument("--merge-key", "-k", 
                       help="If specified, extract this key from each JSON file")
    parser.add_argument("--text-field", "-t", default="content",
                       help="Field to extract for text output (default: content)")
    
    args = parser.parse_args()
    
    # Ensure input directory exists
    if not os.path.isdir(args.input_dir):
        print(f"Error: {args.input_dir} is not a valid directory")
        return
    
    # Set appropriate extension based on format
    output_file = args.output
    if not output_file.endswith(f".{args.format}"):
        base, _ = os.path.splitext(output_file)
        output_file = f"{base}.{args.format}"
    
    # Run the consolidation
    consolidate_json_files(
        args.input_dir, 
        output_file, 
        args.format, 
        args.merge_key,
        args.text_field
    )

if __name__ == "__main__":
    main()