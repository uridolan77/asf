"""
Migration helper for the Medical Research Synthesizer API consolidation.

This script helps with migrating to the consolidated API.
"""

import os
import re
import argparse
from typing import List, Dict, Tuple

def find_files(directory: str, extensions: List[str]) -> List[str]:
    """
    Find files with specific extensions in a directory.
    
    Args:
        directory: Directory to search
        extensions: List of file extensions to include
        
    Returns:
        List of file paths
    """
    file_paths = []
    
    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                file_paths.append(os.path.join(root, file))
    
    return file_paths

def analyze_imports(file_paths: List[str]) -> Dict[str, List[Tuple[str, int]]]:
    """
    Analyze imports in files.
    
    Args:
        file_paths: List of file paths
        
    Returns:
        Dictionary mapping import patterns to file paths and line numbers
    """
    import_patterns = {
        r"from asf\.medical\.api\.auth import": "auth.py",
        r"from asf\.medical\.api\.auth_v2 import": "auth_v2.py",
        r"from asf\.medical\.api\.auth_service import": "auth_service.py",
        r"from asf\.medical\.api\.main import": "main.py",
        r"from asf\.medical\.api\.main_v2 import": "main_v2.py",
        r"from asf\.medical\.api\.consolidated_main import": "consolidated_main.py",
    }
    
    results = {pattern: [] for pattern in import_patterns}
    
    for file_path in file_paths:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            
            for i, line in enumerate(lines):
                for pattern in import_patterns:
                    if re.search(pattern, line):
                        results[pattern].append((file_path, i + 1))
    
    return results

def suggest_replacements(import_analysis: Dict[str, List[Tuple[str, int]]]) -> Dict[str, str]:
    """
    Suggest replacements for imports.
    
    Args:
        import_analysis: Dictionary mapping import patterns to file paths and line numbers
        
    Returns:
        Dictionary mapping old imports to new imports
    """
    replacements = {
        r"from asf\.medical\.api\.auth import (.*)": r"from asf.medical.api.auth import \1",
        r"from asf\.medical\.api\.auth_v2 import (.*)": r"from asf.medical.api.auth import \1",
        r"from asf\.medical\.api\.auth_service import (.*)": r"from asf.medical.api.auth import \1",
        r"from asf\.medical\.api\.main import (.*)": r"from asf.medical.api.main_unified import \1",
        r"from asf\.medical\.api\.main_v2 import (.*)": r"from asf.medical.api.main_unified import \1",
        r"from asf\.medical\.api\.consolidated_main import (.*)": r"from asf.medical.api.main_unified import \1",
    }
    
    return replacements

def apply_replacements(file_path: str, replacements: Dict[str, str]) -> int:
    """
    Apply replacements to a file.
    
    Args:
        file_path: File path
        replacements: Dictionary mapping old patterns to new patterns
        
    Returns:
        Number of replacements made
    """
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    replacement_count = 0
    
    for old_pattern, new_pattern in replacements.items():
        new_content, count = re.subn(old_pattern, new_pattern, content)
        replacement_count += count
        content = new_content
    
    if replacement_count > 0:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
    
    return replacement_count

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Migration helper for the Medical Research Synthesizer API consolidation")
    parser.add_argument("--directory", default="asf/medical", help="Directory to search")
    parser.add_argument("--analyze", action="store_true", help="Analyze imports only")
    parser.add_argument("--apply", action="store_true", help="Apply replacements")
    
    args = parser.parse_args()
    
    # Find Python files
    file_paths = find_files(args.directory, [".py"])
    print(f"Found {len(file_paths)} Python files")
    
    # Analyze imports
    import_analysis = analyze_imports(file_paths)
    
    # Print analysis
    for pattern, occurrences in import_analysis.items():
        if occurrences:
            print(f"\nPattern: {pattern}")
            for file_path, line_number in occurrences:
                print(f"  {file_path}:{line_number}")
    
    # Suggest replacements
    replacements = suggest_replacements(import_analysis)
    
    # Apply replacements if requested
    if args.apply:
        total_replacements = 0
        
        for file_path in file_paths:
            replacement_count = apply_replacements(file_path, replacements)
            total_replacements += replacement_count
            
            if replacement_count > 0:
                print(f"Made {replacement_count} replacements in {file_path}")
        
        print(f"\nTotal replacements: {total_replacements}")
    elif not args.analyze:
        print("\nTo apply replacements, run with --apply")

if __name__ == "__main__":
    main()
