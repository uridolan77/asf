#!/usr/bin/env python
"""
Script to clean sensitive information from Git history.

This script uses git-filter-repo to remove API keys from the repository history.
Make sure git-filter-repo is installed: pip install git-filter-repo
"""

import os
import sys
import shutil
import subprocess
import re
from pathlib import Path

# Define the file path and the pattern to replace
FILE_PATH = "asf/bo/backend/config/llm/llm_gateway_config.yaml"
API_KEY_PATTERN = r"sk-proj-[A-Za-z0-9_]{10,}[A-Za-z0-9]{10,}"

def run_command(command, cwd=None):
    """Run a shell command and return the output."""
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {command}")
        print(f"Error message: {e.stderr}")
        sys.exit(1)

def main():
    """Main function to clean the Git history."""
    # Check if git-filter-repo is installed
    try:
        run_command("git filter-repo --version")
    except:
        print("Error: git-filter-repo is not installed.")
        print("Please install it with: pip install git-filter-repo")
        sys.exit(1)

    # Get the current directory
    current_dir = os.getcwd()
    
    # Create backup directory
    backup_dir = os.path.join(os.path.dirname(current_dir), "asf_backup")
    if os.path.exists(backup_dir):
        shutil.rmtree(backup_dir)
    
    print(f"Creating backup of the repository in {backup_dir}...")
    shutil.copytree(current_dir, backup_dir, symlinks=True)
    
    # Create a temporary directory for the clean repository
    temp_dir = os.path.join(os.path.dirname(current_dir), "asf_clean")
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    
    os.makedirs(temp_dir)
    
    # Clone the repository to the temporary directory
    print(f"Cloning repository to temporary directory {temp_dir}...")
    run_command(f"git clone . {temp_dir}", cwd=current_dir)
    
    # Create a replacement script for git-filter-repo
    replacement_script = f"""
import re

def clean_api_keys(blob, callback_metadata):
    if callback_metadata.filename == '{FILE_PATH}':
        content = blob.data.decode('utf-8')
        # Replace API keys with placeholder
        cleaned_content = re.sub(r'{API_KEY_PATTERN}', 'YOUR_API_KEY_HERE', content)
        blob.data = cleaned_content.encode('utf-8')
        return True
    return False
"""
    
    script_path = os.path.join(temp_dir, "clean_api_keys.py")
    with open(script_path, "w") as f:
        f.write(replacement_script)
    
    # Run git-filter-repo to clean the history
    print("Cleaning repository history...")
    run_command(
        "git filter-repo --force --filename-callback \"lambda x: x\" --blob-callback clean_api_keys.py",
        cwd=temp_dir
    )
    
    # Verify the cleaning
    print("Verifying cleaning...")
    result = run_command(f"git log --all --grep=\"{API_KEY_PATTERN}\" --name-only", cwd=temp_dir)
    if result.strip():
        print("Warning: API keys might still be present in the repository.")
    else:
        print("No API keys found in the repository history.")
    
    print(f"\nClean repository is now in {temp_dir}")
    print(f"Original repository backup is in {backup_dir}")
    print("\nTo use the clean repository:")
    print(f"1. cd {temp_dir}")
    print("2. git remote add origin https://github.com/uridolan77/asf.git")
    print("3. git push -f origin main:main")
    print("\nWARNING: This will force push and overwrite the remote history!")

if __name__ == "__main__":
    main()
