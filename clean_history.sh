#!/bin/bash
# Bash script to clean sensitive information from Git history

# Make sure git-filter-repo is installed
# pip install git-filter-repo

# Create a backup of the repository
echo "Creating backup of the repository..."
cp -r . ../asf_backup

# Define the file path and the pattern to replace
file_path="asf/bo/backend/config/llm/llm_gateway_config.yaml"
api_key_pattern="sk-proj-[A-Za-z0-9_]{10,}[A-Za-z0-9]{10,}"

# Create a temporary directory for the clean repository
temp_dir="../asf_clean"
if [ -d "$temp_dir" ]; then
    rm -rf "$temp_dir"
fi
mkdir -p "$temp_dir"

# Clone the repository to the temporary directory
echo "Cloning repository to temporary directory..."
git clone . "$temp_dir"
cd "$temp_dir"

# Create a replacement script for git-filter-repo
cat > clean_api_keys.py << EOF
import re

def clean_api_keys(blob, callback_metadata):
    if callback_metadata.filename == '$file_path':
        content = blob.data.decode('utf-8')
        # Replace API keys with placeholder
        cleaned_content = re.sub(r'$api_key_pattern', 'YOUR_API_KEY_HERE', content)
        blob.data = cleaned_content.encode('utf-8')
        return True
    return False
EOF

# Run git-filter-repo to clean the history
echo "Cleaning repository history..."
git filter-repo --force --filename-callback "lambda x: x" --blob-callback "clean_api_keys.py"

# Verify the cleaning
echo "Verifying cleaning..."
git log --all --grep="$api_key_pattern" --name-only

echo "Clean repository is now in $temp_dir"
echo "Original repository backup is in ../asf_backup"
echo ""
echo "To use the clean repository:"
echo "1. cd $temp_dir"
echo "2. git remote add origin https://github.com/uridolan77/asf.git"
echo "3. git push -f origin main:main"
echo ""
echo "WARNING: This will force push and overwrite the remote history!"
