# PowerShell script to clean sensitive information from Git history

# Make sure git-filter-repo is installed
# pip install git-filter-repo

# Create a backup of the repository
Write-Host "Creating backup of the repository..."
Copy-Item -Path "." -Destination "../asf_backup" -Recurse -Force

# Define the file path and the pattern to replace
$file_path = "asf/bo/backend/config/llm/llm_gateway_config.yaml"
$api_key_pattern = "sk-proj-[A-Za-z0-9_]{10,}[A-Za-z0-9]{10,}"

# Create a temporary directory for the clean repository
$temp_dir = "../asf_clean"
if (Test-Path $temp_dir) {
    Remove-Item -Path $temp_dir -Recurse -Force
}
New-Item -Path $temp_dir -ItemType Directory

# Clone the repository to the temporary directory
Write-Host "Cloning repository to temporary directory..."
git clone . $temp_dir
Set-Location $temp_dir

# Create a replacement script for git-filter-repo
$replacement_script = @"
import re

def clean_api_keys(blob, callback_metadata):
    if callback_metadata.filename == '$file_path':
        content = blob.data.decode('utf-8')
        # Replace API keys with placeholder
        cleaned_content = re.sub(r'$api_key_pattern', 'YOUR_API_KEY_HERE', content)
        blob.data = cleaned_content.encode('utf-8')
        return True
    return False
"@

# Save the replacement script
$replacement_script | Out-File -FilePath "clean_api_keys.py" -Encoding utf8

# Run git-filter-repo to clean the history
Write-Host "Cleaning repository history..."
git filter-repo --force --filename-callback "lambda x: x" --blob-callback "clean_api_keys.py"

# Verify the cleaning
Write-Host "Verifying cleaning..."
git log --all --grep="$api_key_pattern" --name-only

Write-Host "Clean repository is now in $temp_dir"
Write-Host "Original repository backup is in ../asf_backup"
Write-Host ""
Write-Host "To use the clean repository:"
Write-Host "1. cd $temp_dir"
Write-Host "2. git remote add origin https://github.com/uridolan77/asf.git"
Write-Host "3. git push -f origin main:main"
Write-Host ""
Write-Host "WARNING: This will force push and overwrite the remote history!"
