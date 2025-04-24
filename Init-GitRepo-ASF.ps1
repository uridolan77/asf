# CONFIG
$repoName    = "asf"
$repoDesc    = ""
$githubUser  = "uridolan77"
$licenseType = "mit"  # use lowercase license codes: mit, apache-2.0, gpl-3.0
$forcePush   = $true  # set to $false if you want to pull and merge instead

# STEP 1: Init
git init
echo "# $repoName`n$repoDesc" > README.md

# STEP 2: .gitignore
@"
# Python
__pycache__/
*.py[cod]
.env

# Node
node_modules/
dist/
build/

# VS Code
.vscode/

# System
.DS_Store
*.log
"@ > .gitignore

# STEP 3: LICENSE
try {
    Invoke-WebRequest "https://raw.githubusercontent.com/github/choosealicense.com/gh-pages/_licenses/$licenseType.txt" `
        -OutFile LICENSE.md -UseBasicParsing
} catch {
    Write-Host "⚠️ Failed to download LICENSE. Check licenseType: $licenseType"
}

# STEP 4: Git commit
git add .
git commit -m "Initial commit"

# STEP 5: Link to GitHub
git remote add origin "https://github.com/$githubUser/$repoName.git"
git branch -M main

if ($forcePush) {
    git push --force origin main
    Write-Host "✅ Force-pushed to remote repo: https://github.com/$githubUser/$repoName"
} else {
    git pull origin main --allow-unrelated-histories
    git push origin main
    Write-Host "✅ Pulled & pushed to remote repo"
}
