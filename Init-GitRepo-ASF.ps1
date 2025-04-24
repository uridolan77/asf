# CONFIGURABLE SETTINGS
$repoName = "asf"
$repoDesc = "Autopoietic Semantic Fields Framework"
$githubUser = "uridolan77"
$licenseType = "MIT"

# ASSUMES: GitHub repo already created at https://github.com/$githubUser/$repoName

# STEP 1: Initialize Git
git init
echo "# $repoName`n$repoDesc" > README.md

# STEP 2: Create a common .gitignore (Node+Python+VSCode)
@"
# Python
__pycache__/
*.py[cod]
*.egg-info/
.env

# Node
node_modules/
dist/
build/

# VS Code
.vscode/

# General
.DS_Store
*.log
"@ > .gitignore

# STEP 3: Optional license
Invoke-WebRequest "https://choosealicense.com/licenses/$licenseType/" `
    -OutFile LICENSE.md -UseBasicParsing

# STEP 4: Git basics
git add .
git commit -m "Initial clean project setup"

# STEP 5: Add remote + push
git remote add origin "https://github.com/$githubUser/$repoName.git"
git branch -M main
git push -u origin main
