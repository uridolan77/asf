# Entry point for backend server
import os
import sys
import site

# Add the project root directory to sys.path to import the asf module
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Add to sys.path in multiple ways to ensure it's found
if project_root not in sys.path:
    sys.path.insert(0, project_root)  # Insert at the beginning of sys.path
    print(f"Added {project_root} to Python path")

# Also add the parent directory of the project root
parent_dir = os.path.dirname(project_root)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
    print(f"Added {parent_dir} to Python path")

# Add bo directory to Python path for absolute imports
bo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if bo_dir not in sys.path:
    sys.path.insert(0, bo_dir)
    print(f"Added {bo_dir} to Python path")

# Create __init__.py files if needed to ensure directories are treated as packages
def ensure_package_structure(start_dir):
    for root, dirs, files in os.walk(start_dir):
        # Skip directories that start with . (hidden directories)
        # and directories that are likely not meant to be Python packages
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ('__pycache__', 'node_modules', 'venv', '.git')]

        # Create __init__.py if not exists
        init_file = os.path.join(root, '__init__.py')
        if not os.path.exists(init_file):
            try:
                with open(init_file, 'w') as f:
                    f.write("# Auto-generated __init__.py file\n")
                print(f"Created {init_file}")
            except Exception as e:
                print(f"Could not create {init_file}: {e}")

# Make sure all directories in the project are treated as packages
ensure_package_structure(project_root)

if __name__ == '__main__':
    import uvicorn
    
    print('Running backend server...')
    uvicorn.run("api.endpoints:app", host="0.0.0.0", port=8000, reload=True)
