"""
This script installs the asf module as a development package.
Run this script once to set up the asf module for development.
"""
import os
import sys
import subprocess
import site

def install_asf_as_dev_package():
    # Get the project root directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
    
    print(f"Project root: {project_root}")
    
    # Check if the asf directory exists
    asf_dir = os.path.join(project_root, 'asf')
    if not os.path.exists(asf_dir):
        print(f"Error: asf directory not found at {asf_dir}")
        return False
    
    # Create a setup.py file in the project root
    setup_py_path = os.path.join(project_root, 'setup.py')
    with open(setup_py_path, 'w') as f:
        f.write("""
from setuptools import setup, find_packages

setup(
    name="asf",
    version="0.1",
    packages=find_packages(),
)
""")
    
    print(f"Created setup.py at {setup_py_path}")
    
    # Install the package in development mode
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-e', project_root])
        print(f"Successfully installed asf module in development mode")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing asf module: {e}")
        return False

if __name__ == "__main__":
    install_asf_as_dev_package()
