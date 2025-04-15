"""
This script adds the project root to the Python path.
Run this script once to set up the Python path for the project.
"""
import site
import os
import sys

def add_project_root_to_path():
    # Get the project root directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
    
    # Create a .pth file in the site-packages directory
    site_packages_dir = site.getsitepackages()[0]
    pth_file_path = os.path.join(site_packages_dir, 'asf_project.pth')
    
    # Write the project root to the .pth file
    with open(pth_file_path, 'w') as f:
        f.write(project_root)
    
    print(f"Added {project_root} to Python path via {pth_file_path}")
    return True

if __name__ == "__main__":
    add_project_root_to_path()
