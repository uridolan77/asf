"""
This script sets up the ASF project and runs the backend server.
"""
import os
import sys
import subprocess

def setup_and_run():
    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Run check_imports.py
    print("Running check_imports.py to check if asf module can be imported...")
    subprocess.call([sys.executable, os.path.join(current_dir, 'check_imports.py')])
    
    # Run install_asf.py
    print("\nRunning install_asf.py to install asf module in development mode...")
    subprocess.call([sys.executable, os.path.join(current_dir, 'install_asf.py')])
    
    # Run asf_path.py
    print("\nRunning asf_path.py to add project root to Python path...")
    subprocess.call([sys.executable, os.path.join(current_dir, 'asf_path.py')])
    
    # Run the backend server
    print("\nRunning the backend server...")
    subprocess.call([sys.executable, os.path.join(current_dir, 'run.py')])

if __name__ == "__main__":
    setup_and_run()
