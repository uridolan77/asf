"""
This script checks if the asf module can be imported.
Run this script to verify that the Python path is set up correctly.
"""
import os
import sys
import importlib

def check_imports():
    # Print Python path
    print("Python path:")
    for path in sys.path:
        print(f"  {path}")
    
    # Try to import the asf module
    print("\nTrying to import asf module...")
    try:
        import asf
        print(f"Successfully imported asf module from {asf.__file__}")
        return True
    except ImportError as e:
        print(f"Failed to import asf module: {e}")
        
        # Try to find the asf module
        print("\nLooking for asf module in Python path...")
        for path in sys.path:
            potential_asf_path = os.path.join(path, 'asf')
            if os.path.exists(potential_asf_path):
                print(f"Found potential asf module at {potential_asf_path}")
        
        return False

if __name__ == "__main__":
    check_imports()
