"""
Test script for the application.
"""

import sys
import os

# Add the current directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

try:
    from api.endpoints import app
    print("Successfully imported the application!")
    print(f"App: {app}")
except Exception as e:
    print(f"Error importing the application: {e}")
    import traceback
    traceback.print_exc()

def main():
    """Main function."""
    print("Testing application imports...")

if __name__ == "__main__":
    main()
