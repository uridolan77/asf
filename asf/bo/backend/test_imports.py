"""
Test script to verify import issues.
"""

import sys
import os

# Print current Python path
print("Current Python path:")
for path in sys.path:
    print(f"  - {path}")

# Print current working directory
print(f"\nCurrent working directory: {os.getcwd()}")

# Try importing without modifying sys.path
try:
    print("\nTrying to import without modifying sys.path:")
    from api.routers.llm_gateway import get_gateway_client
    print("  Success!")
except Exception as e:
    print(f"  Failed: {str(e)}")

# Try importing with a timeout
try:
    print("\nTrying to import with a timeout:")
    import threading
    import importlib.util
    import time
    
    def import_with_timeout(module_name, timeout=3):
        result = {"success": False, "error": None}
        
        def _import():
            try:
                if module_name.startswith("."):
                    # Relative import
                    exec(f"from {module_name} import get_gateway_client")
                else:
                    # Absolute import
                    module = __import__(module_name, fromlist=["get_gateway_client"])
                result["success"] = True
            except Exception as e:
                result["error"] = str(e)
        
        thread = threading.Thread(target=_import)
        thread.daemon = True
        thread.start()
        thread.join(timeout)
        
        if thread.is_alive():
            return False, f"Import of {module_name} timed out after {timeout} seconds"
        
        return result["success"], result["error"]
    
    success, error = import_with_timeout("api.routers.llm_gateway", 3)
    if success:
        print("  Success!")
    else:
        print(f"  Failed: {error}")
except Exception as e:
    print(f"  Failed: {str(e)}")

print("\nTest complete.")
