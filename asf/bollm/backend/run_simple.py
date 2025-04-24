"""
Simple script to run the backend server.
"""
import os
import sys
import uvicorn

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)

# Add the asf directory to the Python path
asf_dir = os.path.join(project_root, "asf")
if asf_dir not in sys.path:
    sys.path.append(asf_dir)

# Add the bo directory to the Python path
bo_dir = os.path.join(asf_dir, "bo")
if bo_dir not in sys.path:
    sys.path.append(bo_dir)

print(f"Added {project_root} to Python path")
print(f"Added {asf_dir} to Python path")
print(f"Added {bo_dir} to Python path")

# Run the server
if __name__ == "__main__":
    uvicorn.run("api.endpoints:app", host="0.0.0.0", port=8000, reload=True)
