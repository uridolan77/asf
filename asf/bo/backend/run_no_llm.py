"""
Simple script to run the backend server without the LLM Gateway module.
"""
import os
import sys
import uvicorn

# Set environment variables to disable all observability components
os.environ["DISABLE_MCP_WEBSOCKET_TASKS"] = "1"
os.environ["DISABLE_PROMETHEUS"] = "1"
os.environ["DISABLE_TRACING"] = "1"
os.environ["DISABLE_METRICS"] = "1"
os.environ["DISABLE_OTLP"] = "1"
os.environ["DISABLE_OBSERVABILITY"] = "1"
os.environ["LOG_LEVEL"] = "debug"

# Disable LLM Gateway
os.environ["DISABLE_LLM_GATEWAY"] = "1"

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
    print("Running backend server with LLM Gateway disabled...")
    uvicorn.run(
        "api.endpoints:app",
        host="127.0.0.1",  # Use localhost instead of 0.0.0.0
        port=8000,         # Use port 8000 as requested
        reload=True,
        log_level="debug",
        workers=1,
        lifespan="on"
    )
