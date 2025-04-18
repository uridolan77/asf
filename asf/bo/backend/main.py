import os
import sys

# Add the project root directory to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)  # Insert at the beginning of sys.path
    print(f"Added {project_root} to Python path")

# Also add the parent directory of the project root
parent_dir = os.path.dirname(project_root)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
    print(f"Added {parent_dir} to Python path")

# Import the app from app.py
from api.app import app

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "BO Medical Research Backend API",
        "version": "1.0.0",
        "endpoints": [
            {"path": "/api/knowledge-base", "description": "Knowledge Base Management"},
            {"path": "/api/knowledge-base/search", "description": "Search Medical Literature"},
            {"path": "/api/knowledge-base/search-pico", "description": "PICO Search"},
            {"path": "/api/medical/analysis", "description": "Medical Literature Analysis"},
            {"path": "/api/medical/export", "description": "Export Search and Analysis Results"},
            {"path": "/api/medical/ml", "description": "Machine Learning Services"},
            {"path": "/api/medical/clients", "description": "Medical Clients Management"},
            {"path": "/api/llm", "description": "LLM Services"},
            {"path": "/api/llm/grafana", "description": "Grafana Dashboard Integration"},
            {"path": "/api/dspy", "description": "DSPy Module Management"}
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)