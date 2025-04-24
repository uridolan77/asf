from fastapi import FastAPI, APIRouter
from fastapi.middleware.cors import CORSMiddleware
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

# Import routers
from asf.bollm.backend.api.knowledge_base import router as knowledge_base_router
from asf.bollm.backend.api.analysis import router as analysis_router
from asf.bollm.backend.api.export import router as export_router
from asf.bollm.backend.api.ml import router as ml_router
from asf.bollm.backend.api.ml_router import router as ml_services_router
from asf.bollm.backend.api.clients import router as clients_router
from asf.bollm.backend.api.routers.llm import llm_router
from asf.bollm.backend.api.routers.dspy import router as dspy_router  # Import the DSPy router
from asf.bollm.backend.api.endpoints import router as endpoints_router
from asf.bollm.backend.api.websockets import router as websocket_router

app = FastAPI(title="BO Medical Research Backend")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://localhost:5174", "http://localhost:57104", "http://localhost:57054", "http://10.100.102.28:57104", "http://10.100.102.28:57054"],  # Allow all frontend URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create a main router
main_router = APIRouter()

# Include routers
app.include_router(knowledge_base_router)
app.include_router(analysis_router)
app.include_router(export_router)
app.include_router(ml_router)
app.include_router(ml_services_router)  # Add the new ML services router
app.include_router(clients_router)
app.include_router(llm_router)
app.include_router(dspy_router, prefix="")  # Include the DSPy router with empty prefix
app.include_router(endpoints_router, prefix="/api", tags=["Authentication"])
app.include_router(websocket_router)  # Add the WebSocket router

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
            {"path": "/api/dspy", "description": "DSPy Module Management"}
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)