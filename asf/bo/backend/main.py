from fastapi import FastAPI, APIRouter
from fastapi.middleware.cors import CORSMiddleware
import os
import sys

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

# Import routers
from api.knowledge_base import router as knowledge_base_router

app = FastAPI(title="BO Medical Research Backend")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create a main router
main_router = APIRouter()

# Include routers
app.include_router(knowledge_base_router)

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "BO Medical Research Backend API",
        "version": "1.0.0",
        "endpoints": [
            {"path": "/api/knowledge-base", "description": "Knowledge Base Management"},
            {"path": "/api/knowledge-base/search", "description": "Search Medical Literature"},
            {"path": "/api/knowledge-base/search-pico", "description": "PICO Search"}
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)