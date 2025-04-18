from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
import sys
import logging

# Create router
logger = logging.getLogger(__name__)

# API URLs
MEDICAL_API_URL = os.getenv("MEDICAL_API_URL", "http://localhost:8000")

# Add the project root directory to sys.path to import the asf module
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import routers
from api.routers.medical_kb import router as medical_kb_router
from api.routers.medical_search import router as medical_search_router
from api.routers.medical_contradiction import router as medical_contradiction_router
from api.routers.medical_terminology import router as medical_terminology_router
from api.routers.enhanced_medical_contradiction import router as enhanced_medical_contradiction_router
from api.routers.medical_clinical_data import router as medical_clinical_data_router
from api.routers.llm_gateway import router as llm_gateway_router
from api.routers.document_processing import router as document_processing_router
from api.routers.dspy import router as dspy_router
from api.routers.llm.mcp import router as mcp_router
from api.routers.llm.main import router as llm_router
# Commented out due to missing module
# from api.routers.websockets import router as websocket_router

# Import config routers
from api.routers.config.provider_router import router as provider_router
from api.routers.config.configuration_router import router as configuration_router
from api.routers.config.user_provider_router import router as user_provider_router

# Import clients router
from api.clients import router as clients_router

# Import new modular routers
# Commented out due to missing modules
# from api.auth import router as auth_router
# from api.dashboard import router as dashboard_router
# from api.medical_search import router as med_search_router
# from api.knowledge_base import router as kb_router
# from api.medical_analysis import router as analysis_router
# from api.medical_clients import router as med_clients_router
# from api.export import router as export_router

app = FastAPI()

# Configure CORS - Allow specific origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://localhost:5174",
                  "http://localhost:57104", "http://localhost:57054",
                  "http://10.100.102.28:57104", "http://10.100.102.28:57054"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Include original routers
app.include_router(medical_kb_router)
app.include_router(medical_search_router)
app.include_router(medical_contradiction_router)
app.include_router(medical_terminology_router)
app.include_router(enhanced_medical_contradiction_router)
app.include_router(medical_clinical_data_router)
app.include_router(clients_router)
app.include_router(llm_gateway_router)
app.include_router(document_processing_router)
app.include_router(dspy_router)
app.include_router(llm_router)
app.include_router(mcp_router, prefix="/api/llm")
# Commented out due to missing module
# app.include_router(websocket_router)

# Include config routers
app.include_router(provider_router)
app.include_router(configuration_router)
app.include_router(user_provider_router)

# Include new modular routers
# Commented out due to missing modules
# app.include_router(auth_router)
# app.include_router(dashboard_router)
# app.include_router(med_search_router)
# app.include_router(kb_router)
# app.include_router(analysis_router)
# app.include_router(med_clients_router)
# app.include_router(export_router)

