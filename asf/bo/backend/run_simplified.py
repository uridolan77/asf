"""
Simplified run script that only mocks the specific problematic observability modules.
This version focuses on disabling just the components we know are causing hanging issues.
"""
import os
import sys
import importlib.util
import types

# Set environment variables to disable problematic components
os.environ["DISABLE_MCP_WEBSOCKET_TASKS"] = "1"
os.environ["DISABLE_PROMETHEUS"] = "1"
os.environ["DISABLE_TRACING"] = "1"
os.environ["DISABLE_METRICS"] = "1"
os.environ["DISABLE_OTLP"] = "1"
os.environ["DISABLE_LLM_GATEWAY"] = "1"
os.environ["DISABLE_OBSERVABILITY"] = "1"
os.environ["LOG_LEVEL"] = "debug"

# Add the project root directory to sys.path to import the asf module
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"Added {project_root} to Python path")

# Add parent directory of the project root
parent_dir = os.path.dirname(project_root)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
    print(f"Added {parent_dir} to Python path")

# Add bo directory to Python path for absolute imports
bo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if bo_dir not in sys.path:
    sys.path.insert(0, bo_dir)
    print(f"Added {bo_dir} to Python path")

# ========== SELECTIVE MONKEY PATCHING ==========

# Create fake modules for problematic imports
def create_fake_module(name):
    """Create a fake module that does nothing."""
    module = types.ModuleType(name)
    sys.modules[name] = module
    print(f"Created fake module: {name}")
    return module

# Fake module creator for nested modules
def ensure_fake_module_path(full_module_path):
    """Ensure all parent modules exist for a given module path."""
    parts = full_module_path.split('.')
    current_path = ""
    
    for part in parts:
        if current_path:
            current_path = f"{current_path}.{part}"
        else:
            current_path = part
            
        if current_path not in sys.modules:
            create_fake_module(current_path)

# Only mock the specific observability modules that cause hanging
problematic_modules = [
    # Tracing related
    "mcp_observability",
    "mcp_observability.tracing",
    
    # Prometheus/metrics related
    "prometheus_client",
    "prometheus_client.core",
    "prometheus_client.exposition",
    "asf.medical.llm_gateway.observability.prometheus",
    
    # LLM Gateway resilience
    "llm_gateway",
    "llm_gateway.resilience", 
    "llm_gateway.resilience.metrics",
    "llm_gateway.resilience.tracing"
]

# Create fake modules for these specific problematic modules
for module_name in problematic_modules:
    ensure_fake_module_path(module_name)

# Add dummy implementations for Prometheus metrics
if "prometheus_client" in sys.modules:
    prometheus = sys.modules["prometheus_client"]
    
    class DummyMetric:
        def __init__(self, *args, **kwargs):
            pass
        def inc(self, *args, **kwargs):
            pass
        def dec(self, *args, **kwargs):
            pass
        def set(self, *args, **kwargs):
            pass
        def observe(self, *args, **kwargs):
            pass
        def time(self, *args, **kwargs):
            class DummyTimer:
                def __enter__(self):
                    pass
                def __exit__(self, exc_type, exc_val, exc_tb):
                    pass
            return DummyTimer()
        def labels(self, *args, **kwargs):
            return self
        
    prometheus.Counter = DummyMetric
    prometheus.Gauge = DummyMetric
    prometheus.Summary = DummyMetric
    prometheus.Histogram = DummyMetric
    prometheus.REGISTRY = DummyMetric()
    prometheus.start_http_server = lambda *args, **kwargs: None
    print("Added dummy Prometheus metrics classes")

print("Selective monkey patching complete")

# ========== INITIALIZE DATABASE ==========

print('Initializing database...')
try:
    # Database imports and setup
    from sqlalchemy import text
    from sqlalchemy.exc import OperationalError
    from config.config import engine
    from models.user import Base, Role
    
    try:
        # Import all models to ensure they're registered with Base.metadata
        from models import User, Role, Provider, ProviderModel, Configuration, UserSetting, AuditLog

        # Check if tables exist before creating them
        from sqlalchemy import inspect
        inspector = inspect(engine)
        existing_tables = inspector.get_table_names()

        # Only create tables that don't exist yet
        tables_to_create = []
        for table in Base.metadata.sorted_tables:
            if table.name not in existing_tables:
                tables_to_create.append(table)

        if tables_to_create:
            print(f"Creating tables: {', '.join(t.name for t in tables_to_create)}")
            # Create only the tables that don't exist yet
            Base.metadata.create_all(bind=engine, tables=tables_to_create)
        else:
            print("All tables already exist")

        # Check if roles exist, if not create default roles
        from sqlalchemy.orm import Session
        with Session(engine) as session:
            roles = session.query(Role).all()
            if not roles:
                print("Creating default roles...")
                roles = [
                    Role(name="User", description="Regular user"),
                    Role(name="Admin", description="Administrator")
                ]
                session.add_all(roles)
                session.commit()

        print('Database initialized successfully')
    except OperationalError as e:
        print(f'Database error: {str(e)}')
        print('Make sure MySQL is running and the database exists')
        print('You may need to create the database manually: CREATE DATABASE bo_admin;')
        exit(1)
    except Exception as e:
        print(f'Error initializing database: {str(e)}')
        exit(1)
except ImportError as e:
    print(f"Could not initialize database: {str(e)}")
    print("Continuing without database support...")

# ========== RUN SERVER ==========

# Try to create a simplified app using FastAPI directly that loads the routes from the API
# but avoids some of the problematic components
from fastapi import FastAPI, APIRouter, Depends, HTTPException, Request
from fastapi.security import OAuth2PasswordBearer
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Create a FastAPI app
app = FastAPI(title="Simplified Backend Server")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add a basic health check endpoint
@app.get("/")
async def root():
    return {
        "message": "Simplified backend server is running",
        "status": "online"
    }

@app.get("/ping")
async def ping():
    return {"ping": "pong"}

# Try to import and include the real API routers
try:
    print("Attempting to load API routers...")
    # Import the main routers from the API
    from api.routers.auth import router as auth_router
    app.include_router(auth_router, prefix="/api/auth", tags=["authentication"])
    print("Loaded auth_router")
    
    try:
        from api.routers.users import router as users_router
        app.include_router(users_router, prefix="/api/users", tags=["users"])
        print("Loaded users_router")
    except Exception as e:
        print(f"Could not load users_router: {e}")
    
    try:
        from api.routers.providers import router as providers_router
        app.include_router(providers_router, prefix="/api/providers", tags=["providers"])
        print("Loaded providers_router")
    except Exception as e:
        print(f"Could not load providers_router: {e}")
        
    try:
        from api.routers.models import router as models_router
        app.include_router(models_router, prefix="/api/models", tags=["models"])
        print("Loaded models_router")
    except Exception as e:
        print(f"Could not load models_router: {e}")
        
    try:
        from api.routers.configurations import router as configurations_router
        app.include_router(configurations_router, prefix="/api/configurations", tags=["configurations"])
        print("Loaded configurations_router")
    except Exception as e:
        print(f"Could not load configurations_router: {e}")
    
    # Skip loading the enhanced_medical_contradiction_router which has many dependencies
    print("Skipping enhanced_medical_contradiction_router due to dependencies")
    
    print("API routers loaded successfully")
except Exception as e:
    print(f"Error loading API routers: {e}")
    import traceback
    traceback.print_exc()
    print("Continuing with basic endpoints only")

if __name__ == '__main__':
    print("Starting simplified backend server...")
    
    try:
        uvicorn.run(
            app, 
            host="127.0.0.1",
            port=9000,
            log_level="debug"
        )
    except Exception as e:
        print(f"Error running uvicorn: {e}")
        import traceback
        traceback.print_exc()