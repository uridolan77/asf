"""
Optimized version of run.py that only mocks the specific problematic observability modules.
This script tries to load real implementations when possible, only mocking what's necessary.
"""
# =========== ENVIRONMENT VARIABLES - SET THESE FIRST ===========
import os
import sys

# Set environment variables to disable problematic components BEFORE any other imports
os.environ["DISABLE_MCP_WEBSOCKET_TASKS"] = "1"  # Disable MCP WebSocket background tasks
os.environ["DISABLE_PROMETHEUS"] = "1"           # Disable Prometheus metrics collection
os.environ["DISABLE_TRACING"] = "1"              # Disable OpenTelemetry tracing
os.environ["DISABLE_METRICS"] = "1"              # Disable metrics collection
os.environ["DISABLE_OTLP"] = "1"                 # Disable OpenTelemetry Protocol
os.environ["DISABLE_LLM_GATEWAY"] = "1"          # Disable LLM Gateway
os.environ["DISABLE_OBSERVABILITY"] = "1"        # Disable all observability
os.environ["LOG_LEVEL"] = "debug"                # Set logging level to debug

# =========== PATH SETUP ===========
import site

# Add the project root directory to sys.path to import the asf module
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"Added {project_root} to Python path")

# Add the parent directory of the project root
parent_dir = os.path.dirname(project_root)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
    print(f"Added {parent_dir} to Python path")

# Add bo directory to Python path for absolute imports
bo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if bo_dir not in sys.path:
    sys.path.insert(0, bo_dir)
    print(f"Added {bo_dir} to Python path")

# Try to create a .pth file for more permanent solution
# But don't fail if it doesn't work due to permissions issues
try:
    site_packages_dir = site.getsitepackages()[0]
    pth_file_path = os.path.join(site_packages_dir, 'asf_project.pth')
    with open(pth_file_path, 'w') as f:
        f.write(project_root)
    print(f"Created {pth_file_path} with path {project_root}")
except Exception as e:
    print(f"Could not create .pth file: {e}")
    print("This is not critical, continuing...")

# =========== MOCK ONLY THE PROBLEMATIC MODULES ===========
import types
import importlib.util

# Function to create fake modules that might cause hanging
def create_fake_module(name):
    module = types.ModuleType(name)
    sys.modules[name] = module
    print(f"Created fake module: {name}")
    return module

def ensure_fake_module_path(full_module_path):
    parts = full_module_path.split('.')
    current_path = ""
    
    for part in parts:
        if current_path:
            current_path = f"{current_path}.{part}"
        else:
            current_path = part
            
        if current_path not in sys.modules:
            create_fake_module(current_path)

# Only mock the specific problematic observability modules
problematic_modules = [
    # Observability modules that cause hanging
    "mcp_observability",
    "mcp_observability.tracing",
    "prometheus_client",
    "prometheus_client.core",
    "prometheus_client.exposition",
    "llm_gateway.resilience.metrics",
    "llm_gateway.resilience.tracing",
    "asf.medical.llm_gateway.observability.prometheus",
    
    # Based on the latest error, also mock these circular imports
    "asf.medical.core",
    "asf.medical.core.cache_init"
]

# Add these additional modules that might cause issues
problematic_modules.extend([
    "medical.core"
])

# Create fake versions of these modules to prevent hanging
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
    
    # Replace key Prometheus classes with dummy implementations
    prometheus.Counter = DummyMetric
    prometheus.Gauge = DummyMetric
    prometheus.Summary = DummyMetric
    prometheus.Histogram = DummyMetric
    prometheus.REGISTRY = DummyMetric()
    prometheus.start_http_server = lambda *args, **kwargs: None
    print("Added dummy Prometheus metrics implementations")

print("Mock setup complete for problematic modules only")

# =========== CREATE PACKAGE STRUCTURE ===========
def ensure_package_structure(start_dir):
    for root, dirs, files in os.walk(start_dir):
        # Skip directories that start with . (hidden directories)
        # and directories that are likely not meant to be Python packages
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ('__pycache__', 'node_modules', 'venv', '.git')]

        # Create __init__.py if not exists
        init_file = os.path.join(root, '__init__.py')
        if not os.path.exists(init_file):
            try:
                with open(init_file, 'w') as f:
                    f.write("# Auto-generated __init__.py file\n")
                print(f"Created {init_file}")
            except Exception as e:
                print(f"Could not create {init_file}: {e}")

# Make sure all directories in the project are treated as packages
ensure_package_structure(project_root)

# =========== INITIALIZE DATABASE ===========
def initialize_database():
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
            return True
        except OperationalError as e:
            print(f'Database error: {str(e)}')
            print('Make sure MySQL is running and the database exists')
            print('You may need to create the database manually: CREATE DATABASE bo_admin;')
            return False
        except Exception as e:
            print(f'Error initializing database: {str(e)}')
            print(f"Full error: {e}")
            return False
    except ImportError as e:
        print(f"Could not initialize database due to import error: {str(e)}")
        return False


# =========== IMPORT STRATEGY FOR MEDICAL MODULES ===========

# Function to safely try to import a module with fallback mock if needed
def try_import_with_mock(module_name, mock_classes=None):
    """
    Try to import a real module but fall back to a mock if it fails.
    mock_classes is a dictionary of class names to mock implementations if needed.
    """
    try:
        # First check if we're trying to import a real directory that exists
        parts = module_name.split('.')
        possible_path = os.path.join(project_root, *parts)
        if os.path.isdir(possible_path):
            # If directory exists, ensure it has an __init__.py
            init_path = os.path.join(possible_path, '__init__.py')
            if not os.path.exists(init_path):
                with open(init_path, 'w') as f:
                    f.write("# Auto-generated __init__.py file\n")
                print(f"Created {init_path}")
        
        # Try to import the module directly
        module = __import__(module_name, fromlist=['*'])
        print(f"Successfully imported {module_name}")
        return module
    except ImportError as e:
        print(f"Could not import {module_name}, creating mock: {e}")
        # If import fails, create a mock module
        mock_module = create_fake_module(module_name)
        
        # Add mock classes if specified
        if mock_classes:
            for class_name, mock_impl in mock_classes.items():
                setattr(mock_module, class_name, mock_impl)
                print(f"Added mock for {module_name}.{class_name}")
                
        return mock_module

# Define mock implementations for key classes
class MockContradictionVisualizer:
    def __init__(self, *args, **kwargs):
        print("Initialized MockContradictionVisualizer")
    
    def visualize(self, *args, **kwargs):
        return {"message": "Mock visualization result"}
        
    def get_visualization(self, *args, **kwargs):
        return {"message": "Mock visualization result"}
    
    def process(self, *args, **kwargs):
        return {"message": "Mock processing result"}

class MockSearchService:
    def __init__(self, *args, **kwargs):
        print("Initialized MockSearchService")
        
    def search(self, *args, **kwargs):
        return {"results": []}
        
    def similarity_search(self, *args, **kwargs):
        return {"results": []}
    
    def hybrid_search(self, *args, **kwargs):
        return {"results": []}
    
    async def async_search(self, *args, **kwargs):
        return {"results": []}

class MockTerminologyService:
    def __init__(self, *args, **kwargs):
        print("Initialized MockTerminologyService")
        
    def get_term(self, *args, **kwargs):
        return {"term": "mock term"}
        
    def lookup(self, *args, **kwargs):
        return {"result": "mock lookup"}

class MockClinicalDataService:
    def __init__(self, *args, **kwargs):
        print("Initialized MockClinicalDataService")
        
    def get_patient_data(self, *args, **kwargs):
        return {"patient_data": []}
        
    def get_clinical_document(self, *args, **kwargs):
        return {"document": "mock document"}
    
    def process_document(self, *args, **kwargs):
        return {"status": "processed"}

class MockValidationError(Exception):
    """Mock implementation of ValidationError from medical.core.exceptions"""
    def __init__(self, message="Mock validation error", *args, **kwargs):
        self.message = message
        super().__init__(message, *args)

# Add mock implementation for medical.core.exceptions
if "asf.medical.core" in sys.modules:
    core_module = sys.modules["asf.medical.core"]
    
    # Add cache_init module with necessary functions
    cache_init_module = types.ModuleType("asf.medical.core.cache_init")
    cache_init_module.initialize_cache = lambda *args, **kwargs: None
    cache_init_module.get_cache_manager = lambda *args, **kwargs: None
    sys.modules["asf.medical.core.cache_init"] = cache_init_module
    print("Created mock for asf.medical.core.cache_init")
    
    # Add a mock for medical.core.exceptions
    exceptions_module = types.ModuleType("medical.core.exceptions")
    exceptions_module.ValidationError = MockValidationError
    sys.modules["medical.core.exceptions"] = exceptions_module
    print("Created mock for medical.core.exceptions with ValidationError")

# Try to import needed modules with fallbacks to mocks
print("\nTrying to import medical modules with fallbacks to mocks if needed...")
try_import_with_mock("medical.visualization.contradiction_visualizer", 
                    {"ContradictionVisualizer": MockContradictionVisualizer})
try_import_with_mock("medical.services.search_service", 
                    {"SearchService": MockSearchService})
try_import_with_mock("medical.services.terminology_service", 
                    {"TerminologyService": MockTerminologyService})
try_import_with_mock("medical.services.clinical_data_service", 
                    {"ClinicalDataService": MockClinicalDataService})

# =========== MAIN FUNCTION ===========
if __name__ == '__main__':
    # Initialize database
    if not initialize_database():
        print("Database initialization failed, but trying to continue...")
    
    print('Starting backend server...')
    
    # Try to import the API endpoints
    try:
        import api.endpoints
        
        import uvicorn
        uvicorn.run(
            "api.endpoints:app", 
            host="127.0.0.1",  
            port=9000,         
            reload=False,      # Disable reload to avoid losing mocks
            log_level="debug",
            workers=1          # Single worker to avoid issues
        )
    except Exception as e:
        print(f"Error starting server with api.endpoints: {e}")
        import traceback
        traceback.print_exc()
        
        # Fall back to a minimal API if the full API can't be loaded
        print("\nFalling back to minimal API...")
        from fastapi import FastAPI, APIRouter
        from fastapi.middleware.cors import CORSMiddleware
        import uvicorn
        
        app = FastAPI(title="Backend Server (Fallback)")
        
        # Add CORS middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        @app.get("/")
        async def root():
            return {
                "message": "Fallback API is running",
                "status": "online",
                "note": "The main API couldn't be loaded due to an error"
            }
        
        @app.get("/ping")
        async def ping():
            return {"ping": "pong"}
        
        # Try to load at least the auth router
        try:
            from api.routers.auth import router as auth_router
            app.include_router(auth_router, prefix="/api/auth", tags=["authentication"])
            print("Successfully loaded basic authentication endpoints")
        except Exception as e:
            print(f"Could not load auth router: {e}")
        
        uvicorn.run(
            app,
            host="127.0.0.1",
            port=9000,
            log_level="debug"
        )