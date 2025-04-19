"""
Fixed version of run.py that avoids hanging issues with observability components.
This script sets environment variables and creates lightweight mocks for problematic modules
before they're imported to prevent hanging.
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
# Enable LLM Gateway
if "DISABLE_LLM_GATEWAY" in os.environ:
    del os.environ["DISABLE_LLM_GATEWAY"]
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

# Add backend directory to Python path for relative imports
backend_dir = os.path.dirname(os.path.abspath(__file__))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)
    print(f"Added {backend_dir} to Python path")

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

# =========== MOCK PROBLEMATIC MODULES ===========
import types
import importlib.util

# Function to create fake modules that might cause hanging
def create_fake_module(name):
    module = types.ModuleType(name)
    sys.modules[name] = module
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
            print(f"Created fake module: {current_path}")

# These specific modules are known to cause hanging during initialization
# by trying to connect to services that may not be running
problematic_modules = [
    # Tracing related
    "mcp_observability",
    "mcp_observability.tracing",

    # Metrics related
    "prometheus_client",
    "prometheus_client.core",
    "prometheus_client.exposition",
    "asf.medical.llm_gateway.observability.prometheus",

    # Resilience tracing
    "llm_gateway.resilience.metrics",
    "llm_gateway.resilience.tracing",

    # Module path issues
    "asf.medical.visualization",
    "asf.medical.visualization.contradiction_visualizer"
]


for module_name in modules_to_use_real:
    if module_name in problematic_modules:
        problematic_modules.remove(module_name)

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

# Add mock implementations for medical modules
for module_name in ["medical.visualization.contradiction_visualizer", "medical.services.search_service", "medical.services.terminology_service", "medical.services.clinical_data_service", "medical.ml.cl_peft", "asf.medical.ml.cl_peft"]:
    if module_name in sys.modules:
        mock_module = sys.modules[module_name]

        if module_name == "medical.visualization.contradiction_visualizer":
            mock_module.ContradictionVisualizer = type('ContradictionVisualizer', (), {
                '__init__': lambda self, *args, **kwargs: None,
                'visualize': lambda self, *args, **kwargs: {"message": "Mock visualization"},
                'get_visualization': lambda self, *args, **kwargs: {"message": "Mock visualization"},
                'process': lambda self, *args, **kwargs: {"message": "Mock processing"}
            })
            print(f"Added mock for {module_name}.ContradictionVisualizer")

        elif module_name == "medical.services.search_service":
            mock_module.SearchService = type('SearchService', (), {
                '__init__': lambda self, *args, **kwargs: None,
                'search': lambda self, *args, **kwargs: {"results": []},
                'similarity_search': lambda self, *args, **kwargs: {"results": []},
                'hybrid_search': lambda self, *args, **kwargs: {"results": []}
            })
            print(f"Added mock for {module_name}.SearchService")

        elif module_name == "medical.services.terminology_service":
            mock_module.TerminologyService = type('TerminologyService', (), {
                '__init__': lambda self, *args, **kwargs: None,
                'get_term': lambda self, *args, **kwargs: {"term": "mock term"},
                'lookup': lambda self, *args, **kwargs: {"result": "mock lookup"}
            })
            print(f"Added mock for {module_name}.TerminologyService")

        elif module_name == "medical.services.clinical_data_service":
            mock_module.ClinicalDataService = type('ClinicalDataService', (), {
                '__init__': lambda self, *args, **kwargs: None,
                'get_patient_data': lambda self, *args, **kwargs: {"patient_data": []},
                'get_clinical_document': lambda self, *args, **kwargs: {"document": "mock document"},
                'process_document': lambda self, *args, **kwargs: {"status": "processed"}
            })
            print(f"Added mock for {module_name}.ClinicalDataService")

# Add mock implementation for ContradictionVisualizer
if "asf.medical.visualization.contradiction_visualizer" in sys.modules:
    vis_module = sys.modules["asf.medical.visualization.contradiction_visualizer"]

    class MockContradictionVisualizer:
        def __init__(self, *args, **kwargs):
            print("Initialized mock ContradictionVisualizer")

        def visualize(self, *args, **kwargs):
            return {"message": "Mock visualization result"}

        def get_visualization(self, *args, **kwargs):
            return {"message": "Mock visualization result"}

        def process(self, *args, **kwargs):
            return {"message": "Mock processing result"}

    vis_module.ContradictionVisualizer = MockContradictionVisualizer
    print("Added mock ContradictionVisualizer implementation")

print("Mock setup complete")

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

# =========== CREATE SYMBOLIC LINKS FOR MODULES ===========
# Create symbolic links from asf.medical.ml modules to medical.ml modules
try:
    # Make sure the asf.medical.ml module exists
    asf_medical_ml_dir = os.path.join(project_root, 'asf', 'medical', 'ml')
    os.makedirs(asf_medical_ml_dir, exist_ok=True)

    # Create an __init__.py file if it doesn't exist
    init_file = os.path.join(asf_medical_ml_dir, '__init__.py')
    if not os.path.exists(init_file):
        with open(init_file, 'w') as f:
            f.write("# Auto-generated __init__.py file\n")

    # Define modules to link/copy
    modules_to_link = ['cl_peft', 'document_processing']

    for module_name in modules_to_link:
        # Source and destination paths
        medical_ml_module_dir = os.path.join(project_root, 'medical', 'ml', module_name)
        asf_medical_ml_module_dir = os.path.join(asf_medical_ml_dir, module_name)

        # If the directory already exists, remove it
        if os.path.exists(asf_medical_ml_module_dir):
            if os.path.islink(asf_medical_ml_module_dir):
                os.unlink(asf_medical_ml_module_dir)
            else:
                import shutil
                shutil.rmtree(asf_medical_ml_module_dir)

        # Create the symbolic link or copy the directory
        if os.path.exists(medical_ml_module_dir):
            # On Windows, we need to use a different approach
            if os.name == 'nt':
                # Copy the directory instead of creating a symbolic link
                import shutil
                shutil.copytree(medical_ml_module_dir, asf_medical_ml_module_dir)
                print(f"Copied {medical_ml_module_dir} to {asf_medical_ml_module_dir}")
            else:
                # On Unix-like systems, we can use os.symlink
                os.symlink(medical_ml_module_dir, asf_medical_ml_module_dir, target_is_directory=True)
                print(f"Created symbolic link from {asf_medical_ml_module_dir} to {medical_ml_module_dir}")
        else:
            print(f"Warning: {medical_ml_module_dir} does not exist, cannot create symbolic link/copy")
except Exception as e:
    print(f"Error creating symbolic links: {str(e)}")

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
            port=8000,         # Use port 8000 as requested
            reload=True,       # Enable reload for development
            log_level="debug",
            workers=1,         # Single worker to avoid issues
            lifespan="on"
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
            port=8000,  # Use port 8000 as requested
            log_level="debug"
        )