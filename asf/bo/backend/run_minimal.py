"""
Minimal run script that uses monkey patching to prevent problematic modules from loading.
This is a more aggressive approach to prevent the server from hanging.
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

# ========== MONKEY PATCHING TO PREVENT PROBLEMATIC MODULES ==========

# Create fake modules for problematic imports
def create_fake_module(name):
    """Create a fake module that does nothing."""
    module = types.ModuleType(name)
    sys.modules[name] = module
    
    # Add a print statement to indicate when this fake module is accessed
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

# List of problematic modules to fake
problematic_modules = [
    # Tracing related
    "mcp_observability.tracing",
    "opentelemetry",
    "opentelemetry.sdk",
    "opentelemetry.sdk.trace",
    "opentelemetry.sdk.resources",
    "opentelemetry.sdk.trace.export",
    "opentelemetry.exporter.otlp.proto.grpc",
    
    # Metrics related
    "prometheus_client",
    "prometheus_client.core",
    "prometheus_client.exposition",
    "asf.medical.llm_gateway.observability.prometheus",
    "llm_gateway.resilience.metrics",
    
    # LLM Gateway related
    "llm_gateway",
    "llm_gateway.resilience",
    "llm_gateway.resilience.tracing",
    "medical.ml.dspy.dspy_client",
    "medical.models.biomedlm",
    
    # Visualization modules
    "medical.visualization",
    "medical.visualization.contradiction_visualizer",
    
    # Service modules
    "medical.core",
    "medical.graph",
    "medical.services",
    "medical.services.contradiction_resolution",
    "medical.services.search_service",  # New addition
    "medical.services.document_processor",
    "medical.services.knowledge_base",
    "medical.layer1_knowledge_substrate",
    "medical.orchestration"
]

# Create fake modules for all problematic modules
for module_name in problematic_modules:
    ensure_fake_module_path(module_name)

# Special handling for certain modules if needed
if "prometheus_client" in sys.modules:
    # Create dummy Counter, Gauge, etc. classes
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

# Add dummy ContradictionVisualizer class
if "medical.visualization.contradiction_visualizer" in sys.modules:
    visualization = sys.modules["medical.visualization.contradiction_visualizer"]
    
    class DummyContradictionVisualizer:
        def __init__(self, *args, **kwargs):
            print("Initialized dummy ContradictionVisualizer")
            
        def visualize(self, *args, **kwargs):
            return {"message": "Dummy visualization result"}
            
        def get_visualization(self, *args, **kwargs):
            return {"message": "Dummy visualization result"}
        
        def process(self, *args, **kwargs):
            return {"message": "Dummy processing result"}
    
    visualization.ContradictionVisualizer = DummyContradictionVisualizer
    print("Added dummy ContradictionVisualizer class")

# Add more specific module mocks based on the imports we saw in the error message
if "medical.services.contradiction_resolution" in sys.modules:
    contradiction_module = sys.modules["medical.services.contradiction_resolution"]
    
    # Add any classes or functions that might be imported from this module
    contradiction_module.resolve_contradictions = lambda *args, **kwargs: {"resolved": True, "message": "Dummy resolution"}
    contradiction_module.analyze_contradictions = lambda *args, **kwargs: {"analyzed": True, "message": "Dummy analysis"}
    print("Added dummy contradiction resolution functions")

# Add dummy SearchService class
if "medical.services.search_service" in sys.modules:
    search_service_module = sys.modules["medical.services.search_service"]
    
    class DummySearchService:
        def __init__(self, *args, **kwargs):
            print("Initialized dummy SearchService")
            
        def search(self, *args, **kwargs):
            return {"results": [], "message": "Dummy search results"}
            
        def similarity_search(self, *args, **kwargs):
            return {"results": [], "message": "Dummy similarity search results"}
        
        def hybrid_search(self, *args, **kwargs):
            return {"results": [], "message": "Dummy hybrid search results"}
        
        def configure(self, *args, **kwargs):
            return True
            
        async def async_search(self, *args, **kwargs):
            return {"results": [], "message": "Dummy async search results"}
    
    search_service_module.SearchService = DummySearchService
    print("Added dummy SearchService class")

print("Monkey patching complete - problematic modules have been replaced with dummy versions")

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

print('Running backend server with DISABLED LLM and observability components...')

# First try to load the API without LLM endpoints
try:
    print("Loading API endpoints...")
    
    # This import should now work without hanging since problematic modules are faked
    import api.endpoints
    print("API endpoints loaded successfully!")
    
    import uvicorn
    uvicorn.run(
        "api.endpoints:app",
        host="127.0.0.1",
        port=9000,
        log_level="debug",
        reload=False,  # Don't use reload as it would lose our monkey patches
        workers=1
    )
except Exception as e:
    print(f"Error loading or running the API: {e}")
    import traceback
    traceback.print_exc()
    
    print("\nFalling back to minimal API...")
    from fastapi import FastAPI
    import uvicorn
    
    app = FastAPI(title="Backend Server (Fallback)")
    
    @app.get("/")
    async def root():
        return {
            "message": "Minimal fallback API is running",
            "status": "online",
            "note": "The main API couldn't be loaded due to an error"
        }
    
    @app.get("/ping")
    async def ping():
        return {"ping": "pong"}
    
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=9000,
        log_level="debug"
    )