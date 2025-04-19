"""
Ultra-minimal test server that avoids importing problematic components.
This version doesn't import api.endpoints at all to prevent the server from hanging.
"""
import os
import sys
import site

# Add the project root directory to sys.path to import the asf module
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Add to sys.path in multiple ways to ensure it's found
if project_root not in sys.path:
    sys.path.insert(0, project_root)  # Insert at the beginning of sys.path
    print(f"Added {project_root} to Python path")

# Also add the parent directory of the project root
parent_dir = os.path.dirname(project_root)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
    print(f"Added {parent_dir} to Python path")

# Add bo directory to Python path for absolute imports
bo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if bo_dir not in sys.path:
    sys.path.insert(0, bo_dir)
    print(f"Added {bo_dir} to Python path")

# Set environment variables to disable problematic components
os.environ["DISABLE_MCP_WEBSOCKET_TASKS"] = "1"
os.environ["DISABLE_PROMETHEUS"] = "1"
os.environ["DISABLE_TRACING"] = "1"
os.environ["DISABLE_METRICS"] = "1"
os.environ["DISABLE_OTLP"] = "1"
os.environ["DISABLE_LLM_GATEWAY"] = "1"
os.environ["DISABLE_OBSERVABILITY"] = "1"
os.environ["LOG_LEVEL"] = "debug"

# Initialize database only if explicitly requested via command line arg
init_db = "--init-db" in sys.argv
if init_db:
    try:
        # Database imports and setup
        from sqlalchemy import text
        from sqlalchemy.exc import OperationalError
        from config.config import engine
        from models.user import Base, Role

        # Initialize database tables and default roles
        print('Initializing database...')
        try:
            # Import all models to ensure they're registered with Base.metadata
            try:
                from models import User, Role, Provider, ProviderModel, Configuration, UserSetting, AuditLog
            except ImportError as e:
                print(f"Warning: Some models could not be imported: {e}")
                print("Continuing with available models...")

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
        except Exception as e:
            print(f'Error initializing database: {str(e)}')
            print('Continuing with server startup despite database error...')
    except ImportError as e:
        print(f"Could not initialize database: {str(e)}")
        print("Continuing without database support...")
else:
    print("Skipping database initialization. Use --init-db to initialize the database.")

# We need to set up the app in a module so uvicorn can import it when using reload
# Create a file called test_app.py if it doesn't exist
test_app_path = os.path.join(os.path.dirname(__file__), "test_app.py")
if not os.path.exists(test_app_path) or "--force" in sys.argv:
    print(f"Creating test_app.py for uvicorn reload functionality...")
    with open(test_app_path, "w") as f:
        f.write('''"""
Test app module for uvicorn reload functionality.
"""
from fastapi import FastAPI, APIRouter, Depends, HTTPException, Request
from fastapi.security import OAuth2PasswordBearer
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import time
import threading
import socket
import platform
import os
import sys

# Create our own FastAPI app
app = FastAPI(title="Minimal Test Server")

# Define some basic models
class UserBase(BaseModel):
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None

class UserCreate(UserBase):
    password: str

class User(UserBase):
    id: int
    is_active: bool

    class Config:
        from_attributes = True

# Create a mock users database
fake_users_db = {
    "user1": {
        "id": 1,
        "username": "user1",
        "email": "user1@example.com",
        "full_name": "User One",
        "password": "password",
        "is_active": True
    },
    "user2": {
        "id": 2,
        "username": "user2",
        "email": "user2@example.com",
        "full_name": "User Two",
        "password": "password",
        "is_active": True
    }
}

# Create a basic auth scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Create an API router
router = APIRouter()

@router.get("/")
async def root():
    return {
        "message": "Ultra-minimal test server is running",
        "status": "online",
        "note": "api.endpoints is NOT imported to avoid hanging"
    }

@router.get("/ping")
async def ping():
    return {"ping": "pong"}

@router.get("/users/", response_model=List[User])
async def read_users():
    users = [User(**user) for user in fake_users_db.values()]
    return users

@router.get("/users/{username}", response_model=User)
async def read_user(username: str):
    if username not in fake_users_db:
        raise HTTPException(status_code=404, detail="User not found")
    return User(**fake_users_db[username])

@router.post("/users/", response_model=User)
async def create_user(user: UserCreate):
    if user.username in fake_users_db:
        raise HTTPException(status_code=400, detail="Username already registered")
    
    user_dict = user.model_dump()
    user_dict["id"] = len(fake_users_db) + 1
    user_dict["is_active"] = True
    
    fake_users_db[user.username] = user_dict
    return User(**user_dict)

# Diagnostic endpoints
@router.get("/system-info")
async def system_info():
    """Endpoint to get system information to help diagnose issues."""
    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "sys_path": sys.path,
        "environment_vars": {k: v for k, v in os.environ.items() if not k.startswith("_") 
                             and k.upper() in ["DISABLE_MCP_WEBSOCKET_TASKS", "DISABLE_PROMETHEUS", 
                                              "DISABLE_TRACING", "DISABLE_METRICS", "DISABLE_OTLP", 
                                              "DISABLE_LLM_GATEWAY", "DISABLE_OBSERVABILITY", "LOG_LEVEL"]},
        "hostname": socket.gethostname()
    }

@router.get("/delay/{seconds}")
async def delay(seconds: int):
    """Endpoint to test if long operations block the server."""
    if seconds > 30:
        raise HTTPException(status_code=400, detail="Maximum delay is 30 seconds")
    
    time.sleep(seconds)
    return {"message": f"Delayed for {seconds} seconds"}

@router.post("/test-async")
async def test_async(background_time: int = 5):
    """Test if background tasks run correctly."""
    def background_job(seconds):
        print(f"Background job started, will run for {seconds} seconds")
        time.sleep(seconds)
        print(f"Background job completed after {seconds} seconds")
    
    if background_time > 30:
        raise HTTPException(status_code=400, detail="Maximum background time is 30 seconds")
    
    # Start a background thread
    thread = threading.Thread(target=background_job, args=(background_time,))
    thread.daemon = True
    thread.start()
    
    return {"message": f"Started background job for {background_time} seconds"}

@router.get("/request-info")
async def request_info(request: Request):
    """Get information about the current request."""
    headers = dict(request.headers)
    client = request.client
    
    return {
        "headers": headers,
        "client": {
            "host": client.host if client else None,
            "port": client.port if client else None
        },
        "method": request.method,
        "url": str(request.url)
    }

# Include the router in the app
app.include_router(router)

# Add middleware and exception handlers as needed
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"message": f"An unexpected error occurred: {str(exc)}"}
    )
''')
    print("Created test_app.py")

import uvicorn

if __name__ == "__main__":
    print("Starting ultra-minimal test server (WITHOUT importing api.endpoints)...")
    print("Available endpoints:")
    print("  - GET /")
    print("  - GET /ping")
    print("  - GET /users/")
    print("  - GET /users/{username}")
    print("  - POST /users/")
    print("  - GET /system-info")
    print("  - GET /delay/{seconds}")
    print("  - POST /test-async")
    print("  - GET /request-info")
    
    try:
        # Use the module-based import string for uvicorn with reload=True
        use_reload = "--no-reload" not in sys.argv
        
        if use_reload:
            print("Running with reload enabled (using import string)")
            uvicorn.run(
                "test_app:app",  # Use the module we created/updated
                host="127.0.0.1",
                port=9000,
                log_level="debug",
                reload=True,
                workers=1
            )
        else:
            # Import directly for non-reload mode
            print("Running without reload")
            from test_app import app
            uvicorn.run(
                app,
                host="127.0.0.1",
                port=9000,
                log_level="debug",
                workers=1
            )
    except Exception as e:
        print(f"Error running uvicorn: {e}")
        import traceback
        traceback.print_exc()