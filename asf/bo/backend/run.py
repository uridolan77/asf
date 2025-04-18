# Entry point for backend server
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

# Try to create a .pth file for more permanent solution
try:
    site_packages_dir = site.getsitepackages()[0]
    pth_file_path = os.path.join(site_packages_dir, 'asf_project.pth')
    with open(pth_file_path, 'w') as f:
        f.write(project_root)
    print(f"Created {pth_file_path} with path {project_root}")
except Exception as e:
    print(f"Could not create .pth file: {e}")

# Create __init__.py files if needed to ensure directories are treated as packages
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

if __name__ == '__main__':
    # Set environment variable to disable MCP websocket background tasks
    os.environ["DISABLE_MCP_WEBSOCKET_TASKS"] = "1"
    
    # Set other environment variables to disable potentially problematic components
    os.environ["DISABLE_PROMETHEUS"] = "1"  # Disable Prometheus metrics if they exist
    os.environ["LOG_LEVEL"] = "debug"  # Enable debug logging
    
    import uvicorn
    from sqlalchemy import text
    from sqlalchemy.exc import OperationalError
    from config.config import engine
    from models.user import Base, Role

    # Create database tables
    print('Initializing database...')
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

    print('Running backend server...')
    # Use a different port and configure uvicorn with more explicit settings
    uvicorn.run(
        "api.endpoints:app", 
        host="127.0.0.1",  # Use localhost instead of 0.0.0.0
        port=9000,         # Use a higher port number to avoid permission issues
        reload=True,
        log_level="debug",
        workers=1,
        lifespan="on"
    )
