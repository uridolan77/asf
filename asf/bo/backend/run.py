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

# Try to create a .pth file for more permanent solution
try:
    site_packages_dir = site.getsitepackages()[0]
    pth_file_path = os.path.join(site_packages_dir, 'asf_project.pth')
    with open(pth_file_path, 'w') as f:
        f.write(project_root)
    print(f"Created {pth_file_path} with path {project_root}")
except Exception as e:
    print(f"Could not create .pth file: {e}")

if __name__ == '__main__':
    import uvicorn
    from sqlalchemy import text
    from sqlalchemy.exc import OperationalError
    from config.config import engine
    from models.user import Base, Role

    # Create database tables
    print('Initializing database...')
    try:
        # Try to create all tables
        Base.metadata.create_all(bind=engine)

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
    uvicorn.run("api.endpoints:app", host="0.0.0.0", port=8000, reload=True)
