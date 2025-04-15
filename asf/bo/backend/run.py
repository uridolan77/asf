# Entry point for backend server

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
