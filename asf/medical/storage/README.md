# ASF Medical Research Synthesizer Storage

This directory contains the storage components of the ASF Medical Research Synthesizer, including database models, repositories, and database connection management.

## Database Models

The database models are implemented in `models.py` and provide a SQLAlchemy ORM representation of the database tables.

### Features

- **SQLAlchemy ORM**: Use SQLAlchemy ORM for database operations
- **Type safety**: Type-safe database models
- **Relationships**: Define relationships between models
- **Validation**: Validate data before storing in the database

### Models

- **User**: User model for authentication and authorization
- **Query**: Query model for storing search queries
- **Result**: Result model for storing search results
- **KnowledgeBase**: Knowledge base model for storing knowledge bases

## Repositories

The repositories are implemented in the `repositories` directory and provide a repository pattern for database operations.

### Features

- **Repository pattern**: Separate data access logic from business logic
- **Async operations**: All database operations are async
- **Type safety**: Type-safe repository methods
- **Error handling**: Proper error handling for database operations

### Repositories

- **BaseRepository**: Base repository with common CRUD operations
- **UserRepository**: Repository for user operations
- **QueryRepository**: Repository for query operations
- **ResultRepository**: Repository for result operations
- **KnowledgeBaseRepository**: Repository for knowledge base operations

### Usage

```python
from asf.medical.storage.repositories.user_repository import UserRepository

# Create a repository
user_repository = UserRepository()

# Create a user
user = await user_repository.create_async(
    db=None,  # This will be handled by the repository
    obj_in={
        'username': 'john',
        'email': 'john@example.com',
        'hashed_password': 'hashed_password'
    }
)

# Get a user by ID
user = await user_repository.get_async(db=None, id=1)

# Get a user by username
user = await user_repository.get_by_username_async(db=None, username='john')

# Update a user
user = await user_repository.update_async(
    db=None,
    id=1,
    obj_in={
        'email': 'new_email@example.com'
    }
)

# Delete a user
await user_repository.delete_async(db=None, id=1)
```

## Database Connection

The database connection is managed in `database.py` and provides a SQLAlchemy async engine and session factory.

### Features

- **Async engine**: Use SQLAlchemy async engine for database operations
- **Session factory**: Create async sessions for database operations
- **Connection pooling**: Use connection pooling for efficient database connections
- **Error handling**: Proper error handling for database connections

### Usage

```python
from asf.medical.storage.database import get_db_session

# Get a database session
async with get_db_session() as db:
    # Use the session for database operations
    result = await db.execute("SELECT * FROM users")
    users = result.scalars().all()
```

## Configuration

The database connection can be configured in the `.env` file:

```
# Database settings
DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/medical_research_synthesizer
```

## Migrations

Database migrations are managed using Alembic.

### Features

- **Schema versioning**: Track database schema changes
- **Automatic migrations**: Generate migrations automatically
- **Manual migrations**: Create migrations manually
- **Rollback**: Roll back migrations if needed

### Usage

```bash
# Create a migration
alembic revision --autogenerate -m "Add users table"

# Apply migrations
alembic upgrade head

# Roll back migrations
alembic downgrade -1
```
