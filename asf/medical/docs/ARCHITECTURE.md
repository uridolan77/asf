# ASF Medical Research Synthesizer Architecture

This document provides an overview of the architecture of the ASF Medical Research Synthesizer, describing the purpose of each layer and the design principles that guide the system.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Design Principles](#design-principles)
3. [Layer Descriptions](#layer-descriptions)
   - [API Layer](#api-layer)
   - [Service Layer](#service-layer)
   - [Repository Layer](#repository-layer)
   - [ML Layer](#ml-layer)
   - [Client Layer](#client-layer)
   - [Task Layer](#task-layer)
   - [Core Layer](#core-layer)
4. [Data Flow](#data-flow)
5. [Dependency Injection](#dependency-injection)
6. [Error Handling](#error-handling)
7. [Caching Strategy](#caching-strategy)
8. [Task Processing](#task-processing)
9. [Authentication and Authorization](#authentication-and-authorization)
10. [Monitoring and Observability](#monitoring-and-observability)
11. [Testing Strategy](#testing-strategy)
12. [Deployment Considerations](#deployment-considerations)

## Architecture Overview

The ASF Medical Research Synthesizer follows a layered architecture with clear separation of concerns. The system is designed to be modular, scalable, and maintainable, with each layer having a specific responsibility.

The high-level architecture consists of the following layers:

1. **API Layer**: Handles HTTP requests and responses, input validation, and routing.
2. **Service Layer**: Implements business logic and orchestrates operations across multiple repositories and clients.
3. **Repository Layer**: Provides data access and persistence.
4. **ML Layer**: Implements machine learning models and algorithms.
5. **Client Layer**: Provides interfaces to external services and APIs.
6. **Task Layer**: Handles background and long-running tasks.
7. **Core Layer**: Provides common utilities, configurations, and exceptions.

## Design Principles

The ASF Medical Research Synthesizer follows these design principles:

1. **Separation of Concerns**: Each layer has a specific responsibility and should not be concerned with the implementation details of other layers.
2. **Dependency Injection**: Dependencies are injected rather than created directly, making the code more testable and flexible.
3. **Asynchronous Operations**: The system uses asynchronous operations to improve performance and scalability.
4. **Domain-Driven Design**: The system is designed around the domain of medical research synthesis.
5. **SOLID Principles**: The system follows the SOLID principles of object-oriented design.
6. **Clean Architecture**: The system follows the principles of clean architecture, with dependencies pointing inward.
7. **Stateless Design**: The system is designed to be stateless, with state stored in persistent storage.
8. **Idempotent Operations**: API operations are designed to be idempotent, allowing for safe retries.
9. **Fail Fast**: The system fails fast when errors occur, providing clear error messages.
10. **Graceful Degradation**: The system degrades gracefully when external services are unavailable.

## Layer Descriptions

### API Layer

The API layer is responsible for handling HTTP requests and responses, input validation, and routing. It is implemented using FastAPI, which provides automatic OpenAPI documentation, request validation, and dependency injection.

**Key Components**:
- **Routers**: Define API endpoints and routes.
- **Models**: Define request and response models using Pydantic.
- **Dependencies**: Define dependencies for API endpoints.
- **Middleware**: Implement cross-cutting concerns like authentication, logging, and error handling.

**Design Patterns**:
- **Dependency Injection**: Dependencies are injected into API endpoints.
- **Factory Pattern**: Used to create service instances.
- **Decorator Pattern**: Used for cross-cutting concerns like authentication and validation.

**Example**:
```python
@router.post("/contradiction/detect", response_model=APIResponse[ContradictionResponse])
async def detect_contradiction(
    request: ContradictionRequest,
    current_user: User = Depends(get_current_active_user),
    contradiction_service: UnifiedContradictionService = Depends(get_contradiction_service)
):
    # Implementation
```

### Service Layer

The service layer implements business logic and orchestrates operations across multiple repositories and clients. It is responsible for implementing the core functionality of the system.

**Key Components**:
- **Services**: Implement business logic and orchestrate operations.
- **DTOs (Data Transfer Objects)**: Define data structures for transferring data between layers.
- **Validators**: Validate input data and business rules.
- **Mappers**: Map between different data representations.

**Design Patterns**:
- **Facade Pattern**: Services provide a simplified interface to complex subsystems.
- **Strategy Pattern**: Different strategies can be used for different operations.
- **Template Method Pattern**: Common operations are defined in base classes.
- **Command Pattern**: Operations are encapsulated as commands.

**Example**:
```python
class AnalysisService:
    def __init__(self, search_service: SearchService, contradiction_service: ContradictionService):
        self.search_service = search_service
        self.contradiction_service = contradiction_service
        
    async def analyze_contradictions(self, query: str, max_results: int = 20, threshold: float = 0.7):
        # Implementation
```

### Repository Layer

The repository layer provides data access and persistence. It is responsible for storing and retrieving data from the database.

**Key Components**:
- **Repositories**: Provide data access and persistence.
- **Models**: Define database models using SQLAlchemy.
- **Migrations**: Define database migrations.
- **Queries**: Define complex database queries.

**Design Patterns**:
- **Repository Pattern**: Repositories provide a clean interface to data access.
- **Unit of Work Pattern**: Database operations are grouped into units of work.
- **Data Mapper Pattern**: Maps between database models and domain objects.
- **Active Record Pattern**: Models provide data access methods.

**Example**:
```python
class UserRepository(EnhancedBaseRepository[User]):
    def __init__(self, is_async: bool = True):
        super().__init__(User, is_async)
        
    async def get_by_email(self, db: AsyncSession, email: str) -> Optional[User]:
        # Implementation
```

### ML Layer

The ML layer implements machine learning models and algorithms. It is responsible for providing predictions, classifications, and other ML-based functionality.

**Key Components**:
- **Models**: Implement machine learning models.
- **Algorithms**: Implement machine learning algorithms.
- **Preprocessors**: Preprocess data for ML models.
- **Evaluators**: Evaluate ML model performance.

**Design Patterns**:
- **Strategy Pattern**: Different ML models can be used for different tasks.
- **Factory Pattern**: Used to create ML model instances.
- **Adapter Pattern**: Adapts external ML libraries to the system's interface.
- **Decorator Pattern**: Used for cross-cutting concerns like caching and logging.

**Example**:
```python
class BioMedLMService:
    def __init__(self, model_name: str = "biomedlm-base"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        
    async def detect_contradiction(self, claim1: str, claim2: str) -> Tuple[bool, float]:
        # Implementation
```

### Client Layer

The client layer provides interfaces to external services and APIs. It is responsible for communicating with external systems.

**Key Components**:
- **Clients**: Provide interfaces to external services and APIs.
- **Adapters**: Adapt external APIs to the system's interface.
- **DTOs**: Define data structures for transferring data between systems.
- **Serializers**: Serialize and deserialize data for external APIs.

**Design Patterns**:
- **Adapter Pattern**: Adapts external APIs to the system's interface.
- **Proxy Pattern**: Provides a surrogate for external services.
- **Facade Pattern**: Provides a simplified interface to external services.
- **Retry Pattern**: Retries failed operations.

**Example**:
```python
class NCBIClient:
    def __init__(self, api_key: Optional[str] = None, email: Optional[str] = None):
        self.api_key = api_key
        self.email = email
        
    async def search(self, query: str, max_results: int = 20) -> List[Dict[str, Any]]:
        # Implementation
```

### Task Layer

The task layer handles background and long-running tasks. It is responsible for executing tasks asynchronously.

**Key Components**:
- **Tasks**: Define background and long-running tasks.
- **Workers**: Execute tasks asynchronously.
- **Queues**: Store tasks for execution.
- **Schedulers**: Schedule tasks for execution.

**Design Patterns**:
- **Task Queue Pattern**: Tasks are queued for execution.
- **Worker Pool Pattern**: Multiple workers execute tasks concurrently.
- **Scheduler Pattern**: Tasks are scheduled for execution.
- **Observer Pattern**: Tasks notify observers of their progress.

**Example**:
```python
@dramatiq.actor
def analyze_contradictions_task(task_id: str, query: str, max_results: int = 20, threshold: float = 0.7):
    # Implementation
```

### Core Layer

The core layer provides common utilities, configurations, and exceptions. It is responsible for providing functionality that is used across multiple layers.

**Key Components**:
- **Utilities**: Provide common functionality.
- **Configurations**: Define system configurations.
- **Exceptions**: Define custom exceptions.
- **Constants**: Define system constants.

**Design Patterns**:
- **Singleton Pattern**: Used for shared resources.
- **Factory Pattern**: Used to create instances of common objects.
- **Strategy Pattern**: Different strategies can be used for different operations.
- **Decorator Pattern**: Used for cross-cutting concerns like caching and logging.

**Example**:
```python
class UnifiedTaskStorage:
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(UnifiedTaskStorage, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
        
    # Implementation
```

## Data Flow

The data flow in the ASF Medical Research Synthesizer follows these steps:

1. **API Layer**: Receives HTTP requests, validates input, and routes to the appropriate service.
2. **Service Layer**: Implements business logic and orchestrates operations across multiple repositories and clients.
3. **Repository Layer**: Provides data access and persistence.
4. **ML Layer**: Implements machine learning models and algorithms.
5. **Client Layer**: Provides interfaces to external services and APIs.
6. **Task Layer**: Handles background and long-running tasks.
7. **Core Layer**: Provides common utilities, configurations, and exceptions.

The data flows from the API layer to the service layer, which orchestrates operations across the repository, ML, and client layers. The task layer handles background and long-running tasks, and the core layer provides common functionality used across multiple layers.

## Dependency Injection

The ASF Medical Research Synthesizer uses dependency injection to make the code more testable and flexible. Dependencies are injected rather than created directly, allowing for easier testing and more flexible configuration.

FastAPI's dependency injection system is used for API endpoints, and constructor injection is used for services, repositories, and other components.

**Example**:
```python
# API Layer
@router.post("/contradiction/detect", response_model=APIResponse[ContradictionResponse])
async def detect_contradiction(
    request: ContradictionRequest,
    current_user: User = Depends(get_current_active_user),
    contradiction_service: UnifiedContradictionService = Depends(get_contradiction_service)
):
    # Implementation

# Service Layer
class AnalysisService:
    def __init__(self, search_service: SearchService, contradiction_service: ContradictionService):
        self.search_service = search_service
        self.contradiction_service = contradiction_service
```

## Error Handling

The ASF Medical Research Synthesizer uses a consistent error handling approach across all layers. Custom exceptions are defined in the core layer, and each layer handles exceptions appropriately.

**Key Components**:
- **Custom Exceptions**: Define custom exceptions for different error types.
- **Exception Handlers**: Handle exceptions and return appropriate responses.
- **Error Logging**: Log errors for debugging and monitoring.
- **Error Responses**: Return consistent error responses to clients.

**Example**:
```python
# Custom Exception
class ResourceNotFoundError(Exception):
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)

# Exception Handler
@app.exception_handler(ResourceNotFoundError)
async def resource_not_found_exception_handler(request: Request, exc: ResourceNotFoundError):
    return JSONResponse(
        status_code=404,
        content={"detail": exc.message}
    )
```

## Caching Strategy

The ASF Medical Research Synthesizer uses a multi-level caching strategy to improve performance and reduce load on external services.

**Key Components**:
- **In-Memory Cache**: Provides fast access to frequently used data.
- **Distributed Cache**: Provides shared cache across multiple instances.
- **Cache Invalidation**: Invalidates cache entries when data changes.
- **Cache Eviction**: Evicts cache entries when the cache is full.

**Example**:
```python
# Caching Decorator
@enhanced_cached(prefix="detect_contradiction", data_type="analysis")
async def detect_contradiction(
    self,
    claim1: str,
    claim2: str,
    metadata1: Optional[Dict[str, Any]] = None,
    metadata2: Optional[Dict[str, Any]] = None,
    threshold: float = 0.7,
    use_biomedlm: bool = True,
    use_tsmixer: bool = False,
    use_lorentz: bool = False,
    use_temporal: bool = False,
    skip_cache: bool = False
) -> Dict[str, Any]:
    # Implementation
```

## Task Processing

The ASF Medical Research Synthesizer uses a task processing system to handle background and long-running tasks.

**Key Components**:
- **Task Queue**: Stores tasks for execution.
- **Workers**: Execute tasks asynchronously.
- **Task Storage**: Stores task results and status.
- **Task Monitoring**: Monitors task execution and status.

**Example**:
```python
# Task Definition
@dramatiq.actor
def analyze_contradictions_task(task_id: str, query: str, max_results: int = 20, threshold: float = 0.7):
    # Implementation

# Task Storage
await unified_task_storage.set_task_result(
    task_id=task_id,
    result=result,
    metadata={"status": "completed"}
)
```

## Authentication and Authorization

The ASF Medical Research Synthesizer uses a token-based authentication system with role-based authorization.

**Key Components**:
- **Authentication**: Verifies user identity.
- **Authorization**: Controls access to resources.
- **Token Management**: Manages authentication tokens.
- **Role Management**: Manages user roles and permissions.

**Example**:
```python
# Authentication Dependency
async def get_current_user(token: str = Depends(oauth2_scheme), db: AsyncSession = Depends(get_db)):
    # Implementation

# Authorization Dependency
async def get_current_active_user(current_user: User = Depends(get_current_user)):
    # Implementation
```

## Monitoring and Observability

The ASF Medical Research Synthesizer uses a comprehensive monitoring and observability system to track system performance and health.

**Key Components**:
- **Logging**: Records system events and errors.
- **Metrics**: Measures system performance and health.
- **Tracing**: Tracks request flow through the system.
- **Alerting**: Notifies operators of system issues.

**Example**:
```python
# Logging
logger.info(f"Contradiction detection request: claim1='{request.claim1[:50]}...', claim2='{request.claim2[:50]}...', user_id={current_user.id}")

# Metrics
@async_timed("detect_contradiction_endpoint")
async def detect_contradiction(
    request: ContradictionRequest,
    req: Request = None,
    res: Response = None,
    current_user: User = Depends(get_current_active_user),
    contradiction_service: UnifiedContradictionService = Depends(get_contradiction_service)
):
    # Implementation
```

## Testing Strategy

The ASF Medical Research Synthesizer uses a comprehensive testing strategy to ensure system quality and reliability.

**Key Components**:
- **Unit Tests**: Test individual components in isolation.
- **Integration Tests**: Test interactions between components.
- **End-to-End Tests**: Test the entire system.
- **Performance Tests**: Test system performance under load.

**Example**:
```python
# Unit Test
def test_detect_contradiction():
    # Implementation

# Integration Test
def test_analyze_contradictions():
    # Implementation
```

## Deployment Considerations

The ASF Medical Research Synthesizer is designed to be deployed in a variety of environments, from development to production.

**Key Considerations**:
- **Scalability**: The system can scale horizontally to handle increased load.
- **Reliability**: The system is designed to be reliable and fault-tolerant.
- **Security**: The system implements security best practices.
- **Performance**: The system is optimized for performance.
- **Maintainability**: The system is designed to be maintainable and extensible.

**Deployment Options**:
- **Docker**: The system can be deployed using Docker containers.
- **Kubernetes**: The system can be deployed on Kubernetes for orchestration.
- **Cloud Providers**: The system can be deployed on cloud providers like AWS, Azure, or GCP.
- **On-Premises**: The system can be deployed on-premises in a data center.
