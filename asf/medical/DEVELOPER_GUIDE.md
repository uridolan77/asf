# ASF Medical Research Synthesizer - Developer Guide

This guide provides comprehensive information for developers working on the ASF Medical Research Synthesizer project. It covers the architecture, setup instructions, development workflows, and best practices.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Development Environment Setup](#development-environment-setup)
3. [Project Structure](#project-structure)
4. [Key Components](#key-components)
5. [Development Workflow](#development-workflow)
6. [Testing](#testing)
7. [Logging and Monitoring](#logging-and-monitoring)
8. [API Documentation](#api-documentation)
9. [Deployment](#deployment)
10. [Troubleshooting](#troubleshooting)

## Architecture Overview

The ASF Medical Research Synthesizer is built with a modular, layered architecture that separates concerns and promotes maintainability:

### Layers

1. **API Layer**: FastAPI application with routers for different functionalities
2. **Service Layer**: Business logic for search, analysis, contradiction detection, etc.
3. **Repository Layer**: Data access layer for interacting with databases
4. **ML Layer**: Machine learning models for analysis and synthesis
5. **External Clients**: Interfaces to external APIs and data sources

### Key Technologies

- **FastAPI**: Modern asynchronous web framework
- **SQLAlchemy**: ORM for database interactions
- **Redis**: Distributed caching and task queue backend
- **Ray**: Distributed computing framework for ML tasks
- **Dramatiq**: Task queue for background processing
- **Grafana LGTM Stack**: Observability (Loki, Grafana, Tempo, Mimir/Prometheus)

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        Client Applications                       │
└───────────────────────────────┬─────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────┐
│                           API Layer                              │
│                                                                  │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────┐ │
│  │  Search  │ │ Analysis │ │Screening │ │Contradict│ │  Auth  │ │
│  │  Router  │ │  Router  │ │  Router  │ │  Router  │ │ Router │ │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ └───┬────┘ │
└───────┼───────────┼───────────┼───────────┼───────────┼─────────┘
        │           │           │           │           │
┌───────▼───────────▼───────────▼───────────▼───────────▼─────────┐
│                         Service Layer                            │
│                                                                  │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────┐ │
│  │  Search  │ │ Analysis │ │Screening │ │Contradict│ │  Auth  │ │
│  │ Service  │ │ Service  │ │ Service  │ │ Service  │ │Service │ │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ └───┬────┘ │
└───────┼───────────┼───────────┼───────────┼───────────┼─────────┘
        │           │           │           │           │
┌───────▼───────────▼───────────▼───────────▼───────────▼─────────┐
│                       Repository Layer                           │
│                                                                  │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────┐ │
│  │  Search  │ │ Analysis │ │Screening │ │Contradict│ │  User  │ │
│  │   Repo   │ │   Repo   │ │   Repo   │ │   Repo   │ │  Repo  │ │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ └───┬────┘ │
└───────┼───────────┼───────────┼───────────┼───────────┼─────────┘
        │           │           │           │           │
┌───────▼───────────▼───────────▼───────────▼───────────▼─────────┐
│                        Database Layer                            │
│                                                                  │
│  ┌──────────────────────┐    ┌───────────────────────────────┐  │
│  │      PostgreSQL      │    │             Redis             │  │
│  │  (Primary Storage)   │    │  (Cache & Task Queue)         │  │
│  └──────────────────────┘    └───────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
        │           │           │           │           │
┌───────▼───────────▼───────────▼───────────▼───────────▼─────────┐
│                          ML Layer                                │
│                                                                  │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────┐ │
│  │ BioMedLM │ │ TSMixer  │ │  SHAP    │ │ Lorentz  │ │ Other  │ │
│  │  Model   │ │  Model   │ │Explainer │ │Embeddings│ │ Models │ │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └────────┘ │
└─────────────────────────────────────────────────────────────────┘
        │           │           │           │           │
┌───────▼───────────▼───────────▼───────────▼───────────▼─────────┐
│                      External Data Sources                       │
│                                                                  │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────┐ │
│  │  PubMed  │ │ Clinical │ │ Medical  │ │ Journal  │ │ Other  │ │
│  │   API    │ │ Trials   │ │Guidelines│ │  Data    │ │ APIs   │ │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Development Environment Setup

### Prerequisites

- Docker and Docker Compose
- Python 3.10+
- Git
- NVIDIA GPU (optional, for faster model inference)

### Docker Development Environment (Recommended)

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/asf.git
   cd asf
   ```

2. Start the development environment using Docker Compose:
   ```bash
   cd asf/medical
   docker-compose up -d
   ```

   This will start:
   - FastAPI application on http://localhost:8000
   - PostgreSQL database on localhost:5432
   - Redis on localhost:6379
   - Prometheus on http://localhost:9090
   - Grafana on http://localhost:3000 (admin/admin)

3. Access the API documentation:
   - Swagger UI: http://localhost:8000/docs
   - ReDoc: http://localhost:8000/redoc

### Manual Installation (Alternative)

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/asf.git
   cd asf
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file:
   ```bash
   cp asf/medical/.env.example asf/medical/.env
   ```
   Edit the `.env` file to configure your environment.

5. Initialize the database:
   ```bash
   python -m asf.medical.scripts.init_db
   ```

### Running the API

```bash
python -m asf.medical.scripts.run_api
```

The API will be available at http://localhost:8000. You can access the API documentation at http://localhost:8000/docs.

## Project Structure

```
asf/medical/
├── api/                  # FastAPI application
│   ├── main.py           # Main application entry point
│   ├── middleware/       # Custom middleware
│   ├── models/           # Pydantic models for API
│   ├── routers/          # API route definitions
│   └── dependencies.py   # Dependency injection
├── core/                 # Core modules
│   ├── config.py         # Application configuration
│   ├── cache.py          # Caching functionality
│   ├── logging_config.py # Logging configuration
│   ├── monitoring.py     # Monitoring and metrics
│   ├── security.py       # Security utilities
│   └── exceptions.py     # Custom exceptions
├── ml/                   # Machine learning components
│   ├── models/           # ML model implementations
│   ├── services/         # ML services
│   ├── model_cache.py    # Model caching
│   └── model_registry.py # Model registry
├── services/             # Business logic services
│   ├── search_service.py # Search service
│   ├── analysis_service.py # Analysis service
│   └── knowledge_base_service.py # Knowledge base service
├── storage/              # Data storage
│   ├── database.py       # Database connection
│   ├── models/           # SQLAlchemy models
│   └── repositories/     # Data access repositories
├── orchestration/        # Orchestration components
│   ├── ray_manager.py    # Ray cluster management
│   ├── task_scheduler.py # Task scheduling
│   └── kb_updater.py     # Knowledge base updater
├── data_ingestion_layer/ # Data ingestion components
│   └── enhanced_medical_research_synthesizer.py # Data ingestion
├── scripts/              # Utility scripts
│   ├── run_api.py        # Run API script
│   ├── run_app.py        # Run full application
│   └── init_db.py        # Initialize database
├── tests/                # Test suite
│   ├── unit/             # Unit tests
│   ├── integration/      # Integration tests
│   └── performance/      # Performance tests
├── docker/               # Docker configuration
│   └── observability/    # Observability stack config
├── logs/                 # Log files
├── Dockerfile            # Docker image definition
├── docker-compose.yml    # Docker Compose configuration
└── requirements.txt      # Python dependencies
```

## Key Components

### API Layer

The API layer is built with FastAPI and provides endpoints for interacting with the system. It's organized into routers for different functionalities:

- **Auth Router**: Authentication and user management
- **Search Router**: Medical literature search
- **Analysis Router**: Analysis of medical literature
- **Knowledge Base Router**: Knowledge base management
- **Export Router**: Export functionality for results
- **Screening Router**: PRISMA-guided screening and bias assessment
- **Enhanced Contradiction Router**: Enhanced contradiction detection
- **Contradiction Resolution Router**: Contradiction resolution strategies

### Service Layer

The service layer contains the business logic for the application. It's organized into services for different functionalities:

- **Search Service**: Search functionality
- **Analysis Service**: Analysis functionality
- **Knowledge Base Service**: Knowledge base management
- **Enhanced Contradiction Service**: Enhanced contradiction detection
- **PRISMA Screening Service**: PRISMA-guided screening
- **Bias Assessment Service**: Bias assessment for medical studies

### Repository Layer

The repository layer provides data access functionality. It's organized into repositories for different data types:

- **Search Repository**: Search data
- **Analysis Repository**: Analysis data
- **Knowledge Base Repository**: Knowledge base data
- **User Repository**: User data

### ML Layer

The ML layer contains machine learning models and services:

- **Model Registry**: Central registry for ML models
- **Model Cache**: Caching for ML models
- **BioMedLM**: Biomedical language model
- **TSMixer**: Temporal model
- **SHAP Explainer**: Explainability for ML models
- **Lorentz Embeddings**: Hyperbolic embeddings

### External Clients

The external clients provide interfaces to external APIs and data sources:

- **PubMed Client**: Interface to PubMed API
- **Clinical Trials Client**: Interface to ClinicalTrials.gov API
- **Journal Impact Factor Client**: Interface to journal impact factor data

## Development Workflow

### Git Workflow

1. Create a feature branch from `develop`:
   ```bash
   git checkout develop
   git pull
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and commit them:
   ```bash
   git add .
   git commit -m "Add your feature"
   ```

3. Push your branch to the remote repository:
   ```bash
   git push -u origin feature/your-feature-name
   ```

4. Create a pull request to merge your branch into `develop`.

### Code Style

We follow the PEP 8 style guide for Python code. You can use the following tools to ensure your code follows the style guide:

- **Black**: Code formatter
- **Flake8**: Linter
- **isort**: Import sorter

```bash
# Format code
black asf/medical

# Check imports
isort asf/medical

# Run linting
flake8 asf/medical
```

### Documentation

- Use docstrings for all modules, classes, and functions.
- Follow the Google style for docstrings.
- Keep the API documentation up to date.

## Testing

### Running Tests

Run the test suite using pytest:

```bash
# Inside the Docker container
docker exec -it asf-medical-api-1 pytest

# Or locally with a running database and Redis
pytest
```

Generate a coverage report:

```bash
pytest --cov=asf/medical --cov-report=html
```

### Test Types

- **Unit Tests**: Test individual components in isolation.
- **Integration Tests**: Test interactions between components.
- **Performance Tests**: Test performance and resource usage.

## Logging and Monitoring

### Logging

The application uses a centralized logging configuration in `core/logging_config.py`. It provides:

- Console logging for development
- File logging for production
- Structured logging with context
- Log rotation

To get a logger:

```python
from asf.medical.core.logging_config import get_logger

logger = get_logger(__name__)
logger.info("This is an info message")
logger.error("This is an error message")
```

### Monitoring

The application includes comprehensive monitoring and observability features:

- **Metrics**: Counters, gauges, histograms, and timers
- **Health Checks**: System and application health checks
- **Request Logging**: HTTP request logging with duration and status code
- **Error Tracking**: Error tracking with context

To use monitoring features:

```python
from asf.medical.core.monitoring import (
    increment_counter, set_gauge, record_histogram,
    start_timer, stop_timer, timer, timed, async_timed
)

# Increment a counter
increment_counter("requests", 1, {"endpoint": "/search"})

# Set a gauge
set_gauge("memory_usage", 1024, {"component": "model_registry"})

# Record a histogram value
record_histogram("request_duration", 0.123, {"endpoint": "/search"})

# Use a timer decorator
@async_timed("search_execution")
async def search():
    # ... do something
```

## API Documentation

The API documentation is available at:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- OpenAPI JSON: http://localhost:8000/openapi.json

To ensure your endpoints are properly documented:

1. Use descriptive docstrings for all endpoint functions.
2. Use Pydantic models for request and response schemas.
3. Use appropriate tags for grouping endpoints.

## Deployment

### Docker Deployment

The application can be deployed using Docker and Docker Compose:

```bash
# Build the Docker image
docker build -t asf-medical .

# Run the Docker container
docker run -p 8000:8000 asf-medical
```

### Kubernetes Deployment

For production deployments, we recommend using Kubernetes:

1. Build and push the Docker image to a container registry.
2. Apply the Kubernetes manifests in the `kubernetes` directory.

### CI/CD Pipeline

The CI/CD pipeline is configured using GitHub Actions:

1. On push to a feature branch, run tests and linting.
2. On merge to `develop`, build and push the Docker image to the staging environment.
3. On merge to `main`, build and push the Docker image to the production environment.

## Troubleshooting

### Common Issues

#### API not starting

- Check if the database is running and accessible.
- Check if Redis is running and accessible.
- Check the logs for error messages.

#### Tests failing

- Check if the database is running and accessible.
- Check if Redis is running and accessible.
- Check if the test database is properly configured.

#### Models not loading

- Check if the model files are available in the correct location.
- Check if the GPU is available and configured correctly (if using GPU).
- Check the logs for error messages.

### Getting Help

If you encounter issues not covered in this guide, please:

1. Check the logs for error messages.
2. Search the issue tracker for similar issues.
3. Ask for help in the team chat.
4. Create a new issue in the issue tracker.

## Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [SQLAlchemy Documentation](https://docs.sqlalchemy.org/)
- [Redis Documentation](https://redis.io/documentation)
- [Ray Documentation](https://docs.ray.io/)
- [Dramatiq Documentation](https://dramatiq.io/)
- [Grafana Documentation](https://grafana.com/docs/)
- [Prometheus Documentation](https://prometheus.io/docs/)
