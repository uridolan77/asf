# Medical Research Synthesizer

A comprehensive platform for searching, analyzing, and synthesizing medical research literature.

## Features

- **Advanced Search**: Search PubMed and other medical databases with powerful query capabilities
- **PICO Framework**: Build structured queries using the Population, Intervention, Comparison, Outcome framework
- **Contradiction Detection**: Identify and analyze contradictions in medical literature
- **Knowledge Base Management**: Create and maintain topic-specific knowledge bases with scheduled updates
- **Export Capabilities**: Export results in various formats (JSON, CSV, Excel, PDF)
- **Temporal Analysis**: Analyze how medical knowledge evolves over time
- **GraphRAG**: Graph-based retrieval-augmented generation for enhanced analysis

## Architecture

The Medical Research Synthesizer is built with a modular architecture:

- **API Layer**: FastAPI-based RESTful API
- **Data Ingestion Layer**: Connectors for PubMed, UMLS, ClinicalTrials.gov
- **Storage Layer**: PostgreSQL database with SQLAlchemy ORM
- **Models Layer**: BioMedLM, TSMixer, Lorentz embeddings for contradiction detection
- **Orchestration Layer**: Ray for distributed task processing
- **Graph Layer**: Memgraph/Neo4j for knowledge graph storage

## Getting Started

### Prerequisites

- Docker and Docker Compose
- Python 3.10+ (for local development without Docker)
- Git
- NVIDIA GPU (optional, for faster model inference)

### Docker Development Environment (Recommended)

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/medical-research-synthesizer.git
   cd medical-research-synthesizer
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
   git clone https://github.com/yourusername/medical-research-synthesizer.git
   cd medical-research-synthesizer
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

### Running the API (Manual Installation)

```bash
python -m asf.medical.scripts.run_api
```

The API will be available at http://localhost:8000. You can access the API documentation at http://localhost:8000/docs.

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

### Code Style and Linting

Format code using Black:

```bash
black asf/medical
```

Check imports with isort:

```bash
isort asf/medical
```

Run linting with flake8:

```bash
flake8 asf/medical
```

## API Endpoints

### Authentication

- `POST /token`: Get an access token
- `POST /register`: Register a new user (admin only)

### Search

- `POST /v1/search`: Search medical literature
- `POST /v1/search/pico`: Search using PICO framework
- `GET /v1/search/template/{template_id}`: Get a query from a template

### Analysis

- `POST /v1/analysis/contradictions`: Analyze contradictions in literature
- `GET /v1/analysis/cap`: Analyze Community-Acquired Pneumonia literature
- `GET /v1/analysis/cap/detailed`: Detailed CAP analysis

### Knowledge Base

- `POST /v1/knowledge-base`: Create a knowledge base
- `GET /v1/knowledge-base/{name}`: Get articles from a knowledge base
- `GET /v1/knowledge-base`: List all knowledge bases
- `POST /v1/knowledge-base/{name}/update`: Update a knowledge base

### Export

- `POST /v1/export/{format}`: Export results in various formats (json, csv, excel, pdf)

## License

This project is licensed under the MIT License - see the LICENSE file for details.
