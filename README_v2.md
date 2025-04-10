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

- Python 3.9+
- PostgreSQL (optional, SQLite is used by default)
- Memgraph or Neo4j (optional)
- NVIDIA GPU (optional, for faster model inference)

### Installation

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
   pip install -r asf/medical/requirements.txt
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

### Running with Docker

1. Build and start the containers:
   ```bash
   docker-compose up -d
   ```

2. Initialize the database:
   ```bash
   docker-compose exec api python -m asf.medical.scripts.init_db
   ```

The API will be available at http://localhost:8000.

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

## Development

### Running Tests

```bash
python -m asf.medical.scripts.run_tests
```

### Generating Documentation

```bash
python -m asf.medical.scripts.generate_docs
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
