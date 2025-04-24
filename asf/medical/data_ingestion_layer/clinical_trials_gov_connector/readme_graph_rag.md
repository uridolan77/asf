# Medical GraphRAG: Graph-Based RAG for Medical Research

This project implements a Graph-Based Retrieval-Augmented Generation (GraphRAG) system specifically designed for medical research and clinical trials data. Unlike traditional RAG systems that rely solely on vector similarity search, GraphRAG enhances retrieval by leveraging entity relationships through graph traversal.

## Overview

GraphRAG combines two powerful approaches:

1. **Vector Similarity Search**: Uses embeddings to find semantically similar documents
2. **Graph Traversal**: Follows entity relationships to discover related documents

This provides several benefits over traditional RAG:

- Better contextual awareness through entity relationships
- Ability to discover documents that aren't directly relevant in vector space
- Higher precision with structured traversal of the knowledge graph
- Better handling of complex relationships in medical data

## Features

- **Automatic Document Processing**:
  - Smart chunking strategies for medical documents
  - Named entity recognition for medical concepts
  - Metadata extraction
  - Automated indexing with background monitoring

- **Graph-Based Retrieval**:
  - Entity-relationship tracking
  - Graph traversal with configurable depth
  - Combined vector + graph search

- **Query Processing**:
  - Query rewriting for better retrieval
  - Entity-aware search
  - Explanation of document connections

- **Clinical Trials Integration**:
  - Direct integration with ClinicalTrials.gov API
  - Automatic updates of clinical trial data
  - Entity extraction from trial descriptions, conditions, interventions, and eligibility criteria

## Architecture

The system consists of the following core components:

1. **MedicalEntityExtractor**: Extracts medical entities (conditions, treatments, medications, etc.) from text
2. **TextChunker**: Handles intelligent chunking of medical documents
3. **VectorStore**: Stores document embeddings and provides vector search
4. **GraphRetriever**: Augments vector search with graph traversal
5. **QueryRewriter**: Improves queries for better retrieval
6. **AutoIndexer**: Handles background indexing and updates
7. **ClinicalTrialsGraphRAG**: Main application integrating all components

## Getting Started

### Prerequisites

- Python 3.8+
- OpenAI API key (for embeddings and LLM)
- Required packages (install via `pip install -r requirements.txt`)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/medical-graphrag.git
cd medical-graphrag

# Install dependencies
pip install -r requirements.txt

# Set your OpenAI API key
export OPENAI_API_KEY="your-api-key-here"
```

### Basic Usage

```python
from medical_graphrag import ClinicalTrialsGraphRAG

# Initialize the GraphRAG system
graphrag = ClinicalTrialsGraphRAG(api_key="your-openai-api-key")

# Ingest a clinical trial
trial_data = {
    'nct_id': 'NCT12345678',
    'brief_title': 'Efficacy of Remdesivir for Treatment of COVID-19',
    'brief_summary': 'This study evaluates the efficacy and safety of remdesivir in hospitalized adults diagnosed with COVID-19 pneumonia.',
    'condition': ['COVID-19', 'Pneumonia'],
    'intervention': ['Remdesivir']
}
doc_id = graphrag.ingest_trial(trial_data)

# Search with graph-based retrieval
results = graphrag.search("COVID-19 antiviral treatment efficacy", use_graph=True)

# Generate a response using RAG
response = graphrag.generate_response("What is the efficacy of remdesivir for COVID-19?")
print(response['response'])
```

### Using the CLI

The project includes a command-line interface for easy interaction:

```bash
# Ingest clinical trials
python clinical_trials_client.py ingest --query "COVID-19 treatment" --max 50

# Search for trials
python clinical_trials_client.py search --query "What is the efficacy of remdesivir?"

# Ask a question
python clinical_trials_client.py answer --query "What are the common side effects of remdesivir?"

# Get system statistics
python clinical_trials_client.py stats

# Force reindexing
python clinical_trials_client.py reindex --source-id "api_query:COVID-19_treatment"
```

### Running the Demo

To see GraphRAG in action, run the demo script:

```bash
python demo-graphrag.py
```

This will:
1. Ingest clinical trial data for COVID-19 and diabetes
2. Compare vector search with graph-based search
3. Answer various medical questions using GraphRAG
4. Demonstrate the advantages of graph-based retrieval

## Comparison with Traditional RAG

Traditional RAG systems have several limitations when working with medical data:

1. **Missing Context**: Vector similarity doesn't capture entity relationships
2. **Narrow Focus**: Only finds documents directly related to the query
3. **Lost Connections**: Can't follow connections between medical concepts

GraphRAG addresses these limitations by:

1. **Leveraging Entity Relationships**: Connects documents through shared medical entities
2. **Graph Traversal**: Discovers relevant but non-obvious connections
3. **Structured Knowledge**: Uses medical entity types for better contextual understanding

## How it Works

When a document is ingested:

1. The text is broken into chunks
2. Medical entities are extracted from each chunk
3. The chunks are embedded and stored in the vector database
4. Entity relationships are mapped to documents

During retrieval:

1. The user query is rewritten for better retrieval
2. Initial documents are retrieved using vector search
3. The graph is traversed from these initial documents
4. Additional relevant documents are discovered through entity connections
5. Retrieved documents are used to generate a response

## Future Improvements

- Advanced query rewriting with better medical context
- Integration with UMLS for standardized medical concepts
- More sophisticated graph traversal algorithms
- Support for multi-hop reasoning in medical contexts
- Integration with structured medical databases
- Interactive graph visualization

## License

MIT License



# GraphRAG for Medical Research Dependencies

# Core dependencies
numpy>=1.20.0
pandas>=1.3.0
spacy>=3.5.0
langchain>=0.1.0
langchain-openai>=0.0.2
faiss-cpu>=1.7.0
openai>=1.0.0
transformers>=4.28.0

# Medical NLP
en_core_sci_md @ https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_core_sci_md-0.5.1.tar.gz
scikit-learn>=1.0.0

# Utilities
tqdm>=4.62.0
python-dotenv>=0.19.0
requests>=2.27.0
pydantic>=2.0.0