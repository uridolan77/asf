# BioMedLM and Memgraph Integration

This document describes the integration of BioMedLM for contradiction scoring and Memgraph as an alternative to Neo4j in the ASF project.

## BioMedLM Integration

BioMedLM is a biomedical language model developed by Microsoft that can be used for various NLP tasks in the biomedical domain. In this integration, we use BioMedLM for scoring contradictions between medical claims.

### Components

1. **BioMedLMScorer**: A wrapper class for BioMedLM that provides methods for scoring contradictions between medical claims.
   - Location: `asf/medical/models/biomedlm_wrapper.py`
   - Methods:
     - `get_score(claim1, claim2)`: Returns a contradiction score between 0 and 1
     - `get_detailed_scores(claim1, claim2)`: Returns detailed scores including contradiction score, agreement score, and confidence

2. **Integration with PublicationMetadataExtractor**: The `find_contradictions` method in `PublicationMetadataExtractor` has been updated to use BioMedLM for contradiction scoring.
   - Location: `asf/medical/data_ingestion_layer/metadata_extraction.py`
   - Changes:
     - Added `use_biomedlm` parameter to control whether to use BioMedLM
     - Added `threshold` parameter to set the contradiction score threshold
     - Added fallback to keyword-based approach if BioMedLM is not available

3. **API Updates**: The API has been updated to expose BioMedLM-based contradiction scoring.
   - Location: `asf/medical/api/main.py`
   - Changes:
     - Added `ContradictionAnalysisRequest` model with `use_biomedlm` and `threshold` parameters
     - Updated `ContradictionAnalysisResponse` model to include `detection_method`
     - Updated `analyze_contradictions` endpoint to support BioMedLM

### Usage

To use BioMedLM for contradiction scoring, make a POST request to the `/analyze-contradictions` endpoint with the following parameters:

```json
{
  "query": "aspirin headache",
  "max_results": 20,
  "use_biomedlm": true,
  "threshold": 0.7
}
```

The response will include a `detection_method` field indicating whether BioMedLM or the keyword-based approach was used.

## Memgraph Integration

Memgraph is a graph database that can be used as an alternative to Neo4j in the ChronoGnosisLayer. This integration allows switching between Neo4j and Memgraph based on configuration.

### Components

1. **MemgraphManager**: A manager class for Memgraph database operations.
   - Location: `asf/layer1_knowledge_substrate/memgraph_manager.py`
   - Methods:
     - `connect()`: Establish connection to Memgraph
     - `close()`: Close connection to Memgraph
     - `run_query(query, params)`: Run a Cypher query on Memgraph
     - `fetch_subgraph(entity_id, hops)`: Fetch a subgraph around an entity
     - `get_all_entity_ids()`: Get all entity IDs from the database
     - `create_entity(entity_data)`: Create a new entity in the database
     - `create_relationship(source_id, target_id, rel_type, properties)`: Create a relationship between two entities

2. **Integration with DatabaseManager**: The `DatabaseManager` class in `ChronoGnosisLayer` has been updated to support Memgraph.
   - Location: `asf/layer1_knowledge_substrate/chronograph_gnosis_layer.py`
   - Changes:
     - Added `use_memgraph` parameter to control whether to use Memgraph
     - Added `memgraph_config` parameter for Memgraph configuration
     - Updated methods to use Memgraph if `use_memgraph` is `True`

### Configuration

To use Memgraph instead of Neo4j, set the `use_memgraph` parameter to `True` in the `GnosisConfig`:

```python
config = GnosisConfig(
    use_memgraph=True,
    memgraph=MemgraphConfigModel(
        host="localhost",
        port=7687,
        username="",
        password=""
    )
)
```

## Testing

A test script is provided to verify the integration of BioMedLM and Memgraph:

- Location: `asf/tests/test_biomedlm_memgraph.py`
- Tests:
  - `test_biomedlm()`: Test BioMedLM contradiction scoring
  - `test_memgraph()`: Test Memgraph integration
  - `test_chronognosis_with_memgraph()`: Test ChronoGnosisLayer with Memgraph

To run the tests:

```bash
python -m asf.tests.test_biomedlm_memgraph
```

## Requirements

- BioMedLM integration requires the `transformers` and `torch` packages
- Memgraph integration requires the `mgclient` package

Install the required packages:

```bash
pip install transformers torch mgclient
```
