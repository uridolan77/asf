# NCBI Client Enhancements

## Overview
The NCBI client has been enhanced with additional methods for working with the History server, batch operations with large datasets, and advanced search features.

## History Server Methods

### Create History Session
```python
async def create_history_session()
```
Create a new History server session.

**Returns:**
- Dictionary with WebEnv and query_key

### Post IDs to History
```python
async def post_ids_to_history(ids, db, web_env=None)
```
Post a list of IDs to the History server.

**Parameters:**
- `ids`: UID or list of UIDs to post
- `db`: Database containing the UIDs
- `web_env`: Existing WebEnv to append to (optional)

**Returns:**
- Dictionary with WebEnv and query_key

### Search and Post to History
```python
async def search_and_post_to_history(query, db="pubmed", web_env=None, query_key=None, **search_params)
```
Search a database and post results to the History server.

**Parameters:**
- `query`: Search query
- `db`: Database to search (default: "pubmed")
- `web_env`: Existing WebEnv to append to (optional)
- `query_key`: Existing query_key to use (optional)
- `**search_params`: Additional search parameters

**Returns:**
- Dictionary with search results, WebEnv, and query_key

### Fetch from History
```python
async def fetch_from_history(web_env, query_key, db, retstart=0, retmax=20, rettype=None, retmode=None)
```
Fetch records from the History server.

**Parameters:**
- `web_env`: WebEnv string
- `query_key`: Query key
- `db`: Database to fetch from
- `retstart`: First record to retrieve (default: 0)
- `retmax`: Maximum number of records to retrieve (default: 20)
- `rettype`: Retrieval type (e.g., "abstract", "medline", "gb")
- `retmode`: Retrieval mode (e.g., "text", "xml", "json")

**Returns:**
- Fetched records in the specified format

### Get Summary from History
```python
async def get_summary_from_history(web_env, query_key, db, retstart=0, retmax=20, version="2.0")
```
Get document summaries from the History server.

**Parameters:**
- `web_env`: WebEnv string
- `query_key`: Query key
- `db`: Database to fetch from
- `retstart`: First record to retrieve (default: 0)
- `retmax`: Maximum number of records to retrieve (default: 20)
- `version`: ESummary version (default: "2.0")

**Returns:**
- Document summaries

## Batch Operations Methods

### Batch Fetch Articles
```python
async def batch_fetch_articles(pmids, batch_size=200, include_abstracts=True, max_workers=5)
```
Fetch article details for a large list of PMIDs in batches.

**Parameters:**
- `pmids`: List of PMIDs to fetch
- `batch_size`: Number of PMIDs per batch (default: 200)
- `include_abstracts`: Whether to include abstracts (default: True)
- `max_workers`: Maximum number of concurrent workers (default: 5)

**Returns:**
- List of article details

### Batch Search and Fetch
```python
async def batch_search_and_fetch(query, db="pubmed", max_results=1000, batch_size=200, max_workers=5, **search_params)
```
Search a database and fetch results in batches.

**Parameters:**
- `query`: Search query
- `db`: Database to search (default: "pubmed")
- `max_results`: Maximum number of results to fetch (default: 1000)
- `batch_size`: Number of records per batch (default: 200)
- `max_workers`: Maximum number of concurrent workers (default: 5)
- `**search_params`: Additional search parameters

**Returns:**
- Dictionary with search results and fetched records

### Batch Fetch Sequences
```python
async def batch_fetch_sequences(ids, db="nucleotide", batch_size=50, return_type="fasta", return_mode="text", max_workers=3)
```
Fetch sequences for a large list of IDs in batches.

**Parameters:**
- `ids`: List of sequence IDs to fetch
- `db`: Database to fetch from (default: "nucleotide")
- `batch_size`: Number of IDs per batch (default: 50)
- `return_type`: Return type (default: "fasta")
- `return_mode`: Return mode (default: "text")
- `max_workers`: Maximum number of concurrent workers (default: 3)

**Returns:**
- Dictionary with fetched sequences

## Advanced Search Methods

### Advanced Search
```python
async def advanced_search(db="pubmed", **search_criteria)
```
Perform an advanced search with multiple criteria.

**Parameters:**
- `db`: Database to search (default: "pubmed")
- `**search_criteria`: Search criteria as keyword arguments
  - For PubMed: title, abstract, author, journal, publication_date, publication_type, mesh_terms, keywords, affiliation, doi, free_text, filters
  - For sequence databases: organism, gene, protein, sequence_length, molecule_type, source, free_text

**Returns:**
- Dictionary with search results

### Date Range Search
```python
async def date_range_search(query, db="pubmed", date_type="pdat", start_date=None, end_date=None, relative_date=None, **search_params)
```
Search with date range constraints.

**Parameters:**
- `query`: Search query
- `db`: Database to search (default: "pubmed")
- `date_type`: Type of date (default: "pdat")
- `start_date`: Start date in YYYY/MM/DD format (optional)
- `end_date`: End date in YYYY/MM/DD format (optional)
- `relative_date`: Relative date in days (optional)
- `**search_params`: Additional search parameters

**Returns:**
- Dictionary with search results

### Field Search
```python
async def field_search(terms, db="pubmed", operator="AND", **search_params)
```
Search with field-specific terms.

**Parameters:**
- `terms`: Dictionary mapping fields to search terms
- `db`: Database to search (default: "pubmed")
- `operator`: Operator to combine terms (default: "AND")
- `**search_params`: Additional search parameters

**Returns:**
- Dictionary with search results

### Proximity Search
```python
async def proximity_search(terms, field="Title/Abstract", distance=5, **search_params)
```
Perform a proximity search in PubMed.

**Parameters:**
- `terms`: List of terms to search for
- `field`: Field to search in (default: "Title/Abstract")
- `distance`: Maximum distance between terms (default: 5)
- `**search_params`: Additional search parameters

**Returns:**
- Dictionary with search results

## Usage Examples

### History Server Methods
```python
# Create a new History session
session = await ncbi_client.create_history_session()
web_env = session["web_env"]
query_key = session["query_key"]

# Post IDs to History
result = await ncbi_client.post_ids_to_history(["12345", "67890"], "pubmed", web_env)
new_query_key = result["query_key"]

# Search and post to History
search_result = await ncbi_client.search_and_post_to_history("cancer", "pubmed", web_env)
search_query_key = search_result["query_key"]

# Fetch from History
articles = await ncbi_client.fetch_from_history(web_env, search_query_key, "pubmed", retmax=10, rettype="medline", retmode="text")

# Get summary from History
summaries = await ncbi_client.get_summary_from_history(web_env, search_query_key, "pubmed", retmax=10)
```

### Batch Operations Methods
```python
# Batch fetch articles
articles = await ncbi_client.batch_fetch_articles(["12345", "67890", "54321"], batch_size=2, max_workers=2)

# Batch search and fetch
results = await ncbi_client.batch_search_and_fetch("cancer", max_results=100, batch_size=20)

# Batch fetch sequences
sequences = await ncbi_client.batch_fetch_sequences(["NM_000546", "NM_001126114"], db="nucleotide")
```

### Advanced Search Methods
```python
# Advanced search
results = await ncbi_client.advanced_search(
    title="cancer",
    author="Smith J",
    publication_date="2020:2023",
    filters=["free full text"]
)

# Date range search
results = await ncbi_client.date_range_search(
    "cancer",
    start_date="2020/01/01",
    end_date="2023/12/31"
)

# Field search
results = await ncbi_client.field_search({
    "Title": "cancer",
    "Author": "Smith J"
}, operator="AND")

# Proximity search
results = await ncbi_client.proximity_search(
    ["cancer", "treatment", "novel"],
    field="Title",
    distance=5
)
```
