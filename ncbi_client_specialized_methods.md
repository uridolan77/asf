# NCBI Client Specialized Methods

## Overview
The NCBI client has been enhanced with specialized methods for specific NCBI E-utilities endpoints. These methods provide higher-level functionality for common tasks and make it easier to work with the NCBI API.

## E-utilities Support

### EInfo Methods
- `get_database_info(db=None, version="2.0")` - Get information about NCBI databases or a specific database

### ESpell Methods
- `get_spelling_suggestions(term, db="pubmed")` - Get spelling suggestions for a search term

### EGQuery Methods
- `search_all_databases(term)` - Search across all Entrez databases with a single query

### ECitMatch Methods
- `match_citations(citations)` - Match citation strings to PubMed IDs

### ELink Methods
- `get_links(ids, dbfrom, db, linkname=None, cmd="neighbor")` - Get links between database records
- `get_links_and_post_to_history(ids, dbfrom, db, linkname=None)` - Get links and post results to History server
- `get_links_with_scores(ids, dbfrom, db, linkname=None)` - Get links with similarity scores
- `check_links(ids, dbfrom, db=None)` - Check for the existence of links
- `get_linkout_urls(ids, dbfrom, include_libraries=False)` - Get LinkOut URLs for a set of UIDs
- `get_fulltext_url(pmid)` - Get the full-text URL for a PubMed article

## PubMed-Specific Methods

### Related Articles
```python
async def get_related_articles(pmid, max_results=20)
```
Get related articles for a PubMed article.

**Parameters:**
- `pmid`: PubMed ID to find related articles for
- `max_results`: Maximum number of related articles to return (default: 20)

**Returns:**
- List of related articles with details

### Citing Articles
```python
async def get_citing_articles(pmid, max_results=20)
```
Get articles that cite a PubMed article.

**Parameters:**
- `pmid`: PubMed ID to find citing articles for
- `max_results`: Maximum number of citing articles to return (default: 20)

**Returns:**
- List of citing articles with details

### MeSH Terms
```python
async def get_mesh_terms(pmid)
```
Get MeSH terms for a PubMed article.

**Parameters:**
- `pmid`: PubMed ID to get MeSH terms for

**Returns:**
- List of MeSH terms with their qualifiers

### Journal Information
```python
async def get_journal_info(journal)
```
Get information about a journal from the NLM Catalog.

**Parameters:**
- `journal`: Journal title or abbreviation

**Returns:**
- Dictionary with journal information

## Sequence Database Methods

### Search Sequence Database
```python
async def search_sequence_database(query, db="nucleotide", max_results=20, return_type="gb", return_mode="text")
```
Search a sequence database and retrieve sequences.

**Parameters:**
- `query`: Search query
- `db`: Database to search (default: "nucleotide")
  - Options include: "nucleotide", "protein", "genome", "gene"
- `max_results`: Maximum number of results to return (default: 20)
- `return_type`: Return type (default: "gb")
  - Options for nucleotide: "gb" (GenBank), "fasta", "gbwithparts", "gbc"
  - Options for protein: "gp" (GenPept), "fasta", "gpc"
- `return_mode`: Return mode (default: "text")
  - Options: "text", "xml", "asn.1"

**Returns:**
- Dictionary with search results and sequences

### Fetch Sequence
```python
async def fetch_sequence(id, db="nucleotide", return_type="gb", return_mode="text", strand=None, seq_start=None, seq_stop=None)
```
Fetch a sequence from a sequence database.

**Parameters:**
- `id`: Sequence ID (GI number or accession)
- `db`: Database to fetch from (default: "nucleotide")
  - Options include: "nucleotide", "protein", "genome", "gene"
- `return_type`: Return type (default: "gb")
  - Options for nucleotide: "gb" (GenBank), "fasta", "gbwithparts", "gbc"
  - Options for protein: "gp" (GenPept), "fasta", "gpc"
- `return_mode`: Return mode (default: "text")
  - Options: "text", "xml", "asn.1"
- `strand`: Strand of DNA to retrieve (1 for plus, 2 for minus)
- `seq_start`: First sequence base to retrieve (1-based)
- `seq_stop`: Last sequence base to retrieve (1-based)

**Returns:**
- Sequence in the specified format

### Get Taxonomy
```python
async def get_taxonomy(id)
```
Get taxonomy information for a taxon ID or organism name.

**Parameters:**
- `id`: Taxonomy ID or organism name

**Returns:**
- Dictionary with taxonomy information

## Usage Examples

### Get Database Information
```python
# Get information about all databases
db_info = await ncbi_client.get_database_info()

# Get information about the PubMed database
pubmed_info = await ncbi_client.get_database_info(db="pubmed")
```

### Get Spelling Suggestions
```python
# Get spelling suggestions for a search term
suggestions = await ncbi_client.get_spelling_suggestions("asthmaa")
```

### Search All Databases
```python
# Search all databases for a term
results = await ncbi_client.search_all_databases("cancer")
```

### Match Citations
```python
# Match citation strings to PubMed IDs
citations = [
    {
        "journal": "science",
        "year": "1987",
        "volume": "235",
        "first_page": "182",
        "author": "palmenberg ac",
        "key": "citation1"
    }
]
matches = await ncbi_client.match_citations(citations)
```

### Get Links Between Databases
```python
# Get links from protein to gene
links = await ncbi_client.get_links("15718680", "protein", "gene")

# Get links with scores
links_with_scores = await ncbi_client.get_links_with_scores("19880848", "pubmed", "pubmed")

# Check for the existence of links
link_check = await ncbi_client.check_links("15718680", "protein")

# Get LinkOut URLs
linkout_urls = await ncbi_client.get_linkout_urls("19880848", "pubmed")

# Get full-text URL
fulltext_url = await ncbi_client.get_fulltext_url("19880848")
```

### PubMed-Specific Methods
```python
# Get related articles
related = await ncbi_client.get_related_articles("19880848")

# Get citing articles
citing = await ncbi_client.get_citing_articles("19880848")

# Get MeSH terms
mesh_terms = await ncbi_client.get_mesh_terms("19880848")

# Get journal information
journal_info = await ncbi_client.get_journal_info("science")
```

### Sequence Database Methods
```python
# Search for sequences
sequences = await ncbi_client.search_sequence_database("insulin", db="protein")

# Fetch a sequence
sequence = await ncbi_client.fetch_sequence("NP_000207.2", db="protein", return_type="fasta")

# Get taxonomy information
taxonomy = await ncbi_client.get_taxonomy("9606")  # Human
```
