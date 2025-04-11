# NCBI Client PMC Methods

## Overview
The NCBI client has been enhanced with methods for working with the PubMed Central (PMC) database, which provides free full-text access to biomedical and life sciences journal literature.

## Basic PMC Methods

### Search PMC
```python
async def search_pmc(query, max_results=20, sort="relevance", min_date=None, max_date=None, use_history=False)
```
Search PubMed Central (PMC) with the given query.

**Parameters:**
- `query`: Search query
- `max_results`: Maximum number of results to return (default: 20)
- `sort`: Sort order (default: "relevance")
- `min_date`: Minimum date (YYYY/MM/DD format)
- `max_date`: Maximum date (YYYY/MM/DD format)
- `use_history`: Whether to use the Entrez History server (default: False)

**Returns:**
- Dictionary with search results including PMCIDs and metadata

### Fetch PMC Article
```python
async def fetch_pmc_article(pmcid, format="xml")
```
Fetch a full-text article from PubMed Central (PMC).

**Parameters:**
- `pmcid`: PMC ID (e.g., "PMC1234567" or just "1234567")
- `format`: Format to retrieve (default: "xml")
  - Options: "xml", "medline", "pdf"

**Returns:**
- Full-text article in the specified format

### Extract PMC Article Sections
```python
async def extract_pmc_article_sections(pmcid)
```
Extract sections from a PMC article.

**Parameters:**
- `pmcid`: PMC ID (e.g., "PMC1234567" or just "1234567")

**Returns:**
- Dictionary with article sections including title, abstract, sections, and references

## ID Conversion Methods

### Convert PMID to PMCID
```python
async def convert_pmid_to_pmcid(pmids)
```
Convert PubMed IDs (PMIDs) to PubMed Central IDs (PMCIDs).

**Parameters:**
- `pmids`: PMID or list of PMIDs to convert

**Returns:**
- Dictionary mapping PMIDs to PMCIDs (None if no PMCID exists)

### Convert PMCID to PMID
```python
async def convert_pmcid_to_pmid(pmcids)
```
Convert PubMed Central IDs (PMCIDs) to PubMed IDs (PMIDs).

**Parameters:**
- `pmcids`: PMCID or list of PMCIDs to convert

**Returns:**
- Dictionary mapping PMCIDs to PMIDs (None if no PMID exists)

## Combined Methods

### Search and Fetch PMC
```python
async def search_and_fetch_pmc(query, max_results=20, include_full_text=False, **search_params)
```
Search PMC and fetch articles in one step.

**Parameters:**
- `query`: Search query
- `max_results`: Maximum number of results to return (default: 20)
- `include_full_text`: Whether to include full text (default: False)
- `**search_params`: Additional search parameters

**Returns:**
- List of articles with metadata and optionally full text

### Batch Fetch PMC Articles
```python
async def batch_fetch_pmc_articles(pmcids, include_full_text=False, batch_size=10, max_workers=3)
```
Fetch PMC articles in batches.

**Parameters:**
- `pmcids`: List of PMCIDs to fetch
- `include_full_text`: Whether to include full text (default: False)
- `batch_size`: Number of PMCIDs per batch (default: 10)
- `max_workers`: Maximum number of concurrent workers (default: 3)

**Returns:**
- List of articles with metadata and optionally full text

## Helper Methods

### _fetch_pmc_pdf
```python
async def _fetch_pmc_pdf(pmcid)
```
Fetch a PDF from PubMed Central (PMC).

**Parameters:**
- `pmcid`: PMC ID (with PMC prefix)

**Returns:**
- URL to the PDF file

### _get_element_text
```python
def _get_element_text(element)
```
Get the text content of an XML element, including its children.

**Parameters:**
- `element`: XML element

**Returns:**
- Text content of the element

### _parse_section
```python
def _parse_section(section_elem)
```
Parse a section element from a PMC article.

**Parameters:**
- `section_elem`: Section element

**Returns:**
- Dictionary with section title and content

### _parse_reference
```python
def _parse_reference(ref_elem)
```
Parse a reference element from a PMC article.

**Parameters:**
- `ref_elem`: Reference element

**Returns:**
- Dictionary with reference details

## Usage Examples

### Basic PMC Methods
```python
# Search PMC
results = await ncbi_client.search_pmc("cancer", max_results=10)

# Fetch PMC article
article_xml = await ncbi_client.fetch_pmc_article("PMC1234567", format="xml")

# Extract PMC article sections
article_sections = await ncbi_client.extract_pmc_article_sections("PMC1234567")
```

### ID Conversion Methods
```python
# Convert PMID to PMCID
pmid_to_pmcid = await ncbi_client.convert_pmid_to_pmcid(["12345", "67890"])

# Convert PMCID to PMID
pmcid_to_pmid = await ncbi_client.convert_pmcid_to_pmid(["PMC1234567", "PMC7654321"])
```

### Combined Methods
```python
# Search and fetch PMC
articles = await ncbi_client.search_and_fetch_pmc("cancer", max_results=10, include_full_text=True)

# Batch fetch PMC articles
articles = await ncbi_client.batch_fetch_pmc_articles(["PMC1234567", "PMC7654321"], include_full_text=True)
```

## Article Structure

When using `extract_pmc_article_sections` or `search_and_fetch_pmc` with `include_full_text=True`, the returned article structure is as follows:

```python
{
    "pmcid": "PMC1234567",
    "pmid": "12345678",  # If available
    "title": "Article Title",
    "abstract": "Article abstract...",
    "sections": [
        {
            "title": "Introduction",
            "content": "Introduction content...",
            "subsections": []
        },
        {
            "title": "Methods",
            "content": "Methods content...",
            "subsections": [
                {
                    "title": "Study Design",
                    "content": "Study design details...",
                    "subsections": []
                },
                {
                    "title": "Statistical Analysis",
                    "content": "Statistical analysis details...",
                    "subsections": []
                }
            ]
        },
        {
            "title": "Results",
            "content": "Results content...",
            "subsections": []
        },
        {
            "title": "Discussion",
            "content": "Discussion content...",
            "subsections": []
        }
    ],
    "references": [
        {
            "id": "ref1",
            "title": "Reference Title",
            "authors": ["Smith J", "Doe J"],
            "journal": "Journal Name",
            "year": "2020",
            "volume": "10",
            "issue": "2",
            "pages": "123-145",
            "doi": "10.1234/example",
            "pmid": "98765432"
        }
    ]
}
```
