# Medical Research Synthesizer API Client Examples

This directory contains example scripts demonstrating how to use the Medical Research Synthesizer API client.

## Setup

1. Install the client library:
   ```bash
   pip install -e .
   ```

2. Create a `.env` file with your API credentials:
   ```
   API_BASE_URL=http://localhost:8000
   API_VERSION=v1
   API_EMAIL=user@example.com
   API_PASSWORD=password
   ```

## Examples

### Search Example

The `search_example.py` script demonstrates how to search for medical literature:

```bash
python -m asf.medical.client.examples.search_example
```

### Contradiction Analysis Example

The `contradiction_analysis_example.py` script demonstrates how to analyze contradictions in medical literature:

```bash
python -m asf.medical.client.examples.contradiction_analysis_example
```

### Knowledge Base Example

The `knowledge_base_example.py` script demonstrates how to create, list, update, and delete knowledge bases:

```bash
python -m asf.medical.client.examples.knowledge_base_example
```

### Screening Example

The `screening_example.py` script demonstrates how to screen articles according to PRISMA guidelines and assess risk of bias:

```bash
python -m asf.medical.client.examples.screening_example
```

### Export Example

The `export_example.py` script demonstrates how to export search results in various formats:

```bash
python -m asf.medical.client.examples.export_example
```

## Common Patterns

All examples follow these common patterns:

1. Initialize the client:
   ```python
   client = MedicalResearchSynthesizerClient(
       base_url=os.getenv("API_BASE_URL", "http://localhost:8000"),
       api_version=os.getenv("API_VERSION", "v1")
   )
   ```

2. Authenticate:
   ```python
   login_response = await client.login(email, password)
   if not login_response.success:
       logger.error(f"Login failed: {login_response.message}")
       return
   ```

3. Make API calls:
   ```python
   response = await client.search(query, max_results=max_results)
   if not response.success:
       logger.error(f"Search failed: {response.message}")
       return
   ```

4. Process results:
   ```python
   results = response.data.get("results", [])
   for result in results:
       logger.info(f"Title: {result.get('title')}")
   ```

5. Close the client:
   ```python
   await client.close()
   ```

## Error Handling

All examples include error handling to demonstrate how to handle errors from the API:

```python
try:
    # Make API calls
    response = await client.search(query)
    if not response.success:
        logger.error(f"Search failed: {response.message}")
        return
except Exception as e:
    logger.error(f"Error: {str(e)}")
finally:
    # Close client
    await client.close()
```

## Next Steps

After reviewing these examples, you can:

1. Adapt them to your specific use case
2. Combine multiple API calls to create more complex workflows
3. Integrate the client into your application

For more information, see the [client library documentation](../README.md) and the [migration guide](../MIGRATION_GUIDE.md).
