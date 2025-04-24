# Medical Research Synthesizer API Client

This is a client library for interacting with the Medical Research Synthesizer API.

## Installation

```bash
pip install -e .
```

## Usage

```python
import asyncio
from asf.medical.client.api_client import MedicalResearchSynthesizerClient

async def main():
    # Create client
    client = MedicalResearchSynthesizerClient(
        base_url="http://localhost:8000",
        api_version="v1"
    )
    
    try:
        # Login
        login_response = await client.login("user@example.com", "password")
        
        if not login_response.success:
            print(f"Login failed: {login_response.message}")
            return
        
        print("Login successful!")
        
        # Search for medical literature
        search_response = await client.search("statin therapy cardiovascular", max_results=5)
        
        if not search_response.success:
            print(f"Search failed: {search_response.message}")
            return
        
        print(f"Search successful! Found {len(search_response.data.get('results', []))} results.")
        
    finally:
        # Close client
        await client.close()

if __name__ == "__main__":
    asyncio.run(main())
```

## Environment Variables

You can configure the client using environment variables:

- `API_BASE_URL`: Base URL of the API (default: `http://localhost:8000`)
- `API_VERSION`: API version (default: `v1`)
- `API_EMAIL`: Email for authentication
- `API_PASSWORD`: Password for authentication

## Available Methods

### Authentication

- `login(email, password)`: Log in to the API
- `get_current_user()`: Get the current user

### Search

- `search(query, max_results)`: Search for medical literature
- `search_pico(condition, interventions, outcomes, population, study_design, years, max_results)`: Search using the PICO framework

### Analysis

- `analyze_contradictions(query, max_results, threshold, use_biomedlm, use_tsmixer, use_lorentz)`: Analyze contradictions in medical literature
- `analyze_cap()`: Analyze Community-Acquired Pneumonia (CAP) literature

### Screening

- `screen_articles(query, max_results, stage, criteria)`: Screen articles according to PRISMA guidelines
- `assess_bias(query, max_results, domains)`: Assess risk of bias in articles

### Knowledge Base

- `create_knowledge_base(name, query, update_schedule)`: Create a new knowledge base
- `list_knowledge_bases()`: List all knowledge bases
- `get_knowledge_base(kb_id)`: Get a knowledge base by ID
- `update_knowledge_base(kb_id)`: Update a knowledge base
- `delete_knowledge_base(kb_id)`: Delete a knowledge base

### Export

- `export_results(format, result_id, query, max_results)`: Export search results

## Response Format

All methods return an `APIResponse` object with the following structure:

```python
class APIResponse(BaseModel):
    success: bool  # Whether the request was successful
    message: str  # Response message
    data: Optional[Any]  # Response data
    errors: Optional[List[Dict[str, Any]]]  # Error details
    meta: Optional[Dict[str, Any]]  # Metadata
```

## Example

See `example.py` for a complete example of using the client library.
