# Migration Guide: Updating Client Code to Use the Unified API Client

This guide provides instructions for migrating existing client code to use the new unified Medical Research Synthesizer API client.

## Overview

The new unified API client provides a consistent interface for interacting with the Medical Research Synthesizer API. It handles authentication, request formatting, error handling, and response parsing, making it easier to work with the API.

## Benefits of Migration

- **Consistent Interface**: The client provides a consistent interface for all API endpoints
- **Error Handling**: Comprehensive error handling with detailed error messages
- **Authentication**: Simplified authentication with automatic token management
- **Response Parsing**: Automatic parsing of API responses into structured objects
- **Async Support**: Full support for asynchronous operations
- **Type Hints**: Comprehensive type hints for better IDE support

## Migration Steps

### 1. Install the Client Library

```bash
pip install -e .
```

### 2. Import the Client

```python
from asf.medical.client.api_client import MedicalResearchSynthesizerClient
```

### 3. Initialize the Client

```python
client = MedicalResearchSynthesizerClient(
    base_url="http://localhost:8000",
    api_version="v1"
)
```

### 4. Use Environment Variables (Optional)

You can configure the client using environment variables:

```bash
export API_BASE_URL="http://localhost:8000"
export API_VERSION="v1"
export API_EMAIL="user@example.com"
export API_PASSWORD="password"
```

Then initialize the client without parameters:

```python
from dotenv import load_dotenv
load_dotenv()

client = MedicalResearchSynthesizerClient()
```

### 5. Authenticate

```python
login_response = await client.login("user@example.com", "password")

if not login_response.success:
    print(f"Login failed: {login_response.message}")
    return
```

### 6. Make API Calls

Replace direct HTTP requests with client methods:

#### Before:

```python
import requests

# Authentication
response = requests.post(
    "http://localhost:8000/v1/auth/token",
    data={"username": "user@example.com", "password": "password"}
)
token = response.json()["access_token"]

# Search
response = requests.post(
    "http://localhost:8000/v1/search",
    json={"query": "statin therapy", "max_results": 10},
    headers={"Authorization": f"Bearer {token}"}
)
results = response.json()["results"]
```

#### After:

```python
import asyncio
from asf.medical.client.api_client import MedicalResearchSynthesizerClient

async def main():
    client = MedicalResearchSynthesizerClient()
    
    try:
        # Authentication
        login_response = await client.login("user@example.com", "password")
        if not login_response.success:
            print(f"Login failed: {login_response.message}")
            return
        
        # Search
        search_response = await client.search("statin therapy", max_results=10)
        if not search_response.success:
            print(f"Search failed: {search_response.message}")
            return
        
        results = search_response.data["results"]
        
    finally:
        await client.close()

asyncio.run(main())
```

## Migration Examples

### Example 1: Simple Search Script

#### Before:

```python
import requests
import json

# Authentication
auth_response = requests.post(
    "http://localhost:8000/v1/auth/token",
    data={"username": "user@example.com", "password": "password"}
)
token = auth_response.json()["access_token"]

# Search
search_response = requests.post(
    "http://localhost:8000/v1/search",
    json={"query": "statin therapy", "max_results": 10},
    headers={"Authorization": f"Bearer {token}"}
)

# Print results
results = search_response.json()["results"]
print(f"Found {len(results)} results")
for result in results:
    print(f"- {result['title']}")
```

#### After:

```python
import asyncio
from asf.medical.client.api_client import MedicalResearchSynthesizerClient

async def main():
    client = MedicalResearchSynthesizerClient()
    
    try:
        # Authentication
        login_response = await client.login("user@example.com", "password")
        if not login_response.success:
            print(f"Login failed: {login_response.message}")
            return
        
        # Search
        search_response = await client.search("statin therapy", max_results=10)
        if not search_response.success:
            print(f"Search failed: {search_response.message}")
            return
        
        # Print results
        results = search_response.data["results"]
        print(f"Found {len(results)} results")
        for result in results:
            print(f"- {result['title']}")
        
    finally:
        await client.close()

if __name__ == "__main__":
    asyncio.run(main())
```

### Example 2: Contradiction Analysis

#### Before:

```python
import requests
import json

# Authentication
auth_response = requests.post(
    "http://localhost:8000/v1/auth/token",
    data={"username": "user@example.com", "password": "password"}
)
token = auth_response.json()["access_token"]

# Analyze contradictions
contradiction_response = requests.post(
    "http://localhost:8000/v1/analysis/contradictions",
    json={
        "query": "statin therapy cardiovascular",
        "max_results": 20,
        "threshold": 0.7,
        "use_biomedlm": True,
        "use_tsmixer": False,
        "use_lorentz": False
    },
    headers={"Authorization": f"Bearer {token}"}
)

# Print results
data = contradiction_response.json()["data"]
print(f"Found {data['contradictions_found']} contradictions")
for contradiction in data["contradictions"]:
    print(f"- {contradiction['contradiction_type']}: {contradiction['explanation']}")
```

#### After:

```python
import asyncio
from asf.medical.client.api_client import MedicalResearchSynthesizerClient

async def main():
    client = MedicalResearchSynthesizerClient()
    
    try:
        # Authentication
        login_response = await client.login("user@example.com", "password")
        if not login_response.success:
            print(f"Login failed: {login_response.message}")
            return
        
        # Analyze contradictions
        contradiction_response = await client.analyze_contradictions(
            query="statin therapy cardiovascular",
            max_results=20,
            threshold=0.7,
            use_biomedlm=True,
            use_tsmixer=False,
            use_lorentz=False
        )
        
        if not contradiction_response.success:
            print(f"Contradiction analysis failed: {contradiction_response.message}")
            return
        
        # Print results
        data = contradiction_response.data
        print(f"Found {data['contradictions_found']} contradictions")
        for contradiction in data["contradictions"]:
            print(f"- {contradiction['contradiction_type']}: {contradiction['explanation']}")
        
    finally:
        await client.close()

if __name__ == "__main__":
    asyncio.run(main())
```

### Example 3: Knowledge Base Management

#### Before:

```python
import requests
import json

# Authentication
auth_response = requests.post(
    "http://localhost:8000/v1/auth/token",
    data={"username": "user@example.com", "password": "password"}
)
token = auth_response.json()["access_token"]

# Create knowledge base
kb_response = requests.post(
    "http://localhost:8000/v1/knowledge-base",
    json={
        "name": "cardiovascular_statins",
        "query": "statin therapy cardiovascular",
        "update_schedule": "weekly"
    },
    headers={"Authorization": f"Bearer {token}"}
)

kb_id = kb_response.json()["data"]["kb_id"]
print(f"Created knowledge base with ID: {kb_id}")

# List knowledge bases
list_response = requests.get(
    "http://localhost:8000/v1/knowledge-base",
    headers={"Authorization": f"Bearer {token}"}
)

kbs = list_response.json()["data"]
print(f"Found {len(kbs)} knowledge bases")
for kb in kbs:
    print(f"- {kb['name']}: {kb['query']}")
```

#### After:

```python
import asyncio
from asf.medical.client.api_client import MedicalResearchSynthesizerClient

async def main():
    client = MedicalResearchSynthesizerClient()
    
    try:
        # Authentication
        login_response = await client.login("user@example.com", "password")
        if not login_response.success:
            print(f"Login failed: {login_response.message}")
            return
        
        # Create knowledge base
        kb_response = await client.create_knowledge_base(
            name="cardiovascular_statins",
            query="statin therapy cardiovascular",
            update_schedule="weekly"
        )
        
        if not kb_response.success:
            print(f"Knowledge base creation failed: {kb_response.message}")
            return
        
        kb_id = kb_response.data["kb_id"]
        print(f"Created knowledge base with ID: {kb_id}")
        
        # List knowledge bases
        list_response = await client.list_knowledge_bases()
        
        if not list_response.success:
            print(f"Failed to list knowledge bases: {list_response.message}")
            return
        
        kbs = list_response.data
        print(f"Found {len(kbs)} knowledge bases")
        for kb in kbs:
            print(f"- {kb['name']}: {kb['query']}")
        
    finally:
        await client.close()

if __name__ == "__main__":
    asyncio.run(main())
```

## Handling Responses

All client methods return an `APIResponse` object with the following structure:

```python
class APIResponse(BaseModel):
    success: bool  # Whether the request was successful
    message: str  # Response message
    data: Optional[Any]  # Response data
    errors: Optional[List[Dict[str, Any]]]  # Error details
    meta: Optional[Dict[str, Any]]  # Metadata
```

You can check if a request was successful by checking the `success` field:

```python
response = await client.search("statin therapy")
if response.success:
    # Request was successful
    print(f"Found {len(response.data['results'])} results")
else:
    # Request failed
    print(f"Search failed: {response.message}")
    if response.errors:
        for error in response.errors:
            print(f"- {error['detail']}")
```

## Closing the Client

Always close the client when you're done with it:

```python
await client.close()
```

You can use a `try`/`finally` block to ensure the client is closed even if an error occurs:

```python
try:
    # Use the client
    response = await client.search("statin therapy")
finally:
    # Close the client
    await client.close()
```

## Need Help?

If you need help migrating your code to use the new client library, please contact the ASF Medical Research Synthesizer team.
