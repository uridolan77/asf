# Export API

This module provides endpoints for exporting search results and analyses in various formats (JSON, CSV, Excel, PDF).

## Overview

The export API allows users to export search results and analyses in various formats. It supports both synchronous and asynchronous exports, with background tasks for long-running exports like PDF generation.

## Endpoints

### Export Results

```
POST /export/{format}
```

Export search results or analyses in the specified format.

**Path Parameters:**
- `format`: The export format (json, csv, excel, pdf)

**Request Body:**
```json
{
  "result_id": "string",  // Optional: ID of a stored result
  "query": "string",      // Optional: Query to execute
  "max_results": 100      // Optional: Maximum number of results to return
}
```

**Response:**
```json
{
  "success": true,
  "message": "Export successful",
  "data": {
    "file_url": "string"  // URL to download the exported file
  },
  "meta": {
    "format": "string",
    "query": "string",
    "result_count": 0
  }
}
```

For asynchronous exports (PDF generation with background tasks):
```json
{
  "success": true,
  "message": "PDF generation started. The file will be available shortly.",
  "data": {
    "status": "processing",
    "file_path": "string"  // Path to check the status of the export
  },
  "meta": {
    "format": "string",
    "query": "string",
    "result_count": 0
  }
}
```

### Check Export Status

```
GET /export/status/{file_path}
```

Check the status of an asynchronous export.

**Path Parameters:**
- `file_path`: The path of the file being generated

**Response:**
```json
{
  "success": true,
  "message": "Export status: processing",
  "data": {
    "status": "processing",
    "progress": 50
  }
}
```

When completed:
```json
{
  "success": true,
  "message": "Export completed successfully",
  "data": {
    "status": "completed",
    "file_url": "/export/download/file.pdf"
  }
}
```

### Download Export

```
GET /export/download/{file_name}
```

Download an exported file.

**Path Parameters:**
- `file_name`: The name of the file to download

**Response:**
The file as a binary stream.

## Implementation Details

### Background Tasks

Long-running exports like PDF generation are handled using FastAPI's `BackgroundTasks`. This allows the API to return immediately while the export is processed in the background.

The status of background tasks is tracked in the `background_tasks_status` dictionary, which maps file paths to status information.

### Export Formats

The following export formats are supported:

- **JSON**: Returns a JSON object with the search results or analysis
- **CSV**: Returns a CSV file with the search results or analysis
- **Excel**: Returns an Excel file with the search results or analysis
- **PDF**: Returns a PDF file with the search results or analysis

### Export Utilities

The export utilities are implemented in a single module (`export_utils.py`) to ensure consistent behavior across all export formats. This module provides functions for exporting data in various formats (JSON, CSV, Excel, PDF) with consistent interfaces and error handling.

**Note**: The export utilities have been consolidated into a single comprehensive implementation in `export_utils.py`.

## Usage Examples

### Export Search Results

```python
import httpx

async def export_search_results():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/export/json",
            json={
                "query": "covid-19 treatment"
            }
        )

        data = response.json()
        print(f"Export successful: {data['data']['file_url']}")
```

### Export with Background Tasks

```python
import httpx
import time

async def export_with_background_tasks():
    async with httpx.AsyncClient() as client:
        # Start the export
        response = await client.post(
            "http://localhost:8000/export/pdf",
            json={
                "query": "covid-19 treatment"
            }
        )

        data = response.json()

        if data['data'].get('status') == 'processing':
            # Check the status periodically
            file_path = data['data']['file_path']

            while True:
                status_response = await client.get(
                    f"http://localhost:8000/export/status/{file_path}"
                )

                status_data = status_response.json()

                if status_data['data']['status'] == 'completed':
                    # Download the file
                    file_url = status_data['data']['file_url']
                    download_response = await client.get(
                        f"http://localhost:8000{file_url}"
                    )

                    # Save the file
                    with open("export.pdf", "wb") as f:
                        f.write(download_response.content)

                    print("Export downloaded successfully")
                    break

                elif status_data['data']['status'] == 'failed':
                    print(f"Export failed: {status_data['data'].get('error')}")
                    break

                # Wait before checking again
                time.sleep(1)
        else:
            print(f"Export successful: {data['data']['file_url']}")
```
