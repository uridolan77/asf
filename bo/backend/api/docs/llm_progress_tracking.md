# LLM Progress Tracking API

The LLM Progress Tracking API provides endpoints for monitoring the progress of long-running LLM operations, such as complex requests, batch processing, and model fine-tuning.

## Overview

The Progress Tracking API allows you to:

- List all operations and their status
- Get detailed progress information for a specific operation
- Monitor active operations
- Get a summary of all operations
- Clean up old operations

## Authentication

All endpoints require authentication. Use the standard authentication methods for the BO API.

## Endpoints

### List Operations

```
GET /api/llm/progress/operations
```

Lists all operations with basic information.

**Query Parameters:**
- `status` (optional): Filter by status (pending, running, completed, failed, cancelled)
- `operation_type` (optional): Filter by operation type (LLM_REQUEST, LLM_STREAMING, LLM_BATCH, etc.)
- `limit` (optional): Maximum number of operations to return (default: 100)

**Response:**
```json
[
  {
    "operation_id": "generate_text:a1b2c3d4",
    "operation_type": "LLM_REQUEST",
    "status": "completed",
    "percent_complete": 100.0,
    "message": "Operation completed successfully",
    "start_time": "2023-06-15T10:30:45.123456",
    "end_time": "2023-06-15T10:30:47.654321"
  },
  {
    "operation_id": "batch_generate:e5f6g7h8",
    "operation_type": "LLM_BATCH",
    "status": "running",
    "percent_complete": 45.0,
    "message": "Processing request 45/100 (45.0%)",
    "start_time": "2023-06-15T10:35:12.987654",
    "end_time": null
  }
]
```

### Get Operation Progress

```
GET /api/llm/progress/operations/{operation_id}
```

Gets detailed progress information for a specific operation.

**Path Parameters:**
- `operation_id`: ID of the operation

**Response:**
```json
{
  "operation_id": "generate_text:a1b2c3d4",
  "operation_type": "LLM_REQUEST",
  "total_steps": 5,
  "current_step": 5,
  "status": "completed",
  "message": "Operation completed successfully",
  "percent_complete": 100.0,
  "start_time": "2023-06-15T10:30:45.123456",
  "end_time": "2023-06-15T10:30:47.654321",
  "elapsed_time": 2.530865,
  "estimated_time_remaining": null,
  "steps": [
    {
      "step_number": 1,
      "message": "Initializing request",
      "timestamp": "2023-06-15T10:30:45.223456",
      "details": null
    },
    {
      "step_number": 2,
      "message": "Pre-processing request",
      "timestamp": "2023-06-15T10:30:45.323456",
      "details": null
    },
    {
      "step_number": 3,
      "message": "Selecting provider",
      "timestamp": "2023-06-15T10:30:45.423456",
      "details": null
    },
    {
      "step_number": 4,
      "message": "Generating response",
      "timestamp": "2023-06-15T10:30:46.423456",
      "details": null
    },
    {
      "step_number": 5,
      "message": "Post-processing response",
      "timestamp": "2023-06-15T10:30:47.423456",
      "details": null
    }
  ],
  "metadata": {
    "model": "gpt-4",
    "provider": "openai",
    "prompt_length": 150
  }
}
```

### List Active Operations

```
GET /api/llm/progress/active
```

Lists all active operations (pending, running).

**Query Parameters:**
- `operation_type` (optional): Filter by operation type (LLM_REQUEST, LLM_STREAMING, LLM_BATCH, etc.)

**Response:**
```json
[
  {
    "operation_id": "batch_generate:e5f6g7h8",
    "operation_type": "LLM_BATCH",
    "status": "running",
    "percent_complete": 45.0,
    "message": "Processing request 45/100 (45.0%)",
    "start_time": "2023-06-15T10:35:12.987654",
    "end_time": null
  },
  {
    "operation_id": "generate_stream:i9j0k1l2",
    "operation_type": "LLM_STREAMING",
    "status": "running",
    "percent_complete": 60.0,
    "message": "Streaming response (chunks: 30)",
    "start_time": "2023-06-15T10:36:00.123456",
    "end_time": null
  }
]
```

### Get Progress Summary

```
GET /api/llm/progress/summary
```

Gets a summary of all operations.

**Response:**
```json
{
  "total": 10,
  "active": 2,
  "completed": 7,
  "failed": 1,
  "by_status": {
    "pending": 0,
    "running": 2,
    "completed": 7,
    "failed": 1,
    "cancelled": 0
  },
  "by_type": {
    "LLM_REQUEST": 5,
    "LLM_STREAMING": 3,
    "LLM_BATCH": 2
  }
}
```

### Delete Operation

```
DELETE /api/llm/progress/operations/{operation_id}
```

Deletes a specific operation from the registry.

**Path Parameters:**
- `operation_id`: ID of the operation

**Response:**
```json
{
  "message": "Operation generate_text:a1b2c3d4 deleted"
}
```

### Clean Up Operations

```
POST /api/llm/progress/cleanup
```

Cleans up old completed operations.

**Query Parameters:**
- `max_age_seconds` (optional): Maximum age in seconds for completed operations (default: 3600)

**Response:**
```json
{
  "removed": 5
}
```

## Operation Types

The following operation types are available:

- `LLM_REQUEST`: Standard LLM request
- `LLM_STREAMING`: Streaming LLM request
- `LLM_BATCH`: Batch LLM request
- `LLM_FINE_TUNING`: Fine-tuning operation
- `LLM_EMBEDDING`: Embedding generation
- `PROVIDER_CONNECTION`: Provider connection operation
- `PROVIDER_INITIALIZATION`: Provider initialization
- `SESSION_CREATION`: Session creation
- `SESSION_MANAGEMENT`: Session management
- `CACHE_OPERATION`: Cache operation
- `GENERAL`: General operation
- `CUSTOM`: Custom operation

## Status Values

The following status values are available:

- `pending`: Operation is pending
- `running`: Operation is running
- `completed`: Operation completed successfully
- `failed`: Operation failed
- `cancelled`: Operation was cancelled

## Integration with LLM Gateway

The Progress Tracking API is integrated with the LLM Gateway, allowing you to track the progress of all LLM operations. When you use the LLM Gateway to generate text, the progress of the operation is automatically tracked and available through the Progress Tracking API.

### Example: Generating Text with Progress Tracking

```python
import requests
import time

# Generate text
response = requests.post(
    "http://localhost:8000/api/llm/gateway/generate",
    json={
        "prompt": "Explain quantum computing",
        "model": "gpt-4",
        "provider": "openai"
    },
    headers={"Authorization": "Bearer YOUR_TOKEN"}
)

# Get request ID from response
request_id = response.json()["request_id"]

# Get progress information
progress_response = requests.get(
    f"http://localhost:8000/api/llm/progress/operations/{request_id}",
    headers={"Authorization": "Bearer YOUR_TOKEN"}
)

# Print progress information
print(progress_response.json())
```

### Example: Monitoring Active Operations

```python
import requests
import time

# Get active operations
response = requests.get(
    "http://localhost:8000/api/llm/progress/active",
    headers={"Authorization": "Bearer YOUR_TOKEN"}
)

# Print active operations
active_operations = response.json()
for operation in active_operations:
    print(f"{operation['operation_id']}: {operation['percent_complete']}% - {operation['message']}")
```

## Error Handling

The API returns standard HTTP status codes:

- `200 OK`: The request was successful
- `404 Not Found`: The operation was not found
- `500 Internal Server Error`: An error occurred on the server

Error responses include a detail message:

```json
{
  "detail": "Operation generate_text:a1b2c3d4 not found"
}
```
