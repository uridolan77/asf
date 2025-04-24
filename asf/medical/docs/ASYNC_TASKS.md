# Asynchronous Task Processing with Dramatiq

This document describes the implementation of asynchronous task processing in the Medical Research Synthesizer using Dramatiq.

## Overview

Asynchronous task processing is essential for handling long-running operations without blocking the API server. In the Medical Research Synthesizer, we use Dramatiq for processing background tasks such as PDF generation, which can be time-consuming and resource-intensive.

## Architecture

The asynchronous task processing architecture consists of the following components:

1. **Dramatiq Broker**: Redis is used as the message broker for Dramatiq.
2. **Task Definitions**: Tasks are defined as Dramatiq actors in the `tasks` package.
3. **Task Queue**: The `core/task_queue.py` module configures Dramatiq and provides utilities for task tracking.
4. **API Integration**: The API endpoints use Dramatiq to offload long-running operations to background workers.
5. **Workers**: Dramatiq workers process tasks asynchronously.

## Task Queue Configuration

The `core/task_queue.py` module configures Dramatiq with the following features:

- Redis broker for message passing
- Middleware for task management:
  - Age limit (10 minutes)
  - Time limit (5 minutes)
  - Automatic retries (max 3 retries with exponential backoff)
  - Custom task tracking middleware for monitoring task status

## Task Definitions

Tasks are defined in the `tasks` package. Each task is a Dramatiq actor that can be executed asynchronously. For example, the `export_tasks.py` module defines tasks for generating PDF reports:

```python
@dramatiq.actor(max_retries=3, time_limit=300000)  # 5 minutes
def generate_pdf_report(results: List[Dict[str, Any]], query_text: str, output_path: str):
    """Generate a PDF report for search results in the background."""
    # Task implementation
```

## API Integration

The API endpoints use Dramatiq to offload long-running operations to background workers. For example, the `/export/{format}` endpoint uses Dramatiq for PDF generation:

```python
# Send the task to Dramatiq
message = generate_pdf_report.send(results, query_text, output_path)
task_id = message.message_id

# Return a response with task information
return JSONResponse(content={
    "status": "processing",
    "message": "PDF generation started. The file will be available shortly.",
    "task_id": task_id,
    "file_path": output_path
})
```

The API also provides endpoints for checking task status and downloading completed exports.

## Running Workers

Dramatiq workers are run separately from the API server. The `run_workers.py` script starts Dramatiq workers for processing background tasks:

```bash
python -m asf.medical.run_workers --processes 2 --threads 8
```

This starts 2 worker processes, each with 8 threads, for a total of 16 concurrent task executions.

## Task Status Tracking

Task status is tracked using a custom middleware that stores task status information in memory. The API provides an endpoint for checking task status:

```
GET /export/status/{task_id}
```

This endpoint returns the current status of the task, including progress information and, for completed tasks, a URL for downloading the result.

## Error Handling

Dramatiq provides built-in error handling features:

- Automatic retries for failed tasks (with exponential backoff)
- Error logging
- Task status tracking

The API also handles errors gracefully, providing appropriate error responses to clients.

## Benefits of Dramatiq

Dramatiq offers several advantages over other task queue solutions:

1. **Simplicity**: Dramatiq has a simpler API and configuration compared to Celery.
2. **Performance**: Dramatiq is designed for high performance and low latency.
3. **Reliability**: Dramatiq provides robust error handling and retry mechanisms.
4. **Monitoring**: Dramatiq includes tools for monitoring task execution.
5. **Scalability**: Dramatiq can be scaled horizontally by adding more workers.

## Future Improvements

Future improvements to the asynchronous task processing system could include:

1. **Persistent Task Storage**: Store task status in Redis or a database for durability across restarts.
2. **Task Prioritization**: Implement priority queues for different types of tasks.
3. **Task Scheduling**: Add support for scheduled and periodic tasks.
4. **Web UI**: Implement a web interface for monitoring and managing tasks.
5. **Distributed Tracing**: Integrate with distributed tracing systems for better observability.

## Conclusion

The implementation of asynchronous task processing with Dramatiq significantly improves the performance and responsiveness of the Medical Research Synthesizer API. Long-running operations like PDF generation no longer block the API server, resulting in a better user experience and more efficient resource utilization.
