# Asynchronous ML Inference with Dramatiq and Grafana LGTM Observability

This document describes the implementation of asynchronous ML inference in the Medical Research Synthesizer using Dramatiq and the integration with the Grafana LGTM stack for observability.

## Overview

Machine learning inference operations, such as contradiction detection and embedding generation, can be resource-intensive and time-consuming. To improve the responsiveness of the API and efficiently utilize resources, we've implemented asynchronous processing for these operations using Dramatiq. Additionally, we've integrated with the Grafana LGTM stack (Loki, Grafana, Tempo, Mimir) for comprehensive observability.

## Architecture

The asynchronous ML inference architecture consists of the following components:

1. **Dramatiq Task Queue**: Redis is used as the message broker for Dramatiq.
2. **ML Inference Tasks**: Tasks are defined as Dramatiq actors in the `tasks/ml_inference_tasks.py` module.
3. **API Integration**: The API endpoints in `api/routers/async_ml.py` use Dramatiq to offload ML inference operations to background workers.
4. **Workers**: Dramatiq workers process tasks asynchronously.
5. **Observability**: The Grafana LGTM stack is used for monitoring and observability.

## ML Inference Tasks

The following ML inference tasks are implemented as Dramatiq actors:

1. **Contradiction Detection**: Detects contradiction between two claims using various methods (direct, temporal, statistical).
2. **Contradiction Analysis**: Analyzes contradictions in a list of articles.
3. **Embedding Generation**: Generates embeddings for a list of texts using specified models (BioMedLM, Lorentz).

Each task is instrumented with observability features, including:

- Logging of task start, progress, and completion
- Metrics collection for task duration, memory usage, and error rates
- Distributed tracing for end-to-end visibility

## API Integration

The API provides endpoints for asynchronous ML inference operations in the `api/routers/async_ml.py` module:

- `POST /v1/async-ml/contradiction/detect`: Detect contradiction between two claims asynchronously
- `POST /v1/async-ml/contradiction/analyze`: Analyze contradictions in a list of articles asynchronously
- `POST /v1/async-ml/embeddings/generate`: Generate embeddings for a list of texts asynchronously
- `GET /v1/async-ml/task/{task_id}`: Get the status of an ML task

These endpoints return a task ID that can be used to check the status of the task and retrieve the results when the task is completed.

## Observability with Grafana LGTM Stack

The Grafana LGTM stack provides comprehensive observability for the ML inference operations:

1. **Loki**: Collects and indexes logs from the ML inference tasks.
2. **Tempo**: Collects and stores distributed traces for end-to-end visibility.
3. **Mimir**: Stores metrics for long-term analysis and alerting.
4. **Grafana**: Provides dashboards for visualizing logs, traces, and metrics.

### Metrics

The following metrics are collected for ML inference operations:

- **ml_inference_total**: Total number of ML inference operations, labeled by model and operation
- **ml_inference_errors_total**: Total number of ML inference errors, labeled by model, operation, and error type
- **ml_inference_duration_seconds**: Duration of ML inference operations in seconds, labeled by model and operation
- **ml_inference_queue_size**: Number of ML inference operations in queue, labeled by model
- **ml_model_memory_usage_bytes**: Memory usage of ML models in bytes, labeled by model

### Logs

Structured logs are sent to Loki with the following information:

- **Service**: The name of the service (medical-research-synthesizer)
- **Hostname**: The hostname of the machine running the service
- **Level**: The log level (info, error, etc.)
- **Model**: The name of the ML model
- **Operation**: The name of the operation
- **Event Type**: The type of event (task_start, inference, task_complete, error)
- **Task ID**: The ID of the task
- **Trace ID**: The ID of the trace for correlation with traces in Tempo

### Traces

Distributed traces are sent to Tempo with the following information:

- **Trace ID**: The ID of the trace
- **Span ID**: The ID of the span
- **Name**: The name of the span (model_operation)
- **Start Time**: The start time of the span
- **End Time**: The end time of the span
- **Tags**: Additional information about the span (model, operation, service, hostname, etc.)

## Running Workers

Dramatiq workers are run separately from the API server. The `run_workers.py` script starts Dramatiq workers for processing background tasks:

```bash
python -m asf.medical.run_workers --processes 2 --threads 8
```

This starts 2 worker processes, each with 8 threads, for a total of 16 concurrent task executions.

## Task Status Tracking

Task status is tracked using a custom middleware that stores task status information in memory. The API provides an endpoint for checking task status:

```
GET /v1/async-ml/task/{task_id}
```

This endpoint returns the current status of the task, including progress information and, for completed tasks, the result.

## Error Handling

Dramatiq provides built-in error handling features:

- Automatic retries for failed tasks (with exponential backoff)
- Error logging
- Task status tracking

The API also handles errors gracefully, providing appropriate error responses to clients.

## Benefits of Asynchronous ML Inference

Asynchronous ML inference offers several advantages:

1. **Improved Responsiveness**: The API can respond quickly to client requests, even for long-running ML operations.
2. **Resource Efficiency**: ML operations can be processed in the background, allowing the API server to handle more requests.
3. **Scalability**: The number of worker processes and threads can be adjusted based on the workload.
4. **Fault Tolerance**: Failed tasks can be automatically retried, improving reliability.
5. **Observability**: Comprehensive monitoring and tracing provide visibility into the ML operations.

## Future Improvements

Future improvements to the asynchronous ML inference system could include:

1. **Persistent Task Storage**: Store task status in Redis or a database for durability across restarts.
2. **Task Prioritization**: Implement priority queues for different types of tasks.
3. **Task Scheduling**: Add support for scheduled and periodic tasks.
4. **Resource Limits**: Implement resource limits for ML operations to prevent overloading the system.
5. **Model Caching**: Implement caching of ML models to improve performance.
6. **Distributed Training**: Extend the system to support distributed training of ML models.

## Conclusion

The implementation of asynchronous ML inference with Dramatiq and the Grafana LGTM stack significantly improves the performance, scalability, and observability of the Medical Research Synthesizer. Long-running ML operations no longer block the API server, resulting in a better user experience and more efficient resource utilization.
