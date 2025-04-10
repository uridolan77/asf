# ASF Medical Research Synthesizer Monitoring and Observability

This document describes the monitoring and observability solutions for the ASF Medical Research Synthesizer.

## Monitoring Module

The monitoring module (`monitoring.py`) provides a comprehensive set of tools for monitoring and observability:

- **Metrics**: Counters, gauges, histograms, and timers
- **Health Checks**: System and application health checks
- **Logging**: Structured logging with context
- **Error Tracking**: Error tracking with context
- **Request Logging**: HTTP request logging with duration and status code
- **Performance Monitoring**: Performance monitoring with timers and histograms

## Metrics

The monitoring module provides the following metric types:

- **Counters**: Incrementing values (e.g., number of requests)
- **Gauges**: Current values (e.g., memory usage)
- **Histograms**: Distribution of values (e.g., request duration)
- **Timers**: Duration of operations (e.g., function execution time)

### Using Metrics

```python
from asf.medical.core.monitoring import (
    increment_counter, set_gauge, record_histogram,
    start_timer, stop_timer, timer, timed, async_timed
)

# Increment a counter
increment_counter("requests", 1, {"method": "GET", "path": "/api/v1/search"})

# Set a gauge
set_gauge("memory_usage", 1024, {"unit": "MB"})

# Record a histogram value
record_histogram("request_duration", 0.123, {"method": "GET", "path": "/api/v1/search"})

# Use a timer
timer_id = start_timer("function_execution", {"function": "search"})
# ... do something
elapsed = stop_timer("function_execution", timer_id)

# Use a timer context manager
with timer("function_execution", {"function": "search"}):
    # ... do something

# Use a timer decorator
@timed("function_execution", {"function": "search"})
def my_function():
    # ... do something

# Use an async timer decorator
@async_timed("function_execution", {"function": "search"})
async def my_async_function():
    # ... do something
```

## Health Checks

The monitoring module provides a health check system that can be used to check the health of the application and its dependencies.

### Using Health Checks

```python
from asf.medical.core.monitoring import register_health_check, run_health_checks

# Register a health check
register_health_check("database", lambda: {
    "status": "ok",
    "latency": 0.123,
    "timestamp": datetime.now().isoformat()
})

# Run health checks
health_checks = run_health_checks()
```

## Logging

The monitoring module provides structured logging with context.

### Using Logging

```python
from asf.medical.core.monitoring import log_error, log_request

# Log an error
try:
    # ... do something
except Exception as e:
    log_error(e, {"function": "search", "query": "statin therapy"})

# Log a request
log_request("GET", "/api/v1/search", 200, 0.123, user_id=123)
```

## Middleware

The monitoring middleware (`middleware.py`) provides automatic request logging and error tracking for FastAPI applications.

### Using Middleware

```python
from fastapi import FastAPI
from asf.medical.api.middleware import MonitoringMiddleware

app = FastAPI()
app.add_middleware(MonitoringMiddleware)
```

## API Endpoints

The API provides the following endpoints for monitoring and observability:

- `GET /health`: Health check endpoint
- `GET /metrics`: Metrics endpoint
- `POST /metrics/export`: Export metrics to a JSON file

### Health Check Endpoint

The health check endpoint returns the health status of the application and its dependencies.

```http
GET /health
```

Response:

```json
{
  "status": "ok",
  "checks": {
    "system": {
      "status": "ok",
      "cpu_usage": 0.1,
      "memory_usage": "2.5",
      "timestamp": "2023-04-10T14:30:00.000Z"
    }
  }
}
```

### Metrics Endpoint

The metrics endpoint returns the current metrics.

```http
GET /metrics
```

Response:

```json
{
  "counters": {
    "requests": {
      "value": 100,
      "tags": {
        "method": "GET",
        "path": "/api/v1/search"
      }
    }
  },
  "gauges": {
    "memory_usage": {
      "value": 1024,
      "tags": {
        "unit": "MB"
      }
    }
  },
  "histograms": {
    "request_duration": {
      "values": [0.123, 0.456, 0.789],
      "tags": {
        "method": "GET",
        "path": "/api/v1/search"
      }
    }
  },
  "timers": {
    "function_execution": {
      "timers": {
        "1234567890": {
          "start": 1617979200.0,
          "end": 1617979200.123,
          "elapsed": 0.123
        }
      },
      "tags": {
        "function": "search"
      }
    }
  }
}
```

### Export Metrics Endpoint

The export metrics endpoint exports the current metrics to a JSON file.

```http
POST /metrics/export
```

Request:

```json
{
  "file_path": "logs/metrics.json"
}
```

Response:

```json
{
  "status": "ok",
  "message": "Metrics exported to logs/metrics.json"
}
```

## Integration with External Monitoring Systems

The monitoring module can be integrated with external monitoring systems such as Prometheus, Grafana, and ELK Stack.

### Prometheus Integration

The metrics endpoint can be scraped by Prometheus to collect metrics.

### Grafana Integration

Grafana can be used to visualize metrics collected by Prometheus.

### ELK Stack Integration

The structured logs can be sent to Elasticsearch for indexing and visualization in Kibana.
