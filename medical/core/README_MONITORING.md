# Monitoring Framework

This documentation provides a comprehensive guide to the monitoring capabilities implemented in the Medical Research Synthesizer application.

## Table of Contents
1. [Overview](#overview)
2. [Key Components](#key-components)
3. [Metrics Collection](#metrics-collection)
4. [Logging](#logging)
5. [Alerting](#alerting)
6. [Dashboards](#dashboards)
7. [Health Checks](#health-checks)
8. [Performance Monitoring](#performance-monitoring)
9. [Integration with External Systems](#integration-with-external-systems)
10. [Best Practices](#best-practices)

## Overview

The monitoring framework provides real-time visibility into the health, performance, and behavior of the Medical Research Synthesizer application. It enables developers and operators to detect, diagnose, and resolve issues quickly while providing insights for capacity planning and optimization.

## Key Components

### Monitoring Module (`monitoring.py`)

The main monitoring module provides:
- Metric collection and reporting
- Health check mechanisms
- Performance tracking
- Integration with external monitoring systems

### Observability Module (`observability.py`)

Extends the monitoring framework with:
- Distributed tracing
- Detailed request/response logging
- Event correlation
- System-wide observability features

### Logging Configuration (`logging_config.py`)

Provides structured logging with:
- Consistent log format
- Log level management
- Context enrichment
- Integration with central log management systems

## Metrics Collection

### Available Metrics

The monitoring system collects the following types of metrics:

1. **System Metrics**
   - CPU usage
   - Memory usage
   - Disk I/O
   - Network I/O

2. **Application Metrics**
   - Request rates
   - Response times
   - Error rates
   - Task queue depths
   - Cache hit/miss ratios
   - Database query performance

3. **Business Metrics**
   - Search operations per minute
   - Document processing rates
   - API usage by client
   - User activity

### Metric Collection Methods

```python
# Example: Tracking a custom metric
from asf.medical.core.monitoring import metrics

# Counter example (accumulates)
metrics.increment('api_calls', 1, tags={'endpoint': '/search', 'method': 'GET'})

# Gauge example (current value)
metrics.gauge('active_connections', 42)

# Histogram example (statistical distribution)
metrics.histogram('response_time', 0.23, tags={'endpoint': '/documents'})
```

## Logging

### Log Levels

The system uses the following log levels:
- **DEBUG**: Detailed information for debugging
- **INFO**: General information about system operation
- **WARNING**: Potential issues that don't impact operation
- **ERROR**: Errors that impact specific operations
- **CRITICAL**: Critical issues that require immediate attention

### Contextual Logging

All logs include context to make troubleshooting easier:

```python
from asf.medical.core.logging_config import get_logger

logger = get_logger(__name__)

def process_document(doc_id):
    logger.info("Processing document", extra={
        'document_id': doc_id,
        'process_type': 'text_extraction'
    })
```

## Alerting

The monitoring system integrates with various alerting mechanisms:

1. **Email Alerts**: For non-urgent issues
2. **SMS Notifications**: For critical issues
3. **PagerDuty/OpsGenie Integration**: For on-call escalation
4. **Slack Notifications**: For team awareness

### Alert Configuration

Alerts are configured based on:
- Metric thresholds
- Error rates
- System availability
- Custom business logic

## Dashboards

The monitoring system provides data for various dashboards:

1. **System Overview**: High-level health and performance metrics
2. **API Performance**: Detailed API call metrics
3. **Error Analysis**: Visualization of error rates and types
4. **Resource Utilization**: CPU, memory, and storage metrics
5. **Business Metrics**: Domain-specific usage patterns

## Health Checks

The system implements the following health check types:

1. **Service Liveness**: Is the service running?
2. **Service Readiness**: Is the service ready to handle traffic?
3. **Dependency Checks**: Are required dependencies available?
4. **Deep Health Checks**: Test full functionality paths

### Example Health Check Implementation

```python
from asf.medical.core.monitoring import health_checks

@health_checks.register('database')
def check_database_health():
    try:
        # Perform database check
        return health_checks.HealthStatus.HEALTHY
    except Exception as e:
        return health_checks.HealthStatus.UNHEALTHY, str(e)
```

## Performance Monitoring

The monitoring system tracks performance at various levels:

1. **Request timing**: End-to-end request processing time
2. **Component timing**: Time spent in different components
3. **Database query performance**: Query execution times
4. **External service calls**: API call performance
5. **Resource utilization**: CPU, memory, I/O utilization

### Distributed Tracing

Using OpenTelemetry for distributed tracing:

```python
from asf.medical.core.observability import tracer

@tracer.trace('process_document')
async def process_document(doc_id):
    # Function implementation
    pass
```

## Integration with External Systems

The monitoring framework integrates with:

1. **Prometheus**: Metrics collection and storage
2. **Grafana**: Dashboard visualization
3. **ELK Stack**: Log aggregation and analysis
4. **Jaeger/Zipkin**: Distributed tracing
5. **NewRelic/Datadog**: Application performance monitoring

## Best Practices

1. **Monitor what matters**: Focus on metrics that provide actionable insights
2. **Use structured logging**: Include context in all log messages
3. **Set appropriate thresholds**: Avoid alert fatigue
4. **Implement correlation IDs**: Track requests across services
5. **Document your metrics**: Ensure the team understands what each metric means
6. **Regular review**: Periodically review monitoring effectiveness
7. **Automated testing**: Include monitoring in your test suite

## Configuration

The monitoring system is configurable through environment variables:

```
MONITORING_ENABLED=true
METRICS_REPORTING_INTERVAL=60
LOG_LEVEL=INFO
TRACING_ENABLED=true
HEALTH_CHECK_INTERVAL=30
```

## Troubleshooting

Common monitoring issues and solutions:

1. **High alert volume**: Review and adjust thresholds
2. **Missing metrics**: Check metric collection configuration
3. **Log flooding**: Adjust log levels or implement log rate limiting
4. **Slow dashboards**: Optimize queries or increase refresh intervals

For additional support, contact the DevOps team at devops@example.com
