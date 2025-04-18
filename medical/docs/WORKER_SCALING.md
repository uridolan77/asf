# Worker Scaling Strategies

This document provides strategies for scaling Dramatiq workers based on load.

## Overview

Dramatiq is used for background task processing in the Medical Research Synthesizer. As the application scales, it's important to scale the number of workers to handle the increased load. This document provides strategies for scaling Dramatiq workers based on load.

## Worker Types

The Medical Research Synthesizer uses different types of workers for different tasks:

1. **API Workers**: Process tasks triggered by API endpoints, such as PDF generation and export tasks.
2. **ML Workers**: Process ML inference tasks, such as contradiction detection and bias assessment.
3. **Data Workers**: Process data ingestion tasks, such as importing studies from external sources.

## Scaling Strategies

### Manual Scaling

The simplest scaling strategy is to manually adjust the number of workers based on observed load. This can be done by modifying the `--processes` and `--threads` parameters when starting Dramatiq workers:

```bash
# Start 4 worker processes with 8 threads each
dramatiq asf.medical.tasks --processes 4 --threads 8
```

### Auto-Scaling with Kubernetes

For production deployments, Kubernetes can be used to automatically scale the number of worker pods based on CPU and memory usage:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: dramatiq-workers
spec:
  replicas: 3
  selector:
    matchLabels:
      app: dramatiq-workers
  template:
    metadata:
      labels:
        app: dramatiq-workers
    spec:
      containers:
      - name: dramatiq-worker
        image: medical-research-synthesizer:latest
        command: ["dramatiq", "asf.medical.tasks", "--processes", "2", "--threads", "4"]
        resources:
          requests:
            cpu: 500m
            memory: 1Gi
          limits:
            cpu: 2
            memory: 4Gi
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: dramatiq-workers-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: dramatiq-workers
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### Queue-Based Scaling

For more advanced scaling, you can use queue-based metrics to scale workers. This requires monitoring the Redis queue length and scaling workers based on the number of pending tasks:

```python
import redis
import time
import subprocess

# Connect to Redis
redis_client = redis.Redis.from_url("redis://localhost:6379/0")

# Configuration
MIN_WORKERS = 2
MAX_WORKERS = 10
QUEUE_THRESHOLD = 100  # Scale up if queue length exceeds this threshold
SCALE_UP_STEP = 2  # Number of workers to add when scaling up
SCALE_DOWN_STEP = 1  # Number of workers to remove when scaling down
CHECK_INTERVAL = 60  # Check interval in seconds

# Current number of workers
current_workers = MIN_WORKERS

# Start initial workers
subprocess.run(["dramatiq", "asf.medical.tasks", "--processes", str(current_workers), "--threads", "4"])

while True:
    # Get queue length
    queue_length = redis_client.llen("dramatiq:default")
    
    # Scale up if queue length exceeds threshold
    if queue_length > QUEUE_THRESHOLD and current_workers < MAX_WORKERS:
        new_workers = min(current_workers + SCALE_UP_STEP, MAX_WORKERS)
        if new_workers > current_workers:
            # Stop current workers
            subprocess.run(["pkill", "-f", "dramatiq"])
            
            # Start new workers
            subprocess.run(["dramatiq", "asf.medical.tasks", "--processes", str(new_workers), "--threads", "4"])
            
            current_workers = new_workers
            print(f"Scaled up to {current_workers} workers")
    
    # Scale down if queue length is low
    elif queue_length < QUEUE_THRESHOLD / 2 and current_workers > MIN_WORKERS:
        new_workers = max(current_workers - SCALE_DOWN_STEP, MIN_WORKERS)
        if new_workers < current_workers:
            # Stop current workers
            subprocess.run(["pkill", "-f", "dramatiq"])
            
            # Start new workers
            subprocess.run(["dramatiq", "asf.medical.tasks", "--processes", str(new_workers), "--threads", "4"])
            
            current_workers = new_workers
            print(f"Scaled down to {current_workers} workers")
    
    # Sleep for check interval
    time.sleep(CHECK_INTERVAL)
```

## Monitoring

To effectively scale workers, it's important to monitor the following metrics:

1. **Queue Length**: The number of pending tasks in the queue.
2. **Worker Utilization**: CPU and memory usage of worker processes.
3. **Task Processing Time**: The time it takes to process tasks.
4. **Error Rate**: The rate of task failures.

These metrics can be collected using the Grafana LGTM stack (Loki, Grafana, Tempo, Mimir/Prometheus) and visualized in Grafana dashboards.

## Best Practices

1. **Separate Queues**: Use separate queues for different types of tasks to prevent long-running tasks from blocking short-running tasks.
2. **Resource Limits**: Set appropriate resource limits for worker processes to prevent resource exhaustion.
3. **Graceful Shutdown**: Ensure workers can gracefully shut down when scaling down to prevent task loss.
4. **Retry Mechanism**: Use Dramatiq's built-in retry mechanism to handle transient failures.
5. **Dead Letter Queue**: Configure a dead letter queue to capture failed tasks for later analysis.
6. **Monitoring and Alerting**: Set up monitoring and alerting to detect issues with worker scaling.

## Implementation

The Medical Research Synthesizer uses the following implementation for worker scaling:

1. **Development**: Manual scaling with a fixed number of workers.
2. **Testing**: Manual scaling with a fixed number of workers.
3. **Production**: Auto-scaling with Kubernetes based on CPU and memory usage.

## Conclusion

Proper worker scaling is essential for ensuring the Medical Research Synthesizer can handle varying loads efficiently. By following the strategies outlined in this document, you can ensure that your Dramatiq workers scale appropriately based on load.
