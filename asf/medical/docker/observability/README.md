# Grafana LGTM Stack for Medical Research Synthesizer

This directory contains the Docker Compose configuration for the Grafana LGTM stack (Loki, Grafana, Tempo, Mimir) used for observability in the Medical Research Synthesizer.

## Components

- **Grafana**: Visualization and dashboarding
- **Loki**: Log aggregation
- **Tempo**: Distributed tracing
- **Prometheus**: Metrics collection and storage
- **Promtail**: Log collection agent
- **Pushgateway**: Push-based metrics collection
- **Node Exporter**: Host metrics collection

## Directory Structure

```
observability/
├── docker-compose.yml
├── grafana/
│   ├── dashboards/
│   │   └── ml_inference_dashboard.json
│   └── provisioning/
│       ├── dashboards/
│       │   └── dashboards.yaml
│       └── datasources/
│           └── datasources.yaml
├── loki/
│   └── loki-config.yaml
├── prometheus/
│   └── prometheus.yml
├── promtail/
│   └── promtail-config.yaml
├── tempo/
│   └── tempo-config.yaml
└── README.md
```

## Setup

1. Create the necessary directories:

```bash
mkdir -p logs
mkdir -p grafana/dashboards
mkdir -p grafana/provisioning/dashboards
mkdir -p grafana/provisioning/datasources
mkdir -p loki
mkdir -p prometheus
mkdir -p promtail
mkdir -p tempo
```

2. Start the stack:

```bash
docker-compose up -d
```

3. Access Grafana:

- URL: http://localhost:3000
- Username: admin
- Password: admin

## Dashboards

The following dashboards are included:

- **ML Inference Dashboard**: Metrics and logs for ML inference operations

## Configuration

### Environment Variables

The following environment variables can be set in the application to configure the observability components:

- `LOKI_URL`: URL for Loki log ingestion (default: http://localhost:3100/loki/api/v1/push)
- `TEMPO_URL`: URL for Tempo trace ingestion (default: http://localhost:14268/api/traces)
- `PROMETHEUS_URL`: URL for Prometheus metrics ingestion (default: http://localhost:9090/api/v1/push)
- `PUSH_GATEWAY_URL`: URL for Prometheus Push Gateway (default: localhost:9091)
- `SERVICE_NAME`: Name of the service (default: medical-research-synthesizer)

### Logs

Logs are collected by Promtail and sent to Loki. The application sends structured logs to Loki directly using the `send_log_to_loki` function in the `observability.py` module.

### Traces

Traces are sent to Tempo using the `send_trace_to_tempo` function in the `observability.py` module. The application uses the `trace_ml_operation` context manager to create traces for ML operations.

### Metrics

Metrics are collected by Prometheus and can be pushed to the Prometheus Push Gateway using the `push_metrics` function in the `observability.py` module.

## Integration with the Application

The application integrates with the observability stack using the `observability.py` module, which provides functions for sending logs, traces, and metrics to the respective components.

## Troubleshooting

If you encounter issues with the observability stack, check the following:

1. Make sure all containers are running:

```bash
docker-compose ps
```

2. Check the logs of the containers:

```bash
docker-compose logs grafana
docker-compose logs loki
docker-compose logs tempo
docker-compose logs prometheus
```

3. Make sure the application is configured to send logs, traces, and metrics to the correct URLs.

4. Check the network connectivity between the application and the observability stack.

## References

- [Grafana](https://grafana.com/docs/grafana/latest/)
- [Loki](https://grafana.com/docs/loki/latest/)
- [Tempo](https://grafana.com/docs/tempo/latest/)
- [Prometheus](https://prometheus.io/docs/introduction/overview/)
- [Promtail](https://grafana.com/docs/loki/latest/clients/promtail/)
- [Pushgateway](https://github.com/prometheus/pushgateway)
- [Node Exporter](https://github.com/prometheus/node_exporter)
