# Task Management Architecture

This document explains the architecture and relationships between the different task management mechanisms in the ASF Medical Research Synthesizer.

## Overview

The ASF Medical Research Synthesizer uses multiple task management mechanisms for different purposes:

1. **Dramatiq**: For API-triggered background tasks
2. **Ray**: For distributed ML computation
3. **TaskManager**: Legacy custom task manager

## Components

### 1. Dramatiq (`task_queue.py`)

Dramatiq is used for API-triggered background tasks, such as generating PDF reports and running ML inference tasks. It provides:

- Redis-based message broker
- Task tracking and status monitoring
- Automatic retries and error handling
- Time limits and age limits

**Status**: Active and preferred for API-triggered background tasks.

**Use Cases**:
- PDF generation
- Export tasks
- ML inference tasks triggered by API endpoints

### 2. Ray (`ray_orchestrator.py`)

Ray is used for distributed ML computation, especially for heavy computational tasks. It provides:

- Distributed task scheduling
- Resource management (CPU, GPU)
- Parallel processing
- Integration with ML frameworks

**Status**: Active and preferred for ML computation tasks.

**Use Cases**:
- Distributed ML model training
- Parallel data processing
- Resource-intensive computation

### 3. TaskManager (`task_manager.py`)

The TaskManager is a custom implementation for managing background tasks. It provides:

- Asynchronous task execution
- Task status tracking
- Concurrency control
- Task cancellation

**Status**: Legacy/Deprecated. New code should use Dramatiq or Ray instead.

## Usage Guidelines

1. **For API-triggered background tasks**:
   - Use Dramatiq (`task_queue.py`)
   - Define tasks as Dramatiq actors in the `tasks` package
   - Use the `@dramatiq.actor` decorator

2. **For ML computation tasks**:
   - Use Ray (`ray_orchestrator.py`)
   - Define tasks as Ray remote functions
   - Use the `@ray.remote` decorator

3. **Do not use**:
   - The legacy TaskManager (`task_manager.py`) for new code

## Integration

The task management mechanisms are integrated as follows:

1. **API Layer**:
   - API endpoints use Dramatiq to offload long-running operations to background workers
   - Task status can be checked through API endpoints

2. **ML Layer**:
   - ML services use Ray for distributed computation
   - Ray tasks can be triggered by Dramatiq tasks for complex workflows

## Running Workers

1. **Dramatiq Workers**:
   - Run using the `run_workers.py` script
   - Configure the number of processes and threads

2. **Ray Workers**:
   - Start a Ray cluster using the Ray CLI
   - Connect to the cluster using the `ray_orchestrator.py` module
