# Progress Tracking System for LLM Gateway

This module provides a comprehensive progress tracking system for the LLM Gateway, allowing for detailed tracking of long-running operations such as complex LLM requests, batch processing, and model fine-tuning.

## Features

- **Detailed Progress Tracking**: Track the progress of operations with detailed step-by-step information
- **Centralized Registry**: Manage multiple progress trackers through a centralized registry
- **Decorator-Based Integration**: Easily add progress tracking to functions with decorators
- **Real-Time Updates**: Get real-time updates on operation progress
- **API Access**: Access progress information through a REST API
- **Persistence**: Store progress information in a cache for retrieval by other components
- **Error Handling**: Automatically track and report errors in operations

## Components

### ProgressTracker

The `ProgressTracker` class is the core component of the progress tracking system. It tracks the progress of a single operation, including:

- Current step and total steps
- Status (pending, running, completed, failed, cancelled)
- Progress messages
- Start and end times
- Detailed step history
- Metadata about the operation

### ProgressRegistry

The `ProgressRegistry` class provides a centralized registry for managing multiple progress trackers. It allows for:

- Creating and registering trackers
- Retrieving trackers by operation ID
- Getting all active trackers
- Cleaning up old trackers

### Decorators

The module provides decorators for easily adding progress tracking to functions:

- `@track_progress`: General-purpose decorator for tracking progress
- `@track_llm_progress`: Specialized decorator for LLM operations

### API

The module includes a REST API for accessing progress information:

- List all operations
- Get progress details for a specific operation
- List active operations
- Get a summary of all operations
- Clean up old operations

## Usage Examples

### Basic Usage

```python
from asf.medical.llm_gateway.progress import ProgressTracker

# Create a progress tracker
tracker = ProgressTracker(
    operation_id="example-operation",
    operation_type="general",
    total_steps=5
)

# Update progress
tracker.update(1, "Step 1 completed")
tracker.update(2, "Step 2 completed")
tracker.update(3, "Step 3 completed")
tracker.update(4, "Step 4 completed")
tracker.update(5, "Step 5 completed")

# Mark as completed
tracker.complete("All steps completed")

# Get progress details
progress = tracker.get_progress_details()
print(f"Progress: {progress.percent_complete}% - {progress.message}")
```

### Using the Registry

```python
from asf.medical.llm_gateway.progress import get_progress_registry

# Get the registry
registry = get_progress_registry()

# Create a tracker through the registry
tracker = registry.create_tracker(
    operation_id="example-registry",
    operation_type="general",
    total_steps=3
)

# Update progress
tracker.update(1, "Step 1 completed")
tracker.update(2, "Step 2 completed")
tracker.update(3, "Step 3 completed")

# Mark as completed
tracker.complete("All steps completed")

# Get all trackers
all_trackers = registry.get_all_trackers()
print(f"Number of trackers in registry: {len(all_trackers)}")
```

### Using Decorators

```python
from asf.medical.llm_gateway.progress import track_progress, get_progress_tracker
from asf.medical.llm_gateway.progress.models import OperationType

@track_progress(operation_type=OperationType.GENERAL, total_steps=4)
async def process_data(data):
    # Get the current tracker
    tracker = get_progress_tracker()
    
    # Step 1: Validate data
    tracker.update(1, "Validating data")
    # ... validation logic ...
    
    # Step 2: Process data
    tracker.update(2, "Processing data")
    # ... processing logic ...
    
    # Step 3: Analyze results
    tracker.update(3, "Analyzing results")
    # ... analysis logic ...
    
    # Step 4: Generate report
    tracker.update(4, "Generating report")
    # ... report generation logic ...
    
    return {"status": "success", "message": "Data processed successfully"}
```

### LLM-Specific Tracking

```python
from asf.medical.llm_gateway.progress import track_llm_progress, get_progress_tracker
from asf.medical.llm_gateway.progress.models import OperationType

@track_llm_progress(operation_type=OperationType.LLM_REQUEST, total_steps=3)
async def generate_text(prompt, model="gpt-4", temperature=0.7):
    # Get the current tracker
    tracker = get_progress_tracker()
    
    # Step 1: Prepare request
    tracker.update(1, "Preparing LLM request")
    # ... preparation logic ...
    
    # Step 2: Send request to LLM
    tracker.update(2, "Sending request to LLM")
    # ... API call logic ...
    
    # Step 3: Process response
    tracker.update(3, "Processing LLM response")
    # ... response processing logic ...
    
    return {"model": model, "prompt": prompt, "response": "Generated text..."}
```

### Integrating with LLM Gateway

See the `integration.py` file for examples of how to integrate the progress tracking system with the LLM Gateway client and providers.

## API Endpoints

The module provides the following API endpoints:

- `GET /progress/operations`: List all operation IDs
- `GET /progress/operations/{operation_id}`: Get progress details for a specific operation
- `GET /progress/active`: List all active operations
- `GET /progress/summary`: Get a summary of all operations
- `DELETE /progress/operations/{operation_id}`: Delete a specific operation from the registry
- `POST /progress/cleanup`: Clean up old completed operations

## Testing

The module includes a comprehensive test suite in `test_progress.py`. Run the tests with:

```bash
python -m unittest asf.medical.llm_gateway.progress.test_progress
```
