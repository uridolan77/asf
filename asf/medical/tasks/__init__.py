"""
Tasks Package for the Medical Research Synthesizer.

This package contains background tasks for the Medical Research Synthesizer,
implemented using the Dramatiq task queue. These tasks are designed to run
asynchronously in the background, allowing the API to respond quickly to
requests while long-running operations continue in separate worker processes.

Modules:
- export_tasks: Tasks for exporting data to various formats (PDF, etc.)
- ml_inference_tasks: Tasks for running ML model inference operations

The tasks in this package follow a consistent pattern:
1. They are decorated with @dramatiq.actor to register them with Dramatiq
2. They include proper error handling and logging
3. They store task results in a shared dictionary for status tracking
"""
