"""
Patch module to disable observability components.

This module was previously imported by the observability components to provide dummy implementations.
Now it's been updated to completely disable functionality rather than use dummy implementations.
"""

import logging

logger = logging.getLogger(__name__)
logger.info("Observability components disabled via patch module")

# This module is now empty - no dummy implementations are provided
# Each module (metrics.py, tracing.py, prometheus.py) now has its own simplified no-op implementation
