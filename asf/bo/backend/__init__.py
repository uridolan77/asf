"""
Business Orchestrator (BO) backend package.

This module initializes the BO backend package structure, ensuring proper
import paths for all submodules.
"""

# Make the package structure properly importable
__path__ = __import__('pkgutil').extend_path(__path__, __name__)
