"""
Test script for Prometheus exporter.
"""

import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from asf.medical.llm_gateway.observability.prometheus import PrometheusExporter

def main():
    """Main function."""
    print("Testing Prometheus exporter...")
    exporter = PrometheusExporter()
    print("Prometheus exporter initialized successfully!")

if __name__ == "__main__":
    main()
