"""
Visualization package for the Medical Research Synthesizer.

This package provides visualization tools for different aspects of medical research,
including contradiction analysis, semantic fields, knowledge graphs, and concept clustering.

Components:
- ContradictionVisualizer: Visualizes contradiction analysis results
- SemanticFieldVisualizer: Visualizes Autopoietic Semantic Fields from knowledge graphs

Each visualizer is designed to help researchers understand complex medical knowledge
structures and their relationships. The visualizers can be used independently or
integrated into the larger Medical Research Synthesizer application.
"""

from medical.visualization.contradiction_visualizer import ContradictionVisualizer
from medical.visualization.semantic_field_visualizer import SemanticFieldVisualizer

__all__ = ["ContradictionVisualizer", "SemanticFieldVisualizer"]