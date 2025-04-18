"""
Semantic Field Visualization Example

This script demonstrates how to use the SemanticFieldVisualizer to create
visualizations of Autopoietic Semantic Fields from a medical knowledge graph.

It shows:
1. Basic semantic field visualization
2. Concept clustering visualization
3. Temporal evolution visualization
4. Contradiction hotspot visualization
5. Creating a complete ASF dashboard

The examples use the graph database configured in the application settings.
To run this script, make sure you have a graph database populated with medical
research data and the required Python packages installed.

Required packages:
- networkx
- matplotlib
- pyvis (for interactive visualizations)
- python-louvain (for community detection)
- plotly (optional, for additional visualizations)
"""

import os
import sys
import logging
import argparse
from datetime import datetime, timedelta
from typing import Optional, List, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("asf-visualization-example")

# Import the semantic field visualizer
from ..visualization.semantic_field_visualizer import SemanticFieldVisualizer
from ..graph.graph_service import GraphService

def check_dependencies() -> bool:
    """Check if all required dependencies are installed."""
    missing_deps = []
    
    try:
        import networkx
        logger.info("NetworkX is available")
    except ImportError:
        missing_deps.append("networkx")
    
    try:
        import matplotlib
        logger.info("Matplotlib is available")
    except ImportError:
        missing_deps.append("matplotlib")
    
    try:
        from pyvis.network import Network
        logger.info("PyVis is available")
    except ImportError:
        logger.warning("PyVis is not available. Interactive visualizations will fall back to static images.")
    
    try:
        from community import community_louvain
        logger.info("python-louvain is available")
    except ImportError:
        logger.warning("python-louvain is not available. Concept clustering will be limited.")
    
    if missing_deps:
        logger.error(f"Missing required dependencies: {', '.join(missing_deps)}")
        logger.error("Install them with: pip install " + " ".join(missing_deps))
        return False
    
    return True

def create_basic_visualization(
    output_dir: str, 
    max_nodes: int = 100,
    interactive: bool = True
) -> Optional[str]:
    """
    Create a basic semantic field visualization.
    
    Args:
        output_dir: Directory to save the visualization
        max_nodes: Maximum number of nodes to include
        interactive: Whether to create an interactive visualization
    
    Returns:
        Path to the visualization file or None if visualization fails
    """
    logger.info("Creating basic semantic field visualization...")
    
    # Initialize the visualizer
    visualizer = SemanticFieldVisualizer(output_dir=output_dir)
    
    # Define a query to get interesting subgraph
    # This query finds articles, their claims, and related concepts
    query = """
    MATCH (a:Article)-[:MAKES_CLAIM]->(c:Claim)-[:ABOUT]->(concept)
    RETURN a AS source, {type: 'MAKES_CLAIM'} AS relationship, c AS target
    UNION
    MATCH (c:Claim)-[r:ABOUT]->(concept)
    RETURN c AS source, r AS relationship, concept AS target
    LIMIT $max_nodes
    """
    
    params = {'max_nodes': max_nodes}
    
    # Create the visualization
    vis_path = visualizer.visualize_semantic_field(
        query=query,
        params=params,
        title="Medical Research Semantic Field",
        interactive=interactive
    )
    
    if vis_path:
        logger.info(f"Basic visualization created at: {vis_path}")
    else:
        logger.error("Failed to create basic visualization")
    
    return vis_path

def create_concept_clusters_visualization(
    output_dir: str,
    concept_types: Optional[List[str]] = None,
    max_nodes: int = 150,
    interactive: bool = True
) -> Optional[str]:
    """
    Create a visualization of concept clusters.
    
    Args:
        output_dir: Directory to save the visualization
        concept_types: Types of concepts to include (e.g., ["DISEASE", "DRUG"])
        max_nodes: Maximum number of nodes to include
        interactive: Whether to create an interactive visualization
    
    Returns:
        Path to the visualization file or None if visualization fails
    """
    logger.info("Creating concept clusters visualization...")
    
    # Use default concept types if none provided
    if concept_types is None:
        concept_types = ["Concept", "OntologyConcept", "DISEASE", "DRUG", "GENE"]
    
    # Initialize the visualizer
    visualizer = SemanticFieldVisualizer(output_dir=output_dir)
    
    # Create the visualization
    vis_path = visualizer.visualize_concept_clustering(
        concept_types=concept_types,
        max_nodes=max_nodes,
        title=f"Medical Concept Clusters ({', '.join(concept_types)})",
        interactive=interactive
    )
    
    if vis_path:
        logger.info(f"Concept clusters visualization created at: {vis_path}")
    else:
        logger.error("Failed to create concept clusters visualization")
    
    return vis_path

def create_temporal_evolution_visualization(
    output_dir: str,
    time_range: Optional[Tuple[str, str]] = None,
    interval: str = "month"
) -> Optional[str]:
    """
    Create a visualization of temporal evolution.
    
    Args:
        output_dir: Directory to save the visualization
        time_range: Optional tuple of (start_date, end_date) in ISO format
        interval: Time interval for aggregation ('day', 'week', 'month', 'year')
    
    Returns:
        Path to the visualization file or None if visualization fails
    """
    logger.info("Creating temporal evolution visualization...")
    
    # Initialize the visualizer
    visualizer = SemanticFieldVisualizer(output_dir=output_dir)
    
    # Create default time range if none provided (last 2 years)
    if time_range is None:
        end_date = datetime.now().isoformat()
        start_date = (datetime.now() - timedelta(days=730)).isoformat()  # ~2 years
        time_range = (start_date, end_date)
    
    # Create the visualization
    vis_path = visualizer.visualize_temporal_evolution(
        time_range=time_range,
        interval=interval,
        title="Temporal Evolution of Medical Knowledge"
    )
    
    if vis_path:
        logger.info(f"Temporal evolution visualization created at: {vis_path}")
    else:
        logger.error("Failed to create temporal evolution visualization")
    
    return vis_path

def create_contradiction_hotspots_visualization(
    output_dir: str,
    min_contradictions: int = 1,
    interactive: bool = True
) -> Optional[str]:
    """
    Create a visualization of contradiction hotspots.
    
    Args:
        output_dir: Directory to save the visualization
        min_contradictions: Minimum number of contradictions for inclusion
        interactive: Whether to create an interactive visualization
    
    Returns:
        Path to the visualization file or None if visualization fails
    """
    logger.info("Creating contradiction hotspots visualization...")
    
    # Initialize the visualizer
    visualizer = SemanticFieldVisualizer(output_dir=output_dir)
    
    # Create the visualization
    vis_path = visualizer.visualize_contradiction_hotspots(
        min_contradictions=min_contradictions,
        title="Medical Research Contradiction Hotspots",
        interactive=interactive
    )
    
    if vis_path:
        logger.info(f"Contradiction hotspots visualization created at: {vis_path}")
    else:
        logger.error("Failed to create contradiction hotspots visualization")
    
    return vis_path

def create_full_dashboard(
    output_dir: str,
    time_range: Optional[Tuple[str, str]] = None
) -> Optional[str]:
    """
    Create a full ASF dashboard with multiple visualizations.
    
    Args:
        output_dir: Directory to save the dashboard
        time_range: Optional tuple of (start_date, end_date) in ISO format
    
    Returns:
        Path to the dashboard index file or None if creation fails
    """
    logger.info("Creating full ASF dashboard...")
    
    # Initialize the visualizer
    visualizer = SemanticFieldVisualizer(output_dir=output_dir)
    
    # Create default time range if none provided (last 2 years)
    if time_range is None:
        end_date = datetime.now().isoformat()
        start_date = (datetime.now() - timedelta(days=730)).isoformat()  # ~2 years
        time_range = (start_date, end_date)
    
    # Create the dashboard
    dashboard_path = visualizer.create_asf_dashboard(
        output_dir=os.path.join(output_dir, "asf_dashboard"),
        time_range=time_range
    )
    
    if dashboard_path:
        logger.info(f"Full ASF dashboard created at: {dashboard_path}")
    else:
        logger.error("Failed to create ASF dashboard")
    
    return dashboard_path

def main():
    """Main function to run the example."""
    parser = argparse.ArgumentParser(description="ASF Visualization Example")
    parser.add_argument(
        "--output-dir", 
        default="./asf_visualizations",
        help="Directory to save visualizations"
    )
    parser.add_argument(
        "--max-nodes", 
        type=int, 
        default=150,
        help="Maximum number of nodes to include in visualizations"
    )
    parser.add_argument(
        "--interactive", 
        action="store_true", 
        default=True,
        help="Create interactive visualizations"
    )
    parser.add_argument(
        "--basic", 
        action="store_true",
        help="Create basic semantic field visualization"
    )
    parser.add_argument(
        "--concepts", 
        action="store_true",
        help="Create concept clusters visualization"
    )
    parser.add_argument(
        "--temporal", 
        action="store_true",
        help="Create temporal evolution visualization"
    )
    parser.add_argument(
        "--contradictions", 
        action="store_true",
        help="Create contradiction hotspots visualization"
    )
    parser.add_argument(
        "--dashboard", 
        action="store_true",
        help="Create full ASF dashboard"
    )
    parser.add_argument(
        "--all", 
        action="store_true",
        help="Create all visualizations"
    )
    
    args = parser.parse_args()
    
    # Check if dependencies are installed
    if not check_dependencies():
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check connection to graph database
    try:
        graph_service = GraphService()
        graph_service.connect()
        logger.info("Successfully connected to graph database")
    except Exception as e:
        logger.error(f"Failed to connect to graph database: {e}")
        sys.exit(1)
    
    # Create requested visualizations
    if args.all or args.basic:
        create_basic_visualization(
            output_dir=args.output_dir,
            max_nodes=args.max_nodes,
            interactive=args.interactive
        )
    
    if args.all or args.concepts:
        create_concept_clusters_visualization(
            output_dir=args.output_dir,
            max_nodes=args.max_nodes,
            interactive=args.interactive
        )
    
    if args.all or args.temporal:
        create_temporal_evolution_visualization(
            output_dir=args.output_dir
        )
    
    if args.all or args.contradictions:
        create_contradiction_hotspots_visualization(
            output_dir=args.output_dir,
            interactive=args.interactive
        )
    
    if args.all or args.dashboard:
        create_full_dashboard(
            output_dir=args.output_dir
        )
    
    logger.info("Visualization example completed!")

if __name__ == "__main__":
    main()