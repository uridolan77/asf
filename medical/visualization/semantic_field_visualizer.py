"""
Semantic Field Visualizer for Autopoietic Semantic Fields (ASF).

This module provides visualization capabilities for knowledge graphs representing
Autopoietic Semantic Fields. It visualizes the dynamic evolution of knowledge structures,
semantic relationships, and concept clustering over time.

Features:
- Interactive knowledge graph visualization with hierarchical layouts
- Temporal evolution visualization showing knowledge growth and adaptation
- Concept clustering and community detection visualization
- Contradiction hotspot identification and visualization
- Semantic distance and similarity mapping across the knowledge space
- Dynamic force-directed layouts for exploring evolving semantic relationships
- Export capabilities for web interfaces and static reports

The visualizations help researchers understand how medical knowledge is organized,
how concepts relate to each other, and how knowledge evolves over time in response
to new research findings.
"""

import os
import time
import logging
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from datetime import datetime, timedelta
import json

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap
import networkx as nx

# Import required modules
from ..core.logging_config import get_logger
from ..graph.graph_service import GraphService

# Visualization libraries - imported conditionally to handle optional dependencies
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    from pyvis.network import Network
    PYVIS_AVAILABLE = True
except ImportError:
    PYVIS_AVAILABLE = False

# Configure logger
logger = get_logger(__name__)

# Constants
DEFAULT_OUTPUT_DIR = "./semantic_field_visualizations"
DEFAULT_MAX_NODES = 150  # Limit for clearer visualizations
DEFAULT_SEED = 42  # For reproducible layouts
DEFAULT_COLORMAP = "viridis"  # Default colormap for node coloring
DEFAULT_EDGE_COLORMAP = "plasma"  # Default colormap for edge coloring

# Node colors by type
NODE_COLORS = {
    "Article": "#1f77b4",  # Blue
    "Concept": "#ff7f0e",  # Orange
    "Claim": "#2ca02c",    # Green
    "Finding": "#d62728",  # Red
    "Entity": "#9467bd",   # Purple
    "OntologyConcept": "#8c564b",  # Brown
    "DRUG": "#e377c2",     # Pink
    "DISEASE": "#7f7f7f",  # Gray
    "GENE": "#bcbd22",     # Yellow-green
    "ANATOMY": "#17becf",  # Cyan
    "PROCEDURE": "#aec7e8" # Light blue
}

class SemanticFieldVisualizer:
    """
    Visualizer for Autopoietic Semantic Fields.
    
    This class provides methods for visualizing knowledge graphs as semantic fields,
    showing concept clustering, temporal evolution, and contradiction analysis.
    It supports both static and interactive visualizations using different backends.
    """
    
    def __init__(
        self, 
        graph_service: Optional[GraphService] = None,
        output_dir: str = DEFAULT_OUTPUT_DIR
    ):
        """
        Initialize the semantic field visualizer.
        
        Args:
            graph_service: Service for interacting with the graph database
            output_dir: Directory to save visualizations
        """
        self.graph_service = graph_service or GraphService()
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Check for visualization libraries
        if not PLOTLY_AVAILABLE:
            logger.warning(
                "Plotly not available. Install with: pip install plotly"
            )
        
        if not PYVIS_AVAILABLE:
            logger.warning(
                "PyVis not available. Install with: pip install pyvis"
            )
    
    def _run_graph_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Run a graph query and return the results.
        
        Args:
            query: The Cypher query to run
            params: Optional query parameters
            
        Returns:
            List of query results
        """
        try:
            return self.graph_service.execute_query(query, params or {})
        except Exception as e:
            logger.error(f"Error executing graph query: {e}")
            return []
    
    def _convert_to_networkx(self, results: List[Dict[str, Any]]) -> nx.Graph:
        """
        Convert graph query results to a NetworkX graph.
        
        Args:
            results: List of results from a graph query
            
        Returns:
            NetworkX graph representation
        """
        G = nx.Graph()
        
        # Track nodes and edges to avoid duplicates
        added_nodes = set()
        added_edges = set()
        
        for record in results:
            if 'source' in record and 'target' in record and 'relationship' in record:
                source = record['source']
                target = record['target']
                relationship = record['relationship']
                
                # Extract node properties
                source_id = source.get('id', str(hash(str(source))))
                source_labels = source.get('labels', ['Node'])
                source_props = source.get('properties', {})
                
                target_id = target.get('id', str(hash(str(target))))
                target_labels = target.get('labels', ['Node'])
                target_props = target.get('properties', {})
                
                # Extract relationship properties
                rel_type = relationship.get('type', 'RELATED_TO')
                rel_props = relationship.get('properties', {})
                
                # Add nodes if not already added
                if source_id not in added_nodes:
                    # Choose node color based on first label
                    node_color = NODE_COLORS.get(source_labels[0], "#1f77b4")
                    
                    G.add_node(
                        source_id, 
                        labels=source_labels,
                        **source_props,
                        color=node_color
                    )
                    added_nodes.add(source_id)
                
                if target_id not in added_nodes:
                    # Choose node color based on first label
                    node_color = NODE_COLORS.get(target_labels[0], "#1f77b4")
                    
                    G.add_node(
                        target_id, 
                        labels=target_labels,
                        **target_props,
                        color=node_color
                    )
                    added_nodes.add(target_id)
                
                # Add edge if not already added
                edge_key = (source_id, target_id)
                if edge_key not in added_edges:
                    G.add_edge(
                        source_id, 
                        target_id, 
                        type=rel_type,
                        **rel_props
                    )
                    added_edges.add(edge_key)
        
        return G
    
    def visualize_semantic_field(
        self,
        query: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        title: str = "Semantic Field Visualization",
        output_path: Optional[str] = None,
        max_nodes: int = DEFAULT_MAX_NODES,
        interactive: bool = True
    ) -> str:
        """
        Visualize a semantic field from a knowledge graph.
        
        Args:
            query: The Cypher query to extract the semantic field (if None, uses a default query)
            params: Optional query parameters
            title: Title for the visualization
            output_path: Path to save the visualization (if None, generates a default path)
            max_nodes: Maximum number of nodes to include (for readability)
            interactive: Whether to create an interactive visualization
            
        Returns:
            Path to the saved visualization file
        """
        # Use default query if none provided
        if query is None:
            query = """
            MATCH (n)-[r]->(m)
            RETURN n AS source, r AS relationship, m AS target
            LIMIT $max_nodes
            """
            params = params or {}
            params['max_nodes'] = max_nodes
        
        # Execute the query
        results = self._run_graph_query(query, params)
        
        if not results:
            logger.warning("No results returned from the graph query.")
            return ""
        
        # Convert to NetworkX graph
        G = self._convert_to_networkx(results)
        
        if len(G.nodes) == 0:
            logger.warning("Graph contains no nodes.")
            return ""
        
        logger.info(f"Creating semantic field visualization with {len(G.nodes)} nodes and {len(G.edges)} edges")
        
        # Generate output path if not provided
        if output_path is None:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            output_path = os.path.join(self.output_dir, f"semantic_field_{timestamp}")
        
        # Create interactive visualization
        if interactive and PYVIS_AVAILABLE:
            return self._create_interactive_semantic_field(G, title, output_path)
        else:
            return self._create_static_semantic_field(G, title, output_path)
    
    def _create_static_semantic_field(
        self, 
        G: nx.Graph, 
        title: str, 
        output_path: str
    ) -> str:
        """
        Create a static visualization of a semantic field.
        
        Args:
            G: NetworkX graph
            title: Title for the visualization
            output_path: Path to save the visualization
            
        Returns:
            Path to the saved visualization file
        """
        plt.figure(figsize=(12, 10))
        plt.title(title, fontsize=16)
        
        # Get node positions using a force-directed layout
        pos = nx.spring_layout(G, seed=DEFAULT_SEED)
        
        # Get node colors, sizes, and labels
        node_colors = [G.nodes[node].get('color', '#1f77b4') for node in G.nodes]
        
        # Scale node sizes based on their degree (number of connections)
        node_sizes = [30 + 5 * G.degree[node] for node in G.nodes]
        
        # Create node labels using the most informative attribute
        node_labels = {}
        for node in G.nodes:
            node_data = G.nodes[node]
            if 'name' in node_data:
                node_labels[node] = node_data['name']
            elif 'title' in node_data:
                node_labels[node] = node_data['title'][:30] + '...' if len(node_data['title']) > 30 else node_data['title']
            elif 'text' in node_data:
                node_labels[node] = node_data['text'][:30] + '...' if len(node_data['text']) > 30 else node_data['text']
            else:
                # Use first label and id as fallback
                label_text = node_data.get('labels', ['Node'])[0]
                node_id = str(node)[:8]
                node_labels[node] = f"{label_text}_{node_id}"
        
        # Draw the graph
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8)
        nx.draw_networkx_edges(G, pos, alpha=0.5, edge_color="#aaaaaa")
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8, font_color="black")
        
        # Create legend for node types
        unique_node_types = set()
        for node in G.nodes:
            labels = G.nodes[node].get('labels', ['Node'])
            if labels:
                unique_node_types.add(labels[0])
        
        # Add legend
        legend_handles = []
        for node_type in unique_node_types:
            color = NODE_COLORS.get(node_type, "#1f77b4")
            legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', 
                                  markerfacecolor=color, markersize=10, label=node_type))
        
        if legend_handles:
            plt.legend(handles=legend_handles, loc='upper right', fontsize=10)
        
        plt.axis('off')
        plt.tight_layout()
        
        # Save the visualization
        if not output_path.endswith('.png') and not output_path.endswith('.pdf'):
            output_path += '.png'
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Static semantic field visualization saved to {output_path}")
        return output_path
    
    def _create_interactive_semantic_field(
        self, 
        G: nx.Graph, 
        title: str, 
        output_path: str
    ) -> str:
        """
        Create an interactive visualization of a semantic field.
        
        Args:
            G: NetworkX graph
            title: Title for the visualization
            output_path: Path to save the visualization
            
        Returns:
            Path to the saved visualization file
        """
        if not PYVIS_AVAILABLE:
            logger.warning("PyVis not available. Falling back to static visualization.")
            return self._create_static_semantic_field(G, title, output_path)
        
        # Create a PyVis network
        net = Network(
            height="800px", 
            width="100%", 
            bgcolor="#ffffff", 
            font_color="black",
            heading=title
        )
        
        # Set network options
        net.barnes_hut(gravity=-80000, central_gravity=0.3, spring_length=250)
        net.toggle_physics(True)
        net.show_buttons(filter_=['physics'])
        
        # Add nodes
        for node_id in G.nodes:
            node_data = G.nodes[node_id]
            
            # Get node label
            if 'name' in node_data:
                label = node_data['name']
            elif 'title' in node_data:
                label = node_data['title'][:50] + '...' if len(node_data['title']) > 50 else node_data['title']
            elif 'text' in node_data:
                label = node_data['text'][:50] + '...' if len(node_data['text']) > 50 else node_data['text']
            else:
                # Use first label and id as fallback
                label_text = node_data.get('labels', ['Node'])[0]
                node_short_id = str(node_id)[:8]
                label = f"{label_text}_{node_short_id}"
            
            # Get node type and color
            node_type = node_data.get('labels', ['Node'])[0]
            color = node_data.get('color', NODE_COLORS.get(node_type, "#1f77b4"))
            
            # Scale node size based on degree (number of connections)
            size = 15 + 2 * G.degree[node_id]
            
            # Prepare title (tooltip) with key attributes
            tooltip_attrs = []
            for key, value in node_data.items():
                if key not in ['color', 'labels'] and not isinstance(value, (list, dict)):
                    tooltip_attrs.append(f"{key}: {value}")
            
            title = "<br>".join(tooltip_attrs[:10])  # Limit to first 10 attributes
            
            # Add node to network
            net.add_node(
                node_id, 
                label=label, 
                title=title,
                color=color,
                size=size,
                shape="dot"
            )
        
        # Add edges
        for source, target, edge_data in G.edges(data=True):
            # Get edge type
            edge_type = edge_data.get('type', 'RELATED_TO')
            
            # Prepare tooltip with key attributes
            tooltip_attrs = [f"type: {edge_type}"]
            for key, value in edge_data.items():
                if key != 'type' and not isinstance(value, (list, dict)):
                    tooltip_attrs.append(f"{key}: {value}")
            
            title = "<br>".join(tooltip_attrs[:5])  # Limit to first 5 attributes
            
            # Add edge to network
            net.add_edge(
                source, 
                target, 
                title=title,
                label=edge_type,
                arrowStrikethrough=False,
                width=1.5
            )
        
        # Save the visualization
        if not output_path.endswith('.html'):
            output_path += '.html'
        
        net.save_graph(output_path)
        logger.info(f"Interactive semantic field visualization saved to {output_path}")
        return output_path
    
    def visualize_temporal_evolution(
        self,
        time_range: Optional[Tuple[str, str]] = None,
        interval: str = "month",
        concept_types: Optional[List[str]] = None,
        title: str = "Temporal Evolution of Semantic Fields",
        output_path: Optional[str] = None
    ) -> str:
        """
        Visualize the temporal evolution of the semantic field.
        
        Args:
            time_range: Optional tuple of (start_date, end_date) in ISO format
            interval: Time interval for aggregation ('day', 'week', 'month', 'year')
            concept_types: List of concept types to include
            title: Title for the visualization
            output_path: Path to save the visualization
            
        Returns:
            Path to the saved visualization file
        """
        # Generate output path if not provided
        if output_path is None:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            output_path = os.path.join(self.output_dir, f"temporal_evolution_{timestamp}.png")
        
        # Build query parameters
        params = {}
        date_condition = ""
        
        if time_range:
            start_date, end_date = time_range
            date_condition = "WHERE date.timestamp >= $start_date AND date.timestamp <= $end_date"
            params['start_date'] = start_date
            params['end_date'] = end_date
        
        # Build type filter
        type_filter = ""
        if concept_types:
            type_list = ", ".join([f"'{t}'" for t in concept_types])
            type_filter = f"AND any(label IN labels(n) WHERE label IN [{type_list}])"
        
        # Query to get temporal data
        query = f"""
        MATCH (n)-[r]->(date:Date) {date_condition}
        {type_filter}
        RETURN date.timestamp AS timestamp, count(distinct n) AS node_count,
               labels(n) AS node_types, count(distinct r) AS connection_count
        ORDER BY timestamp
        """
        
        # Execute the query
        results = self._run_graph_query(query, params)
        
        if not results:
            logger.warning("No temporal data found.")
            return ""
        
        # Process results
        timestamps = []
        node_counts = []
        connection_counts = []
        node_type_counts = {}
        
        for record in results:
            timestamp = record.get('timestamp')
            node_count = record.get('node_count', 0)
            connection_count = record.get('connection_count', 0)
            node_types = record.get('node_types', [])
            
            timestamps.append(datetime.fromisoformat(timestamp))
            node_counts.append(node_count)
            connection_counts.append(connection_count)
            
            # Count nodes by type
            for node_type in node_types:
                if node_type not in node_type_counts:
                    node_type_counts[node_type] = []
                node_type_counts[node_type].append(node_count)
        
        # Create the visualization
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        # Plot node growth
        ax1.plot(timestamps, node_counts, 'b-', linewidth=2, label='Total Nodes')
        ax1.set_ylabel('Number of Nodes')
        ax1.set_title(f'{title} - Node Growth', fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        # Plot connection growth
        ax2.plot(timestamps, connection_counts, 'r-', linewidth=2, label='Total Connections')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Number of Connections')
        ax2.set_title('Connection Growth', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        # Format x-axis dates
        if interval == 'day':
            ax2.xaxis.set_major_locator(mdates.DayLocator(interval=7))
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        elif interval == 'week':
            ax2.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%W'))
        elif interval == 'month':
            ax2.xaxis.set_major_locator(mdates.MonthLocator())
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        else:  # year
            ax2.xaxis.set_major_locator(mdates.YearLocator())
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        
        plt.xticks(rotation=45)
        
        # Add legends
        ax1.legend()
        ax2.legend()
        
        plt.tight_layout()
        
        # Save the visualization
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Temporal evolution visualization saved to {output_path}")
        return output_path
    
    def visualize_concept_clustering(
        self,
        concept_types: Optional[List[str]] = None,
        max_nodes: int = DEFAULT_MAX_NODES,
        title: str = "Concept Clustering Analysis",
        output_path: Optional[str] = None,
        interactive: bool = True
    ) -> str:
        """
        Visualize concept clustering in the semantic field.
        
        Args:
            concept_types: List of concept types to include
            max_nodes: Maximum number of nodes to include
            title: Title for the visualization
            output_path: Path to save the visualization
            interactive: Whether to create an interactive visualization
            
        Returns:
            Path to the saved visualization file
        """
        # Generate output path if not provided
        if output_path is None:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            output_path = os.path.join(
                self.output_dir, 
                f"concept_clusters_{timestamp}" + ('.html' if interactive and PYVIS_AVAILABLE else '.png')
            )
        
        # Build type filter
        type_filter = ""
        params = {'max_nodes': max_nodes}
        
        if concept_types:
            type_list = ", ".join([f"'{t}'" for t in concept_types])
            type_filter = f"WHERE any(label IN labels(n) WHERE label IN [{type_list}]) " + \
                          f"AND any(label IN labels(m) WHERE label IN [{type_list}])"
        
        # Query to get concept relationships
        query = f"""
        MATCH (n)-[r]->(m)
        {type_filter}
        RETURN n AS source, r AS relationship, m AS target
        LIMIT $max_nodes
        """
        
        # Execute the query
        results = self._run_graph_query(query, params)
        
        if not results:
            logger.warning("No concept clustering data found.")
            return ""
        
        # Convert to NetworkX graph
        G = self._convert_to_networkx(results)
        
        if len(G.nodes) == 0:
            logger.warning("Graph contains no nodes for concept clustering.")
            return ""
        
        # Detect communities using Louvain method
        try:
            from community import community_louvain
            partition = community_louvain.best_partition(G)
            
            # Add community information to nodes
            nx.set_node_attributes(G, partition, 'community')
            
            # Get unique communities
            communities = set(partition.values())
            logger.info(f"Detected {len(communities)} concept clusters")
            
            # Generate colors for communities
            colormap = plt.cm.get_cmap(DEFAULT_COLORMAP, len(communities))
            community_colors = {i: f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}" 
                              for i, (r, g, b, _) in enumerate(colormap(range(len(communities))))}
            
            # Update node colors based on communities
            for node in G.nodes:
                community = G.nodes[node].get('community', 0)
                G.nodes[node]['color'] = community_colors.get(community, "#1f77b4")
            
        except ImportError:
            logger.warning("python-louvain package not available. Install with: pip install python-louvain")
        
        # Create the visualization
        if interactive and PYVIS_AVAILABLE:
            return self._create_interactive_semantic_field(G, f"{title} - {len(G.nodes)} nodes", output_path)
        else:
            return self._create_static_semantic_field(G, f"{title} - {len(G.nodes)} nodes", output_path)
    
    def visualize_contradiction_hotspots(
        self,
        min_contradictions: int = 1,
        title: str = "Contradiction Hotspots",
        output_path: Optional[str] = None,
        interactive: bool = True
    ) -> str:
        """
        Visualize contradiction hotspots in the semantic field.
        
        Args:
            min_contradictions: Minimum number of contradictions for inclusion
            title: Title for the visualization
            output_path: Path to save the visualization
            interactive: Whether to create an interactive visualization
            
        Returns:
            Path to the saved visualization file
        """
        # Generate output path if not provided
        if output_path is None:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            output_path = os.path.join(
                self.output_dir, 
                f"contradiction_hotspots_{timestamp}" + ('.html' if interactive and PYVIS_AVAILABLE else '.png')
            )
        
        # Query to find contradiction hotspots
        query = """
        MATCH (a:Article)-[:MAKES_CLAIM]->(c1:Claim)
        MATCH (b:Article)-[:MAKES_CLAIM]->(c2:Claim)
        MATCH (c1)-[r:CONTRADICTS]->(c2)
        WHERE a <> b
        WITH c1, c2, r, count(r) AS contradiction_count
        WHERE contradiction_count >= $min_contradictions
        MATCH (c1)-[:ABOUT]->(topic)
        RETURN c1 AS source, r AS relationship, c2 AS target, topic
        """
        
        params = {'min_contradictions': min_contradictions}
        
        # Execute the query
        results = self._run_graph_query(query, params)
        
        if not results:
            logger.warning("No contradiction hotspots found.")
            return ""
        
        # Convert to NetworkX graph
        G = self._convert_to_networkx(results)
        
        if len(G.nodes) == 0:
            logger.warning("Graph contains no nodes for contradiction hotspots.")
            return ""
        
        # Highlight contradiction relationships
        for u, v, data in G.edges(data=True):
            if data.get('type') == 'CONTRADICTS':
                data['color'] = '#ff0000'  # Red for contradictions
                data['width'] = 3
        
        # Create the visualization
        if interactive and PYVIS_AVAILABLE:
            return self._create_interactive_semantic_field(G, f"{title} - {len(G.nodes)} nodes", output_path)
        else:
            return self._create_static_semantic_field(G, f"{title} - {len(G.nodes)} nodes", output_path)
    
    def create_asf_dashboard(
        self, 
        output_dir: Optional[str] = None,
        time_range: Optional[Tuple[str, str]] = None
    ) -> str:
        """
        Create a comprehensive ASF dashboard with multiple visualizations.
        
        Args:
            output_dir: Directory to save the dashboard files
            time_range: Optional tuple of (start_date, end_date) in ISO format
            
        Returns:
            Path to the dashboard index file
        """
        # Use provided output directory or create a new one
        if output_dir is None:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            output_dir = os.path.join(self.output_dir, f"asf_dashboard_{timestamp}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate all visualization types
        logger.info("Generating ASF dashboard visualizations...")
        
        # 1. Overall semantic field
        semantic_field_path = self.visualize_semantic_field(
            title="Current Semantic Field",
            output_path=os.path.join(output_dir, "semantic_field.html"),
            interactive=True
        )
        
        # 2. Concept clustering
        concept_clusters_path = self.visualize_concept_clustering(
            title="Concept Clustering", 
            output_path=os.path.join(output_dir, "concept_clusters.html"),
            interactive=True
        )
        
        # 3. Temporal evolution
        temporal_evolution_path = self.visualize_temporal_evolution(
            time_range=time_range,
            title="Knowledge Evolution Over Time",
            output_path=os.path.join(output_dir, "temporal_evolution.png")
        )
        
        # 4. Contradiction hotspots
        contradictions_path = self.visualize_contradiction_hotspots(
            title="Contradiction Hotspots",
            output_path=os.path.join(output_dir, "contradiction_hotspots.html"),
            interactive=True
        )
        
        # Create dashboard index.html
        index_path = os.path.join(output_dir, "index.html")
        
        with open(index_path, "w") as f:
            f.write(f"""<!DOCTYPE html>
<html>
<head>
    <title>Autopoietic Semantic Field Dashboard</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {{ font-family: Arial, sans-serif; margin: 0; padding: 0; }}
        .dashboard {{ display: flex; flex-direction: column; padding: 20px; }}
        .header {{ background-color: #2c3e50; color: white; padding: 20px; text-align: center; }}
        .row {{ display: flex; flex-wrap: wrap; margin: 0 -10px; }}
        .card {{ flex: 1; margin: 10px; background-color: white; border-radius: 5px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }}
        .card-header {{ background-color: #3498db; color: white; padding: 10px; border-radius: 5px 5px 0 0; }}
        .card-body {{ padding: 15px; }}
        iframe {{ width: 100%; height: 600px; border: none; }}
        img {{ max-width: 100%; height: auto; }}
        @media (max-width: 768px) {{ 
            .row {{ flex-direction: column; }} 
            .card {{ margin: 10px 0; }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Autopoietic Semantic Field Dashboard</h1>
        <p>Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    </div>
    
    <div class="dashboard">
        <div class="row">
            <div class="card">
                <div class="card-header">
                    <h2>Current Semantic Field</h2>
                </div>
                <div class="card-body">
                    <iframe src="semantic_field.html"></iframe>
                </div>
            </div>
        </div>
        
        <div class="row">
            <div class="card">
                <div class="card-header">
                    <h2>Concept Clusters</h2>
                </div>
                <div class="card-body">
                    <iframe src="concept_clusters.html"></iframe>
                </div>
            </div>
        </div>
        
        <div class="row">
            <div class="card">
                <div class="card-header">
                    <h2>Knowledge Evolution Over Time</h2>
                </div>
                <div class="card-body">
                    <img src="temporal_evolution.png" alt="Temporal Evolution">
                </div>
            </div>
        </div>
        
        <div class="row">
            <div class="card">
                <div class="card-header">
                    <h2>Contradiction Hotspots</h2>
                </div>
                <div class="card-body">
                    <iframe src="contradiction_hotspots.html"></iframe>
                </div>
            </div>
        </div>
    </div>
</body>
</html>
""")
        
        logger.info(f"ASF dashboard created at {index_path}")
        return index_path