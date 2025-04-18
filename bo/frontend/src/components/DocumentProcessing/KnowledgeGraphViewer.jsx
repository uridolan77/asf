import React, { useState, useEffect, useRef } from 'react';
import {
  Box,
  Paper,
  Typography,
  Slider,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Chip,
  ToggleButton,
  ToggleButtonGroup,
  TextField,
  IconButton,
  Tooltip,
  CircularProgress,
  Alert
} from '@mui/material';
import {
  ZoomIn as ZoomInIcon,
  ZoomOut as ZoomOutIcon,
  FilterList as FilterListIcon,
  Search as SearchIcon,
  Save as SaveIcon,
  Refresh as RefreshIcon,
  LocalHospital as DiseaseIcon,
  MedicalInformation as DrugIcon,
  Biotech as GeneIcon,
  Science as ChemicalIcon,
  Psychology as SymptomIcon,
  Healing as TreatmentIcon
} from '@mui/icons-material';
import ForceGraph2D from 'react-force-graph-2d';
import ForceGraph3D from 'react-force-graph-3d';
import { saveAs } from 'file-saver';

/**
 * Interactive Knowledge Graph Viewer component
 *
 * This component visualizes knowledge graphs extracted from documents
 * with interactive features like filtering, searching, and zooming.
 */
const KnowledgeGraphViewer = ({ graphData, isLoading, error }) => {
  const [dimensions, setDimensions] = useState({ width: 800, height: 600 });
  const [viewMode, setViewMode] = useState('2d');
  const [layoutAlgorithm, setLayoutAlgorithm] = useState('force');
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.5);
  const [selectedEntityTypes, setSelectedEntityTypes] = useState([]);
  const [selectedRelationTypes, setSelectedRelationTypes] = useState([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [highlightedNodes, setHighlightedNodes] = useState(new Set());
  const [filteredGraph, setFilteredGraph] = useState(null);
  const [availableEntityTypes, setAvailableEntityTypes] = useState([]);
  const [availableRelationTypes, setAvailableRelationTypes] = useState([]);
  const [nodeSize, setNodeSize] = useState(5);
  const [linkDistance, setLinkDistance] = useState(100);
  const [chargeStrength, setChargeStrength] = useState(-30);
  const [enableClustering, setEnableClustering] = useState(false);
  const [clusterThreshold, setClusterThreshold] = useState(10);
  const [clusterRadius, setClusterRadius] = useState(50);
  const [clusteredGraph, setClusteredGraph] = useState(null);

  const graphRef = useRef();
  const containerRef = useRef();

  // Entity type to color mapping
  const entityColors = {
    'DISEASE': '#f44336', // red
    'DRUG': '#2196f3',    // blue
    'GENE': '#4caf50',    // green
    'CHEMICAL': '#9c27b0', // purple
    'SYMPTOM': '#ff9800',  // orange
    'TREATMENT': '#00bcd4', // cyan
    'PROCEDURE': '#795548', // brown
    'ORGANISM': '#607d8b',  // blue-grey
    'DEFAULT': '#9e9e9e'    // grey
  };

  // Relation type to color mapping
  const relationColors = {
    'treats': '#4caf50',      // green
    'causes': '#f44336',      // red
    'diagnoses': '#2196f3',   // blue
    'prevents': '#00bcd4',    // cyan
    'complicates': '#ff9800', // orange
    'predisposes': '#9c27b0', // purple
    'associated_with': '#607d8b', // blue-grey
    'DEFAULT': '#9e9e9e'      // grey
  };

  // Icons for entity types
  const entityIcons = {
    'DISEASE': <DiseaseIcon />,
    'DRUG': <DrugIcon />,
    'GENE': <GeneIcon />,
    'CHEMICAL': <ChemicalIcon />,
    'SYMPTOM': <SymptomIcon />,
    'TREATMENT': <TreatmentIcon />
  };

  // Update dimensions when container size changes
  useEffect(() => {
    if (containerRef.current) {
      const updateDimensions = () => {
        if (containerRef.current) {
          setDimensions({
            width: containerRef.current.offsetWidth,
            height: Math.max(500, window.innerHeight * 0.6)
          });
        }
      };

      updateDimensions();
      window.addEventListener('resize', updateDimensions);

      return () => {
        window.removeEventListener('resize', updateDimensions);
      };
    }
  }, [containerRef]);

  // Extract available entity and relation types from graph data
  useEffect(() => {
    if (graphData && graphData.nodes && graphData.links) {
      // Extract entity types
      const entityTypes = [...new Set(graphData.nodes.map(node => node.type))];
      setAvailableEntityTypes(entityTypes);

      // Extract relation types
      const relationTypes = [...new Set(graphData.links.map(link => link.type))];
      setAvailableRelationTypes(relationTypes);

      // Set all types as selected by default
      setSelectedEntityTypes(entityTypes);
      setSelectedRelationTypes(relationTypes);
    }
  }, [graphData]);

  // Filter graph based on selected criteria
  useEffect(() => {
    if (!graphData) return;

    // Filter nodes based on entity types and search term
    const filteredNodes = graphData.nodes.filter(node => {
      const matchesType = selectedEntityTypes.length === 0 || selectedEntityTypes.includes(node.type);
      const matchesSearch = !searchTerm || node.id.toLowerCase().includes(searchTerm.toLowerCase());
      return matchesType && matchesSearch;
    });

    // Get IDs of filtered nodes
    const nodeIds = new Set(filteredNodes.map(node => node.id));

    // Filter links based on relation types, confidence threshold, and connected to filtered nodes
    const filteredLinks = graphData.links.filter(link => {
      const matchesType = selectedRelationTypes.length === 0 || selectedRelationTypes.includes(link.type);
      const matchesConfidence = link.confidence >= confidenceThreshold;
      const nodesExist = nodeIds.has(link.source.id || link.source) && nodeIds.has(link.target.id || link.target);
      return matchesType && matchesConfidence && nodesExist;
    });

    // Create filtered graph
    const newFilteredGraph = {
      nodes: filteredNodes,
      links: filteredLinks
    };

    setFilteredGraph(newFilteredGraph);

    // Apply clustering if enabled
    if (enableClustering && filteredNodes.length > clusterThreshold) {
      const clustered = clusterGraph(newFilteredGraph);
      setClusteredGraph(clustered);
    } else {
      setClusteredGraph(null);
    }

    // Update highlighted nodes based on search
    if (searchTerm) {
      const highlighted = new Set(
        filteredNodes
          .filter(node => node.id.toLowerCase().includes(searchTerm.toLowerCase()))
          .map(node => node.id)
      );
      setHighlightedNodes(highlighted);
    } else {
      setHighlightedNodes(new Set());
    }
  }, [graphData, selectedEntityTypes, selectedRelationTypes, confidenceThreshold, searchTerm, enableClustering, clusterThreshold, clusterRadius]);

  // Handle view mode change
  const handleViewModeChange = (event, newMode) => {
    if (newMode !== null) {
      setViewMode(newMode);
    }
  };

  // Handle layout algorithm change
  const handleLayoutChange = (event) => {
    setLayoutAlgorithm(event.target.value);

    // Reset the graph physics when changing layout
    if (graphRef.current) {
      setTimeout(() => {
        graphRef.current.d3Force('charge', null);
        graphRef.current.d3Force('link', null);
        graphRef.current.d3Force('center', null);

        // Apply new forces based on selected layout
        applyLayoutForces();

        // Reheat the simulation
        graphRef.current.d3ReheatSimulation();
      }, 100);
    }
  };

  // Apply forces based on selected layout algorithm
  const applyLayoutForces = () => {
    if (!graphRef.current) return;

    const d3 = graphRef.current.d3Force;

    switch (layoutAlgorithm) {
      case 'force':
        // Standard force-directed layout
        d3('charge').strength(chargeStrength);
        d3('link').distance(linkDistance);
        break;
      case 'radial':
        // Radial layout
        d3('charge').strength(chargeStrength * 0.7);
        d3('link').distance(linkDistance * 0.8);
        // Add radial force
        d3('radial', d3.forceRadial()
          .radius(Math.min(dimensions.width, dimensions.height) / 3)
          .strength(0.8)
        );
        break;
      case 'circular':
        // Circular layout
        d3('charge').strength(-10);
        d3('link').distance(30);
        // Position nodes in a circle
        if (filteredGraph && filteredGraph.nodes.length > 0) {
          const radius = Math.min(dimensions.width, dimensions.height) / 2.5;
          const angleStep = (2 * Math.PI) / filteredGraph.nodes.length;

          filteredGraph.nodes.forEach((node, i) => {
            const angle = i * angleStep;
            node.x = dimensions.width / 2 + radius * Math.cos(angle);
            node.y = dimensions.height / 2 + radius * Math.sin(angle);
            node.fx = node.x;
            node.fy = node.y;
          });

          // Release fixed positions after a delay
          setTimeout(() => {
            if (filteredGraph && filteredGraph.nodes) {
              filteredGraph.nodes.forEach(node => {
                node.fx = null;
                node.fy = null;
              });
            }
          }, 2000);
        }
        break;
      case 'hierarchical':
        // Hierarchical layout
        d3('charge').strength(chargeStrength * 0.5);
        d3('link').distance(linkDistance * 1.5);

        // Arrange nodes in levels based on their connections
        if (filteredGraph && filteredGraph.nodes.length > 0) {
          // Find root nodes (nodes with only outgoing links)
          const nodeLinks = {};
          filteredGraph.links.forEach(link => {
            const sourceId = link.source.id || link.source;
            const targetId = link.target.id || link.target;

            if (!nodeLinks[sourceId]) nodeLinks[sourceId] = { in: 0, out: 0 };
            if (!nodeLinks[targetId]) nodeLinks[targetId] = { in: 0, out: 0 };

            nodeLinks[sourceId].out++;
            nodeLinks[targetId].in++;
          });

          // Assign levels
          const levels = {};
          const assignedNodes = new Set();

          // Start with root nodes (nodes with no incoming links)
          const rootNodes = filteredGraph.nodes.filter(node =>
            !nodeLinks[node.id] || nodeLinks[node.id].in === 0
          );

          rootNodes.forEach(node => {
            levels[node.id] = 0;
            assignedNodes.add(node.id);
          });

          // Assign levels to the rest of the nodes
          let changed = true;
          while (changed) {
            changed = false;
            filteredGraph.links.forEach(link => {
              const sourceId = link.source.id || link.source;
              const targetId = link.target.id || link.target;

              if (assignedNodes.has(sourceId) && !assignedNodes.has(targetId)) {
                levels[targetId] = levels[sourceId] + 1;
                assignedNodes.add(targetId);
                changed = true;
              }
            });
          }

          // Position nodes based on levels
          const levelCounts = {};
          const levelPositions = {};

          // Count nodes per level
          Object.entries(levels).forEach(([nodeId, level]) => {
            if (!levelCounts[level]) levelCounts[level] = 0;
            levelCounts[level]++;
          });

          // Calculate positions
          filteredGraph.nodes.forEach(node => {
            const level = levels[node.id] || 0;
            if (!levelPositions[level]) levelPositions[level] = 0;

            const levelWidth = dimensions.width * 0.8;
            const levelHeight = dimensions.height * 0.8;
            const levelCount = levelCounts[level] || 1;

            node.fx = (levelPositions[level] + 0.5) * (levelWidth / levelCount) + dimensions.width * 0.1;
            node.fy = level * (levelHeight / (Object.keys(levelCounts).length || 1)) + dimensions.height * 0.1;

            levelPositions[level]++;
          });

          // Release fixed positions after a delay
          setTimeout(() => {
            if (filteredGraph && filteredGraph.nodes) {
              filteredGraph.nodes.forEach(node => {
                node.fx = null;
                node.fy = null;
              });
            }
          }, 3000);
        }
        break;
      default:
        // Default force-directed layout
        d3('charge').strength(chargeStrength);
        d3('link').distance(linkDistance);
    }
  };

  // Handle confidence threshold change
  const handleConfidenceChange = (event, newValue) => {
    setConfidenceThreshold(newValue);
  };

  // Handle node size change
  const handleNodeSizeChange = (event, newValue) => {
    setNodeSize(newValue);
  };

  // Handle link distance change
  const handleLinkDistanceChange = (event, newValue) => {
    setLinkDistance(newValue);

    // Update force simulation
    if (graphRef.current) {
      graphRef.current.d3Force('link').distance(newValue);
      graphRef.current.d3ReheatSimulation();
    }
  };

  // Handle charge strength change
  const handleChargeStrengthChange = (event, newValue) => {
    setChargeStrength(newValue);

    // Update force simulation
    if (graphRef.current) {
      graphRef.current.d3Force('charge').strength(newValue);
      graphRef.current.d3ReheatSimulation();
    }
  };

  // Handle clustering toggle
  const handleClusteringToggle = (event) => {
    setEnableClustering(event.target.checked);
  };

  // Handle cluster threshold change
  const handleClusterThresholdChange = (event, newValue) => {
    setClusterThreshold(newValue);
  };

  // Handle cluster radius change
  const handleClusterRadiusChange = (event, newValue) => {
    setClusterRadius(newValue);
  };

  // Cluster graph nodes by type
  const clusterGraph = (graph) => {
    if (!graph || !graph.nodes || graph.nodes.length <= clusterThreshold) {
      return graph;
    }

    // Group nodes by type
    const nodesByType = {};
    graph.nodes.forEach(node => {
      if (!nodesByType[node.type]) {
        nodesByType[node.type] = [];
      }
      nodesByType[node.type].push(node);
    });

    // Create cluster nodes and map original nodes to clusters
    const clusters = [];
    const nodeToCluster = {};
    const clusterNodes = [];

    Object.entries(nodesByType).forEach(([type, nodes]) => {
      // If there are enough nodes of this type, create clusters
      if (nodes.length > clusterThreshold) {
        // Group nodes into clusters using a simple spatial partitioning
        // This is a simplified approach - in a real app, you might use a more sophisticated algorithm
        const clustersOfType = [];

        // Create initial cluster centers
        const numClusters = Math.ceil(nodes.length / clusterThreshold);
        for (let i = 0; i < numClusters; i++) {
          clustersOfType.push({
            id: `cluster-${type}-${i}`,
            type: `${type}-Cluster`,
            nodes: [],
            x: Math.random() * dimensions.width,
            y: Math.random() * dimensions.height,
            clusterId: clusters.length + i
          });
        }

        // Assign nodes to nearest cluster
        nodes.forEach(node => {
          let nearestCluster = clustersOfType[0];
          let minDistance = Infinity;

          clustersOfType.forEach(cluster => {
            // Use Euclidean distance if node has position, otherwise random assignment
            let distance;
            if (node.x !== undefined && node.y !== undefined) {
              distance = Math.sqrt(Math.pow(node.x - cluster.x, 2) + Math.pow(node.y - cluster.y, 2));
            } else {
              distance = Math.random() * 1000;
            }

            if (distance < minDistance) {
              minDistance = distance;
              nearestCluster = cluster;
            }
          });

          nearestCluster.nodes.push(node);
          nodeToCluster[node.id] = nearestCluster.id;
        });

        // Add clusters to the list
        clusters.push(...clustersOfType);
      } else {
        // If not enough nodes, keep them as individual nodes
        nodes.forEach(node => {
          clusterNodes.push(node);
        });
      }
    });

    // Create cluster nodes for visualization
    const visualClusterNodes = clusters.map(cluster => ({
      id: cluster.id,
      type: cluster.type,
      nodeCount: cluster.nodes.length,
      entityType: cluster.type.split('-')[0],
      x: cluster.x,
      y: cluster.y,
      isCluster: true
    }));

    // Add individual nodes and cluster nodes to the final node list
    const finalNodes = [...clusterNodes, ...visualClusterNodes];

    // Create links between clusters and non-clustered nodes
    const finalLinks = [];
    graph.links.forEach(link => {
      const sourceId = link.source.id || link.source;
      const targetId = link.target.id || link.target;

      const sourceClusterId = nodeToCluster[sourceId];
      const targetClusterId = nodeToCluster[targetId];

      // If both nodes are in clusters, create a link between clusters
      if (sourceClusterId && targetClusterId) {
        // Only add link if clusters are different
        if (sourceClusterId !== targetClusterId) {
          // Check if this link already exists
          const existingLink = finalLinks.find(l =>
            (l.source === sourceClusterId && l.target === targetClusterId) ||
            (l.source === targetClusterId && l.target === sourceClusterId)
          );

          if (!existingLink) {
            finalLinks.push({
              source: sourceClusterId,
              target: targetClusterId,
              type: link.type,
              confidence: link.confidence,
              isClusterLink: true
            });
          }
        }
      }
      // If only source is in a cluster
      else if (sourceClusterId) {
        finalLinks.push({
          source: sourceClusterId,
          target: targetId,
          type: link.type,
          confidence: link.confidence,
          isClusterLink: true
        });
      }
      // If only target is in a cluster
      else if (targetClusterId) {
        finalLinks.push({
          source: sourceId,
          target: targetClusterId,
          type: link.type,
          confidence: link.confidence,
          isClusterLink: true
        });
      }
      // If neither node is in a cluster, keep the original link
      else {
        finalLinks.push(link);
      }
    });

    return { nodes: finalNodes, links: finalLinks };
  };

  // Handle entity type selection
  const handleEntityTypeChange = (event) => {
    setSelectedEntityTypes(event.target.value);
  };

  // Handle relation type selection
  const handleRelationTypeChange = (event) => {
    setSelectedRelationTypes(event.target.value);
  };

  // Handle search
  const handleSearch = (event) => {
    setSearchTerm(event.target.value);
  };

  // Handle zoom in
  const handleZoomIn = () => {
    if (graphRef.current) {
      graphRef.current.zoomIn();
    }
  };

  // Handle zoom out
  const handleZoomOut = () => {
    if (graphRef.current) {
      graphRef.current.zoomOut();
    }
  };

  // Handle graph export
  const handleExport = (format = 'json') => {
    // Determine which graph to export (clustered or filtered)
    const graphToExport = enableClustering && clusteredGraph ? clusteredGraph : filteredGraph;
    if (!graphToExport) return;

    switch (format) {
      case 'json':
        // Export as JSON
        const graphJson = JSON.stringify(graphToExport, null, 2);
        const jsonBlob = new Blob([graphJson], { type: 'application/json' });
        saveAs(jsonBlob, 'knowledge_graph.json');
        break;

      case 'csv':
        // Export nodes as CSV
        const nodesCsv = ['id,type,label'];
        graphToExport.nodes.forEach(node => {
          nodesCsv.push(`${node.id},${node.type},${node.id}`);
        });

        // Export links as CSV
        const linksCsv = ['source,target,type,confidence'];
        graphToExport.links.forEach(link => {
          const sourceId = link.source.id || link.source;
          const targetId = link.target.id || link.target;
          linksCsv.push(`${sourceId},${targetId},${link.type},${link.confidence}`);
        });

        // Create and save nodes CSV
        const nodesCsvBlob = new Blob([nodesCsv.join('\n')], { type: 'text/csv' });
        saveAs(nodesCsvBlob, 'knowledge_graph_nodes.csv');

        // Create and save links CSV
        const linksCsvBlob = new Blob([linksCsv.join('\n')], { type: 'text/csv' });
        saveAs(linksCsvBlob, 'knowledge_graph_links.csv');
        break;

      case 'graphml':
        // Export as GraphML
        let graphml = '<?xml version="1.0" encoding="UTF-8"?>\n';
        graphml += '<graphml xmlns="http://graphml.graphdrawing.org/xmlns">\n';

        // Define node attributes
        graphml += '  <key id="type" for="node" attr.name="type" attr.type="string"/>\n';
        graphml += '  <key id="label" for="node" attr.name="label" attr.type="string"/>\n';

        // Define edge attributes
        graphml += '  <key id="relation" for="edge" attr.name="relation" attr.type="string"/>\n';
        graphml += '  <key id="confidence" for="edge" attr.name="confidence" attr.type="double"/>\n';

        // Start graph
        graphml += '  <graph id="G" edgedefault="directed">\n';

        // Add nodes
        graphToExport.nodes.forEach(node => {
          graphml += `    <node id="${node.id}">\n`;
          graphml += `      <data key="type">${node.type}</data>\n`;
          graphml += `      <data key="label">${node.id}</data>\n`;
          graphml += '    </node>\n';
        });

        // Add edges
        graphToExport.links.forEach((link, index) => {
          const sourceId = link.source.id || link.source;
          const targetId = link.target.id || link.target;
          graphml += `    <edge id="e${index}" source="${sourceId}" target="${targetId}">\n`;
          graphml += `      <data key="relation">${link.type}</data>\n`;
          graphml += `      <data key="confidence">${link.confidence}</data>\n`;
          graphml += '    </edge>\n';
        });

        // Close graph and graphml
        graphml += '  </graph>\n';
        graphml += '</graphml>';

        // Create and save GraphML file
        const graphmlBlob = new Blob([graphml], { type: 'application/xml' });
        saveAs(graphmlBlob, 'knowledge_graph.graphml');
        break;

      case 'png':
        // Export as PNG image
        if (graphRef.current) {
          try {
            // For 2D graph
            if (viewMode === '2d' && graphRef.current.canvas) {
              const canvas = graphRef.current.canvas();
              canvas.toBlob(blob => {
                saveAs(blob, 'knowledge_graph.png');
              });
            }
            // For 3D graph
            else if (viewMode === '3d' && graphRef.current.renderer) {
              const renderer = graphRef.current.renderer();
              renderer.domElement.toBlob(blob => {
                saveAs(blob, 'knowledge_graph.png');
              });
            }
          } catch (error) {
            console.error('Error exporting graph as PNG:', error);
          }
        }
        break;

      default:
        console.error(`Unsupported export format: ${format}`);
    }
  };

  // Node painting function
  const paintNode = (node, ctx, globalScale) => {
    const isCluster = node.isCluster;
    const label = isCluster ? `${node.entityType} (${node.nodeCount})` : node.id;
    const fontSize = 12/globalScale;
    const nodeColor = isCluster
      ? entityColors[node.entityType] || entityColors.DEFAULT
      : entityColors[node.type] || entityColors.DEFAULT;
    const isHighlighted = highlightedNodes.has(node.id);
    const size = isCluster
      ? Math.min(Math.max(Math.sqrt(node.nodeCount) * 2, nodeSize), nodeSize * 3)
      : (isHighlighted ? nodeSize * 1.5 : nodeSize);

    // Node circle or cluster
    if (isCluster) {
      // Draw cluster as a larger circle with a border
      ctx.beginPath();
      ctx.arc(node.x, node.y, size, 0, 2 * Math.PI);
      ctx.fillStyle = nodeColor;
      ctx.fill();

      // Add a border
      ctx.strokeStyle = '#000';
      ctx.lineWidth = 1.5/globalScale;
      ctx.stroke();

      // Add a count indicator
      ctx.font = `${fontSize * 1.2}px Sans-Serif`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillStyle = '#fff';
      ctx.fillText(node.nodeCount.toString(), node.x, node.y);
    } else {
      // Regular node
      ctx.beginPath();
      ctx.arc(node.x, node.y, size, 0, 2 * Math.PI);
      ctx.fillStyle = nodeColor;
      ctx.fill();

      if (isHighlighted) {
        ctx.strokeStyle = '#000';
        ctx.lineWidth = 2/globalScale;
        ctx.stroke();
      }
    }

    // Node label
    ctx.font = `${fontSize}px Sans-Serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillStyle = '#000';
    ctx.fillText(label, node.x, node.y + size + 5);
  };

  // Link painting function
  const paintLink = (link, ctx, globalScale) => {
    const start = link.source;
    const end = link.target;
    const color = relationColors[link.type] || relationColors.DEFAULT;
    const lineWidth = link.confidence * 3;

    // Draw line
    ctx.beginPath();
    ctx.moveTo(start.x, start.y);
    ctx.lineTo(end.x, end.y);
    ctx.strokeStyle = color;
    ctx.lineWidth = lineWidth / globalScale;
    ctx.stroke();

    // Draw arrow
    const dx = end.x - start.x;
    const dy = end.y - start.y;
    const angle = Math.atan2(dy, dx);
    const length = Math.sqrt(dx * dx + dy * dy);
    const arrowLength = 10 / globalScale;
    const arrowPos = length - 10 / globalScale;

    const arrowX = start.x + Math.cos(angle) * arrowPos;
    const arrowY = start.y + Math.sin(angle) * arrowPos;

    ctx.beginPath();
    ctx.moveTo(arrowX, arrowY);
    ctx.lineTo(
      arrowX - arrowLength * Math.cos(angle - Math.PI / 6),
      arrowY - arrowLength * Math.sin(angle - Math.PI / 6)
    );
    ctx.lineTo(
      arrowX - arrowLength * Math.cos(angle + Math.PI / 6),
      arrowY - arrowLength * Math.sin(angle + Math.PI / 6)
    );
    ctx.closePath();
    ctx.fillStyle = color;
    ctx.fill();

    // Draw relation label if zoomed in enough
    if (globalScale > 1.5) {
      const labelX = start.x + dx / 2;
      const labelY = start.y + dy / 2;
      const fontSize = 10 / globalScale;

      ctx.font = `${fontSize}px Sans-Serif`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillStyle = '#000';
      ctx.fillText(link.type, labelX, labelY - 5);
    }
  };

  // Render entity type chips
  const renderEntityTypeChips = () => {
    return availableEntityTypes.map(type => (
      <Chip
        key={type}
        label={type}
        icon={entityIcons[type] || <FilterListIcon />}
        style={{
          backgroundColor: entityColors[type] || entityColors.DEFAULT,
          margin: '0 4px 4px 0',
          color: '#fff'
        }}
        onClick={() => {
          if (selectedEntityTypes.includes(type)) {
            setSelectedEntityTypes(selectedEntityTypes.filter(t => t !== type));
          } else {
            setSelectedEntityTypes([...selectedEntityTypes, type]);
          }
        }}
        variant={selectedEntityTypes.includes(type) ? 'filled' : 'outlined'}
      />
    ));
  };

  // Render relation type chips
  const renderRelationTypeChips = () => {
    return availableRelationTypes.map(type => (
      <Chip
        key={type}
        label={type}
        style={{
          backgroundColor: relationColors[type] || relationColors.DEFAULT,
          margin: '0 4px 4px 0',
          color: '#fff'
        }}
        onClick={() => {
          if (selectedRelationTypes.includes(type)) {
            setSelectedRelationTypes(selectedRelationTypes.filter(t => t !== type));
          } else {
            setSelectedRelationTypes([...selectedRelationTypes, type]);
          }
        }}
        variant={selectedRelationTypes.includes(type) ? 'filled' : 'outlined'}
      />
    ));
  };

  // Render graph
  const renderGraph = () => {
    if (isLoading) {
      return (
        <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: dimensions.height }}>
          <CircularProgress />
          <Typography variant="body1" sx={{ ml: 2 }}>
            Loading knowledge graph...
          </Typography>
        </Box>
      );
    }

    if (error) {
      return (
        <Alert severity="error" sx={{ mt: 2 }}>
          {error}
        </Alert>
      );
    }

    if (!filteredGraph || filteredGraph.nodes.length === 0) {
      return (
        <Alert severity="info" sx={{ mt: 2 }}>
          No graph data available or no nodes match the current filters.
        </Alert>
      );
    }

    // Determine which graph to render (clustered or filtered)
    const graphToRender = enableClustering && clusteredGraph ? clusteredGraph : filteredGraph;

    if (viewMode === '2d') {
      return (
        <ForceGraph2D
          ref={graphRef}
          graphData={graphToRender}
          width={dimensions.width}
          height={dimensions.height}
          nodeLabel={node => node.isCluster
            ? `Cluster: ${node.entityType} (${node.nodeCount} nodes)`
            : `${node.id} (${node.type})`
          }
          linkLabel={link => `${link.type} (${link.confidence.toFixed(2)})`}
          nodeCanvasObject={paintNode}
          linkCanvasObject={paintLink}
          linkDirectionalArrowLength={3}
          linkDirectionalArrowRelPos={1}
          linkCurvature={0.25}
          cooldownTicks={100}
          d3AlphaDecay={0.02}
          d3VelocityDecay={0.3}
          onEngineStop={() => {
            // Apply layout forces when simulation stops
            applyLayoutForces();
          }}
          onNodeClick={node => {
            // If it's a cluster node, expand it
            if (node.isCluster) {
              setEnableClustering(false);
              // Filter to show only nodes of this type
              const entityType = node.entityType;
              setSelectedEntityTypes([entityType]);
            } else {
              // Center view on node
              graphRef.current.centerAt(node.x, node.y, 1000);
              graphRef.current.zoom(2, 1000);
            }
          }}
        />
      );
    } else {
      return (
        <ForceGraph3D
          ref={graphRef}
          graphData={graphToRender}
          width={dimensions.width}
          height={dimensions.height}
          nodeLabel={node => node.isCluster
            ? `Cluster: ${node.entityType} (${node.nodeCount} nodes)`
            : `${node.id} (${node.type})`
          }
          linkLabel={link => `${link.type} (${link.confidence.toFixed(2)})`}
          nodeColor={node => node.isCluster
            ? entityColors[node.entityType] || entityColors.DEFAULT
            : entityColors[node.type] || entityColors.DEFAULT
          }
          linkColor={link => relationColors[link.type] || relationColors.DEFAULT}
          linkWidth={link => link.confidence * 2}
          linkDirectionalArrowLength={3}
          linkDirectionalArrowRelPos={1}
          cooldownTicks={100}
          nodeRelSize={node => node.isCluster
            ? Math.min(Math.max(Math.sqrt(node.nodeCount) * 2, nodeSize), nodeSize * 3)
            : nodeSize
          }
          d3AlphaDecay={0.02}
          d3VelocityDecay={0.3}
          onEngineStop={() => {
            // Apply layout forces when simulation stops
            applyLayoutForces();
          }}
          onNodeClick={node => {
            // Center view on node
            const distance = 40;
            const distRatio = 1 + distance/Math.hypot(node.x, node.y, node.z);
            graphRef.current.cameraPosition(
              { x: node.x * distRatio, y: node.y * distRatio, z: node.z * distRatio },
              node,
              1000
            );
          }}
        />
      );
    }
  };

  return (
    <Box ref={containerRef} sx={{ width: '100%' }}>
      <Paper sx={{ p: 2, mb: 2 }}>
        <Typography variant="h6" gutterBottom>
          Knowledge Graph Visualization
        </Typography>

        {/* Controls */}
        <Box sx={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', mb: 2 }}>
          {/* View mode toggle */}
          <ToggleButtonGroup
            value={viewMode}
            exclusive
            onChange={handleViewModeChange}
            size="small"
            sx={{ mr: 2, mb: 1 }}
          >
            <ToggleButton value="2d">2D</ToggleButton>
            <ToggleButton value="3d">3D</ToggleButton>
          </ToggleButtonGroup>

          {/* Layout algorithm selector */}
          <FormControl size="small" sx={{ mr: 2, mb: 1, minWidth: 120 }}>
            <InputLabel id="layout-algorithm-label">Layout</InputLabel>
            <Select
              labelId="layout-algorithm-label"
              id="layout-algorithm"
              value={layoutAlgorithm}
              label="Layout"
              onChange={handleLayoutChange}
            >
              <MenuItem value="force">Force-directed</MenuItem>
              <MenuItem value="radial">Radial</MenuItem>
              <MenuItem value="circular">Circular</MenuItem>
              <MenuItem value="hierarchical">Hierarchical</MenuItem>
            </Select>
          </FormControl>

          {/* Zoom controls */}
          <Box sx={{ display: 'flex', mr: 2, mb: 1 }}>
            <Tooltip title="Zoom out">
              <IconButton onClick={handleZoomOut} size="small">
                <ZoomOutIcon />
              </IconButton>
            </Tooltip>
            <Tooltip title="Zoom in">
              <IconButton onClick={handleZoomIn} size="small">
                <ZoomInIcon />
              </IconButton>
            </Tooltip>
          </Box>

          {/* Search */}
          <TextField
            label="Search entities"
            variant="outlined"
            size="small"
            value={searchTerm}
            onChange={handleSearch}
            InputProps={{
              startAdornment: <SearchIcon fontSize="small" sx={{ mr: 1 }} />
            }}
            sx={{ mr: 2, mb: 1, width: 200 }}
          />

          {/* Export options */}
          <Box sx={{ display: 'flex', mb: 1 }}>
            <Tooltip title="Export as JSON">
              <IconButton
                onClick={() => handleExport('json')}
                disabled={!filteredGraph}
                size="small"
              >
                <SaveIcon />
              </IconButton>
            </Tooltip>
            <Tooltip title="Export as CSV">
              <IconButton
                onClick={() => handleExport('csv')}
                disabled={!filteredGraph}
                size="small"
              >
                <SaveIcon />
              </IconButton>
            </Tooltip>
            <Tooltip title="Export as GraphML">
              <IconButton
                onClick={() => handleExport('graphml')}
                disabled={!filteredGraph}
                size="small"
              >
                <SaveIcon />
              </IconButton>
            </Tooltip>
            <Tooltip title="Export as PNG">
              <IconButton
                onClick={() => handleExport('png')}
                disabled={!filteredGraph}
                size="small"
              >
                <SaveIcon />
              </IconButton>
            </Tooltip>
          </Box>
        </Box>

        {/* Filters and Graph Settings */}
        <Box sx={{ display: 'flex', flexWrap: 'wrap', mb: 2 }}>
          <Box sx={{ mr: 4, mb: 2, minWidth: 200 }}>
            <Typography variant="subtitle2" gutterBottom>
              Confidence Threshold: {confidenceThreshold.toFixed(2)}
            </Typography>
            <Slider
              value={confidenceThreshold}
              onChange={handleConfidenceChange}
              min={0}
              max={1}
              step={0.05}
              valueLabelDisplay="auto"
              sx={{ maxWidth: 300 }}
            />
          </Box>

          <Box sx={{ mr: 4, mb: 2, minWidth: 200 }}>
            <Typography variant="subtitle2" gutterBottom>
              Node Size: {nodeSize}
            </Typography>
            <Slider
              value={nodeSize}
              onChange={handleNodeSizeChange}
              min={1}
              max={15}
              step={1}
              valueLabelDisplay="auto"
              sx={{ maxWidth: 300 }}
            />
          </Box>

          <Box sx={{ mr: 4, mb: 2, minWidth: 200 }}>
            <Typography variant="subtitle2" gutterBottom>
              Link Distance: {linkDistance}
            </Typography>
            <Slider
              value={linkDistance}
              onChange={handleLinkDistanceChange}
              min={30}
              max={300}
              step={10}
              valueLabelDisplay="auto"
              sx={{ maxWidth: 300 }}
            />
          </Box>

          <Box sx={{ mb: 2, minWidth: 200 }}>
            <Typography variant="subtitle2" gutterBottom>
              Charge Strength: {chargeStrength}
            </Typography>
            <Slider
              value={chargeStrength}
              onChange={handleChargeStrengthChange}
              min={-100}
              max={0}
              step={5}
              valueLabelDisplay="auto"
              sx={{ maxWidth: 300 }}
            />
          </Box>
        </Box>

        {/* Entity type filters */}
        <Box sx={{ mb: 2 }}>
          <Typography variant="subtitle2" gutterBottom>
            Entity Types
          </Typography>
          <Box sx={{ display: 'flex', flexWrap: 'wrap' }}>
            {renderEntityTypeChips()}
          </Box>
        </Box>

        {/* Relation type filters */}
        <Box sx={{ mb: 2 }}>
          <Typography variant="subtitle2" gutterBottom>
            Relation Types
          </Typography>
          <Box sx={{ display: 'flex', flexWrap: 'wrap' }}>
            {renderRelationTypeChips()}
          </Box>
        </Box>
      </Paper>

      {/* Graph visualization */}
      <Paper sx={{ p: 0, overflow: 'hidden' }}>
        {renderGraph()}
      </Paper>
    </Box>
  );
};

export default KnowledgeGraphViewer;
