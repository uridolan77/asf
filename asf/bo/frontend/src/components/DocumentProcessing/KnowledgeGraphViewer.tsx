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
  Alert,
  SelectChangeEvent
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

import { useKnowledgeGraph, KnowledgeGraph, GraphNode, GraphLink } from '../../hooks/useKnowledgeGraph';
import { useFeatureFlags } from '../../context/FeatureFlagContext';

interface KnowledgeGraphViewerProps {
  graphData?: KnowledgeGraph;
  isLoading?: boolean;
  error?: Error | null;
  documentId?: string;
}

/**
 * Interactive Knowledge Graph Viewer component
 *
 * This component visualizes knowledge graphs extracted from documents
 * with interactive features like filtering, searching, and zooming.
 */
const KnowledgeGraphViewer: React.FC<KnowledgeGraphViewerProps> = ({ 
  graphData, 
  isLoading = false, 
  error = null,
  documentId
}) => {
  // Feature flags
  const { isEnabled } = useFeatureFlags();
  const useMockData = isEnabled('useMockData');

  // Knowledge graph hook
  const {
    getDocumentGraph,
    getEntityTypes,
    getRelationTypes,
    exportGraph
  } = useKnowledgeGraph();

  // Fetch document graph if documentId is provided
  const {
    data: fetchedGraphData,
    isLoading: isLoadingGraph,
    isError: isErrorGraph,
    error: graphError
  } = getDocumentGraph(documentId || '');

  // Combine props and fetched data
  const combinedGraphData = graphData || fetchedGraphData;
  const combinedIsLoading = isLoading || isLoadingGraph;
  const combinedError = error || (isErrorGraph ? graphError : null);

  // State for graph dimensions
  const [dimensions, setDimensions] = useState<{ width: number; height: number }>({ width: 800, height: 600 });
  
  // State for view options
  const [viewMode, setViewMode] = useState<'2d' | '3d'>('2d');
  const [layoutAlgorithm, setLayoutAlgorithm] = useState<string>('force');
  const [confidenceThreshold, setConfidenceThreshold] = useState<number>(0.5);
  const [selectedEntityTypes, setSelectedEntityTypes] = useState<string[]>([]);
  const [selectedRelationTypes, setSelectedRelationTypes] = useState<string[]>([]);
  const [searchTerm, setSearchTerm] = useState<string>('');
  const [enableClustering, setEnableClustering] = useState<boolean>(false);
  const [nodeSize, setNodeSize] = useState<number>(5);
  const [showLabels, setShowLabels] = useState<boolean>(true);

  // Refs
  const containerRef = useRef<HTMLDivElement>(null);
  const graphRef = useRef<any>(null);

  // Derived state
  const [filteredGraph, setFilteredGraph] = useState<KnowledgeGraph | null>(null);
  const [clusteredGraph, setClusteredGraph] = useState<KnowledgeGraph | null>(null);

  // Entity and relation types
  const {
    data: entityTypesData,
    isLoading: isLoadingEntityTypes
  } = getEntityTypes();

  const {
    data: relationTypesData,
    isLoading: isLoadingRelationTypes
  } = getRelationTypes();

  const entityTypes = entityTypesData?.entity_types || [];
  const relationTypes = relationTypesData?.relation_types || [];

  // Colors for entities and relations
  const entityColors: Record<string, string> = {
    'Disease': '#e41a1c',
    'Drug': '#377eb8',
    'Gene': '#4daf4a',
    'Chemical': '#984ea3',
    'Symptom': '#ff7f00',
    'Treatment': '#a65628',
    'DEFAULT': '#999999'
  };

  const relationColors: Record<string, string> = {
    'TREATS': '#4c78a8',
    'CAUSES': '#f58518',
    'ASSOCIATED_WITH': '#e45756',
    'INTERACTS_WITH': '#72b7b2',
    'PART_OF': '#54a24b',
    'HAS_EFFECT': '#eeca3b',
    'DEFAULT': '#b279a2'
  };

  // Update dimensions when container size changes
  useEffect(() => {
    const updateDimensions = () => {
      if (containerRef.current) {
        const { width, height } = containerRef.current.getBoundingClientRect();
        setDimensions({ width, height: Math.max(height, 500) });
      }
    };

    updateDimensions();
    window.addEventListener('resize', updateDimensions);
    return () => window.removeEventListener('resize', updateDimensions);
  }, []);

  // Filter graph based on confidence threshold, entity types, relation types, and search term
  useEffect(() => {
    if (!combinedGraphData) return;

    const { nodes, links } = combinedGraphData;

    // Filter links by confidence threshold
    const filteredLinks = links.filter(link => link.confidence >= confidenceThreshold);

    // Get node IDs from filtered links
    const nodeIdsInLinks = new Set<string>();
    filteredLinks.forEach(link => {
      nodeIdsInLinks.add(link.source.toString());
      nodeIdsInLinks.add(link.target.toString());
    });

    // Filter nodes by entity types and search term
    const filteredNodes = nodes.filter(node => {
      // Filter by entity type if any are selected
      const passesEntityTypeFilter = selectedEntityTypes.length === 0 || 
        selectedEntityTypes.includes(node.type);
      
      // Filter by search term
      const passesSearchFilter = !searchTerm || 
        node.label.toLowerCase().includes(searchTerm.toLowerCase()) ||
        node.id.toLowerCase().includes(searchTerm.toLowerCase());
      
      // Node must be in filtered links or have no links
      const isInLinks = nodeIdsInLinks.has(node.id);
      
      return passesEntityTypeFilter && passesSearchFilter && isInLinks;
    });

    // Get node IDs from filtered nodes
    const filteredNodeIds = new Set(filteredNodes.map(node => node.id));

    // Filter links again to only include links between filtered nodes
    const finalFilteredLinks = filteredLinks.filter(link => {
      // Filter by relation type if any are selected
      const passesRelationTypeFilter = selectedRelationTypes.length === 0 || 
        selectedRelationTypes.includes(link.type);
      
      // Link must connect two filtered nodes
      const connectsFilteredNodes = 
        filteredNodeIds.has(link.source.toString()) && 
        filteredNodeIds.has(link.target.toString());
      
      return passesRelationTypeFilter && connectsFilteredNodes;
    });

    setFilteredGraph({
      nodes: filteredNodes,
      links: finalFilteredLinks
    });
  }, [combinedGraphData, confidenceThreshold, selectedEntityTypes, selectedRelationTypes, searchTerm]);

  // Handle view mode change
  const handleViewModeChange = (_event: React.MouseEvent<HTMLElement>, newViewMode: '2d' | '3d' | null) => {
    if (newViewMode !== null) {
      setViewMode(newViewMode);
    }
  };

  // Handle layout algorithm change
  const handleLayoutChange = (event: SelectChangeEvent<string>) => {
    setLayoutAlgorithm(event.target.value);
  };

  // Handle confidence threshold change
  const handleConfidenceChange = (_event: Event, newValue: number | number[]) => {
    setConfidenceThreshold(newValue as number);
  };

  // Handle entity type selection
  const handleEntityTypeChange = (event: SelectChangeEvent<string[]>) => {
    setSelectedEntityTypes(event.target.value as string[]);
  };

  // Handle relation type selection
  const handleRelationTypeChange = (event: SelectChangeEvent<string[]>) => {
    setSelectedRelationTypes(event.target.value as string[]);
  };

  // Handle search term change
  const handleSearchChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setSearchTerm(event.target.value);
  };

  // Handle clustering toggle
  const handleClusteringChange = (_event: React.MouseEvent<HTMLElement>, newEnableClustering: boolean) => {
    setEnableClustering(newEnableClustering);
  };

  // Handle node size change
  const handleNodeSizeChange = (_event: Event, newValue: number | number[]) => {
    setNodeSize(newValue as number);
  };

  // Handle label toggle
  const handleLabelToggle = (_event: React.MouseEvent<HTMLElement>, newShowLabels: boolean) => {
    setShowLabels(newShowLabels);
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
  const handleExport = (format: 'json' | 'graphml' | 'png' = 'json') => {
    // Determine which graph to export (clustered or filtered)
    const graphToExport = enableClustering && clusteredGraph ? clusteredGraph : filteredGraph;
    if (!graphToExport) return;

    if (format === 'png') {
      // Export as PNG image
      if (graphRef.current) {
        try {
          // For 2D graph
          if (viewMode === '2d' && graphRef.current.canvas) {
            const canvas = graphRef.current.canvas();
            canvas.toBlob((blob: Blob | null) => {
              if (blob) {
                saveAs(blob, 'knowledge_graph.png');
              }
            });
          }
          // For 3D graph
          else if (viewMode === '3d' && graphRef.current.renderer) {
            const renderer = graphRef.current.renderer();
            renderer.domElement.toBlob((blob: Blob | null) => {
              if (blob) {
                saveAs(blob, 'knowledge_graph.png');
              }
            });
          }
        } catch (error) {
          console.error('Error exporting graph as PNG:', error);
        }
      }
    } else {
      // Use the exportGraph API for other formats
      const { mutate: exportMutate } = exportGraph();
      
      if (documentId) {
        exportMutate({
          graph_id: documentId,
          format: format,
          include_properties: true
        });
      } else {
        // If no documentId, create a blob and save it directly
        const graphJson = JSON.stringify(graphToExport, null, 2);
        const jsonBlob = new Blob([graphJson], { type: 'application/json' });
        saveAs(jsonBlob, `knowledge_graph.${format}`);
      }
    }
  };

  // Apply layout forces
  const applyLayoutForces = () => {
    // This would be implemented based on the selected layout algorithm
    console.log(`Applying ${layoutAlgorithm} layout`);
  };

  // Render loading state
  if (combinedIsLoading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '400px' }}>
        <CircularProgress />
      </Box>
    );
  }

  // Render error state
  if (combinedError) {
    return (
      <Alert severity="error">
        Error loading knowledge graph: {combinedError.message || 'Unknown error'}
      </Alert>
    );
  }

  // Render empty state
  if (!combinedGraphData || !filteredGraph || filteredGraph.nodes.length === 0) {
    return (
      <Alert severity="info">
        No knowledge graph data available. The document may not have any extracted entities or relations.
      </Alert>
    );
  }

  // Determine which graph to render
  const graphToRender = enableClustering && clusteredGraph ? clusteredGraph : filteredGraph;

  // Paint node function for 2D graph
  const paintNode = (node: GraphNode, ctx: CanvasRenderingContext2D) => {
    const { x, y } = node as any;
    const color = entityColors[node.type] || entityColors.DEFAULT;
    const size = node.isCluster
      ? Math.min(Math.max(Math.sqrt(node.nodeCount || 1) * 2, nodeSize), nodeSize * 3)
      : nodeSize;

    // Draw node
    ctx.beginPath();
    ctx.arc(x, y, size, 0, 2 * Math.PI);
    ctx.fillStyle = color;
    ctx.fill();
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 0.5;
    ctx.stroke();

    // Draw label if enabled
    if (showLabels) {
      ctx.font = '4px Arial';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillStyle = '#ffffff';
      ctx.fillText(node.label || node.id, x, y);
    }
  };

  // Paint link function for 2D graph
  const paintLink = (link: GraphLink, ctx: CanvasRenderingContext2D) => {
    const { source, target } = link as any;
    const color = relationColors[link.type] || relationColors.DEFAULT;
    const width = link.confidence * 2;

    // Draw link
    ctx.beginPath();
    ctx.moveTo(source.x, source.y);
    ctx.lineTo(target.x, target.y);
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.stroke();

    // Draw arrow
    const dx = target.x - source.x;
    const dy = target.y - source.y;
    const angle = Math.atan2(dy, dx);
    const length = Math.sqrt(dx * dx + dy * dy);
    const arrowLength = 5;
    const arrowPos = length - 10;
    const arrowX = source.x + Math.cos(angle) * arrowPos;
    const arrowY = source.y + Math.sin(angle) * arrowPos;

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
  };

  // Render graph
  const renderGraph = () => {
    if (!graphToRender) return null;

    if (viewMode === '2d') {
      return (
        <ForceGraph2D
          ref={graphRef}
          graphData={graphToRender as any}
          width={dimensions.width}
          height={dimensions.height}
          nodeLabel={(node: GraphNode) => node.isCluster
            ? `Cluster: ${node.entityType} (${node.nodeCount} nodes)`
            : `${node.id} (${node.type})`
          }
          linkLabel={(link: GraphLink) => `${link.type} (${link.confidence.toFixed(2)})`}
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
        />
      );
    } else {
      return (
        <ForceGraph3D
          ref={graphRef}
          graphData={graphToRender as any}
          width={dimensions.width}
          height={dimensions.height}
          nodeLabel={(node: GraphNode) => node.isCluster
            ? `Cluster: ${node.entityType} (${node.nodeCount} nodes)`
            : `${node.id} (${node.type})`
          }
          linkLabel={(link: GraphLink) => `${link.type} (${link.confidence.toFixed(2)})`}
          nodeColor={(node: GraphNode) => entityColors[node.type] || entityColors.DEFAULT}
          linkColor={(link: GraphLink) => relationColors[link.type] || relationColors.DEFAULT}
          linkWidth={(link: GraphLink) => link.confidence * 2}
          linkDirectionalArrowLength={3}
          linkDirectionalArrowRelPos={1}
          cooldownTicks={100}
          nodeRelSize={(node: GraphNode) => node.isCluster
            ? Math.min(Math.max(Math.sqrt(node.nodeCount || 1) * 2, nodeSize), nodeSize * 3)
            : nodeSize
          }
          d3AlphaDecay={0.02}
          d3VelocityDecay={0.3}
          onEngineStop={() => {
            // Apply layout forces when simulation stops
            applyLayoutForces();
          }}
          onNodeClick={(node: GraphNode) => {
            // Center view on node
            const distance = 40;
            const nodeAny = node as any;
            const distRatio = 1 + distance/Math.hypot(nodeAny.x, nodeAny.y, nodeAny.z);
            graphRef.current.cameraPosition(
              { x: nodeAny.x * distRatio, y: nodeAny.y * distRatio, z: nodeAny.z * distRatio },
              node,
              1000
            );
          }}
        />
      );
    }
  };

  return (
    <Box ref={containerRef} sx={{ width: '100%', height: '100%', minHeight: '500px' }}>
      {useMockData && (
        <Alert severity="info" sx={{ mb: 2 }}>
          Using mock data. Toggle the "Use Mock Data" feature flag to use real API data.
        </Alert>
      )}
      
      <Paper sx={{ p: 2, mb: 2 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Typography variant="h6">Knowledge Graph Viewer</Typography>
          
          <Box sx={{ display: 'flex', gap: 1 }}>
            <Tooltip title="Zoom In">
              <IconButton onClick={handleZoomIn} size="small">
                <ZoomInIcon />
              </IconButton>
            </Tooltip>
            
            <Tooltip title="Zoom Out">
              <IconButton onClick={handleZoomOut} size="small">
                <ZoomOutIcon />
              </IconButton>
            </Tooltip>
            
            <Tooltip title="Export as JSON">
              <IconButton onClick={() => handleExport('json')} size="small">
                <SaveIcon />
              </IconButton>
            </Tooltip>
          </Box>
        </Box>
        
        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 2, mb: 2 }}>
          <FormControl size="small" sx={{ minWidth: 120 }}>
            <InputLabel id="layout-label">Layout</InputLabel>
            <Select
              labelId="layout-label"
              value={layoutAlgorithm}
              label="Layout"
              onChange={handleLayoutChange}
              size="small"
            >
              <MenuItem value="force">Force-Directed</MenuItem>
              <MenuItem value="radial">Radial</MenuItem>
              <MenuItem value="hierarchical">Hierarchical</MenuItem>
            </Select>
          </FormControl>
          
          <FormControl size="small" sx={{ minWidth: 200 }}>
            <InputLabel id="entity-type-label">Entity Types</InputLabel>
            <Select
              labelId="entity-type-label"
              multiple
              value={selectedEntityTypes}
              label="Entity Types"
              onChange={handleEntityTypeChange}
              size="small"
              renderValue={(selected) => (
                <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                  {(selected as string[]).map((value) => (
                    <Chip 
                      key={value} 
                      label={value} 
                      size="small" 
                      sx={{ 
                        backgroundColor: entityColors[value] || entityColors.DEFAULT,
                        color: '#ffffff'
                      }} 
                    />
                  ))}
                </Box>
              )}
            >
              {entityTypes.map((type) => (
                <MenuItem key={type} value={type}>
                  <Box sx={{ display: 'flex', alignItems: 'center' }}>
                    <Box 
                      sx={{ 
                        width: 12, 
                        height: 12, 
                        borderRadius: '50%', 
                        backgroundColor: entityColors[type] || entityColors.DEFAULT,
                        mr: 1 
                      }} 
                    />
                    {type}
                  </Box>
                </MenuItem>
              ))}
            </Select>
          </FormControl>
          
          <FormControl size="small" sx={{ minWidth: 200 }}>
            <InputLabel id="relation-type-label">Relation Types</InputLabel>
            <Select
              labelId="relation-type-label"
              multiple
              value={selectedRelationTypes}
              label="Relation Types"
              onChange={handleRelationTypeChange}
              size="small"
              renderValue={(selected) => (
                <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                  {(selected as string[]).map((value) => (
                    <Chip 
                      key={value} 
                      label={value} 
                      size="small" 
                      sx={{ 
                        backgroundColor: relationColors[value] || relationColors.DEFAULT,
                        color: '#ffffff'
                      }} 
                    />
                  ))}
                </Box>
              )}
            >
              {relationTypes.map((type) => (
                <MenuItem key={type} value={type}>
                  <Box sx={{ display: 'flex', alignItems: 'center' }}>
                    <Box 
                      sx={{ 
                        width: 12, 
                        height: 12, 
                        borderRadius: '50%', 
                        backgroundColor: relationColors[type] || relationColors.DEFAULT,
                        mr: 1 
                      }} 
                    />
                    {type}
                  </Box>
                </MenuItem>
              ))}
            </Select>
          </FormControl>
          
          <TextField
            label="Search"
            value={searchTerm}
            onChange={handleSearchChange}
            size="small"
            InputProps={{
              startAdornment: <SearchIcon fontSize="small" sx={{ mr: 1 }} />,
            }}
          />
        </Box>
        
        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 2, mb: 2 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Typography variant="body2">Confidence:</Typography>
            <Slider
              value={confidenceThreshold}
              onChange={handleConfidenceChange}
              min={0}
              max={1}
              step={0.05}
              valueLabelDisplay="auto"
              sx={{ width: 120 }}
            />
          </Box>
          
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Typography variant="body2">Node Size:</Typography>
            <Slider
              value={nodeSize}
              onChange={handleNodeSizeChange}
              min={1}
              max={10}
              step={1}
              valueLabelDisplay="auto"
              sx={{ width: 120 }}
            />
          </Box>
          
          <ToggleButtonGroup
            value={viewMode}
            exclusive
            onChange={handleViewModeChange}
            size="small"
          >
            <ToggleButton value="2d">2D</ToggleButton>
            <ToggleButton value="3d">3D</ToggleButton>
          </ToggleButtonGroup>
          
          <ToggleButtonGroup
            value={enableClustering}
            onChange={handleClusteringChange}
            size="small"
          >
            <ToggleButton value={true}>Cluster</ToggleButton>
          </ToggleButtonGroup>
          
          <ToggleButtonGroup
            value={showLabels}
            onChange={handleLabelToggle}
            size="small"
          >
            <ToggleButton value={true}>Show Labels</ToggleButton>
          </ToggleButtonGroup>
        </Box>
        
        <Box sx={{ display: 'flex', gap: 1, mb: 2 }}>
          <Chip 
            icon={<DiseaseIcon />} 
            label="Disease" 
            size="small" 
            sx={{ backgroundColor: entityColors.Disease, color: '#ffffff' }} 
          />
          <Chip 
            icon={<DrugIcon />} 
            label="Drug" 
            size="small" 
            sx={{ backgroundColor: entityColors.Drug, color: '#ffffff' }} 
          />
          <Chip 
            icon={<GeneIcon />} 
            label="Gene" 
            size="small" 
            sx={{ backgroundColor: entityColors.Gene, color: '#ffffff' }} 
          />
          <Chip 
            icon={<ChemicalIcon />} 
            label="Chemical" 
            size="small" 
            sx={{ backgroundColor: entityColors.Chemical, color: '#ffffff' }} 
          />
          <Chip 
            icon={<SymptomIcon />} 
            label="Symptom" 
            size="small" 
            sx={{ backgroundColor: entityColors.Symptom, color: '#ffffff' }} 
          />
          <Chip 
            icon={<TreatmentIcon />} 
            label="Treatment" 
            size="small" 
            sx={{ backgroundColor: entityColors.Treatment, color: '#ffffff' }} 
          />
        </Box>
      </Paper>
      
      <Paper sx={{ height: dimensions.height }}>
        {renderGraph()}
      </Paper>
      
      <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
        {graphToRender?.nodes.length} nodes, {graphToRender?.links.length} relationships
      </Typography>
    </Box>
  );
};

export default KnowledgeGraphViewer;
