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
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.5);
  const [selectedEntityTypes, setSelectedEntityTypes] = useState([]);
  const [selectedRelationTypes, setSelectedRelationTypes] = useState([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [highlightedNodes, setHighlightedNodes] = useState(new Set());
  const [filteredGraph, setFilteredGraph] = useState(null);
  const [availableEntityTypes, setAvailableEntityTypes] = useState([]);
  const [availableRelationTypes, setAvailableRelationTypes] = useState([]);
  
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
    setFilteredGraph({
      nodes: filteredNodes,
      links: filteredLinks
    });
    
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
  }, [graphData, selectedEntityTypes, selectedRelationTypes, confidenceThreshold, searchTerm]);
  
  // Handle view mode change
  const handleViewModeChange = (event, newMode) => {
    if (newMode !== null) {
      setViewMode(newMode);
    }
  };
  
  // Handle confidence threshold change
  const handleConfidenceChange = (event, newValue) => {
    setConfidenceThreshold(newValue);
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
  const handleExport = () => {
    if (!filteredGraph) return;
    
    const graphJson = JSON.stringify(filteredGraph, null, 2);
    const blob = new Blob([graphJson], { type: 'application/json' });
    saveAs(blob, 'knowledge_graph.json');
  };
  
  // Node painting function
  const paintNode = (node, ctx, globalScale) => {
    const label = node.id;
    const fontSize = 12/globalScale;
    const nodeColor = entityColors[node.type] || entityColors.DEFAULT;
    const isHighlighted = highlightedNodes.has(node.id);
    
    // Node circle
    ctx.beginPath();
    ctx.arc(node.x, node.y, isHighlighted ? 8 : 5, 0, 2 * Math.PI);
    ctx.fillStyle = nodeColor;
    ctx.fill();
    
    if (isHighlighted) {
      ctx.strokeStyle = '#000';
      ctx.lineWidth = 2/globalScale;
      ctx.stroke();
    }
    
    // Node label
    ctx.font = `${fontSize}px Sans-Serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillStyle = '#000';
    ctx.fillText(label, node.x, node.y + 12);
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
    
    if (viewMode === '2d') {
      return (
        <ForceGraph2D
          ref={graphRef}
          graphData={filteredGraph}
          width={dimensions.width}
          height={dimensions.height}
          nodeLabel={node => `${node.id} (${node.type})`}
          linkLabel={link => `${link.type} (${link.confidence.toFixed(2)})`}
          nodeCanvasObject={paintNode}
          linkCanvasObject={paintLink}
          linkDirectionalArrowLength={3}
          linkDirectionalArrowRelPos={1}
          linkCurvature={0.25}
          cooldownTicks={100}
          onNodeClick={node => {
            // Center view on node
            graphRef.current.centerAt(node.x, node.y, 1000);
            graphRef.current.zoom(2, 1000);
          }}
        />
      );
    } else {
      return (
        <ForceGraph3D
          ref={graphRef}
          graphData={filteredGraph}
          width={dimensions.width}
          height={dimensions.height}
          nodeLabel={node => `${node.id} (${node.type})`}
          linkLabel={link => `${link.type} (${link.confidence.toFixed(2)})`}
          nodeColor={node => entityColors[node.type] || entityColors.DEFAULT}
          linkColor={link => relationColors[link.type] || relationColors.DEFAULT}
          linkWidth={link => link.confidence * 2}
          linkDirectionalArrowLength={3}
          linkDirectionalArrowRelPos={1}
          cooldownTicks={100}
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
          
          {/* Export button */}
          <Tooltip title="Export graph as JSON">
            <IconButton onClick={handleExport} disabled={!filteredGraph} size="small" sx={{ mb: 1 }}>
              <SaveIcon />
            </IconButton>
          </Tooltip>
        </Box>
        
        {/* Filters */}
        <Box sx={{ mb: 2 }}>
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
