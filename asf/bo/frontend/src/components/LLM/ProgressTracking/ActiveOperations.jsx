import React, { useState } from 'react';
import { 
  Box, 
  Typography, 
  Paper, 
  List, 
  ListItem, 
  ListItemText, 
  ListItemIcon, 
  ListItemSecondaryAction,
  IconButton, 
  Collapse, 
  Divider,
  LinearProgress,
  Chip,
  Button,
  CircularProgress,
  Alert
} from '@mui/material';
import { 
  ExpandMore as ExpandMoreIcon,
  ExpandLess as ExpandLessIcon,
  Refresh as RefreshIcon,
  Info as InfoIcon,
  CheckCircle as CheckCircleIcon,
  Error as ErrorIcon,
  HourglassEmpty as PendingIcon,
  PlayArrow as RunningIcon,
  Cancel as CancelIcon
} from '@mui/icons-material';
import { useQuery } from 'react-query';
import { apiClient } from '../../../services/apiClient';

// Status chip colors
const statusColors = {
  pending: 'default',
  running: 'primary',
  completed: 'success',
  failed: 'error',
  cancelled: 'warning'
};

// Status icons
const StatusIcon = ({ status }) => {
  switch (status) {
    case 'pending':
      return <PendingIcon />;
    case 'running':
      return <RunningIcon />;
    case 'completed':
      return <CheckCircleIcon color="success" />;
    case 'failed':
      return <ErrorIcon color="error" />;
    case 'cancelled':
      return <CancelIcon color="warning" />;
    default:
      return <InfoIcon />;
  }
};

/**
 * Active Operations component for displaying currently active LLM operations
 * 
 * @param {Object} props
 * @param {string} props.operationType - Filter by operation type
 * @param {function} props.onOperationSelect - Callback when an operation is selected
 * @param {number} props.maxItems - Maximum number of items to display
 */
const ActiveOperations = ({ 
  operationType = null, 
  onOperationSelect = () => {}, 
  maxItems = 5 
}) => {
  const [expandedItems, setExpandedItems] = useState({});
  
  // Fetch active operations
  const { 
    data: operations, 
    isLoading, 
    error,
    refetch
  } = useQuery(
    ['active-operations', operationType],
    () => {
      const url = '/api/llm/progress/active';
      const params = operationType ? { operation_type: operationType } : {};
      return apiClient.get(url, { params }).then(res => res.data);
    },
    { 
      refetchInterval: 3000,
      keepPreviousData: true
    }
  );
  
  // Toggle item expansion
  const toggleExpand = (operationId) => {
    setExpandedItems(prev => ({
      ...prev,
      [operationId]: !prev[operationId]
    }));
  };
  
  // Handle refresh
  const handleRefresh = () => {
    refetch();
  };
  
  // Handle operation select
  const handleSelect = (operation) => {
    onOperationSelect(operation);
  };
  
  // Format time
  const formatTime = (timestamp) => {
    if (!timestamp) return 'N/A';
    try {
      const date = new Date(timestamp);
      return date.toLocaleString();
    } catch (e) {
      return timestamp;
    }
  };
  
  return (
    <Paper variant="outlined" sx={{ mb: 3 }}>
      <Box display="flex" justifyContent="space-between" alignItems="center" p={2} pb={1}>
        <Typography variant="h6">Active Operations</Typography>
        <Button 
          startIcon={<RefreshIcon />} 
          size="small" 
          onClick={handleRefresh}
        >
          Refresh
        </Button>
      </Box>
      
      <Divider />
      
      {isLoading ? (
        <Box display="flex" justifyContent="center" alignItems="center" p={3}>
          <CircularProgress size={24} sx={{ mr: 2 }} />
          <Typography>Loading active operations...</Typography>
        </Box>
      ) : error ? (
        <Alert severity="error" sx={{ m: 2 }}>
          Error loading operations: {error.message}
        </Alert>
      ) : !operations || operations.length === 0 ? (
        <Alert severity="info" sx={{ m: 2 }}>
          No active operations found.
        </Alert>
      ) : (
        <List>
          {operations.slice(0, maxItems).map((operation) => (
            <React.Fragment key={operation.operation_id}>
              <ListItem button onClick={() => toggleExpand(operation.operation_id)}>
                <ListItemIcon>
                  <StatusIcon status={operation.status} />
                </ListItemIcon>
                <ListItemText 
                  primary={operation.operation_type}
                  secondary={operation.message}
                />
                <Box sx={{ width: '30%', mr: 2 }}>
                  <Box display="flex" justifyContent="space-between" alignItems="center" mb={0.5}>
                    <Typography variant="caption" color="text.secondary">
                      {Math.round(operation.percent_complete)}%
                    </Typography>
                  </Box>
                  <LinearProgress 
                    variant="determinate" 
                    value={operation.percent_complete} 
                    color={
                      operation.status === 'failed' ? 'error' : 
                      operation.status === 'cancelled' ? 'warning' : 'primary'
                    }
                  />
                </Box>
                <Chip 
                  label={operation.status} 
                  color={statusColors[operation.status] || 'default'} 
                  size="small" 
                  sx={{ mr: 1 }}
                />
                <ListItemSecondaryAction>
                  <IconButton 
                    edge="end" 
                    onClick={(e) => {
                      e.stopPropagation();
                      handleSelect(operation);
                    }}
                  >
                    <InfoIcon />
                  </IconButton>
                  {expandedItems[operation.operation_id] ? <ExpandLessIcon /> : <ExpandMoreIcon />}
                </ListItemSecondaryAction>
              </ListItem>
              
              <Collapse in={expandedItems[operation.operation_id]} timeout="auto" unmountOnExit>
                <Box sx={{ pl: 9, pr: 2, pb: 2, pt: 1 }}>
                  <Typography variant="body2" gutterBottom>
                    <strong>Operation ID:</strong> {operation.operation_id}
                  </Typography>
                  <Typography variant="body2" gutterBottom>
                    <strong>Started:</strong> {formatTime(operation.start_time)}
                  </Typography>
                  <Button 
                    variant="outlined" 
                    size="small" 
                    onClick={() => handleSelect(operation)}
                  >
                    View Details
                  </Button>
                </Box>
              </Collapse>
              
              <Divider />
            </React.Fragment>
          ))}
          
          {operations.length > maxItems && (
            <ListItem button onClick={() => onOperationSelect()}>
              <ListItemText 
                primary={`View all ${operations.length} active operations`}
                primaryTypographyProps={{ align: 'center', color: 'primary' }}
              />
            </ListItem>
          )}
        </List>
      )}
    </Paper>
  );
};

export default ActiveOperations;
