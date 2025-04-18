import React from 'react';
import { 
  Box, 
  Typography, 
  Paper, 
  Grid, 
  Divider,
  LinearProgress,
  Chip,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  CircularProgress,
  Alert
} from '@mui/material';
import { 
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
      return null;
  }
};

/**
 * Operation Details component for displaying detailed information about an LLM operation
 * 
 * @param {Object} props
 * @param {string} props.operationId - The ID of the operation to display
 */
const OperationDetails = ({ operationId }) => {
  // Fetch operation details
  const { 
    data: operation, 
    isLoading, 
    error 
  } = useQuery(
    ['operation-details', operationId],
    () => apiClient.get(`/api/llm/progress/operations/${operationId}`).then(res => res.data),
    { 
      enabled: !!operationId,
      refetchInterval: (data) => {
        if (!data) return 1000;
        if (data.status === 'running' || data.status === 'pending') return 1000;
        return false;
      }
    }
  );
  
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
  
  // Format elapsed time
  const formatElapsedTime = (seconds) => {
    if (seconds === null || seconds === undefined) return 'N/A';
    
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const remainingSeconds = Math.floor(seconds % 60);
    
    let result = '';
    if (hours > 0) result += `${hours}h `;
    if (minutes > 0 || hours > 0) result += `${minutes}m `;
    result += `${remainingSeconds}s`;
    
    return result;
  };
  
  if (!operationId) {
    return (
      <Alert severity="info">
        No operation selected. Please select an operation to view its details.
      </Alert>
    );
  }
  
  if (isLoading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" p={3}>
        <CircularProgress size={24} sx={{ mr: 2 }} />
        <Typography>Loading operation details...</Typography>
      </Box>
    );
  }
  
  if (error) {
    return (
      <Alert severity="error">
        Error loading operation details: {error.message}
      </Alert>
    );
  }
  
  if (!operation) {
    return (
      <Alert severity="warning">
        Operation not found. It may have been deleted or expired.
      </Alert>
    );
  }
  
  return (
    <Box>
      <Paper sx={{ p: 3, mb: 3 }}>
        <Typography variant="h5" gutterBottom>
          Operation Details
        </Typography>
        
        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <Box mb={2}>
              <Typography variant="subtitle1" gutterBottom>
                Operation ID
              </Typography>
              <Typography variant="body1">
                {operation.operation_id}
              </Typography>
            </Box>
            
            <Box mb={2}>
              <Typography variant="subtitle1" gutterBottom>
                Type
              </Typography>
              <Typography variant="body1">
                {operation.operation_type}
              </Typography>
            </Box>
            
            <Box mb={2}>
              <Typography variant="subtitle1" gutterBottom>
                Status
              </Typography>
              <Chip 
                label={operation.status} 
                color={statusColors[operation.status] || 'default'} 
                icon={<StatusIcon status={operation.status} />}
              />
            </Box>
            
            <Box mb={2}>
              <Typography variant="subtitle1" gutterBottom>
                Progress
              </Typography>
              <Box sx={{ display: 'flex', alignItems: 'center' }}>
                <Box sx={{ width: '100%', mr: 1 }}>
                  <LinearProgress 
                    variant="determinate" 
                    value={operation.percent_complete} 
                    color={
                      operation.status === 'failed' ? 'error' : 
                      operation.status === 'cancelled' ? 'warning' : 'primary'
                    }
                  />
                </Box>
                <Typography variant="body2" color="text.secondary">
                  {Math.round(operation.percent_complete)}%
                </Typography>
              </Box>
              <Typography variant="body2" color="text.secondary" mt={1}>
                {operation.message}
              </Typography>
            </Box>
          </Grid>
          
          <Grid item xs={12} md={6}>
            <Box mb={2}>
              <Typography variant="subtitle1" gutterBottom>
                Started
              </Typography>
              <Typography variant="body1">
                {formatTime(operation.start_time)}
              </Typography>
            </Box>
            
            <Box mb={2}>
              <Typography variant="subtitle1" gutterBottom>
                Ended
              </Typography>
              <Typography variant="body1">
                {operation.end_time ? formatTime(operation.end_time) : 'In progress...'}
              </Typography>
            </Box>
            
            <Box mb={2}>
              <Typography variant="subtitle1" gutterBottom>
                Elapsed Time
              </Typography>
              <Typography variant="body1">
                {formatElapsedTime(operation.elapsed_time)}
              </Typography>
            </Box>
            
            <Box mb={2}>
              <Typography variant="subtitle1" gutterBottom>
                Estimated Time Remaining
              </Typography>
              <Typography variant="body1">
                {operation.estimated_time_remaining ? 
                  formatElapsedTime(operation.estimated_time_remaining) : 
                  'N/A'
                }
              </Typography>
            </Box>
          </Grid>
        </Grid>
      </Paper>
      
      <Paper sx={{ p: 3, mb: 3 }}>
        <Typography variant="h6" gutterBottom>
          Progress Steps
        </Typography>
        
        <TableContainer>
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell>Step</TableCell>
                <TableCell>Message</TableCell>
                <TableCell>Time</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {operation.steps.map((step) => (
                <TableRow key={step.step_number}>
                  <TableCell>{step.step_number}</TableCell>
                  <TableCell>{step.message}</TableCell>
                  <TableCell>{formatTime(step.timestamp)}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      </Paper>
      
      {operation.metadata && Object.keys(operation.metadata).length > 0 && (
        <Paper sx={{ p: 3 }}>
          <Typography variant="h6" gutterBottom>
            Metadata
          </Typography>
          
          <TableContainer>
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell>Key</TableCell>
                  <TableCell>Value</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {Object.entries(operation.metadata).map(([key, value]) => (
                  <TableRow key={key}>
                    <TableCell>{key}</TableCell>
                    <TableCell>
                      {typeof value === 'object' ? 
                        JSON.stringify(value) : 
                        String(value)
                      }
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </Paper>
      )}
    </Box>
  );
};

export default OperationDetails;
