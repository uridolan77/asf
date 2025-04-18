import React, { useState, useEffect } from 'react';
import { 
  Box, 
  LinearProgress, 
  Typography, 
  Paper, 
  Chip,
  Tooltip,
  CircularProgress
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
 * Progress Bar component for tracking LLM operations
 * 
 * @param {Object} props
 * @param {string} props.operationId - The ID of the operation to track
 * @param {boolean} props.showDetails - Whether to show additional details
 * @param {function} props.onComplete - Callback when operation completes
 * @param {function} props.onError - Callback when operation fails
 */
const ProgressBar = ({ 
  operationId, 
  showDetails = false, 
  onComplete = () => {}, 
  onError = () => {} 
}) => {
  const [isComplete, setIsComplete] = useState(false);
  const [hasFailed, setHasFailed] = useState(false);
  
  // Fetch operation progress
  const { 
    data: operation, 
    isLoading, 
    error 
  } = useQuery(
    ['operation-progress', operationId],
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
  
  // Handle completion and failure
  useEffect(() => {
    if (!operation) return;
    
    if (operation.status === 'completed' && !isComplete) {
      setIsComplete(true);
      onComplete(operation);
    }
    
    if (operation.status === 'failed' && !hasFailed) {
      setHasFailed(true);
      onError(operation);
    }
  }, [operation, isComplete, hasFailed, onComplete, onError]);
  
  if (isLoading) {
    return (
      <Box display="flex" alignItems="center" justifyContent="center" p={1}>
        <CircularProgress size={20} sx={{ mr: 1 }} />
        <Typography variant="body2">Loading progress...</Typography>
      </Box>
    );
  }
  
  if (error) {
    return (
      <Box p={1}>
        <Typography variant="body2" color="error">
          Error loading progress: {error.message}
        </Typography>
      </Box>
    );
  }
  
  if (!operation) {
    return (
      <Box p={1}>
        <Typography variant="body2" color="text.secondary">
          No progress information available
        </Typography>
      </Box>
    );
  }
  
  return (
    <Paper variant="outlined" sx={{ p: 2, mb: 2 }}>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
        <Box display="flex" alignItems="center">
          <StatusIcon status={operation.status} />
          <Typography variant="body1" sx={{ ml: 1 }}>
            {operation.message}
          </Typography>
        </Box>
        <Chip 
          label={operation.status} 
          color={statusColors[operation.status] || 'default'} 
          size="small" 
        />
      </Box>
      
      <Box sx={{ width: '100%', mb: showDetails ? 2 : 0 }}>
        <Box display="flex" justifyContent="space-between" alignItems="center" mb={0.5}>
          <Typography variant="body2" color="text.secondary">
            {operation.current_step > 0 ? `Step ${operation.current_step} of ${operation.total_steps}` : ''}
          </Typography>
          <Typography variant="body2" color="text.secondary">
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
      
      {showDetails && (
        <Box mt={2}>
          <Typography variant="body2" color="text.secondary">
            Operation ID: {operation.operation_id}
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Type: {operation.operation_type}
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Elapsed Time: {formatElapsedTime(operation.elapsed_time)}
          </Typography>
          {operation.estimated_time_remaining && (
            <Typography variant="body2" color="text.secondary">
              Estimated Time Remaining: {formatElapsedTime(operation.estimated_time_remaining)}
            </Typography>
          )}
        </Box>
      )}
    </Paper>
  );
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

export default ProgressBar;
