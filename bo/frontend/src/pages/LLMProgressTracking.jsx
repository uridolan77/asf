import React, { useState, useEffect } from 'react';
import PageLayout from '../components/Layout/PageLayout';
import { useQuery, useQueryClient } from 'react-query';
import {
  Box,
  Typography,
  Paper,
  Grid,
  Card,
  CardContent,
  CardHeader,
  Divider,
  LinearProgress,
  Chip,
  Button,
  IconButton,
  Tabs,
  Tab,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Tooltip,
  CircularProgress,
  Alert,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions
} from '@mui/material';
import {
  Refresh as RefreshIcon,
  Delete as DeleteIcon,
  Info as InfoIcon,
  CheckCircle as CheckCircleIcon,
  Error as ErrorIcon,
  HourglassEmpty as PendingIcon,
  PlayArrow as RunningIcon,
  Cancel as CancelIcon
} from '@mui/icons-material';
import { formatDistanceToNow, formatDistance } from 'date-fns';
import { apiClient } from '../services/apiClient';

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

// Progress Tracker component
const ProgressTracker = ({ operation }) => {
  return (
    <Card variant="outlined" sx={{ mb: 2 }}>
      <CardHeader
        title={
          <Box display="flex" alignItems="center">
            <StatusIcon status={operation.status} />
            <Typography variant="h6" sx={{ ml: 1 }}>
              {operation.operation_id}
            </Typography>
          </Box>
        }
        subheader={`Type: ${operation.operation_type}`}
        action={
          <Chip
            label={operation.status}
            color={statusColors[operation.status] || 'default'}
            size="small"
          />
        }
      />
      <CardContent>
        <Box sx={{ width: '100%', mb: 2 }}>
          <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
            <Typography variant="body2" color="text.secondary">
              {operation.message}
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

        <Grid container spacing={2}>
          <Grid item xs={6}>
            <Typography variant="body2" color="text.secondary">
              Started: {formatTime(operation.start_time)}
            </Typography>
          </Grid>
          <Grid item xs={6}>
            <Typography variant="body2" color="text.secondary">
              {operation.end_time ?
                `Ended: ${formatTime(operation.end_time)}` :
                'In progress...'
              }
            </Typography>
          </Grid>
        </Grid>
      </CardContent>
    </Card>
  );
};

// Operation Details Dialog
const OperationDetailsDialog = ({ open, onClose, operationId }) => {
  const { data: operation, isLoading, error } = useQuery(
    ['operation', operationId],
    () => apiClient.get(`/api/llm/progress/operations/${operationId}`).then(res => res.data),
    {
      enabled: !!operationId && open,
      refetchInterval: (data) =>
        data && (data.status === 'running' || data.status === 'pending') ? 2000 : false
    }
  );

  return (
    <Dialog open={open} onClose={onClose} maxWidth="md" fullWidth>
      <DialogTitle>
        Operation Details
        {isLoading && <CircularProgress size={20} sx={{ ml: 2 }} />}
      </DialogTitle>
      <DialogContent>
        {error && (
          <Alert severity="error" sx={{ mb: 2 }}>
            Error loading operation details: {error.message}
          </Alert>
        )}

        {operation && (
          <>
            <Grid container spacing={2} sx={{ mb: 3 }}>
              <Grid item xs={12} md={6}>
                <Typography variant="subtitle1">Operation ID</Typography>
                <Typography variant="body1" gutterBottom>{operation.operation_id}</Typography>

                <Typography variant="subtitle1">Type</Typography>
                <Typography variant="body1" gutterBottom>{operation.operation_type}</Typography>

                <Typography variant="subtitle1">Status</Typography>
                <Chip
                  label={operation.status}
                  color={statusColors[operation.status] || 'default'}
                  icon={<StatusIcon status={operation.status} />}
                  sx={{ mb: 1 }}
                />

                <Typography variant="subtitle1">Progress</Typography>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
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
              </Grid>

              <Grid item xs={12} md={6}>
                <Typography variant="subtitle1">Started</Typography>
                <Typography variant="body1" gutterBottom>{formatTime(operation.start_time)}</Typography>

                <Typography variant="subtitle1">Ended</Typography>
                <Typography variant="body1" gutterBottom>
                  {operation.end_time ? formatTime(operation.end_time) : 'In progress...'}
                </Typography>

                <Typography variant="subtitle1">Elapsed Time</Typography>
                <Typography variant="body1" gutterBottom>
                  {formatElapsedTime(operation.elapsed_time)}
                </Typography>

                <Typography variant="subtitle1">Estimated Time Remaining</Typography>
                <Typography variant="body1" gutterBottom>
                  {operation.estimated_time_remaining ?
                    formatElapsedTime(operation.estimated_time_remaining) :
                    'N/A'
                  }
                </Typography>
              </Grid>
            </Grid>

            <Typography variant="h6" gutterBottom>Progress Steps</Typography>
            <TableContainer component={Paper} variant="outlined" sx={{ mb: 3 }}>
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

            {operation.metadata && Object.keys(operation.metadata).length > 0 && (
              <>
                <Typography variant="h6" gutterBottom>Metadata</Typography>
                <TableContainer component={Paper} variant="outlined">
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
              </>
            )}
          </>
        )}
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>Close</Button>
      </DialogActions>
    </Dialog>
  );
};

// Main component
const LLMProgressTracking = () => {
  const [tabValue, setTabValue] = useState(0);
  const [selectedOperation, setSelectedOperation] = useState(null);
  const [detailsOpen, setDetailsOpen] = useState(false);
  const queryClient = useQueryClient();

  // Fetch operations
  const {
    data: operations,
    isLoading: operationsLoading,
    error: operationsError,
    refetch: refetchOperations
  } = useQuery(
    ['operations', tabValue],
    () => {
      if (tabValue === 0) {
        return apiClient.get('/api/llm/progress/active').then(res => res.data);
      } else {
        return apiClient.get('/api/llm/progress/operations').then(res => res.data);
      }
    },
    {
      refetchInterval: tabValue === 0 ? 2000 : false,
      keepPreviousData: true
    }
  );

  // Fetch summary
  const {
    data: summary,
    isLoading: summaryLoading,
    error: summaryError,
    refetch: refetchSummary
  } = useQuery(
    ['progress-summary'],
    () => apiClient.get('/api/llm/progress/summary').then(res => res.data),
    { refetchInterval: 5000 }
  );

  // Handle tab change
  const handleTabChange = (event, newValue) => {
    setTabValue(newValue);
  };

  // Handle refresh
  const handleRefresh = () => {
    refetchOperations();
    refetchSummary();
  };

  // Handle view details
  const handleViewDetails = (operationId) => {
    setSelectedOperation(operationId);
    setDetailsOpen(true);
  };

  // Handle delete operation
  const handleDeleteOperation = async (operationId) => {
    try {
      await apiClient.delete(`/api/llm/progress/operations/${operationId}`);
      queryClient.invalidateQueries(['operations']);
      queryClient.invalidateQueries(['progress-summary']);
    } catch (error) {
      console.error('Error deleting operation:', error);
    }
  };

  // Handle cleanup
  const handleCleanup = async () => {
    try {
      await apiClient.post('/api/llm/progress/cleanup');
      queryClient.invalidateQueries(['operations']);
      queryClient.invalidateQueries(['progress-summary']);
    } catch (error) {
      console.error('Error cleaning up operations:', error);
    }
  };

  return (
    <PageLayout title="LLM Progress Tracking">
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h4">LLM Progress Tracking</Typography>
        <Box>
          <Button
            variant="outlined"
            startIcon={<RefreshIcon />}
            onClick={handleRefresh}
            sx={{ mr: 1 }}
          >
            Refresh
          </Button>
          <Button
            variant="outlined"
            color="warning"
            onClick={handleCleanup}
          >
            Cleanup Old Operations
          </Button>
        </Box>
      </Box>

      {/* Summary Cards */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        {summaryLoading ? (
          <Grid item xs={12}>
            <CircularProgress />
          </Grid>
        ) : summaryError ? (
          <Grid item xs={12}>
            <Alert severity="error">
              Error loading summary: {summaryError.message}
            </Alert>
          </Grid>
        ) : summary && (
          <>
            <Grid item xs={12} sm={6} md={3}>
              <Card>
                <CardContent>
                  <Typography color="text.secondary" gutterBottom>
                    Total Operations
                  </Typography>
                  <Typography variant="h4">{summary.total}</Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <Card>
                <CardContent>
                  <Typography color="text.secondary" gutterBottom>
                    Active Operations
                  </Typography>
                  <Typography variant="h4" color="primary">{summary.active}</Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <Card>
                <CardContent>
                  <Typography color="text.secondary" gutterBottom>
                    Completed Operations
                  </Typography>
                  <Typography variant="h4" color="success.main">{summary.completed}</Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <Card>
                <CardContent>
                  <Typography color="text.secondary" gutterBottom>
                    Failed Operations
                  </Typography>
                  <Typography variant="h4" color="error.main">{summary.failed}</Typography>
                </CardContent>
              </Card>
            </Grid>
          </>
        )}
      </Grid>

      {/* Operations List */}
      <Paper sx={{ mb: 3 }}>
        <Tabs
          value={tabValue}
          onChange={handleTabChange}
          indicatorColor="primary"
          textColor="primary"
          sx={{ borderBottom: 1, borderColor: 'divider' }}
        >
          <Tab label="Active Operations" />
          <Tab label="All Operations" />
        </Tabs>

        <Box sx={{ p: 2 }}>
          {operationsLoading ? (
            <Box display="flex" justifyContent="center" p={3}>
              <CircularProgress />
            </Box>
          ) : operationsError ? (
            <Alert severity="error" sx={{ m: 2 }}>
              Error loading operations: {operationsError.message}
            </Alert>
          ) : operations && operations.length === 0 ? (
            <Alert severity="info" sx={{ m: 2 }}>
              No {tabValue === 0 ? 'active ' : ''}operations found.
            </Alert>
          ) : (
            <TableContainer>
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell>Status</TableCell>
                    <TableCell>Operation ID</TableCell>
                    <TableCell>Type</TableCell>
                    <TableCell>Progress</TableCell>
                    <TableCell>Message</TableCell>
                    <TableCell>Started</TableCell>
                    <TableCell>Actions</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {operations && operations.map((operation) => (
                    <TableRow key={operation.operation_id}>
                      <TableCell>
                        <Chip
                          label={operation.status}
                          color={statusColors[operation.status] || 'default'}
                          size="small"
                          icon={<StatusIcon status={operation.status} />}
                        />
                      </TableCell>
                      <TableCell>{operation.operation_id}</TableCell>
                      <TableCell>{operation.operation_type}</TableCell>
                      <TableCell>
                        <Box sx={{ display: 'flex', alignItems: 'center' }}>
                          <Box sx={{ width: 100, mr: 1 }}>
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
                      </TableCell>
                      <TableCell>{operation.message}</TableCell>
                      <TableCell>
                        <Tooltip title={formatTime(operation.start_time)}>
                          <span>
                            {formatDistanceToNow(new Date(operation.start_time), { addSuffix: true })}
                          </span>
                        </Tooltip>
                      </TableCell>
                      <TableCell>
                        <Tooltip title="View Details">
                          <IconButton
                            size="small"
                            onClick={() => handleViewDetails(operation.operation_id)}
                          >
                            <InfoIcon />
                          </IconButton>
                        </Tooltip>
                        <Tooltip title="Delete">
                          <IconButton
                            size="small"
                            color="error"
                            onClick={() => handleDeleteOperation(operation.operation_id)}
                          >
                            <DeleteIcon />
                          </IconButton>
                        </Tooltip>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          )}
        </Box>
      </Paper>

      {/* Operation Details Dialog */}
      <OperationDetailsDialog
        open={detailsOpen}
        onClose={() => setDetailsOpen(false)}
        operationId={selectedOperation}
      />
    </PageLayout>
  );
};

export default LLMProgressTracking;
