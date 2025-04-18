import React, { useState, useEffect } from 'react';
import {
  Box,
  Button,
  Card,
  CardContent,
  Typography,
  TextField,
  Grid,
  Paper,
  Divider,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  CircularProgress,
  Alert,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip,
  IconButton,
  Tooltip,
  LinearProgress,
  Switch,
  FormControlLabel,
  Stack,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions
} from '@mui/material';
import {
  Upload as UploadIcon,
  Delete as DeleteIcon,
  CloudUpload as CloudUploadIcon,
  Refresh as RefreshIcon,
  CheckCircle as CheckCircleIcon,
  Error as ErrorIcon,
  HourglassEmpty as HourglassEmptyIcon,
  Speed as SpeedIcon,
  Settings as SettingsIcon,
  Info as InfoIcon
} from '@mui/icons-material';
import { useDropzone } from 'react-dropzone';
import apiService from '../../services/api';
import useWebSocket from '../../hooks/useWebSocket';
import ProcessingLog from './ProcessingLog';

/**
 * Component for batch processing multiple documents
 */
const BatchDocumentProcessor = () => {
  const [files, setFiles] = useState([]);
  const [batchSize, setBatchSize] = useState(4);
  const [useEnhanced, setUseEnhanced] = useState(true);
  const [useStreaming, setUseStreaming] = useState(true);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [taskIds, setTaskIds] = useState([]);
  const [taskStatuses, setTaskStatuses] = useState({});
  const [pollingInterval, setPollingInterval] = useState(null);
  const [overallProgress, setOverallProgress] = useState(0);
  const [taskProgress, setTaskProgress] = useState({});
  const [processingLogs, setProcessingLogs] = useState({});
  const [selectedTaskLogs, setSelectedTaskLogs] = useState(null);
  const [logDialogOpen, setLogDialogOpen] = useState(false);

  // WebSocket hook
  const {
    connected: wsConnected,
    subscribeToTask,
    unsubscribeFromTask,
    onProgress,
    onCompleted,
    onFailed,
    onIntermediateResult
  } = useWebSocket();

  // Dropzone configuration
  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    accept: {
      'application/pdf': ['.pdf'],
      'text/plain': ['.txt'],
      'text/markdown': ['.md'],
      'application/json': ['.json']
    },
    onDrop: acceptedFiles => {
      if (acceptedFiles.length > 0) {
        setFiles(prevFiles => [...prevFiles, ...acceptedFiles]);
        setError(null);
      }
    }
  });

  // Clear the polling interval when component unmounts
  useEffect(() => {
    return () => {
      if (pollingInterval) {
        clearInterval(pollingInterval);
      }
    };
  }, [pollingInterval]);

  // Subscribe to WebSocket updates for tasks
  useEffect(() => {
    if (taskIds.length > 0 && wsConnected) {
      // Subscribe to all tasks
      const subscribeToTasks = async () => {
        for (const taskId of taskIds) {
          try {
            await subscribeToTask(taskId);
          } catch (err) {
            console.error(`Error subscribing to task ${taskId}:`, err);
          }
        }
      };

      subscribeToTasks();

      // Set up handlers for each task
      const progressHandlers = {};
      const completedHandlers = {};
      const failedHandlers = {};

      for (const taskId of taskIds) {
        // Progress handler
        progressHandlers[taskId] = (message) => {
          setTaskProgress(prev => ({
            ...prev,
            [taskId]: {
              progress: message.progress * 100,
              stage: message.stage
            }
          }));

          // Add log entry
          setProcessingLogs(prev => ({
            ...prev,
            [taskId]: [
              ...(prev[taskId] || []),
              {
                timestamp: new Date().toISOString(),
                level: 'info',
                message: `Progress: ${Math.round(message.progress * 100)}% - Stage: ${message.stage}`,
                details: `File: ${taskStatuses[taskId]?.file_name || 'Unknown'}
Task ID: ${taskId}
Metrics: ${JSON.stringify(message.metrics || {}).substring(0, 100)}`,
                stage: message.stage
              }
            ]
          }));

          // Update overall progress
          updateOverallProgress();
        };

        // Completed handler
        completedHandlers[taskId] = (message) => {
          setTaskStatuses(prev => ({
            ...prev,
            [taskId]: {
              ...prev[taskId],
              status: 'completed',
              completed_at: new Date().toISOString(),
              entity_count: message.result?.entity_count || 0,
              relation_count: message.result?.relation_count || 0,
              processing_time: message.result?.processing_time || 0
            }
          }));

          setTaskProgress(prev => ({
            ...prev,
            [taskId]: {
              progress: 100,
              stage: 'completed'
            }
          }));

          // Add log entry
          setProcessingLogs(prev => ({
            ...prev,
            [taskId]: [
              ...(prev[taskId] || []),
              {
                timestamp: new Date().toISOString(),
                level: 'success',
                message: `Processing completed successfully. Entities: ${message.result?.entity_count || 0}, Relations: ${message.result?.relation_count || 0}`,
                details: `File: ${taskStatuses[taskId]?.file_name || 'Unknown'}
Task ID: ${taskId}
Processing time: ${message.result?.processing_time ? `${message.result.processing_time.toFixed(2)}s` : 'N/A'}
Result path: ${message.result?.result_path || 'Unknown'}`,
                stage: 'completed'
              }
            ]
          }));

          // Update overall progress
          updateOverallProgress();
        };

        // Failed handler
        failedHandlers[taskId] = (message) => {
          setTaskStatuses(prev => ({
            ...prev,
            [taskId]: {
              ...prev[taskId],
              status: 'failed',
              completed_at: new Date().toISOString(),
              error_message: message.error
            }
          }));

          // Add log entry
          setProcessingLogs(prev => ({
            ...prev,
            [taskId]: [
              ...(prev[taskId] || []),
              {
                timestamp: new Date().toISOString(),
                level: 'error',
                message: `Processing failed: ${message.error || 'Unknown error'}`,
                details: `File: ${taskStatuses[taskId]?.file_name || 'Unknown'}
Task ID: ${taskId}
Status: ${taskStatuses[taskId]?.status || 'Unknown'}
Error details: ${message.error || 'No additional details available'}`,
                stage: 'failed'
              }
            ]
          }));

          // Update overall progress
          updateOverallProgress();
        };

        // Intermediate result handler
        const intermediateResultHandler = (message) => {
          // Add log entry
          setProcessingLogs(prev => ({
            ...prev,
            [taskId]: [
              ...(prev[taskId] || []),
              {
                timestamp: new Date().toISOString(),
                level: 'info',
                message: `Intermediate result from ${message.stage}`,
                details: `File: ${taskStatuses[taskId]?.file_name || 'Unknown'}
Task ID: ${taskId}
Timestamp: ${message.timestamp || new Date().toISOString()}
Result type: ${typeof message.result}
Preview: ${JSON.stringify(message.result).substring(0, 150)}...`,
                stage: message.stage
              }
            ]
          }));
        };

        // Register handlers
        onProgress(taskId, progressHandlers[taskId]);
        onCompleted(taskId, completedHandlers[taskId]);
        onFailed(taskId, failedHandlers[taskId]);
        onIntermediateResult(taskId, intermediateResultHandler);
      }

      // Cleanup function
      return () => {
        // Unsubscribe from all tasks
        for (const taskId of taskIds) {
          try {
            unsubscribeFromTask(taskId);
          } catch (err) {
            console.error(`Error unsubscribing from task ${taskId}:`, err);
          }
        }
      };
    }
  }, [taskIds, wsConnected, subscribeToTask, unsubscribeFromTask, onProgress, onCompleted, onFailed]);

  // Update overall progress based on individual task progress
  const updateOverallProgress = () => {
    if (taskIds.length === 0) return;

    let totalProgress = 0;
    let completedCount = 0;

    for (const taskId of taskIds) {
      const status = taskStatuses[taskId]?.status;
      const progress = taskProgress[taskId]?.progress || 0;

      if (status === 'completed' || status === 'failed') {
        totalProgress += 100;
        completedCount++;
      } else {
        totalProgress += progress;
      }
    }

    const avgProgress = totalProgress / taskIds.length;
    setOverallProgress(avgProgress);

    // If all tasks are completed or failed, set loading to false
    if (completedCount === taskIds.length) {
      setLoading(false);
    }
  };

  // Fallback to polling if WebSocket is not connected
  useEffect(() => {
    if (taskIds.length > 0 && !wsConnected) {
      const interval = setInterval(async () => {
        let allCompleted = true;
        let completedCount = 0;

        for (const taskId of taskIds) {
          try {
            const response = await apiService.documentProcessing.getTask(taskId);
            setTaskStatuses(prev => ({
              ...prev,
              [taskId]: response.data
            }));

            if (response.data.status !== 'completed' && response.data.status !== 'failed') {
              allCompleted = false;
            } else {
              completedCount++;
            }
          } catch (err) {
            console.error(`Error polling task status for ${taskId}:`, err);
          }
        }

        // Update progress
        setOverallProgress((completedCount / taskIds.length) * 100);

        if (allCompleted) {
          clearInterval(interval);
          setLoading(false);
        }
      }, 2000);

      setPollingInterval(interval);

      return () => clearInterval(interval);
    }
  }, [taskIds, wsConnected]);

  // Handle form submission
  const handleSubmit = async (e) => {
    e.preventDefault();

    if (files.length === 0) {
      setError('Please select at least one file to process');
      return;
    }

    setLoading(true);
    setError(null);
    setTaskIds([]);
    setTaskStatuses({});
    setOverallProgress(0);

    try {
      // Get default settings first
      const settingsResponse = await apiService.documentProcessing.getSettings();

      if (!settingsResponse.success) {
        throw new Error('Failed to get processing settings');
      }

      // Create settings object with our preferences
      const settings = {
        ...settingsResponse.data,
        use_enhanced_synthesizer: useEnhanced,
        use_streaming: useStreaming
      };

      // Create form data with files and settings
      const formData = new FormData();
      files.forEach(file => {
        formData.append('files', file);
      });
      formData.append('batch_size', batchSize.toString());
      formData.append('settings_json', JSON.stringify(settings));

      const response = await apiService.documentProcessing.processBatch(formData);

      const newTaskIds = response.data.map(task => task.task_id);
      setTaskIds(newTaskIds);

      // Initialize task statuses
      const initialStatuses = {};
      newTaskIds.forEach((taskId, index) => {
        initialStatuses[taskId] = {
          task_id: taskId,
          status: 'queued',
          file_name: files[index].name,
          created_at: new Date().toISOString()
        };
      });
      setTaskStatuses(initialStatuses);
    } catch (err) {
      console.error('Error processing documents:', err);
      setError('Error processing documents. Please try again.');
      setLoading(false);
    }
  };

  // Remove a file from the list
  const handleRemoveFile = (index) => {
    setFiles(prevFiles => prevFiles.filter((_, i) => i !== index));
  };

  // Clear all files
  const handleClearFiles = () => {
    setFiles([]);
  };

  // Refresh task statuses
  const handleRefreshStatuses = async () => {
    if (taskIds.length === 0) return;

    let completedCount = 0;

    for (const taskId of taskIds) {
      try {
        const response = await apiService.documentProcessing.getTask(taskId);
        setTaskStatuses(prev => ({
          ...prev,
          [taskId]: response.data
        }));

        if (response.data.status === 'completed' || response.data.status === 'failed') {
          completedCount++;
        }
      } catch (err) {
        console.error(`Error refreshing task status for ${taskId}:`, err);
      }
    }

    // Update progress
    setOverallProgress((completedCount / taskIds.length) * 100);
  };

  // Get status chip for a task
  const getStatusChip = (status) => {
    switch (status) {
      case 'queued':
        return <Chip icon={<HourglassEmptyIcon />} label="Queued" color="default" size="small" />;
      case 'processing':
        return <Chip icon={<CircularProgress size={16} />} label="Processing" color="primary" size="small" />;
      case 'completed':
        return <Chip icon={<CheckCircleIcon />} label="Completed" color="success" size="small" />;
      case 'failed':
        return <Chip icon={<ErrorIcon />} label="Failed" color="error" size="small" />;
      default:
        return <Chip label={status} size="small" />;
    }
  };

  return (
    <>
      <Box>
        <form onSubmit={handleSubmit}>
        <Grid container spacing={3}>
          <Grid item xs={12}>
            <Paper
              {...getRootProps()}
              sx={{
                p: 3,
                border: '2px dashed',
                borderColor: isDragActive ? 'primary.main' : 'grey.300',
                backgroundColor: isDragActive ? 'rgba(0, 0, 0, 0.05)' : 'background.paper',
                textAlign: 'center',
                cursor: 'pointer',
                transition: 'all 0.2s ease-in-out',
                '&:hover': {
                  borderColor: 'primary.main',
                  backgroundColor: 'rgba(0, 0, 0, 0.05)'
                }
              }}
            >
              <input {...getInputProps()} />
              <UploadIcon sx={{ fontSize: 48, color: 'primary.main', mb: 2 }} />
              <Typography variant="h6" gutterBottom>
                {isDragActive ? 'Drop the files here' : 'Drag & drop files here, or click to select'}
              </Typography>
              <Typography variant="body2" color="textSecondary">
                Supported formats: PDF, TXT, MD, JSON
              </Typography>
            </Paper>
          </Grid>

          {files.length > 0 && (
            <Grid item xs={12}>
              <Paper sx={{ p: 2 }}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                  <Typography variant="subtitle1">
                    Selected Files ({files.length})
                  </Typography>
                  <Button
                    variant="outlined"
                    color="error"
                    size="small"
                    startIcon={<DeleteIcon />}
                    onClick={handleClearFiles}
                  >
                    Clear All
                  </Button>
                </Box>
                <TableContainer sx={{ maxHeight: 300 }}>
                  <Table size="small" stickyHeader>
                    <TableHead>
                      <TableRow>
                        <TableCell>File Name</TableCell>
                        <TableCell>Size</TableCell>
                        <TableCell>Type</TableCell>
                        <TableCell align="right">Actions</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {files.map((file, index) => (
                        <TableRow key={index}>
                          <TableCell>{file.name}</TableCell>
                          <TableCell>{(file.size / 1024).toFixed(1)} KB</TableCell>
                          <TableCell>{file.type || 'Unknown'}</TableCell>
                          <TableCell align="right">
                            <IconButton
                              size="small"
                              color="error"
                              onClick={() => handleRemoveFile(index)}
                            >
                              <DeleteIcon fontSize="small" />
                            </IconButton>
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              </Paper>
            </Grid>
          )}

          <Grid item xs={12}>
            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 2, mb: 2 }}>
              <FormControl sx={{ minWidth: 120 }}>
                <InputLabel id="batch-size-label">Batch Size</InputLabel>
                <Select
                  labelId="batch-size-label"
                  id="batch-size"
                  value={batchSize}
                  label="Batch Size"
                  onChange={(e) => setBatchSize(e.target.value)}
                  disabled={loading}
                >
                  <MenuItem value={1}>1</MenuItem>
                  <MenuItem value={2}>2</MenuItem>
                  <MenuItem value={4}>4</MenuItem>
                  <MenuItem value={8}>8</MenuItem>
                </Select>
              </FormControl>

              <FormControlLabel
                control={
                  <Switch
                    checked={useEnhanced}
                    onChange={(e) => setUseEnhanced(e.target.checked)}
                    color="primary"
                    disabled={loading}
                  />
                }
                label={
                  <Tooltip title="Use the enhanced document processor with improved performance and features">
                    <Box sx={{ display: 'flex', alignItems: 'center' }}>
                      <SettingsIcon fontSize="small" sx={{ mr: 0.5 }} />
                      <span>Use enhanced processor</span>
                    </Box>
                  </Tooltip>
                }
              />

              <FormControlLabel
                control={
                  <Switch
                    checked={useStreaming}
                    onChange={(e) => setUseStreaming(e.target.checked)}
                    disabled={!useEnhanced || loading}
                    color="primary"
                  />
                }
                label={
                  <Tooltip title="Stream results as they become available">
                    <Box sx={{ display: 'flex', alignItems: 'center' }}>
                      <SpeedIcon fontSize="small" sx={{ mr: 0.5 }} />
                      <span>Use streaming</span>
                    </Box>
                  </Tooltip>
                }
              />
            </Box>

            <Box sx={{ display: 'flex', justifyContent: 'flex-end' }}>
              <Button
                type="submit"
                variant="contained"
                color="primary"
                disabled={files.length === 0 || loading}
                startIcon={loading ? <CircularProgress size={20} /> : <CloudUploadIcon />}
              >
                {loading ? 'Processing...' : 'Process Batch'}
              </Button>
            </Box>
          </Grid>
        </Grid>
      </form>

      {error && (
        <Alert severity="error" sx={{ mt: 3 }}>
          {error}
        </Alert>
      )}

      {taskIds.length > 0 && (
        <Paper sx={{ mt: 3, p: 2 }}>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
            <Typography variant="subtitle1">
              Batch Processing Status
            </Typography>
            <Tooltip title="Refresh Status">
              <IconButton
                color="primary"
                onClick={handleRefreshStatuses}
                disabled={loading}
              >
                <RefreshIcon />
              </IconButton>
            </Tooltip>
          </Box>

          <Box sx={{ mb: 2 }}>
            <Typography variant="body2" color="textSecondary" gutterBottom>
              Overall Progress: {Math.round(overallProgress)}%
            </Typography>
            <LinearProgress variant="determinate" value={overallProgress} sx={{ height: 8, borderRadius: 4 }} />
          </Box>

          <TableContainer sx={{ maxHeight: 400 }}>
            <Table size="small" stickyHeader>
              <TableHead>
                <TableRow>
                  <TableCell>File Name</TableCell>
                  <TableCell>Status</TableCell>
                  <TableCell>Created</TableCell>
                  <TableCell>Completed</TableCell>
                  <TableCell>Processing Time</TableCell>
                  <TableCell>Entities</TableCell>
                  <TableCell>Relations</TableCell>
                  <TableCell>Progress</TableCell>
                  <TableCell align="right">Actions</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {taskIds.map((taskId) => {
                  const task = taskStatuses[taskId] || {
                    task_id: taskId,
                    status: 'unknown',
                    file_name: 'Unknown',
                    created_at: null,
                    completed_at: null
                  };

                  return (
                    <TableRow key={taskId}>
                      <TableCell>{task.file_name}</TableCell>
                      <TableCell>{getStatusChip(task.status)}</TableCell>
                      <TableCell>
                        {task.created_at ? new Date(task.created_at).toLocaleTimeString() : 'N/A'}
                      </TableCell>
                      <TableCell>
                        {task.completed_at ? new Date(task.completed_at).toLocaleTimeString() : 'N/A'}
                      </TableCell>
                      <TableCell>
                        {task.processing_time ? `${task.processing_time.toFixed(2)}s` : 'N/A'}
                      </TableCell>
                      <TableCell>
                        {task.entity_count !== undefined ? task.entity_count : 'N/A'}
                      </TableCell>
                      <TableCell>
                        {task.relation_count !== undefined ? task.relation_count : 'N/A'}
                      </TableCell>
                      <TableCell>
                        {task.status === 'processing' && (
                          <Box sx={{ width: '100%' }}>
                            <Typography variant="caption" color="textSecondary" gutterBottom>
                              {taskProgress[taskId]?.stage || 'Processing'} - {Math.round(taskProgress[taskId]?.progress || 0)}%
                            </Typography>
                            <LinearProgress
                              variant="determinate"
                              value={taskProgress[taskId]?.progress || 0}
                              sx={{ height: 4, borderRadius: 2 }}
                            />
                          </Box>
                        )}
                      </TableCell>
                      <TableCell align="right">
                        <Tooltip title="View Logs">
                          <IconButton
                            color="info"
                            onClick={() => {
                              setSelectedTaskLogs(taskId);
                              setLogDialogOpen(true);
                            }}
                            size="small"
                            sx={{ mr: 1 }}
                          >
                            <InfoIcon />
                          </IconButton>
                        </Tooltip>
                      </TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
          </TableContainer>
        </Paper>
      )}
      </Box>

      {/* Processing Logs Dialog */}
    <Dialog
      open={logDialogOpen}
      onClose={() => setLogDialogOpen(false)}
      maxWidth="md"
      fullWidth
    >
      <DialogTitle>
        Processing Logs
        {selectedTaskLogs && taskStatuses[selectedTaskLogs] && (
          <Typography variant="subtitle2" color="textSecondary">
            {taskStatuses[selectedTaskLogs].file_name}
          </Typography>
        )}
      </DialogTitle>
      <DialogContent dividers sx={{ height: '70vh' }}>
        {selectedTaskLogs ? (
          <ProcessingLog
            logs={processingLogs[selectedTaskLogs] || []}
            isLoading={false}
            error={null}
            title={`Task ID: ${selectedTaskLogs}`}
          />
        ) : (
          <Typography>No logs available</Typography>
        )}
      </DialogContent>
      <DialogActions>
        <Button onClick={() => setLogDialogOpen(false)}>Close</Button>
      </DialogActions>
    </Dialog>
    </>
  );
};

export default BatchDocumentProcessor;
