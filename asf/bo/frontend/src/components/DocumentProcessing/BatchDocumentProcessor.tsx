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
  DialogActions,
  SelectChangeEvent
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

import { useDocumentProcessing } from '../../hooks/useDocumentProcessing';
import useWebSocket from '../../hooks/useWebSocket';
import { useNotification } from '../../context/NotificationContext';
import { useFeatureFlags } from '../../context/FeatureFlagContext';
import ProcessingLog from './ProcessingLog';
import { ButtonLoader } from '../UI/LoadingIndicators';

interface ProcessingLog {
  timestamp: string;
  level: 'info' | 'success' | 'error' | 'warning';
  message: string;
  details?: string;
  stage?: string;
}

interface TaskStatus {
  task_id: string;
  status: string;
  file_name: string;
  created_at: string;
  completed_at?: string;
  error_message?: string;
  entity_count?: number;
  relation_count?: number;
  processing_time?: number;
}

interface TaskProgress {
  progress: number;
  stage: string;
}

/**
 * Component for batch processing multiple documents
 */
const BatchDocumentProcessor: React.FC = () => {
  const [files, setFiles] = useState<File[]>([]);
  const [batchSize, setBatchSize] = useState<number>(4);
  const [useEnhanced, setUseEnhanced] = useState<boolean>(true);
  const [useStreaming, setUseStreaming] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [taskIds, setTaskIds] = useState<string[]>([]);
  const [taskStatuses, setTaskStatuses] = useState<Record<string, TaskStatus>>({});
  const [pollingInterval, setPollingInterval] = useState<NodeJS.Timeout | null>(null);
  const [overallProgress, setOverallProgress] = useState<number>(0);
  const [taskProgress, setTaskProgress] = useState<Record<string, TaskProgress>>({});
  const [processingLogs, setProcessingLogs] = useState<Record<string, ProcessingLog[]>>({});
  const [selectedTaskLogs, setSelectedTaskLogs] = useState<string | null>(null);
  const [logDialogOpen, setLogDialogOpen] = useState<boolean>(false);

  const { showSuccess, showError } = useNotification();
  const { isEnabled } = useFeatureFlags();
  const useMockData = isEnabled('useMockData');

  // Document processing hooks
  const {
    uploadDocument,
    batchProcessDocuments
  } = useDocumentProcessing();

  // Get the mutation functions
  const { mutate: uploadMutate, isPending: isUploading } = uploadDocument();
  const { mutate: batchProcessMutate, isPending: isProcessing } = batchProcessDocuments();

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
      const progressHandlers: Record<string, (message: any) => void> = {};
      const completedHandlers: Record<string, (message: any) => void> = {};
      const failedHandlers: Record<string, (message: any) => void> = {};
      const cleanupFunctions: (() => void)[] = [];

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
        const intermediateResultHandler = (message: any) => {
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
        const unsubProgress = onProgress(taskId, progressHandlers[taskId]);
        const unsubCompleted = onCompleted(taskId, completedHandlers[taskId]);
        const unsubFailed = onFailed(taskId, failedHandlers[taskId]);
        const unsubIntermediate = onIntermediateResult(taskId, intermediateResultHandler);

        cleanupFunctions.push(unsubProgress);
        cleanupFunctions.push(unsubCompleted);
        cleanupFunctions.push(unsubFailed);
        cleanupFunctions.push(unsubIntermediate);
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

        // Call all cleanup functions
        cleanupFunctions.forEach(fn => fn());
      };
    }
  }, [taskIds, wsConnected, subscribeToTask, unsubscribeFromTask, onProgress, onCompleted, onFailed, onIntermediateResult, taskStatuses]);

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
  };

  // Handle form submission
  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();

    if (files.length === 0) {
      setError('Please select at least one file to process');
      return;
    }

    setError(null);
    setTaskIds([]);
    setTaskStatuses({});
    setOverallProgress(0);

    // Create form data with files
    const formData = new FormData();
    files.forEach(file => {
      formData.append('files', file);
    });

    // Upload files first
    uploadMutate(formData, {
      onSuccess: (data) => {
        // Now process the uploaded documents
        const documentIds = data.document_ids || [];
        
        if (documentIds.length === 0) {
          showError('No documents were uploaded successfully');
          return;
        }

        const batchParams = {
          document_ids: documentIds,
          options: {
            extract_entities: true,
            extract_relations: true,
            detect_sections: true,
            parse_references: true,
            generate_summary: true,
            use_enhanced_synthesizer: useEnhanced
          }
        };

        batchProcessMutate(batchParams, {
          onSuccess: (data) => {
            const newTaskIds = data.task_ids || [];
            setTaskIds(newTaskIds);

            // Initialize task statuses
            const initialStatuses: Record<string, TaskStatus> = {};
            newTaskIds.forEach((taskId, index) => {
              initialStatuses[taskId] = {
                task_id: taskId,
                status: 'queued',
                file_name: files[index]?.name || `Document ${index + 1}`,
                created_at: new Date().toISOString()
              };
            });
            setTaskStatuses(initialStatuses);

            showSuccess(`Started batch processing of ${newTaskIds.length} documents`);
          }
        });
      }
    });
  };

  // Remove a file from the list
  const handleRemoveFile = (index: number) => {
    setFiles(prevFiles => prevFiles.filter((_, i) => i !== index));
  };

  // Clear all files
  const handleClearFiles = () => {
    setFiles([]);
  };

  // Handle batch size change
  const handleBatchSizeChange = (e: SelectChangeEvent<number>) => {
    setBatchSize(Number(e.target.value));
  };

  // View logs for a task
  const handleViewLogs = (taskId: string) => {
    setSelectedTaskLogs(taskId);
    setLogDialogOpen(true);
  };

  // Close log dialog
  const handleCloseLogDialog = () => {
    setLogDialogOpen(false);
    setSelectedTaskLogs(null);
  };

  // Get status chip for a task
  const getStatusChip = (status: string) => {
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
        {useMockData && (
          <Alert severity="info" sx={{ mb: 2 }}>
            Using mock data. Toggle the "Use Mock Data" feature flag to use real API data.
          </Alert>
        )}

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
                  <TableContainer>
                    <Table size="small">
                      <TableHead>
                        <TableRow>
                          <TableCell>File Name</TableCell>
                          <TableCell>Type</TableCell>
                          <TableCell>Size</TableCell>
                          <TableCell align="right">Actions</TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {files.map((file, index) => (
                          <TableRow key={index}>
                            <TableCell>{file.name}</TableCell>
                            <TableCell>{file.type || 'Unknown'}</TableCell>
                            <TableCell>{(file.size / 1024).toFixed(2)} KB</TableCell>
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
              <Paper sx={{ p: 2 }}>
                <Typography variant="subtitle1" gutterBottom>
                  Processing Options
                </Typography>
                <Grid container spacing={2}>
                  <Grid item xs={12} md={4}>
                    <FormControl fullWidth>
                      <InputLabel id="batch-size-label">Batch Size</InputLabel>
                      <Select
                        labelId="batch-size-label"
                        value={batchSize}
                        label="Batch Size"
                        onChange={handleBatchSizeChange}
                      >
                        <MenuItem value={1}>1 (Sequential)</MenuItem>
                        <MenuItem value={2}>2</MenuItem>
                        <MenuItem value={4}>4</MenuItem>
                        <MenuItem value={8}>8</MenuItem>
                        <MenuItem value={16}>16 (Maximum)</MenuItem>
                      </Select>
                    </FormControl>
                  </Grid>
                  <Grid item xs={12} md={4}>
                    <FormControlLabel
                      control={
                        <Switch
                          checked={useEnhanced}
                          onChange={(e) => setUseEnhanced(e.target.checked)}
                        />
                      }
                      label="Use Enhanced Synthesizer"
                    />
                  </Grid>
                  <Grid item xs={12} md={4}>
                    <FormControlLabel
                      control={
                        <Switch
                          checked={useStreaming}
                          onChange={(e) => setUseStreaming(e.target.checked)}
                        />
                      }
                      label="Use Streaming Processing"
                    />
                  </Grid>
                </Grid>
              </Paper>
            </Grid>

            <Grid item xs={12}>
              <Box sx={{ display: 'flex', justifyContent: 'center' }}>
                <Button
                  type="submit"
                  variant="contained"
                  color="primary"
                  size="large"
                  startIcon={isUploading || isProcessing ? <ButtonLoader size={20} /> : <CloudUploadIcon />}
                  disabled={files.length === 0 || isUploading || isProcessing}
                >
                  {isUploading ? 'Uploading...' : isProcessing ? 'Processing...' : 'Process Documents'}
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
          <Box sx={{ mt: 4 }}>
            <Paper sx={{ p: 2 }}>
              <Typography variant="h6" gutterBottom>
                Processing Status
              </Typography>
              <Box sx={{ mb: 2 }}>
                <Typography variant="body2" gutterBottom>
                  Overall Progress: {Math.round(overallProgress)}%
                </Typography>
                <LinearProgress
                  variant="determinate"
                  value={overallProgress}
                  sx={{ height: 10, borderRadius: 5 }}
                />
              </Box>

              <TableContainer>
                <Table>
                  <TableHead>
                    <TableRow>
                      <TableCell>File</TableCell>
                      <TableCell>Status</TableCell>
                      <TableCell>Progress</TableCell>
                      <TableCell>Started</TableCell>
                      <TableCell>Completed</TableCell>
                      <TableCell>Actions</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {taskIds.map((taskId) => {
                      const task = taskStatuses[taskId] || {};
                      const progress = taskProgress[taskId] || { progress: 0, stage: '' };
                      
                      return (
                        <TableRow key={taskId}>
                          <TableCell>{task.file_name || 'Unknown'}</TableCell>
                          <TableCell>{getStatusChip(task.status)}</TableCell>
                          <TableCell>
                            <Box sx={{ display: 'flex', alignItems: 'center' }}>
                              <Box sx={{ width: '100%', mr: 1 }}>
                                <LinearProgress
                                  variant="determinate"
                                  value={progress.progress}
                                  sx={{ height: 8, borderRadius: 4 }}
                                />
                              </Box>
                              <Box sx={{ minWidth: 35 }}>
                                <Typography variant="body2" color="text.secondary">
                                  {Math.round(progress.progress)}%
                                </Typography>
                              </Box>
                            </Box>
                            <Typography variant="caption" color="text.secondary">
                              {progress.stage || 'Waiting...'}
                            </Typography>
                          </TableCell>
                          <TableCell>
                            {task.created_at ? new Date(task.created_at).toLocaleTimeString() : 'N/A'}
                          </TableCell>
                          <TableCell>
                            {task.completed_at ? new Date(task.completed_at).toLocaleTimeString() : 'N/A'}
                          </TableCell>
                          <TableCell>
                            <Button
                              size="small"
                              variant="outlined"
                              onClick={() => handleViewLogs(taskId)}
                            >
                              View Logs
                            </Button>
                          </TableCell>
                        </TableRow>
                      );
                    })}
                  </TableBody>
                </Table>
              </TableContainer>
            </Paper>
          </Box>
        )}
      </Box>

      {/* Log Dialog */}
      <Dialog
        open={logDialogOpen}
        onClose={handleCloseLogDialog}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>
          Processing Logs
          {selectedTaskLogs && taskStatuses[selectedTaskLogs] && (
            <Typography variant="subtitle2" color="text.secondary">
              {taskStatuses[selectedTaskLogs].file_name} - Task ID: {selectedTaskLogs}
            </Typography>
          )}
        </DialogTitle>
        <DialogContent dividers>
          {selectedTaskLogs && processingLogs[selectedTaskLogs] ? (
            <ProcessingLog logs={processingLogs[selectedTaskLogs]} />
          ) : (
            <Typography>No logs available</Typography>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={handleCloseLogDialog}>Close</Button>
        </DialogActions>
      </Dialog>
    </>
  );
};

export default BatchDocumentProcessor;
