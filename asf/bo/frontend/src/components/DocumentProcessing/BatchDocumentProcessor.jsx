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
  LinearProgress
} from '@mui/material';
import {
  Upload as UploadIcon,
  Delete as DeleteIcon,
  CloudUpload as CloudUploadIcon,
  Refresh as RefreshIcon,
  CheckCircle as CheckCircleIcon,
  Error as ErrorIcon,
  HourglassEmpty as HourglassEmptyIcon
} from '@mui/icons-material';
import { useDropzone } from 'react-dropzone';
import apiService from '../../services/api';

/**
 * Component for batch processing multiple documents
 */
const BatchDocumentProcessor = () => {
  const [files, setFiles] = useState([]);
  const [batchSize, setBatchSize] = useState(4);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [taskIds, setTaskIds] = useState([]);
  const [taskStatuses, setTaskStatuses] = useState({});
  const [pollingInterval, setPollingInterval] = useState(null);
  const [overallProgress, setOverallProgress] = useState(0);

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

  // Poll for task statuses when taskIds is set
  useEffect(() => {
    if (taskIds.length > 0) {
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
  }, [taskIds]);

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
      const formData = new FormData();
      files.forEach(file => {
        formData.append('files', file);
      });
      formData.append('batch_size', batchSize);

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
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
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
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
          </TableContainer>
        </Paper>
      )}
    </Box>
  );
};

export default BatchDocumentProcessor;
