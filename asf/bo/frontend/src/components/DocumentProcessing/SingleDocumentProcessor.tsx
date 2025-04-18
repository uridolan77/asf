import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Paper,
  Button,
  Grid,
  Divider,
  Alert,
  CircularProgress
} from '@mui/material';
import {
  PlayArrow as StartIcon,
  Stop as StopIcon,
  Refresh as RefreshIcon
} from '@mui/icons-material';

import { useDocumentProcessing, useDocumentProcessingTask } from '../../hooks/useDocumentProcessing';
import { useNotification } from '../../context/NotificationContext';
import { useFeatureFlags } from '../../context/FeatureFlagContext';

import DocumentUploader from './DocumentUploader';
import ProcessingLogs from './ProcessingLogs';
import ProcessingResults from './ProcessingResults';

/**
 * SingleDocumentProcessor component
 * 
 * A component for processing a single document with various options.
 */
const SingleDocumentProcessor: React.FC = () => {
  // State
  const [file, setFile] = useState<File | null>(null);
  const [taskId, setTaskId] = useState<string | null>(null);
  const [logs, setLogs] = useState<string[]>([]);
  const [useEnhanced, setUseEnhanced] = useState<boolean>(true);
  const [useStreaming, setUseStreaming] = useState<boolean>(false);
  const [useParallel, setUseParallel] = useState<boolean>(false);
  
  // Hooks
  const { showSuccess, showError } = useNotification();
  const { isEnabled } = useFeatureFlags();
  const useMockData = isEnabled('useMockData');
  
  // Document processing hooks
  const {
    uploadDocument,
    processDocument,
    exportDocument
  } = useDocumentProcessing();
  
  // Task tracking hook
  const {
    progress,
    stage,
    status,
    result,
    error,
    loading,
    task,
    intermediateResults
  } = useDocumentProcessingTask(taskId);
  
  // Get mutation functions
  const { mutate: uploadMutate, isPending: isUploading } = uploadDocument();
  const { mutate: processMutate, isPending: isProcessing } = processDocument();
  const { mutate: exportMutate, isPending: isExporting } = exportDocument();
  
  // Add log entry
  const addLog = (message: string) => {
    const timestamp = new Date().toISOString().slice(11, 19);
    setLogs(prev => [...prev, `[${timestamp}] ${message}`]);
  };
  
  // Handle file upload and processing
  const handleProcess = async () => {
    if (!file) {
      showError('Please select a file to process');
      return;
    }
    
    // Clear previous logs and results
    setLogs([]);
    setTaskId(null);
    
    try {
      // Create form data
      const formData = new FormData();
      formData.append('file', file);
      formData.append('use_enhanced_synthesizer', useEnhanced.toString());
      formData.append('use_streaming', useStreaming.toString());
      formData.append('use_parallel', useParallel.toString());
      
      // Log start
      addLog(`Starting processing of ${file.name}`);
      addLog('Uploading file...');
      
      // Upload file
      uploadMutate(formData, {
        onSuccess: (data) => {
          const uploadTaskId = data.task_id;
          setTaskId(uploadTaskId);
          
          addLog('File uploaded successfully');
          addLog(`Task ID: ${uploadTaskId}`);
          addLog('Starting document processing...');
          
          // Process document
          processMutate({
            task_id: uploadTaskId,
            extract_entities: true,
            extract_relations: true,
            detect_sections: true,
            parse_references: true,
            use_biomedlm: true,
            use_enhanced_synthesizer: useEnhanced,
            use_streaming: useStreaming,
            use_parallel: useParallel
          }, {
            onSuccess: () => {
              addLog('Processing started successfully');
            },
            onError: (error) => {
              addLog(`Error starting processing: ${error.message}`);
              showError(`Error starting processing: ${error.message}`);
            }
          });
        },
        onError: (error) => {
          addLog(`Error uploading file: ${error.message}`);
          showError(`Error uploading file: ${error.message}`);
        }
      });
    } catch (error: any) {
      addLog(`Error: ${error.message}`);
      showError(`Error: ${error.message}`);
    }
  };
  
  // Handle export
  const handleExport = (format: string) => {
    if (!taskId) return;
    
    exportMutate({
      task_id: taskId,
      format
    }, {
      onSuccess: (data) => {
        showSuccess(`Document exported successfully as ${format.toUpperCase()}`);
        
        // If URL is returned, open it
        if (data.url) {
          window.open(data.url, '_blank');
        }
      },
      onError: (error) => {
        showError(`Error exporting document: ${error.message}`);
      }
    });
  };
  
  // Handle cancel
  const handleCancel = () => {
    // Not implemented yet
    showError('Cancel operation not implemented yet');
  };
  
  // Update logs based on task status
  useEffect(() => {
    if (!task) return;
    
    if (status === 'processing' && stage) {
      addLog(`Processing stage: ${stage} (${progress.toFixed(0)}%)`);
    } else if (status === 'completed') {
      addLog('Processing completed successfully');
      addLog(`Processed ${task.entity_count || 0} entities and ${task.relation_count || 0} relations`);
      addLog(`Total processing time: ${task.processing_time?.toFixed(2) || 0}s`);
    } else if (status === 'failed') {
      addLog(`Processing failed: ${error || 'Unknown error'}`);
    }
  }, [status, stage, progress, task, error]);
  
  // Update logs based on intermediate results
  useEffect(() => {
    if (!intermediateResults) return;
    
    Object.entries(intermediateResults).forEach(([stage, result]) => {
      if (typeof result === 'object') {
        const counts = Object.entries(result)
          .filter(([key, value]) => typeof value === 'number')
          .map(([key, value]) => `${key}: ${value}`)
          .join(', ');
        
        if (counts) {
          addLog(`Intermediate result for ${stage}: ${counts}`);
        }
      }
    });
  }, [intermediateResults]);
  
  return (
    <Box sx={{ width: '100%' }}>
      {useMockData && (
        <Alert severity="info" sx={{ mb: 2 }}>
          Using mock data. Toggle the "Use Mock Data" feature flag to use real API data.
        </Alert>
      )}
      
      <Paper elevation={3} sx={{ p: 3, mb: 3 }}>
        <Typography variant="h5" gutterBottom>
          Document Processor
        </Typography>
        <Typography variant="body2" color="textSecondary" paragraph>
          Upload a document to extract entities, relations, and other information using our advanced document processing pipeline.
        </Typography>
        
        <Divider sx={{ my: 2 }} />
        
        <DocumentUploader
          file={file}
          setFile={setFile}
          useEnhanced={useEnhanced}
          setUseEnhanced={setUseEnhanced}
          useStreaming={useStreaming}
          setUseStreaming={setUseStreaming}
          useParallel={useParallel}
          setUseParallel={setUseParallel}
        />
        
        <Box sx={{ mt: 2, display: 'flex', gap: 2 }}>
          <Button
            variant="contained"
            color="primary"
            startIcon={isUploading || isProcessing ? <CircularProgress size={20} color="inherit" /> : <StartIcon />}
            onClick={handleProcess}
            disabled={!file || isUploading || isProcessing || status === 'processing'}
          >
            {isUploading || isProcessing ? 'Processing...' : 'Process Document'}
          </Button>
          
          {status === 'processing' && (
            <Button
              variant="outlined"
              color="secondary"
              startIcon={<StopIcon />}
              onClick={handleCancel}
            >
              Cancel
            </Button>
          )}
          
          <Button
            variant="outlined"
            startIcon={<RefreshIcon />}
            onClick={() => {
              setFile(null);
              setTaskId(null);
              setLogs([]);
            }}
            disabled={isUploading || isProcessing}
          >
            Reset
          </Button>
        </Box>
      </Paper>
      
      <Grid container spacing={3} sx={{ height: 'calc(100vh - 400px)', minHeight: '500px' }}>
        <Grid item xs={12} md={4} sx={{ height: '100%' }}>
          <ProcessingLogs
            logs={logs}
            progress={progress}
            stage={stage}
            status={status}
            error={error}
          />
        </Grid>
        
        <Grid item xs={12} md={8} sx={{ height: '100%' }}>
          <ProcessingResults
            result={result}
            onExport={handleExport}
          />
        </Grid>
      </Grid>
    </Box>
  );
};

export default SingleDocumentProcessor;
