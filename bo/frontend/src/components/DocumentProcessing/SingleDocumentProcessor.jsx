import React, { useState, useEffect } from 'react';
import {
  Box,
  Button,
  Card,
  CardContent,
  Typography,
  Link,
  TextField,
  Grid,
  Paper,
  Divider,
  FormControlLabel,
  Switch,
  CircularProgress,
  Alert,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Chip,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  LinearProgress,
  Tooltip,
  Tab,
  Tabs
} from '@mui/material';
import {
  Upload as UploadIcon,
  ExpandMore as ExpandMoreIcon,
  Science as ScienceIcon,
  LocalHospital as LocalHospitalIcon,
  Biotech as BiotechIcon,
  MedicalInformation as MedicalInformationIcon,
  Speed as SpeedIcon,
  Settings as SettingsIcon,
  Info as InfoIcon
} from '@mui/icons-material';
import { useDropzone } from 'react-dropzone';
import apiService from '../../services/api';
import KnowledgeGraphViewer from './KnowledgeGraphViewer';
import ProcessingLog from './ProcessingLog';
import useDocumentProcessing from '../../hooks/useDocumentProcessing';

/**
 * Component for processing a single document
 */
const SingleDocumentProcessor = () => {
  const [file, setFile] = useState(null);
  const [useParallel, setUseParallel] = useState(true);
  const [useEnhanced, setUseEnhanced] = useState(true);
  const [useStreaming, setUseStreaming] = useState(true);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [taskId, setTaskId] = useState(null);
  const [taskStatus, setTaskStatus] = useState(null);
  const [processingResults, setProcessingResults] = useState(null);
  const [pollingInterval, setPollingInterval] = useState(null);
  const [progress, setProgress] = useState(0);
  const [currentStage, setCurrentStage] = useState('');
  const [activeTab, setActiveTab] = useState(0);
  const [processingLogs, setProcessingLogs] = useState([]);

  // Dropzone configuration
  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    accept: {
      'application/pdf': ['.pdf'],
      'text/plain': ['.txt'],
      'text/markdown': ['.md'],
      'application/json': ['.json']
    },
    maxFiles: 1,
    onDrop: acceptedFiles => {
      if (acceptedFiles.length > 0) {
        setFile(acceptedFiles[0]);
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

  // Use the document processing hook for real-time updates
  const {
    progress: wsProgress,
    stage: wsStage,
    status: wsStatus,
    result: wsResult,
    error: wsError,
    loading: wsLoading,
    task: wsTask,
    intermediateResults: wsIntermediateResults
  } = useDocumentProcessing(taskId);

  // Add error to logs when it occurs
  useEffect(() => {
    if (wsError) {
      // Get detailed error information if available
      const errorDetails = wsTask?.error_details || {};

      // Create a detailed error message for the logs
      let detailedErrorInfo = `Task ID: ${taskId}\nStatus: ${wsStatus}\nStage: ${wsStage || 'Unknown'}`;

      // Add detailed error information if available
      if (Object.keys(errorDetails).length > 0) {
        detailedErrorInfo += `\n\nError Type: ${errorDetails.error_type || 'Unknown'}\n`;

        if (errorDetails.file_name) {
          detailedErrorInfo += `File: ${errorDetails.file_name}\n`;
        }

        if (errorDetails.file_size) {
          detailedErrorInfo += `File Size: ${(errorDetails.file_size / 1024).toFixed(2)} KB\n`;
        }

        if (errorDetails.is_pdf !== undefined) {
          detailedErrorInfo += `Is PDF: ${errorDetails.is_pdf}\n`;
        }

        if (errorDetails.settings) {
          detailedErrorInfo += `Settings: ${JSON.stringify(errorDetails.settings, null, 2)}\n`;
        }

        if (errorDetails.stage) {
          detailedErrorInfo += `Failed Stage: ${errorDetails.stage}\n`;
        }

        if (errorDetails.traceback) {
          detailedErrorInfo += `\nTraceback: ${errorDetails.traceback}`;
        }
      }

      // Add installation instructions for common errors
      if (wsError.includes("No PDF parser is available")) {
        detailedErrorInfo += `\n\nInstallation Instructions:\nTo fix this error, install PDF parsing libraries:\n\npip install pymupdf pdfminer.six`;
      } else if (wsError.includes("GLiNER-biomed model not initialized") || wsError.includes("GLiNER package not installed")) {
        detailedErrorInfo += `\n\nInstallation Instructions:\nTo fix this error, install entity extraction libraries:\n\npip install scispacy==0.5.3 spacy==3.5.4 gliner==0.2.17\npip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_core_sci_md-0.5.1.tar.gz`;
      } else if (wsError.includes("sacremoses")) {
        detailedErrorInfo += `\n\nInstallation Instructions:\nTo fix this error, install relation extraction libraries:\n\npip install sacremoses`;
      } else if (wsError.includes("SciFive model") || wsError.includes("Model not initialized")) {
        detailedErrorInfo += `\n\nInstallation Instructions:\nTo fix this error, install summarization libraries:\n\npip install transformers==4.30.2 torch==2.0.1`;
      }

      // Add dependency conflict warning
      detailedErrorInfo += `\n\nNote: There may be dependency conflicts in your Python environment. Consider creating a separate environment for document processing.`;

      setProcessingLogs(prev => [
        ...prev,
        {
          timestamp: new Date().toISOString(),
          level: 'error',
          message: `Processing Error: ${wsError}`,
          details: detailedErrorInfo,
          stage: 'error'
        }
      ]);
    }
  }, [wsError, taskId, wsStatus, wsStage, wsTask]);

  // Update local state when WebSocket updates are received
  useEffect(() => {
    if (taskId && wsTask) {
      // Update progress state
      setProgress(wsProgress);
      setCurrentStage(wsStage);

      // Update task status
      setTaskStatus({
        task_id: taskId,
        status: wsStatus,
        file_name: file?.name || 'Unknown file',
        created_at: wsTask.created_at || new Date().toISOString(),
        entity_count: wsTask.entity_count || 0,
        relation_count: wsTask.relation_count || 0,
        current_stage: wsStage,
        progress: wsProgress / 100
      });

      // Add log entry for stage change
      if (wsStage && wsStage !== currentStage) {
        setProcessingLogs(prev => [
          ...prev,
          {
            timestamp: new Date().toISOString(),
            level: 'info',
            message: `Processing stage: ${wsStage}`,
            details: `File: ${file?.name || 'Unknown'}
Task ID: ${taskId}
Status: ${wsStatus}`,
            stage: wsStage
          }
        ]);
      }

      // Add log entry for progress update (but not too frequently)
      if (wsProgress % 10 === 0 && wsProgress !== 0) {
        setProcessingLogs(prev => [
          ...prev,
          {
            timestamp: new Date().toISOString(),
            level: 'info',
            message: `Progress: ${Math.round(wsProgress)}%`,
            details: `Stage: ${wsStage}
File: ${file?.name || 'Unknown'}
Task ID: ${taskId}`,
            stage: wsStage
          }
        ]);
      }

      // If processing is complete, fetch the full results
      if (wsStatus === 'completed' && !processingResults) {
        setProcessingLogs(prev => [
          ...prev,
          {
            timestamp: new Date().toISOString(),
            level: 'success',
            message: `Processing completed successfully. Entities: ${wsTask.entity_count || 0}, Relations: ${wsTask.relation_count || 0}`,
            details: `File: ${file?.name || 'Unknown'}
Task ID: ${taskId}
Processing time: ${wsTask.processing_time ? `${wsTask.processing_time.toFixed(2)}s` : 'N/A'}
Status: ${wsStatus}`,
            stage: 'completed'
          }
        ]);
        fetchResults(taskId);
        setLoading(false);
      } else if (wsStatus === 'failed') {
        setProcessingLogs(prev => [
          ...prev,
          {
            timestamp: new Date().toISOString(),
            level: 'error',
            message: `Processing failed: ${wsError || 'Unknown error'}`,
            details: `File: ${file?.name || 'Unknown'}
Task ID: ${taskId}
Status: ${wsStatus}
Stage: ${wsStage || 'Unknown'}`,
            stage: 'failed'
          }
        ]);
        setError(wsError || 'Unknown error');
        setLoading(false);
      }
    }
  }, [taskId, file, wsProgress, wsStage, wsStatus, wsResult, wsError, wsTask]);

  // Handle intermediate results
  useEffect(() => {
    if (wsIntermediateResults && Object.keys(wsIntermediateResults).length > 0) {
      // Log intermediate results
      Object.entries(wsIntermediateResults).forEach(([stage, result]) => {
        setProcessingLogs(prev => [
          ...prev,
          {
            timestamp: new Date().toISOString(),
            level: 'info',
            message: `Intermediate result from ${stage}`,
            details: `Result type: ${typeof result}
Task ID: ${taskId}
File: ${file?.name || 'Unknown'}
Preview: ${JSON.stringify(result).substring(0, 150)}...`,
            stage: stage
          }
        ]);
      });
    }
  }, [wsIntermediateResults]);

  // Fetch processing results
  const fetchResults = async (id) => {
    try {
      setProcessingLogs(prev => [
        ...prev,
        {
          timestamp: new Date().toISOString(),
          level: 'info',
          message: `Fetching processing results for task ${id}...`,
          details: `File: ${file?.name || 'Unknown'}
Endpoint: /api/document-processing/results/${id}`,
          stage: 'fetching_results'
        }
      ]);

      const response = await apiService.documentProcessing.getResults(id);
      if (response.success) {
        setProcessingResults(response.data);

        // Log some information about the results
        const results = response.data.results;
        setProcessingLogs(prev => [
          ...prev,
          {
            timestamp: new Date().toISOString(),
            level: 'success',
            message: `Results fetched successfully. Title: "${results.title || 'Untitled'}", Entities: ${results.entities?.length || 0}, Relations: ${results.relations?.length || 0}`,
            details: `File: ${file?.name || 'Unknown'}
Task ID: ${id}
Document sections: ${results.sections?.length || 0}
References: ${results.references?.length || 0}`,
            stage: 'results_fetched'
          }
        ]);

        // Log entity types if available
        if (results.entities && results.entities.length > 0) {
          const entityTypes = {};
          results.entities.forEach(entity => {
            entityTypes[entity.label] = (entityTypes[entity.label] || 0) + 1;
          });

          setProcessingLogs(prev => [
            ...prev,
            {
              timestamp: new Date().toISOString(),
              level: 'info',
              message: `Entity types: ${Object.entries(entityTypes).map(([type, count]) => `${type}=${count}`).join(', ')}`,
              details: `File: ${file?.name || 'Unknown'}
Task ID: ${id}
Total entities: ${results.entities.length}
Entity extraction method: ${results.metadata?.entity_extraction_method || 'Unknown'}`,
              stage: 'results_analysis'
            }
          ]);
        } else {
          setProcessingLogs(prev => [
            ...prev,
            {
              timestamp: new Date().toISOString(),
              level: 'warning',
              message: 'No entities were extracted from the document.',
              details: `File: ${file?.name || 'Unknown'}
Task ID: ${id}
Document type: ${results.metadata?.document_type || 'Unknown'}
Possible reasons: Document may not contain medical entities, text extraction issues, or entity recognition model limitations.`,
              stage: 'results_analysis'
            }
          ]);
        }
      } else {
        setError(response.error || 'Failed to fetch results');
        setProcessingLogs(prev => [
          ...prev,
          {
            timestamp: new Date().toISOString(),
            level: 'error',
            message: `Failed to fetch results: ${response.error || 'Unknown error'}`,
            details: `File: ${file?.name || 'Unknown'}
Task ID: ${id}
Endpoint: /api/document-processing/results/${id}
Status code: ${response.status || 'Unknown'}`,
            stage: 'error'
          }
        ]);
      }
      setLoading(false);
    } catch (err) {
      console.error('Error fetching results:', err);
      setError('Error fetching processing results. Please try again.');
      setProcessingLogs(prev => [
        ...prev,
        {
          timestamp: new Date().toISOString(),
          level: 'error',
          message: `Error fetching results: ${err.message || 'Unknown error'}`,
          details: `File: ${file?.name || 'Unknown'}
Task ID: ${id}
Endpoint: /api/document-processing/results/${id}
Error type: ${err.name || 'Unknown'}
Stack trace: ${err.stack?.split('\n')[0] || 'Not available'}`,
          stage: 'error'
        }
      ]);
      setLoading(false);
    }
  };

  // Handle form submission
  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!file) {
      setError('Please select a file to process');
      return;
    }

    setLoading(true);
    setError(null);
    setTaskId(null);
    setTaskStatus(null);
    setProcessingResults(null);
    setActiveTab(0);
    setProcessingLogs([{
      timestamp: new Date().toISOString(),
      level: 'info',
      message: `Starting document processing for ${file.name}`,
      details: `File type: ${file.type || file.name.split('.').pop() || 'Unknown'}
File size: ${(file.size / 1024).toFixed(2)} KB
Settings: Enhanced=${useEnhanced}, Streaming=${useStreaming}, Parallel=${useParallel}`,
      stage: 'initializing'
    }]);

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
        use_streaming: useStreaming,
        use_parallel: useParallel
      };

      // Create form data with file and settings
      const formData = new FormData();
      formData.append('file', file);
      formData.append('use_parallel', useParallel.toString());
      formData.append('settings_json', JSON.stringify(settings));

      // Log settings
      setProcessingLogs(prev => [
        ...prev,
        {
          timestamp: new Date().toISOString(),
          level: 'info',
          message: `Processing settings configured`,
          details: `Enhanced processor: ${useEnhanced ? 'Enabled' : 'Disabled'}
Streaming: ${useStreaming ? 'Enabled' : 'Disabled'}
Parallel processing: ${useParallel ? 'Enabled' : 'Disabled'}
File: ${file.name}
Size: ${(file.size / 1024).toFixed(2)} KB`,
          stage: 'initializing'
        }
      ]);

      const response = await apiService.documentProcessing.processSingle(formData);

      if (response.success) {
        setTaskId(response.data.task_id);
        setTaskStatus({
          task_id: response.data.task_id,
          status: response.data.status,
          file_name: file.name,
          created_at: response.data.created_at,
          current_stage: 'queued',
          progress: 0
        });

        setProcessingLogs(prev => [
          ...prev,
          {
            timestamp: new Date().toISOString(),
            level: 'info',
            message: `Document queued for processing. Task ID: ${response.data.task_id}`,
            details: `File: ${file.name}
Size: ${(file.size / 1024).toFixed(2)} KB
Created at: ${response.data.created_at}
Status: ${response.data.status}
Endpoint: /api/document-processing/process`,
            stage: 'queued'
          }
        ]);
      } else {
        setError(response.error || 'Failed to start processing');
        setProcessingLogs(prev => [
          ...prev,
          {
            timestamp: new Date().toISOString(),
            level: 'error',
            message: `Failed to start processing: ${response.error || 'Unknown error'}`,
            details: `File: ${file.name}
Size: ${(file.size / 1024).toFixed(2)} KB
Settings: Enhanced=${useEnhanced}, Streaming=${useStreaming}, Parallel=${useParallel}
Status code: ${response.status || 'Unknown'}
Endpoint: /api/document-processing/process`,
            stage: 'error'
          }
        ]);
        setLoading(false);
      }
    } catch (err) {
      console.error('Error processing document:', err);
      setError('Error processing document. Please try again.');
      setProcessingLogs(prev => [
        ...prev,
        {
          timestamp: new Date().toISOString(),
          level: 'error',
          message: `Error processing document: ${err.message || 'Unknown error'}`,
          details: `File: ${file.name}
Size: ${(file.size / 1024).toFixed(2)} KB
Settings: Enhanced=${useEnhanced}, Streaming=${useStreaming}, Parallel=${useParallel}
Error type: ${err.name || 'Unknown'}
Stack trace: ${err.stack?.split('\n')[0] || 'Not available'}
Endpoint: /api/document-processing/process`,
          stage: 'error'
        }
      ]);
      setLoading(false);
    }
  };

  // Render entity list
  const renderEntities = () => {
    if (!processingResults || !processingResults.results || !processingResults.results.entities) {
      return <Typography>No entities found</Typography>;
    }

    const entities = processingResults.results.entities;
    const entityTypes = [...new Set(entities.map(entity => entity.label))];

    return (
      <Box>
        <Box sx={{ mb: 2, display: 'flex', flexWrap: 'wrap', gap: 1 }}>
          {entityTypes.map(type => (
            <Chip
              key={type}
              label={`${type} (${entities.filter(e => e.label === type).length})`}
              color="primary"
              variant="outlined"
            />
          ))}
        </Box>

        <List dense>
          {entities.slice(0, 10).map((entity, index) => (
            <ListItem key={index}>
              <ListItemIcon>
                {entity.label === 'DISEASE' ? <LocalHospitalIcon color="error" /> :
                 entity.label === 'DRUG' ? <MedicalInformationIcon color="primary" /> :
                 entity.label === 'GENE' ? <BiotechIcon color="success" /> :
                 <ScienceIcon color="secondary" />}
              </ListItemIcon>
              <ListItemText
                primary={entity.text}
                secondary={entity.label}
              />
            </ListItem>
          ))}
          {entities.length > 10 && (
            <ListItem>
              <ListItemText
                primary={`... and ${entities.length - 10} more entities`}
                secondary="Showing first 10 entities only"
              />
            </ListItem>
          )}
        </List>
      </Box>
    );
  };

  // Render relations list
  const renderRelations = () => {
    if (!processingResults || !processingResults.results || !processingResults.results.relations) {
      return <Typography>No relations found</Typography>;
    }

    const relations = processingResults.results.relations;

    return (
      <List dense>
        {relations.slice(0, 10).map((relation, index) => (
          <ListItem key={index}>
            <ListItemText
              primary={`${relation.head} → ${relation.relation} → ${relation.tail}`}
              secondary={relation.confidence ? `Confidence: ${(relation.confidence * 100).toFixed(1)}%` : ''}
            />
          </ListItem>
        ))}
        {relations.length > 10 && (
          <ListItem>
            <ListItemText
              primary={`... and ${relations.length - 10} more relations`}
              secondary="Showing first 10 relations only"
            />
          </ListItem>
        )}
      </List>
    );
  };

  // Render summary
  const renderSummary = () => {
    if (!processingResults || !processingResults.results || !processingResults.results.summary) {
      return <Typography>No summary available</Typography>;
    }

    const summary = processingResults.results.summary;

    return (
      <Box>
        {summary.abstract && (
          <Box sx={{ mb: 2 }}>
            <Typography variant="subtitle1" gutterBottom>Abstract</Typography>
            <Typography variant="body2">{summary.abstract}</Typography>
          </Box>
        )}

        {summary.key_findings && (
          <Box sx={{ mb: 2 }}>
            <Typography variant="subtitle1" gutterBottom>Key Findings</Typography>
            <Typography variant="body2">{summary.key_findings}</Typography>
          </Box>
        )}

        {summary.conclusion && (
          <Box>
            <Typography variant="subtitle1" gutterBottom>Conclusion</Typography>
            <Typography variant="body2">{summary.conclusion}</Typography>
          </Box>
        )}
      </Box>
    );
  };

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Typography variant="h5">Document Processing</Typography>
        <Button
          variant="outlined"
          color="info"
          startIcon={<InfoIcon />}
          onClick={() => window.open('/document_processing_dependencies.html', '_blank')}
        >
          Dependencies
        </Button>
      </Box>

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
                {isDragActive ? 'Drop the file here' : 'Drag & drop a file here, or click to select'}
              </Typography>
              <Typography variant="body2" color="textSecondary">
                Supported formats: PDF, TXT, MD, JSON
              </Typography>
              {file && (
                <Box sx={{ mt: 2 }}>
                  <Chip
                    label={file.name}
                    onDelete={() => setFile(null)}
                    color="primary"
                  />
                </Box>
              )}
            </Paper>
          </Grid>

          <Grid item xs={12}>
            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 2, mb: 2 }}>
              <FormControlLabel
                control={
                  <Switch
                    checked={useEnhanced}
                    onChange={(e) => setUseEnhanced(e.target.checked)}
                    color="primary"
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
                    disabled={!useEnhanced}
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

              <FormControlLabel
                control={
                  <Switch
                    checked={useParallel}
                    onChange={(e) => setUseParallel(e.target.checked)}
                    color="primary"
                  />
                }
                label="Use parallel processing"
              />
            </Box>

            <Box sx={{ display: 'flex', justifyContent: 'flex-end' }}>
              <Button
                type="submit"
                variant="contained"
                color="primary"
                disabled={!file || loading}
                startIcon={loading ? <CircularProgress size={20} /> : <ScienceIcon />}
              >
                {loading ? 'Processing...' : 'Process Document'}
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

      {taskStatus && (
        <Paper sx={{ mt: 3, p: 2 }}>
          <Typography variant="subtitle1" gutterBottom>
            Processing Status
          </Typography>
          <Grid container spacing={2}>
            <Grid item xs={12} sm={6} md={3}>
              <Typography variant="body2" color="textSecondary">
                Status:
              </Typography>
              <Typography variant="body1">
                {taskStatus.status === 'queued' && 'Queued'}
                {taskStatus.status === 'processing' && 'Processing...'}
                {taskStatus.status === 'completed' && 'Completed'}
                {taskStatus.status === 'failed' && 'Failed'}
              </Typography>
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <Typography variant="body2" color="textSecondary">
                File:
              </Typography>
              <Typography variant="body1" noWrap>
                {taskStatus.file_name}
              </Typography>
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <Typography variant="body2" color="textSecondary">
                Created:
              </Typography>
              <Typography variant="body1">
                {new Date(taskStatus.created_at).toLocaleString()}
              </Typography>
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <Typography variant="body2" color="textSecondary">
                Task ID:
              </Typography>
              <Typography variant="body1" noWrap>
                {taskStatus.task_id}
              </Typography>
            </Grid>
          </Grid>

          {taskStatus.status === 'processing' && (
            <Box sx={{ mt: 2 }}>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                <Typography variant="body2" color="textSecondary">
                  {currentStage ? currentStage.charAt(0).toUpperCase() + currentStage.slice(1) : 'Processing'}
                </Typography>
                <Typography variant="body2" color="textSecondary">
                  {progress.toFixed(0)}%
                </Typography>
              </Box>
              <LinearProgress variant="determinate" value={progress} sx={{ height: 8, borderRadius: 4 }} />
            </Box>
          )}

          {/* Show processing logs */}
          <Box sx={{ mt: 2 }}>
            <Typography variant="subtitle2" gutterBottom>
              Processing Log
            </Typography>
            <Paper variant="outlined" sx={{ height: 350, overflow: 'auto', p: 1 }}>
              <ProcessingLog
                logs={processingLogs}
                isLoading={loading && processingLogs.length === 0}
                error={null}
                title=""
              />
            </Paper>
          </Box>
        </Paper>
      )}

      {processingResults && (
        <Box sx={{ mt: 3 }}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Processing Results
            </Typography>

            <Grid container spacing={3} sx={{ mb: 3 }}>
              <Grid item xs={12} sm={6} md={3}>
                <Card>
                  <CardContent>
                    <Typography variant="subtitle2" color="textSecondary">
                      Processing Time
                    </Typography>
                    <Typography variant="h5">
                      {processingResults.processing_time ? `${processingResults.processing_time.toFixed(2)}s` : 'N/A'}
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
              <Grid item xs={12} sm={6} md={3}>
                <Card>
                  <CardContent>
                    <Typography variant="subtitle2" color="textSecondary">
                      Entities Extracted
                    </Typography>
                    <Typography variant="h5">
                      {processingResults.entity_count || 0}
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
              <Grid item xs={12} sm={6} md={3}>
                <Card>
                  <CardContent>
                    <Typography variant="subtitle2" color="textSecondary">
                      Relations Extracted
                    </Typography>
                    <Typography variant="h5">
                      {processingResults.relation_count || 0}
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
              <Grid item xs={12} sm={6} md={3}>
                <Card>
                  <CardContent>
                    <Typography variant="subtitle2" color="textSecondary">
                      Document Title
                    </Typography>
                    <Typography variant="body1" noWrap>
                      {processingResults.results?.title || 'Untitled Document'}
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
            </Grid>

            <Divider sx={{ mb: 3 }} />

            <Accordion defaultExpanded>
              <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                <Typography variant="subtitle1">Document Summary</Typography>
              </AccordionSummary>
              <AccordionDetails>
                {renderSummary()}
              </AccordionDetails>
            </Accordion>

            <Accordion>
              <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                <Typography variant="subtitle1">Entities ({processingResults.entity_count || 0})</Typography>
              </AccordionSummary>
              <AccordionDetails>
                {renderEntities()}
              </AccordionDetails>
            </Accordion>

            <Accordion>
              <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                <Typography variant="subtitle1">Relations ({processingResults.relation_count || 0})</Typography>
              </AccordionSummary>
              <AccordionDetails>
                {renderRelations()}
              </AccordionDetails>
            </Accordion>

            <Accordion>
              <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                <Typography variant="subtitle1">Knowledge Graph</Typography>
              </AccordionSummary>
              <AccordionDetails>
                <Box sx={{ height: 600 }}>
                  <KnowledgeGraphViewer
                    graphData={{
                      nodes: processingResults.results?.entities?.map(entity => ({
                        id: entity.id || entity.text,
                        type: entity.label,
                        confidence: entity.confidence || 1.0
                      })) || [],
                      links: processingResults.results?.relations?.map(relation => ({
                        source: relation.head_id || relation.head,
                        target: relation.tail_id || relation.tail,
                        type: relation.relation,
                        confidence: relation.confidence || 0.8
                      })) || []
                    }}
                    isLoading={false}
                    error={null}
                  />
                </Box>
              </AccordionDetails>
            </Accordion>

            <Accordion>
              <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                <Typography variant="subtitle1">Processing Log</Typography>
              </AccordionSummary>
              <AccordionDetails>
                <Box sx={{ height: 600 }}>
                  <ProcessingLog
                    logs={processingLogs}
                    isLoading={loading && processingLogs.length === 0}
                    error={error}
                    title={`Processing Log: ${file?.name || 'Document'}`}
                  />
                </Box>
              </AccordionDetails>
            </Accordion>

            <Accordion>
              <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                <Typography variant="subtitle1">Raw JSON</Typography>
              </AccordionSummary>
              <AccordionDetails>
                <Box sx={{ maxHeight: '400px', overflow: 'auto' }}>
                  <pre>{JSON.stringify(processingResults.results, null, 2)}</pre>
                </Box>
              </AccordionDetails>
            </Accordion>
          </Paper>
        </Box>
      )}
    </Box>
  );
};

export default SingleDocumentProcessor;
