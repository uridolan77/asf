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
  Tooltip
} from '@mui/material';
import {
  Upload as UploadIcon,
  ExpandMore as ExpandMoreIcon,
  Science as ScienceIcon,
  LocalHospital as LocalHospitalIcon,
  Biotech as BiotechIcon,
  MedicalInformation as MedicalInformationIcon,
  Speed as SpeedIcon,
  Settings as SettingsIcon
} from '@mui/icons-material';
import { useDropzone } from 'react-dropzone';
import apiService from '../../services/api';
import KnowledgeGraphViewer from './KnowledgeGraphViewer';

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

  // Poll for task status and progress when taskId is set
  useEffect(() => {
    if (taskId) {
      // Use the new polling function for progress updates
      apiService.documentProcessing.pollProgress(taskId, (progressData) => {
        // Update progress state
        setProgress(progressData.progress * 100);
        setCurrentStage(progressData.current_stage);

        // Update task status
        setTaskStatus({
          task_id: progressData.task_id,
          status: progressData.status,
          file_name: file?.name || 'Unknown file',
          created_at: new Date().toISOString(),
          entity_count: progressData.entity_count,
          relation_count: progressData.relation_count,
          current_stage: progressData.current_stage,
          progress: progressData.progress
        });

        // If processing is complete, fetch the full results
        if (progressData.status === 'completed') {
          fetchResults(taskId);
          setLoading(false);
        } else if (progressData.status === 'failed') {
          setError(`Processing failed: ${progressData.error_message || 'Unknown error'}`);
          setLoading(false);
        }
      }, 1000);

      return () => {
        // No need to clear interval as the pollProgress function handles cleanup
      };
    }
  }, [taskId, file]);

  // Fetch processing results
  const fetchResults = async (id) => {
    try {
      const response = await apiService.documentProcessing.getResults(id);
      setProcessingResults(response.data);
      setLoading(false);
    } catch (err) {
      console.error('Error fetching results:', err);
      setError('Error fetching processing results. Please try again.');
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

      const response = await apiService.documentProcessing.processSingle(formData);

      setTaskId(response.data.task_id);
    } catch (err) {
      console.error('Error processing document:', err);
      setError('Error processing document. Please try again.');
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
