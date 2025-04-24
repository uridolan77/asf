import React, { useState, useEffect } from 'react';
import {
  Box,
  Paper,
  Typography,
  Button,
  Grid,
  Card,
  CardHeader,
  CardContent,
  CardActions,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Alert,
  Chip,
  Divider,
  CircularProgress,
  IconButton,
  Tooltip,
  Accordion,
  AccordionSummary,
  AccordionDetails
} from '@mui/material';
import {
  Refresh as RefreshIcon,
  ExpandMore as ExpandMoreIcon,
  ContentCopy as ContentCopyIcon,
  CheckCircle as CheckCircleIcon,
  Description as DescriptionIcon,
  Settings as SettingsIcon,
  Biotech as BiotechIcon
} from '@mui/icons-material';

import apiService from '../../services/api';
import { useNotification } from '../../context/NotificationContext.jsx';
import ModelConfigDialog from '../../components/LLM/Models/ModelConfigDialog';

/**
 * Dashboard for BiomedLM models and operations
 */
const BiomedLMDashboard = ({ status, onRefresh }) => {
  const [loading, setLoading] = useState(false);
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState('');
  const [prompt, setPrompt] = useState('');
  const [responseText, setResponseText] = useState('');
  const [generating, setGenerating] = useState(false);
  const [configOpen, setConfigOpen] = useState(false);
  const [currentConfig, setCurrentConfig] = useState(null);
  const [copySuccess, setCopySuccess] = useState(false);

  const { showSuccess, showError } = useNotification();

  // Load models on mount
  useEffect(() => {
    loadModels();
  }, []);

  // Load BiomedLM models
  const loadModels = async () => {
    setLoading(true);

    try {
      const result = await apiService.llm.getBiomedLMModels();

      if (result.success) {
        setModels(result.data);
        if (result.data.length > 0 && !selectedModel) {
          setSelectedModel(result.data[0].id);
        }
      } else {
        showError(`Failed to load BiomedLM models: ${result.error}`);
      }
    } catch (error) {
      console.error('Error loading BiomedLM models:', error);
      showError(`Error loading BiomedLM models: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  // Get BiomedLM configuration
  const getModelConfig = async () => {
    try {
      const result = await apiService.llm.getBiomedLMConfig();

      if (result.success) {
        setCurrentConfig(result.data);
        setConfigOpen(true);
      } else {
        showError(`Failed to load BiomedLM configuration: ${result.error}`);
      }
    } catch (error) {
      console.error('Error loading BiomedLM configuration:', error);
      showError(`Error loading BiomedLM configuration: ${error.message}`);
    }
  };

  // Update BiomedLM configuration
  const updateModelConfig = async (config) => {
    try {
      const result = await apiService.llm.updateBiomedLMConfig(config);

      if (result.success) {
        setCurrentConfig(result.data);
        showSuccess('BiomedLM configuration updated successfully');
        setConfigOpen(false);
        if (onRefresh) onRefresh();
      } else {
        showError(`Failed to update BiomedLM configuration: ${result.error}`);
      }
    } catch (error) {
      console.error('Error updating BiomedLM configuration:', error);
      showError(`Error updating BiomedLM configuration: ${error.message}`);
    }
  };

  // Generate text with BiomedLM
  const generateText = async () => {
    if (!selectedModel || !prompt.trim()) {
      showError('Please select a model and enter a prompt');
      return;
    }

    setGenerating(true);
    setResponseText('');

    try {
      const result = await apiService.llm.generateBiomedLMText(
        selectedModel,
        prompt,
        {
          max_tokens: 500,
          temperature: 0.7
        }
      );

      if (result.success) {
        setResponseText(result.data.text);
        showSuccess('Text generated successfully');
      } else {
        showError(`Failed to generate text: ${result.error}`);
      }
    } catch (error) {
      console.error('Error generating text:', error);
      showError(`Error generating text: ${error.message}`);
    } finally {
      setGenerating(false);
    }
  };

  // Handle model change
  const handleModelChange = (event) => {
    setSelectedModel(event.target.value);
  };

  // Handle prompt change
  const handlePromptChange = (event) => {
    setPrompt(event.target.value);
  };

  // Copy response text to clipboard
  const copyToClipboard = () => {
    navigator.clipboard.writeText(responseText).then(
      () => {
        setCopySuccess(true);
        setTimeout(() => setCopySuccess(false), 2000);
      },
      (err) => {
        console.error('Failed to copy text: ', err);
        showError('Failed to copy text to clipboard');
      }
    );
  };

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h5">
          <BiotechIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
          BiomedLM Dashboard
        </Typography>
        <Box>
          <Button
            variant="outlined"
            startIcon={<SettingsIcon />}
            onClick={getModelConfig}
            sx={{ mr: 1 }}
          >
            Configuration
          </Button>
          <Button
            variant="outlined"
            startIcon={loading ? <CircularProgress size={20} /> : <RefreshIcon />}
            onClick={loadModels}
            disabled={loading}
          >
            Refresh Models
          </Button>
        </Box>
      </Box>

      {status?.status !== 'available' && (
        <Alert severity="error" sx={{ mb: 3 }}>
          BiomedLM service is currently unavailable. Please check the server status.
        </Alert>
      )}

      <Grid container spacing={3}>
        {/* Models Overview */}
        <Grid item xs={12} md={4}>
          <Paper sx={{ p: 2, height: '100%' }}>
            <Typography variant="h6" gutterBottom>
              Available Models
            </Typography>

            {loading ? (
              <Box sx={{ display: 'flex', justifyContent: 'center', mt: 3 }}>
                <CircularProgress />
              </Box>
            ) : models.length === 0 ? (
              <Alert severity="info">No BiomedLM models available</Alert>
            ) : (
              <Box>
                {models.map((model) => (
                  <Card
                    key={model.id}
                    variant="outlined"
                    sx={{
                      mb: 2,
                      border: selectedModel === model.id ? '2px solid' : '1px solid',
                      borderColor: selectedModel === model.id ? 'primary.main' : 'divider',
                    }}
                  >
                    <CardHeader
                      title={model.name}
                      subheader={`${model.size} parameters`}
                      action={
                        <Chip
                          label={model.status}
                          color={model.status === 'active' ? 'success' : 'default'}
                          size="small"
                        />
                      }
                    />
                    <CardContent sx={{ pt: 0 }}>
                      <Typography variant="body2" color="text.secondary">
                        {model.description}
                      </Typography>

                      <Box sx={{ mt: 1 }}>
                        {model.tags && model.tags.map(tag => (
                          <Chip
                            key={tag}
                            label={tag}
                            size="small"
                            sx={{ mr: 0.5, mt: 0.5 }}
                          />
                        ))}
                      </Box>
                    </CardContent>
                    <CardActions>
                      <Button
                        size="small"
                        variant={selectedModel === model.id ? 'contained' : 'outlined'}
                        onClick={() => setSelectedModel(model.id)}
                      >
                        {selectedModel === model.id ? 'Selected' : 'Select'}
                      </Button>
                      {model.documentation_url && (
                        <Button
                          size="small"
                          startIcon={<DescriptionIcon />}
                          onClick={() => window.open(model.documentation_url, '_blank')}
                        >
                          Docs
                        </Button>
                      )}
                    </CardActions>
                  </Card>
                ))}
              </Box>
            )}
          </Paper>
        </Grid>

        {/* Text Generation */}
        <Grid item xs={12} md={8}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Medical Text Generation
            </Typography>

            <Box sx={{ mb: 3 }}>
              <FormControl fullWidth sx={{ mb: 2 }}>
                <InputLabel id="model-select-label">Select BiomedLM Model</InputLabel>
                <Select
                  labelId="model-select-label"
                  id="model-select"
                  value={selectedModel}
                  label="Select BiomedLM Model"
                  onChange={handleModelChange}
                  disabled={models.length === 0 || generating}
                >
                  {models.map((model) => (
                    <MenuItem key={model.id} value={model.id}>
                      {model.name} {model.status !== 'active' && `(${model.status})`}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>

              <TextField
                label="Enter your medical prompt"
                multiline
                rows={4}
                fullWidth
                variant="outlined"
                value={prompt}
                onChange={handlePromptChange}
                placeholder="Enter a medical question or prompt for BiomedLM..."
                disabled={generating}
                sx={{ mb: 2 }}
              />

              <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                <Button
                  variant="contained"
                  onClick={generateText}
                  disabled={!selectedModel || !prompt.trim() || generating}
                  startIcon={generating ? <CircularProgress size={20} /> : null}
                >
                  {generating ? 'Generating...' : 'Generate Medical Text'}
                </Button>

                <Button
                  variant="outlined"
                  onClick={() => {
                    setPrompt('');
                    setResponseText('');
                  }}
                  disabled={generating}
                >
                  Clear
                </Button>
              </Box>
            </Box>

            <Divider sx={{ my: 2 }} />

            <Box>
              <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
                Response
                {responseText && (
                  <Tooltip title="Copy to clipboard">
                    <IconButton onClick={copyToClipboard} sx={{ ml: 1 }}>
                      {copySuccess ? <CheckCircleIcon color="success" /> : <ContentCopyIcon />}
                    </IconButton>
                  </Tooltip>
                )}
              </Typography>

              {generating ? (
                <Box sx={{ display: 'flex', justifyContent: 'center', my: 3 }}>
                  <CircularProgress />
                </Box>
              ) : responseText ? (
                <Paper
                  variant="outlined"
                  sx={{
                    p: 2,
                    backgroundColor: 'grey.50',
                    maxHeight: '300px',
                    overflowY: 'auto',
                    whiteSpace: 'pre-wrap'
                  }}
                >
                  {responseText}
                </Paper>
              ) : (
                <Alert severity="info">
                  Response will appear here after generation
                </Alert>
              )}
            </Box>

            <Accordion sx={{ mt: 3 }}>
              <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                <Typography>Medical Applications</Typography>
              </AccordionSummary>
              <AccordionDetails>
                <Typography variant="body2" paragraph>
                  BiomedLM is specialized for medical text processing. Here are some example prompts:
                </Typography>

                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                  <Button
                    variant="outlined"
                    size="small"
                    onClick={() => setPrompt("Explain the pathophysiology of type 2 diabetes mellitus in simple terms.")}
                  >
                    Explain diabetes pathophysiology
                  </Button>

                  <Button
                    variant="outlined"
                    size="small"
                    onClick={() => setPrompt("What are the key differences between ACE inhibitors and ARBs for treating hypertension?")}
                  >
                    Compare antihypertensives
                  </Button>

                  <Button
                    variant="outlined"
                    size="small"
                    onClick={() => setPrompt("Summarize the latest evidence for managing acute ischemic stroke in the emergency setting.")}
                  >
                    Stroke management
                  </Button>

                  <Button
                    variant="outlined"
                    size="small"
                    onClick={() => setPrompt("Statement 1: Aspirin reduces risk of myocardial infarction.\nStatement 2: Aspirin increases risk of gastrointestinal bleeding.\nAre these statements contradictory? Explain.")}
                  >
                    Contradiction analysis
                  </Button>
                </Box>
              </AccordionDetails>
            </Accordion>
          </Paper>
        </Grid>
      </Grid>

      {/* Configuration Dialog */}
      {configOpen && currentConfig && (
        <ModelConfigDialog
          open={configOpen}
          onClose={() => setConfigOpen(false)}
          config={currentConfig}
          onSave={updateModelConfig}
          title="BiomedLM Configuration"
        />
      )}
    </Box>
  );
};

export default BiomedLMDashboard;
