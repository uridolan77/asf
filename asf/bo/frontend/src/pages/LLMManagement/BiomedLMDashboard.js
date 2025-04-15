import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Paper,
  Grid,
  Card,
  CardHeader,
  CardContent,
  CardActions,
  Button,
  Chip,
  Divider,
  TextField,
  CircularProgress,
  Alert,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Slider,
  FormControl,
  InputLabel,
  Select,
  MenuItem
} from '@mui/material';
import {
  ExpandMore as ExpandMoreIcon,
  Settings as SettingsIcon,
  PlayArrow as PlayArrowIcon,
  Add as AddIcon,
  Science as ScienceIcon,
  Tune as TuneIcon
} from '@mui/icons-material';

import { useNotification } from '../../context/NotificationContext';
import apiService from '../../services/api';
import { ContentLoader } from '../../components/UI/LoadingIndicators';

/**
 * BiomedLM Dashboard component
 */
const BiomedLMDashboard = ({ status, onRefresh }) => {
  const { showSuccess, showError } = useNotification();
  
  const [models, setModels] = useState([]);
  const [loading, setLoading] = useState(true);
  const [generatingText, setGeneratingText] = useState(false);
  const [selectedModel, setSelectedModel] = useState(null);
  const [prompt, setPrompt] = useState('');
  const [generationResult, setGenerationResult] = useState(null);
  const [generationParams, setGenerationParams] = useState({
    temperature: 0.2,
    max_tokens: 512,
    top_p: 0.95,
    top_k: 50,
    repetition_penalty: 1.1
  });
  
  // Load models on mount
  useEffect(() => {
    loadModels();
  }, []);
  
  // Load models
  const loadModels = async () => {
    setLoading(true);
    
    try {
      const result = await apiService.llm.getBiomedLMModels();
      
      if (result.success) {
        setModels(result.data);
        if (result.data.length > 0) {
          setSelectedModel(result.data[0].model_id);
        }
      } else {
        showError(`Failed to load models: ${result.error}`);
      }
    } catch (error) {
      console.error('Error loading models:', error);
      showError(`Error loading models: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };
  
  // Handle text generation
  const handleGenerateText = async () => {
    if (!selectedModel || !prompt.trim()) {
      showError('Please select a model and enter a prompt');
      return;
    }
    
    setGeneratingText(true);
    setGenerationResult(null);
    
    try {
      const result = await apiService.llm.generateBiomedLMText(selectedModel, prompt, generationParams);
      
      if (result.success) {
        setGenerationResult(result.data);
        showSuccess('Text generated successfully');
      } else {
        showError(`Failed to generate text: ${result.error}`);
      }
    } catch (error) {
      console.error('Error generating text:', error);
      showError(`Error generating text: ${error.message}`);
    } finally {
      setGeneratingText(false);
    }
  };
  
  // Handle parameter change
  const handleParamChange = (param, value) => {
    setGenerationParams({
      ...generationParams,
      [param]: value
    });
  };
  
  // Render model cards
  const renderModelCards = () => {
    return (
      <Grid container spacing={3}>
        {models.map((model) => (
          <Grid item xs={12} md={6} lg={4} key={model.model_id}>
            <Card 
              sx={{ 
                height: '100%', 
                display: 'flex', 
                flexDirection: 'column',
                transition: 'all 0.3s ease',
                border: selectedModel === model.model_id ? '2px solid' : 'none',
                borderColor: 'primary.main',
                '&:hover': {
                  boxShadow: 6,
                  transform: 'translateY(-4px)'
                }
              }}
              onClick={() => setSelectedModel(model.model_id)}
            >
              <CardHeader
                title={model.display_name}
                subheader={`Base: ${model.base_model}`}
                action={
                  model.adapter_type && (
                    <Chip
                      label={model.adapter_type.toUpperCase()}
                      color="secondary"
                      size="small"
                    />
                  )
                }
              />
              <CardContent sx={{ flexGrow: 1 }}>
                <Typography variant="body2" color="text.secondary" gutterBottom>
                  {model.description || "No description available"}
                </Typography>
                
                <Typography variant="body2" sx={{ mt: 1 }}>
                  Parameters: {(model.parameters / 1000000000).toFixed(1)}B
                </Typography>
                
                {model.fine_tuned_for && model.fine_tuned_for.length > 0 && (
                  <>
                    <Typography variant="subtitle2" gutterBottom sx={{ mt: 2 }}>
                      Fine-tuned for:
                    </Typography>
                    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                      {model.fine_tuned_for.map((task) => (
                        <Chip
                          key={task}
                          label={task}
                          size="small"
                          variant="outlined"
                        />
                      ))}
                    </Box>
                  </>
                )}
              </CardContent>
              <CardActions>
                <Button
                  size="small"
                  startIcon={<PlayArrowIcon />}
                  onClick={(e) => {
                    e.stopPropagation();
                    setSelectedModel(model.model_id);
                    window.scrollTo({
                      top: document.getElementById('generation-section').offsetTop - 20,
                      behavior: 'smooth'
                    });
                  }}
                >
                  Use for Generation
                </Button>
                {model.adapter_type && (
                  <Button
                    size="small"
                    startIcon={<TuneIcon />}
                    onClick={(e) => e.stopPropagation()}
                  >
                    Adapter Settings
                  </Button>
                )}
              </CardActions>
            </Card>
          </Grid>
        ))}
        
        {/* Add new fine-tuning card */}
        <Grid item xs={12} md={6} lg={4}>
          <Card 
            sx={{ 
              height: '100%', 
              display: 'flex', 
              flexDirection: 'column',
              justifyContent: 'center',
              alignItems: 'center',
              p: 3,
              bgcolor: 'grey.100',
              border: '2px dashed',
              borderColor: 'grey.300',
              cursor: 'pointer',
              transition: 'all 0.3s ease',
              '&:hover': {
                borderColor: 'primary.main',
                bgcolor: 'grey.200'
              }
            }}
          >
            <ScienceIcon sx={{ fontSize: 48, color: 'grey.500', mb: 2 }} />
            <Typography variant="h6" color="text.secondary">
              Create New Fine-Tuning
            </Typography>
          </Card>
        </Grid>
      </Grid>
    );
  };
  
  // Render text generation section
  const renderGenerationSection = () => {
    return (
      <Paper sx={{ p: 3, mt: 4 }} id="generation-section">
        <Typography variant="h6" gutterBottom>
          Text Generation
        </Typography>
        
        <Grid container spacing={3}>
          <Grid item xs={12}>
            <FormControl fullWidth variant="outlined">
              <InputLabel id="model-select-label">Model</InputLabel>
              <Select
                labelId="model-select-label"
                value={selectedModel || ''}
                onChange={(e) => setSelectedModel(e.target.value)}
                label="Model"
              >
                {models.map((model) => (
                  <MenuItem key={model.model_id} value={model.model_id}>
                    {model.display_name}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </Grid>
          
          <Grid item xs={12}>
            <TextField
              fullWidth
              label="Prompt"
              multiline
              rows={4}
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="Enter your prompt here..."
              variant="outlined"
            />
          </Grid>
          
          <Grid item xs={12} md={6}>
            <Typography gutterBottom>Temperature: {generationParams.temperature}</Typography>
            <Slider
              value={generationParams.temperature}
              onChange={(_, value) => handleParamChange('temperature', value)}
              min={0}
              max={2}
              step={0.1}
              valueLabelDisplay="auto"
            />
          </Grid>
          
          <Grid item xs={12} md={6}>
            <Typography gutterBottom>Max Tokens: {generationParams.max_tokens}</Typography>
            <Slider
              value={generationParams.max_tokens}
              onChange={(_, value) => handleParamChange('max_tokens', value)}
              min={1}
              max={1024}
              step={1}
              valueLabelDisplay="auto"
            />
          </Grid>
          
          <Grid item xs={12} md={4}>
            <Typography gutterBottom>Top P: {generationParams.top_p}</Typography>
            <Slider
              value={generationParams.top_p}
              onChange={(_, value) => handleParamChange('top_p', value)}
              min={0}
              max={1}
              step={0.01}
              valueLabelDisplay="auto"
            />
          </Grid>
          
          <Grid item xs={12} md={4}>
            <Typography gutterBottom>Top K: {generationParams.top_k}</Typography>
            <Slider
              value={generationParams.top_k}
              onChange={(_, value) => handleParamChange('top_k', value)}
              min={1}
              max={100}
              step={1}
              valueLabelDisplay="auto"
            />
          </Grid>
          
          <Grid item xs={12} md={4}>
            <Typography gutterBottom>Repetition Penalty: {generationParams.repetition_penalty}</Typography>
            <Slider
              value={generationParams.repetition_penalty}
              onChange={(_, value) => handleParamChange('repetition_penalty', value)}
              min={1}
              max={2}
              step={0.01}
              valueLabelDisplay="auto"
            />
          </Grid>
          
          <Grid item xs={12}>
            <Button
              variant="contained"
              color="primary"
              onClick={handleGenerateText}
              disabled={generatingText || !selectedModel || !prompt.trim()}
              startIcon={generatingText ? <CircularProgress size={20} /> : <PlayArrowIcon />}
            >
              {generatingText ? 'Generating...' : 'Generate Text'}
            </Button>
          </Grid>
          
          {generationResult && (
            <Grid item xs={12}>
              <Paper variant="outlined" sx={{ p: 2, bgcolor: 'grey.50' }}>
                <Typography variant="subtitle2" gutterBottom>
                  Generated Text:
                </Typography>
                <Typography variant="body1" sx={{ whiteSpace: 'pre-wrap' }}>
                  {generationResult.generated_text}
                </Typography>
                <Divider sx={{ my: 2 }} />
                <Typography variant="caption" color="text.secondary">
                  Generation time: {generationResult.generation_time_ms.toFixed(2)}ms | 
                  Tokens generated: {generationResult.tokens_generated}
                </Typography>
              </Paper>
            </Grid>
          )}
        </Grid>
      </Paper>
    );
  };
  
  if (loading) {
    return <ContentLoader height={200} message="Loading BiomedLM models..." />;
  }
  
  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        BiomedLM Model Management
      </Typography>
      
      <Typography paragraph>
        Manage BiomedLM models and adapters for medical-specific LLM capabilities.
        BiomedLM is a specialized language model for biomedical text.
      </Typography>
      
      {/* Model cards */}
      {models.length > 0 ? (
        renderModelCards()
      ) : (
        <Alert severity="info" sx={{ mb: 3 }}>
          No BiomedLM models found. Please check your installation.
        </Alert>
      )}
      
      {/* Text generation section */}
      {models.length > 0 && renderGenerationSection()}
      
      {/* Fine-tuning section */}
      <Accordion sx={{ mt: 4 }}>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography variant="subtitle1">Fine-Tuning</Typography>
        </AccordionSummary>
        <AccordionDetails>
          <Typography paragraph>
            Fine-tune BiomedLM models for specific medical tasks.
          </Typography>
          
          <Typography variant="subtitle2" gutterBottom>
            Coming soon...
          </Typography>
        </AccordionDetails>
      </Accordion>
    </Box>
  );
};

export default BiomedLMDashboard;
