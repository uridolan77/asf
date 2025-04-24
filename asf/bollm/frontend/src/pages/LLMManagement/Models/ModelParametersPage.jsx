// filepath: c:\code\asf\asf\bollm\frontend\src\pages\LLMManagement\Models\ModelParametersPage.jsx
import React, { useState, useEffect } from 'react';
import { 
  Box, 
  Typography, 
  Paper,
  TextField,
  Button,
  Divider,
  Grid,
  Alert,
  IconButton,
  Tooltip,
  Slider,
  FormControlLabel,
  Switch,
  Card,
  CardContent,
  Accordion,
  AccordionSummary,
  AccordionDetails
} from '@mui/material';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';
import InfoIcon from '@mui/icons-material/Info';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import { useNavigate, useParams } from 'react-router-dom';
import apiService from '../../../services/api';

// Import the PageLayout component for consistent navigation
import PageLayout from '../../../components/Layout/PageLayout';

/**
 * Model Parameters Page
 * For configuring detailed parameters of an LLM model
 */
const ModelParametersPage = () => {
  const navigate = useNavigate();
  const { modelId, providerId } = useParams();
  
  // State for model data
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [model, setModel] = useState(null);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  
  // Form state for parameters
  const [parameters, setParameters] = useState({
    temperature: 0.7,
    top_p: 1.0,
    presence_penalty: 0.0,
    frequency_penalty: 0.0,
    max_tokens: 1000,
    stop_sequences: [],
    repetition_penalty: 1.0,
    use_beam_search: false,
    top_k: 50,
  });
  
  // Default presets
  const presets = [
    { name: 'Default', temperature: 0.7, top_p: 1.0, presence_penalty: 0.0, frequency_penalty: 0.0 },
    { name: 'Creative', temperature: 1.0, top_p: 0.9, presence_penalty: 0.6, frequency_penalty: 0.1 },
    { name: 'Precise', temperature: 0.3, top_p: 0.85, presence_penalty: 0.0, frequency_penalty: 0.2 },
    { name: 'Balanced', temperature: 0.5, top_p: 0.95, presence_penalty: 0.1, frequency_penalty: 0.1 },
  ];
  
  // Load model data
  useEffect(() => {
    const loadModel = async () => {
      setLoading(true);
      try {
        if (!modelId || !providerId) {
          setError('Model ID and Provider ID are required');
          return;
        }
        
        const response = await apiService.llm.getModelById(modelId, providerId);
        if (response.success) {
          setModel(response.data);
          // Initialize parameters with model defaults if available
          setParameters({
            temperature: response.data.default_temperature || 0.7,
            top_p: response.data.top_p || 1.0,
            presence_penalty: response.data.presence_penalty || 0.0,
            frequency_penalty: response.data.frequency_penalty || 0.0,
            max_tokens: response.data.max_output_tokens || 1000,
            stop_sequences: response.data.stop_sequences || [],
            repetition_penalty: response.data.repetition_penalty || 1.0,
            use_beam_search: response.data.use_beam_search || false,
            top_k: response.data.top_k || 50,
          });
        } else {
          setError(`Failed to load model data: ${response.error}`);
        }
      } catch (err) {
        setError(`Error loading model: ${err.message}`);
      } finally {
        setLoading(false);
      }
    };
    
    loadModel();
  }, [modelId, providerId]);
  
  // Handle parameter change
  const handleParameterChange = (param, value) => {
    setParameters(prev => ({
      ...prev,
      [param]: value
    }));
  };
  
  // Handle slider change
  const handleSliderChange = (param) => (_, value) => {
    handleParameterChange(param, value);
  };
  
  // Handle slider input change
  const handleSliderInputChange = (param) => (e) => {
    const value = e.target.value === '' ? '' : Number(e.target.value);
    if (value !== '' && !isNaN(value)) {
      handleParameterChange(param, value);
    }
  };
  
  // Apply preset
  const applyPreset = (preset) => {
    setParameters(prev => ({
      ...prev,
      temperature: preset.temperature,
      top_p: preset.top_p,
      presence_penalty: preset.presence_penalty,
      frequency_penalty: preset.frequency_penalty,
    }));
    setSuccess(`Applied "${preset.name}" preset`);
    setTimeout(() => setSuccess(''), 3000);
  };
  
  // Handle save
  const handleSave = async () => {
    setSaving(true);
    setError('');
    
    try {
      // Prepare update data with just the parameters
      const updateData = {
        default_temperature: parameters.temperature,
        top_p: parameters.top_p,
        presence_penalty: parameters.presence_penalty,
        frequency_penalty: parameters.frequency_penalty,
        max_output_tokens: parameters.max_tokens,
        stop_sequences: parameters.stop_sequences,
        repetition_penalty: parameters.repetition_penalty,
        use_beam_search: parameters.use_beam_search,
        top_k: parameters.top_k,
      };
      
      const result = await apiService.llm.updateModel(
        modelId,
        providerId,
        updateData
      );
      
      if (result.success) {
        setSuccess('Model parameters saved successfully');
        setTimeout(() => setSuccess(''), 3000);
      } else {
        setError(`Failed to save parameters: ${result.error}`);
      }
    } catch (err) {
      setError(`Error saving parameters: ${err.message}`);
    } finally {
      setSaving(false);
    }
  };
  
  // Test the model with current parameters
  const handleTest = async () => {
    navigate(`/llm/models/test/${providerId}/${modelId}`, { 
      state: { parameters } 
    });
  };
  
  return (
    <PageLayout>
      <Box sx={{ pt: 3, pb: 5 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 4 }}>
          <IconButton 
            onClick={() => navigate('/llm/models')}
            sx={{ mr: 2 }}
          >
            <ArrowBackIcon />
          </IconButton>
          <Typography variant="h4" component="h1">
            Model Parameters
          </Typography>
        </Box>
        
        {success && (
          <Alert severity="success" sx={{ mb: 3 }}>
            {success}
          </Alert>
        )}
        
        {error && (
          <Alert severity="error" sx={{ mb: 3 }}>
            {error}
          </Alert>
        )}
        
        {loading ? (
          <Paper sx={{ p: 3, textAlign: 'center' }}>
            <Typography>Loading model data...</Typography>
          </Paper>
        ) : model ? (
          <>
            {/* Model Info Header */}
            <Paper sx={{ p: 3, mb: 3 }}>
              <Grid container spacing={2}>
                <Grid item xs={12} sm={8}>
                  <Typography variant="h5">{model.display_name || model.model_id}</Typography>
                  <Typography variant="body2" color="text.secondary">
                    Provider: {model.provider_id} | Type: {model.model_type || 'chat'}
                  </Typography>
                </Grid>
                <Grid item xs={12} sm={4} sx={{ display: 'flex', justifyContent: 'flex-end', alignItems: 'center' }}>
                  <Button 
                    variant="contained"
                    color="primary"
                    onClick={handleTest}
                    sx={{ mr: 2 }}
                  >
                    Test Model
                  </Button>
                </Grid>
              </Grid>
            </Paper>
            
            {/* Parameter Presets */}
            <Paper sx={{ p: 3, mb: 3 }}>
              <Typography variant="h6" gutterBottom>
                Parameter Presets
              </Typography>
              <Grid container spacing={2}>
                {presets.map((preset) => (
                  <Grid item xs={12} sm={6} md={3} key={preset.name}>
                    <Card 
                      sx={{ 
                        cursor: 'pointer',
                        '&:hover': { boxShadow: 3 }
                      }}
                      onClick={() => applyPreset(preset)}
                    >
                      <CardContent>
                        <Typography variant="h6">{preset.name}</Typography>
                        <Typography variant="body2" color="text.secondary">
                          Temp: {preset.temperature} | Top-p: {preset.top_p}
                        </Typography>
                      </CardContent>
                    </Card>
                  </Grid>
                ))}
              </Grid>
            </Paper>
            
            {/* Parameter Controls */}
            <Paper sx={{ p: 3, mb: 3 }}>
              <Typography variant="h6" gutterBottom>
                Generation Parameters
              </Typography>
              <Divider sx={{ mb: 3 }} />
              
              <Grid container spacing={3}>
                {/* Temperature */}
                <Grid item xs={12} md={6}>
                  <Typography gutterBottom>
                    Temperature
                    <Tooltip title="Controls randomness: Higher values produce more diverse outputs, lower values are more focused and deterministic">
                      <IconButton size="small">
                        <InfoIcon fontSize="small" />
                      </IconButton>
                    </Tooltip>
                  </Typography>
                  <Grid container spacing={2} alignItems="center">
                    <Grid item xs>
                      <Slider
                        value={parameters.temperature}
                        onChange={handleSliderChange('temperature')}
                        aria-labelledby="temperature-slider"
                        step={0.05}
                        min={0}
                        max={2}
                      />
                    </Grid>
                    <Grid item>
                      <TextField
                        value={parameters.temperature}
                        onChange={handleSliderInputChange('temperature')}
                        inputProps={{
                          step: 0.05,
                          min: 0,
                          max: 2,
                          type: 'number',
                        }}
                        sx={{ width: '80px' }}
                        size="small"
                      />
                    </Grid>
                  </Grid>
                </Grid>
                
                {/* Top P */}
                <Grid item xs={12} md={6}>
                  <Typography gutterBottom>
                    Top P (Nucleus Sampling)
                    <Tooltip title="Controls diversity via nucleus sampling: Only consider tokens comprising the top P probability mass">
                      <IconButton size="small">
                        <InfoIcon fontSize="small" />
                      </IconButton>
                    </Tooltip>
                  </Typography>
                  <Grid container spacing={2} alignItems="center">
                    <Grid item xs>
                      <Slider
                        value={parameters.top_p}
                        onChange={handleSliderChange('top_p')}
                        aria-labelledby="top-p-slider"
                        step={0.01}
                        min={0}
                        max={1}
                      />
                    </Grid>
                    <Grid item>
                      <TextField
                        value={parameters.top_p}
                        onChange={handleSliderInputChange('top_p')}
                        inputProps={{
                          step: 0.01,
                          min: 0,
                          max: 1,
                          type: 'number',
                        }}
                        sx={{ width: '80px' }}
                        size="small"
                      />
                    </Grid>
                  </Grid>
                </Grid>
                
                {/* Presence Penalty */}
                <Grid item xs={12} md={6}>
                  <Typography gutterBottom>
                    Presence Penalty
                    <Tooltip title="Positive values penalize new tokens based on whether they appear in the text so far">
                      <IconButton size="small">
                        <InfoIcon fontSize="small" />
                      </IconButton>
                    </Tooltip>
                  </Typography>
                  <Grid container spacing={2} alignItems="center">
                    <Grid item xs>
                      <Slider
                        value={parameters.presence_penalty}
                        onChange={handleSliderChange('presence_penalty')}
                        aria-labelledby="presence-penalty-slider"
                        step={0.1}
                        min={-2}
                        max={2}
                      />
                    </Grid>
                    <Grid item>
                      <TextField
                        value={parameters.presence_penalty}
                        onChange={handleSliderInputChange('presence_penalty')}
                        inputProps={{
                          step: 0.1,
                          min: -2,
                          max: 2,
                          type: 'number',
                        }}
                        sx={{ width: '80px' }}
                        size="small"
                      />
                    </Grid>
                  </Grid>
                </Grid>
                
                {/* Frequency Penalty */}
                <Grid item xs={12} md={6}>
                  <Typography gutterBottom>
                    Frequency Penalty
                    <Tooltip title="Positive values penalize tokens based on their frequency in the text so far">
                      <IconButton size="small">
                        <InfoIcon fontSize="small" />
                      </IconButton>
                    </Tooltip>
                  </Typography>
                  <Grid container spacing={2} alignItems="center">
                    <Grid item xs>
                      <Slider
                        value={parameters.frequency_penalty}
                        onChange={handleSliderChange('frequency_penalty')}
                        aria-labelledby="frequency-penalty-slider"
                        step={0.1}
                        min={-2}
                        max={2}
                      />
                    </Grid>
                    <Grid item>
                      <TextField
                        value={parameters.frequency_penalty}
                        onChange={handleSliderInputChange('frequency_penalty')}
                        inputProps={{
                          step: 0.1,
                          min: -2,
                          max: 2,
                          type: 'number',
                        }}
                        sx={{ width: '80px' }}
                        size="small"
                      />
                    </Grid>
                  </Grid>
                </Grid>
                
                {/* Max Tokens */}
                <Grid item xs={12} md={6}>
                  <Typography gutterBottom>
                    Max Output Tokens
                    <Tooltip title="Maximum number of tokens to generate in the response">
                      <IconButton size="small">
                        <InfoIcon fontSize="small" />
                      </IconButton>
                    </Tooltip>
                  </Typography>
                  <Grid container spacing={2} alignItems="center">
                    <Grid item xs>
                      <Slider
                        value={parameters.max_tokens}
                        onChange={handleSliderChange('max_tokens')}
                        aria-labelledby="max-tokens-slider"
                        step={100}
                        min={100}
                        max={model.max_output_tokens || 4000}
                      />
                    </Grid>
                    <Grid item>
                      <TextField
                        value={parameters.max_tokens}
                        onChange={handleSliderInputChange('max_tokens')}
                        inputProps={{
                          step: 100,
                          min: 100,
                          max: model.max_output_tokens || 4000,
                          type: 'number',
                        }}
                        sx={{ width: '80px' }}
                        size="small"
                      />
                    </Grid>
                  </Grid>
                </Grid>
                
                {/* Advanced Parameters */}
                <Grid item xs={12}>
                  <Accordion>
                    <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                      <Typography>Advanced Parameters</Typography>
                    </AccordionSummary>
                    <AccordionDetails>
                      <Grid container spacing={3}>
                        {/* Repetition Penalty */}
                        <Grid item xs={12} md={6}>
                          <Typography gutterBottom>
                            Repetition Penalty
                            <Tooltip title="Penalizes repetition. Higher values reduce repetition. 1.0 = no penalty">
                              <IconButton size="small">
                                <InfoIcon fontSize="small" />
                              </IconButton>
                            </Tooltip>
                          </Typography>
                          <Grid container spacing={2} alignItems="center">
                            <Grid item xs>
                              <Slider
                                value={parameters.repetition_penalty}
                                onChange={handleSliderChange('repetition_penalty')}
                                step={0.05}
                                min={1.0}
                                max={2.0}
                              />
                            </Grid>
                            <Grid item>
                              <TextField
                                value={parameters.repetition_penalty}
                                onChange={handleSliderInputChange('repetition_penalty')}
                                inputProps={{
                                  step: 0.05,
                                  min: 1.0,
                                  max: 2.0,
                                  type: 'number',
                                }}
                                sx={{ width: '80px' }}
                                size="small"
                              />
                            </Grid>
                          </Grid>
                        </Grid>
                        
                        {/* Top K */}
                        <Grid item xs={12} md={6}>
                          <Typography gutterBottom>
                            Top K
                            <Tooltip title="Limits token selection to the top K options. Only applicable when beam search is disabled">
                              <IconButton size="small">
                                <InfoIcon fontSize="small" />
                              </IconButton>
                            </Tooltip>
                          </Typography>
                          <Grid container spacing={2} alignItems="center">
                            <Grid item xs>
                              <Slider
                                value={parameters.top_k}
                                onChange={handleSliderChange('top_k')}
                                step={1}
                                min={1}
                                max={100}
                                disabled={parameters.use_beam_search}
                              />
                            </Grid>
                            <Grid item>
                              <TextField
                                value={parameters.top_k}
                                onChange={handleSliderInputChange('top_k')}
                                inputProps={{
                                  step: 1,
                                  min: 1,
                                  max: 100,
                                  type: 'number',
                                }}
                                sx={{ width: '80px' }}
                                size="small"
                                disabled={parameters.use_beam_search}
                              />
                            </Grid>
                          </Grid>
                        </Grid>
                        
                        {/* Use Beam Search */}
                        <Grid item xs={12}>
                          <FormControlLabel 
                            control={
                              <Switch
                                checked={parameters.use_beam_search}
                                onChange={(e) => handleParameterChange('use_beam_search', e.target.checked)}
                              />
                            }
                            label="Use beam search (may not be supported by all models)"
                          />
                        </Grid>
                      </Grid>
                    </AccordionDetails>
                  </Accordion>
                </Grid>
              </Grid>
              
              <Box sx={{ mt: 4, display: 'flex', justifyContent: 'flex-end' }}>
                <Button 
                  variant="contained" 
                  onClick={handleSave}
                  disabled={saving}
                >
                  {saving ? 'Saving...' : 'Save as Default Parameters'}
                </Button>
              </Box>
            </Paper>
          </>
        ) : (
          <Alert severity="error">Model not found</Alert>
        )}
      </Box>
    </PageLayout>
  );
};

export default ModelParametersPage;