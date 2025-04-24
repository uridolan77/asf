// filepath: c:\code\asf\asf\bollm\frontend\src\pages\LLMManagement\Models\ModelConfigPage.jsx
import React, { useState, useEffect } from 'react';
import { 
  Box, 
  Typography, 
  Container, 
  Paper,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Button,
  FormHelperText,
  Divider,
  Grid,
  Alert,
  IconButton,
  Tooltip,
  Chip,
  Autocomplete
} from '@mui/material';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';
import InfoIcon from '@mui/icons-material/Info';
import { useNavigate, useParams } from 'react-router-dom';
import apiService from '../../../services/api';

/**
 * Model Configuration Page
 * For adding/editing LLM model configurations
 */
const ModelConfigPage = () => {
  const navigate = useNavigate();
  const { modelId, providerId } = useParams();
  const isEditMode = !!modelId;
  
  // State for providers list
  const [providers, setProviders] = useState([]);
  const [loading, setLoading] = useState(true);
  
  // Form state
  const [formData, setFormData] = useState({
    modelId: '',
    providerId: '',
    displayName: '',
    modelType: 'chat',
    capabilities: [],
    contextWindow: 4096,
    maxTokens: 1000,
    temperature: 0.7,
    topP: 1.0,
    description: ''
  });
  
  const [saved, setSaved] = useState(false);
  const [error, setError] = useState('');
  
  // Model types
  const modelTypes = [
    { value: 'chat', label: 'Chat Completion' },
    { value: 'completion', label: 'Text Completion' },
    { value: 'embedding', label: 'Embedding' },
    { value: 'image', label: 'Image Generation' },
    { value: 'vision', label: 'Vision' }
  ];
  
  // Capabilities
  const capabilities = [
    { value: 'function_calling', label: 'Function Calling' },
    { value: 'json_mode', label: 'JSON Mode' },
    { value: 'vision', label: 'Vision' },
    { value: 'streaming', label: 'Streaming' },
    { value: 'code', label: 'Code Generation' },
    { value: 'reasoning', label: 'Reasoning' },
    { value: 'tool_use', label: 'Tool Use' }
  ];
  
  // Load providers and model data if in edit mode
  useEffect(() => {
    const loadData = async () => {
      setLoading(true);
      try {
        // Load providers
        const providersResponse = await apiService.llm.getProviders();
        if (providersResponse.success) {
          setProviders(providersResponse.data);
          
          // If not in edit mode, pre-select first provider
          if (!isEditMode && providersResponse.data.length > 0) {
            setFormData(prev => ({
              ...prev,
              providerId: providersResponse.data[0].provider_id
            }));
          }
        } else {
          setError('Failed to load providers');
        }
        
        // If in edit mode, load model data
        if (isEditMode) {
          try {
            const modelResponse = await apiService.llm.getModelById(modelId, providerId);
            if (modelResponse.success) {
              const model = modelResponse.data;
              setFormData({
                modelId: model.model_id,
                providerId: model.provider_id,
                displayName: model.display_name || model.model_id,
                modelType: model.model_type || 'chat',
                capabilities: model.capabilities || [],
                contextWindow: model.context_window || 4096,
                maxTokens: model.max_output_tokens || 1000,
                temperature: model.default_temperature || 0.7,
                topP: model.top_p || 1.0,
                description: model.description || ''
              });
            } else {
              setError(`Failed to load model data: ${modelResponse.error}`);
            }
          } catch (err) {
            setError(`Error loading model data: ${err.message}`);
          }
        }
      } catch (err) {
        setError(`Error loading data: ${err.message}`);
      } finally {
        setLoading(false);
      }
    };
    
    loadData();
  }, [modelId, providerId, isEditMode]);
  
  // Handle input change
  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };
  
  // Handle number input change
  const handleNumberChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value === '' ? '' : Number(value)
    }));
  };
  
  // Handle capabilities change
  const handleCapabilitiesChange = (_, newValue) => {
    setFormData(prev => ({
      ...prev,
      capabilities: newValue.map(item => typeof item === 'string' ? item : item.value)
    }));
  };
  
  // Handle form submission
  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    
    // Validate form
    if (!formData.modelId) {
      setError('Model ID is required');
      return;
    }
    
    if (!formData.providerId) {
      setError('Provider is required');
      return;
    }
    
    if (!formData.displayName) {
      setError('Display name is required');
      return;
    }
    
    try {
      // Format data for API
      const modelData = {
        model_id: formData.modelId,
        provider_id: formData.providerId,
        display_name: formData.displayName,
        model_type: formData.modelType,
        capabilities: formData.capabilities,
        context_window: formData.contextWindow,
        max_output_tokens: formData.maxTokens,
        default_temperature: formData.temperature,
        top_p: formData.topP,
        description: formData.description
      };
      
      let result;
      if (isEditMode) {
        // Update existing model
        result = await apiService.llm.updateModel(
          formData.modelId,
          formData.providerId,
          {
            display_name: formData.displayName,
            model_type: formData.modelType,
            capabilities: formData.capabilities,
            context_window: formData.contextWindow,
            max_output_tokens: formData.maxTokens,
            default_temperature: formData.temperature,
            top_p: formData.topP,
            description: formData.description
          }
        );
      } else {
        // Create new model
        result = await apiService.llm.createModel(modelData);
      }
      
      if (result.success) {
        setSaved(true);
        setTimeout(() => {
          navigate('/llm/models');
        }, 2000);
      } else {
        setError(`Failed to save model: ${result.error}`);
      }
    } catch (err) {
      setError(`Error saving model: ${err.message}`);
    }
  };
  
  return (
    <Container maxWidth="md">
      <Box sx={{ pt: 3, pb: 5 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 4 }}>
          <IconButton 
            onClick={() => navigate('/llm/models')}
            sx={{ mr: 2 }}
          >
            <ArrowBackIcon />
          </IconButton>
          <Typography variant="h4" component="h1">
            {isEditMode ? 'Edit Model' : 'Add New Model'}
          </Typography>
        </Box>
        
        {saved && (
          <Alert severity="success" sx={{ mb: 3 }}>
            Model {isEditMode ? 'updated' : 'created'} successfully! Redirecting...
          </Alert>
        )}
        
        {error && (
          <Alert severity="error" sx={{ mb: 3 }}>
            {error}
          </Alert>
        )}
        
        <Paper sx={{ p: 3 }}>
          {loading ? (
            <Box sx={{ textAlign: 'center', py: 3 }}>
              <Typography>Loading...</Typography>
            </Box>
          ) : (
            <form onSubmit={handleSubmit}>
              <Grid container spacing={3}>
                {/* Basic Info */}
                <Grid item xs={12}>
                  <Typography variant="h6">Basic Information</Typography>
                  <Divider sx={{ mt: 1, mb: 2 }} />
                </Grid>
                
                <Grid item xs={12} md={6}>
                  <TextField
                    required
                    fullWidth
                    label="Model ID"
                    name="modelId"
                    value={formData.modelId}
                    onChange={handleChange}
                    helperText="Unique identifier for this model"
                    disabled={isEditMode}
                  />
                </Grid>
                
                <Grid item xs={12} md={6}>
                  <TextField
                    required
                    fullWidth
                    label="Display Name"
                    name="displayName"
                    value={formData.displayName}
                    onChange={handleChange}
                    helperText="Human-readable name for this model"
                  />
                </Grid>
                
                <Grid item xs={12} md={6}>
                  <FormControl fullWidth required>
                    <InputLabel>Provider</InputLabel>
                    <Select
                      name="providerId"
                      value={formData.providerId}
                      onChange={handleChange}
                      label="Provider"
                      disabled={isEditMode}
                    >
                      {providers.map(provider => (
                        <MenuItem key={provider.provider_id} value={provider.provider_id}>
                          {provider.display_name || provider.provider_id}
                        </MenuItem>
                      ))}
                    </Select>
                    <FormHelperText>Select the provider this model belongs to</FormHelperText>
                  </FormControl>
                </Grid>
                
                <Grid item xs={12} md={6}>
                  <FormControl fullWidth required>
                    <InputLabel>Model Type</InputLabel>
                    <Select
                      name="modelType"
                      value={formData.modelType}
                      onChange={handleChange}
                      label="Model Type"
                    >
                      {modelTypes.map(type => (
                        <MenuItem key={type.value} value={type.value}>
                          {type.label}
                        </MenuItem>
                      ))}
                    </Select>
                    <FormHelperText>Type of model capability</FormHelperText>
                  </FormControl>
                </Grid>
                
                <Grid item xs={12}>
                  <TextField
                    fullWidth
                    label="Description"
                    name="description"
                    value={formData.description}
                    onChange={handleChange}
                    multiline
                    rows={2}
                  />
                </Grid>
                
                {/* Model Parameters */}
                <Grid item xs={12} sx={{ mt: 2 }}>
                  <Typography variant="h6">
                    Model Parameters
                    <Tooltip title="Technical specifications and capabilities of this model">
                      <IconButton size="small">
                        <InfoIcon fontSize="small" />
                      </IconButton>
                    </Tooltip>
                  </Typography>
                  <Divider sx={{ mt: 1, mb: 2 }} />
                </Grid>
                
                <Grid item xs={12} md={6}>
                  <TextField
                    fullWidth
                    type="number"
                    label="Context Window (tokens)"
                    name="contextWindow"
                    value={formData.contextWindow}
                    onChange={handleNumberChange}
                    inputProps={{ min: 0 }}
                    helperText="Maximum context window in tokens"
                  />
                </Grid>
                
                <Grid item xs={12} md={6}>
                  <TextField
                    fullWidth
                    type="number"
                    label="Max Output Tokens"
                    name="maxTokens"
                    value={formData.maxTokens}
                    onChange={handleNumberChange}
                    inputProps={{ min: 0 }}
                    helperText="Maximum number of output tokens"
                  />
                </Grid>
                
                <Grid item xs={12} md={6}>
                  <TextField
                    fullWidth
                    type="number"
                    label="Default Temperature"
                    name="temperature"
                    value={formData.temperature}
                    onChange={handleNumberChange}
                    inputProps={{ min: 0, max: 2, step: 0.1 }}
                    helperText="Default temperature parameter (0.0-2.0)"
                  />
                </Grid>
                
                <Grid item xs={12} md={6}>
                  <TextField
                    fullWidth
                    type="number"
                    label="Default Top P"
                    name="topP"
                    value={formData.topP}
                    onChange={handleNumberChange}
                    inputProps={{ min: 0, max: 1, step: 0.01 }}
                    helperText="Default top_p nucleus sampling parameter (0.0-1.0)"
                  />
                </Grid>
                
                <Grid item xs={12}>
                  <Autocomplete
                    multiple
                    options={capabilities}
                    getOptionLabel={(option) => typeof option === 'string' ? option : option.label}
                    value={formData.capabilities.map(cap => {
                      const option = capabilities.find(c => c.value === cap);
                      return option || cap;
                    })}
                    onChange={handleCapabilitiesChange}
                    renderInput={(params) => (
                      <TextField
                        {...params}
                        label="Capabilities"
                        helperText="Special capabilities of this model"
                      />
                    )}
                    renderTags={(value, getTagProps) =>
                      value.map((option, index) => (
                        <Chip 
                          label={typeof option === 'string' ? option : option.label} 
                          {...getTagProps({ index })} 
                          size="small"
                        />
                      ))
                    }
                  />
                </Grid>
                
                {/* Submit */}
                <Grid item xs={12} sx={{ mt: 3 }}>
                  <Box sx={{ display: 'flex', justifyContent: 'flex-end', gap: 2 }}>
                    <Button 
                      variant="outlined" 
                      onClick={() => navigate('/llm/models')}
                    >
                      Cancel
                    </Button>
                    <Button 
                      type="submit" 
                      variant="contained" 
                      color="primary"
                      disabled={loading}
                    >
                      {isEditMode ? 'Update Model' : 'Create Model'}
                    </Button>
                  </Box>
                </Grid>
              </Grid>
            </form>
          )}
        </Paper>
      </Box>
    </Container>
  );
};

export default ModelConfigPage;