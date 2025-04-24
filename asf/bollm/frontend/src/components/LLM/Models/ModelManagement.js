import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Paper,
  Button,
  IconButton,
  Card,
  CardHeader,
  CardContent,
  CardActions,
  Grid,
  Chip,
  TextField,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  MenuItem,
  Divider,
  Alert,
  CircularProgress,
  Tooltip,
  TableContainer,
  Table,
  TableHead,
  TableBody,
  TableRow,
  TableCell,
  FormControl,
  InputLabel,
  Select,
  Switch,
  FormControlLabel,
  Autocomplete
} from '@mui/material';
import {
  Add as AddIcon,
  Edit as EditIcon,
  Delete as DeleteIcon,
  Refresh as RefreshIcon,
  Check as CheckIcon,
  Error as ErrorIcon,
  Code as CodeIcon,
  Image as ImageIcon,
  TextFields as TextFieldsIcon,
  TimerOutlined as TimerIcon
} from '@mui/icons-material';

import { useNotification } from '../../../context/NotificationContext';
import apiService from '../../../services/api';

/**
 * Model management component
 * Allows adding, editing, and removing LLM models
 */
const ModelManagement = () => {
  const { showSuccess, showError } = useNotification();
  
  // State
  const [models, setModels] = useState([]);
  const [providers, setProviders] = useState([]);
  const [loading, setLoading] = useState(true);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [dialogMode, setDialogMode] = useState('add'); // 'add' or 'edit'
  const [selectedModel, setSelectedModel] = useState(null);
  const [formValues, setFormValues] = useState({
    model_id: '',
    provider_id: '',
    display_name: '',
    model_type: 'chat',
    capabilities: [],
    context_window: null,
    max_output_tokens: null,
    parameters: {}
  });
  const [deleteConfirmOpen, setDeleteConfirmOpen] = useState(false);
  const [modelToDelete, setModelToDelete] = useState(null);
  const [refreshing, setRefreshing] = useState(false);
  const [filterProviderId, setFilterProviderId] = useState('');
  const [filterModelType, setFilterModelType] = useState('');
  const [usingMockData, setUsingMockData] = useState(false);
  
  // Model types
  const modelTypes = [
    { value: 'chat', label: 'Chat Completion' },
    { value: 'completion', label: 'Text Completion' },
    { value: 'embedding', label: 'Embedding' },
    { value: 'image', label: 'Image Generation' },
    { value: 'vision', label: 'Vision' }
  ];
  
  // Capabilities
  const capabilityOptions = [
    { value: 'function_calling', label: 'Function Calling' },
    { value: 'json_mode', label: 'JSON Mode' },
    { value: 'vision', label: 'Vision' },
    { value: 'streaming', label: 'Streaming' },
    { value: 'code', label: 'Code Generation' },
    { value: 'reasoning', label: 'Reasoning' },
    { value: 'tool_use', label: 'Tool Use' }
  ];
  
  // Load data on mount
  useEffect(() => {
    loadData();
  }, []);
  
  // Load data
  const loadData = async () => {
    setLoading(true);
    
    try {
      // Load providers first
      const providersResult = await apiService.llm.getProviders();
      let providersList = [];
      
      if (providersResult.success) {
        // Ensure providers data is an array
        providersList = Array.isArray(providersResult.data) ? providersResult.data : [];
        setProviders(providersList);
      } else {
        console.error('Failed to load providers:', providersResult.error);
        // Continue with empty providers list but don't show error yet
        console.warn('Could not load providers from API');
        providersList = [];
      }
      
      // Then load models
      try {
        // Use the gateway/models endpoint specifically
        const modelsResult = await apiService.llm.getModels();
        
        console.log('Models API response:', modelsResult);
        
        if (modelsResult.success && Array.isArray(modelsResult.data) && modelsResult.data.length > 0) {
          // Process models from the API response
          console.log('Models data from API:', modelsResult.data);
          console.log('Setting models from API data');
          setModels(modelsResult.data);
          setUsingMockData(false);
        } else {
          console.error('Failed to load models or empty response:', modelsResult.error || 'No models returned');
          
          // Create mock models based on available providers
          console.warn('Using mock model data since API request failed or returned no models');
          const mockModels = createMockModels(providersList);
          console.log('Setting mock models:', mockModels);
          setModels(mockModels);
          setUsingMockData(true);
          
          // Only show error if we couldn't load both providers and models
          if (providersList.length === 0) {
            showError(`Failed to load models and providers. Using mock data.`);
          } else {
            showError(`Failed to load models: ${modelsResult.error || 'No models returned'}. Using mock data.`);
          }
        }
      } catch (modelError) {
        console.error('Error loading models:', modelError);
        
        // Create mock models based on available providers
        console.warn('Using mock model data due to API error');
        const mockModels = createMockModels(providersList);
        setModels(mockModels);
        setUsingMockData(true);
        showError(`Error loading models: ${modelError.message}. Using mock data.`);
      }
    } catch (error) {
      console.error('Error loading data:', error);
      
      // Use mock data for both providers and models
      console.warn('Using mock data for both providers and models');
      const mockProviders = [
        {
          provider_id: 'openai',
          provider_type: 'openai',
          display_name: 'OpenAI',
          status: 'operational'
        },
        {
          provider_id: 'anthropic',
          provider_type: 'anthropic',
          display_name: 'Anthropic',
          status: 'operational'
        },
        {
          provider_id: 'mistral',
          provider_type: 'mistral',
          display_name: 'Mistral AI',
          status: 'operational'
        }
      ];
      
      setProviders(mockProviders);
      const mockModels = createMockModels(mockProviders);
      setModels(mockModels);
      setUsingMockData(true);
      showError(`Error connecting to API: ${error.message}. Using mock data.`);
    } finally {
      setLoading(false);
    }
  };
  
  // Helper function to create mock models based on providers
  const createMockModels = (providers) => {
    const mockModels = [];
    
    // For each provider, add a generic model
    providers.forEach(provider => {
      // Create a generic model for each provider
      mockModels.push({
        model_id: `${provider.provider_id}-model-1`,
        provider_id: provider.provider_id,
        display_name: `${provider.display_name || provider.provider_id} Model 1`,
        model_type: 'chat',
        capabilities: ['function_calling'],
        context_window: 16000,
        max_output_tokens: 2048
      });
      
      // Add a second model with different capabilities
      mockModels.push({
        model_id: `${provider.provider_id}-model-2`,
        provider_id: provider.provider_id,
        display_name: `${provider.display_name || provider.provider_id} Model 2`,
        model_type: 'chat',
        capabilities: ['function_calling', 'json_mode'],
        context_window: 32000,
        max_output_tokens: 4096
      });
    });
    
    return mockModels;
  };
  
  // Refresh data
  const handleRefresh = async () => {
    setRefreshing(true);
    await loadData();
    setRefreshing(false);
  };
  
  // Open add dialog
  const handleOpenAddDialog = () => {
    setDialogMode('add');
    setSelectedModel(null);
    setFormValues({
      model_id: '',
      provider_id: providers.length > 0 ? providers[0].provider_id : '',
      display_name: '',
      model_type: 'chat',
      capabilities: [],
      context_window: null,
      max_output_tokens: null,
      parameters: {}
    });
    setDialogOpen(true);
  };
  
  // Open edit dialog
  const handleOpenEditDialog = (model) => {
    setDialogMode('edit');
    setSelectedModel(model);
    
    // Transform model data to form values
    setFormValues({
      model_id: model.model_id,
      provider_id: model.provider_id,
      display_name: model.display_name || model.model_id,
      model_type: model.model_type || 'chat',
      capabilities: model.capabilities || [],
      context_window: model.context_window || null,
      max_output_tokens: model.max_output_tokens || null,
      parameters: model.parameters || {}
    });
    
    setDialogOpen(true);
  };
  
  // Handle dialog close
  const handleDialogClose = () => {
    setDialogOpen(false);
  };
  
  // Handle form input change
  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormValues((prev) => ({
      ...prev,
      [name]: value
    }));
  };
  
  // Handle number input change
  const handleNumberInputChange = (e) => {
    const { name, value } = e.target;
    setFormValues((prev) => ({
      ...prev,
      [name]: value === '' ? null : Number(value)
    }));
  };
  
  // Handle capabilities change
  const handleCapabilitiesChange = (_, newValue) => {
    setFormValues((prev) => ({
      ...prev,
      capabilities: newValue.map(item => typeof item === 'string' ? item : item.value)
    }));
  };
  
  // Handle form submit
  const handleSubmit = async () => {
    // Validate form
    if (!formValues.model_id) {
      showError('Model ID is required');
      return;
    }
    
    if (!formValues.provider_id) {
      showError('Provider is required');
      return;
    }
    
    if (!formValues.display_name) {
      showError('Display name is required');
      return;
    }
    
    try {
      if (dialogMode === 'add') {
        // Create model
        const result = await apiService.llm.createModel(formValues);
        
        if (result.success) {
          showSuccess(`Model "${formValues.display_name}" created successfully`);
          handleDialogClose();
          loadData();
        } else {
          showError(`Failed to create model: ${result.error}`);
        }
      } else {
        // Update model
        const result = await apiService.llm.updateModel(
          formValues.model_id,
          formValues.provider_id,
          {
            display_name: formValues.display_name,
            model_type: formValues.model_type,
            capabilities: formValues.capabilities,
            context_window: formValues.context_window,
            max_output_tokens: formValues.max_output_tokens,
            parameters: formValues.parameters
          }
        );
        
        if (result.success) {
          showSuccess(`Model "${formValues.display_name}" updated successfully`);
          handleDialogClose();
          loadData();
        } else {
          showError(`Failed to update model: ${result.error}`);
        }
      }
    } catch (error) {
      console.error('Error saving model:', error);
      showError(`Error saving model: ${error.message}`);
    }
  };
  
  // Open delete confirm dialog
  const handleOpenDeleteConfirm = (model) => {
    setModelToDelete(model);
    setDeleteConfirmOpen(true);
  };
  
  // Handle delete confirm
  const handleDeleteConfirm = async () => {
    if (!modelToDelete) return;
    
    try {
      const result = await apiService.llm.deleteModel(
        modelToDelete.model_id,
        modelToDelete.provider_id
      );
      
      if (result.success) {
        showSuccess(`Model "${modelToDelete.display_name}" deleted successfully`);
        setDeleteConfirmOpen(false);
        setModelToDelete(null);
        loadData();
      } else {
        showError(`Failed to delete model: ${result.error}`);
      }
    } catch (error) {
      console.error('Error deleting model:', error);
      showError(`Error deleting model: ${error.message}`);
    }
  };
  
  // Get filtered models
  const getFilteredModels = () => {
    return models.filter(model => {
      // Filter by provider
      if (filterProviderId && model.provider_id !== filterProviderId) {
        return false;
      }
      
      // Filter by model type
      if (filterModelType && model.model_type !== filterModelType) {
        return false;
      }
      
      return true;
    });
  };
  
  // Get provider name by ID
  const getProviderName = (providerId) => {
    const provider = providers.find(p => p.provider_id === providerId);
    return provider ? (provider.display_name || provider.provider_id) : providerId;
  };
  
  // Get capability icon
  const getCapabilityIcon = (capability) => {
    switch (capability) {
      case 'function_calling':
      case 'code':
        return <CodeIcon fontSize="small" />;
      case 'vision':
        return <ImageIcon fontSize="small" />;
      case 'streaming':
        return <TimerIcon fontSize="small" />;
      default:
        return <TextFieldsIcon fontSize="small" />;
    }
  };
  
  // Get model type color
  const getModelTypeColor = (modelType) => {
    switch (modelType) {
      case 'chat':
        return 'primary';
      case 'completion':
        return 'secondary';
      case 'embedding':
        return 'info';
      case 'image':
        return 'success';
      case 'vision':
        return 'warning';
      default:
        return 'default';
    }
  };
  
  // Render model table
  const renderModelTable = () => {
    const filteredModels = getFilteredModels();
    
    return (
      <TableContainer component={Paper}>
        <Table>
          <TableHead>
            <TableRow>
              <TableCell>ID</TableCell>
              <TableCell>Name</TableCell>
              <TableCell>Provider</TableCell>
              <TableCell>Type</TableCell>
              <TableCell>Context Window</TableCell>
              <TableCell>Capabilities</TableCell>
              <TableCell>Actions</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {filteredModels.map((model) => (
              <TableRow key={`${model.provider_id}-${model.model_id}`}>
                <TableCell>{model.model_id}</TableCell>
                <TableCell>{model.display_name || model.model_id}</TableCell>
                <TableCell>{getProviderName(model.provider_id)}</TableCell>
                <TableCell>
                  <Chip
                    label={model.model_type || 'chat'}
                    color={getModelTypeColor(model.model_type)}
                    size="small"
                  />
                </TableCell>
                <TableCell>{model.context_window || 'N/A'}</TableCell>
                <TableCell>
                  {model.capabilities && model.capabilities.length > 0 ? (
                    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                      {model.capabilities.slice(0, 2).map((capability) => (
                        <Chip
                          key={capability}
                          label={capability}
                          size="small"
                          variant="outlined"
                          icon={getCapabilityIcon(capability)}
                        />
                      ))}
                      {model.capabilities.length > 2 && (
                        <Tooltip title={model.capabilities.slice(2).join(', ')}>
                          <Chip label={`+${model.capabilities.length - 2}`} size="small" />
                        </Tooltip>
                      )}
                    </Box>
                  ) : (
                    <Typography variant="body2" color="text.secondary">
                      None
                    </Typography>
                  )}
                </TableCell>
                <TableCell>
                  <Box sx={{ display: 'flex', gap: 1 }}>
                    <IconButton
                      size="small"
                      onClick={() => handleOpenEditDialog(model)}
                      title="Edit model"
                    >
                      <EditIcon fontSize="small" />
                    </IconButton>
                    <IconButton
                      size="small"
                      color="error"
                      onClick={() => handleOpenDeleteConfirm(model)}
                      title="Delete model"
                    >
                      <DeleteIcon fontSize="small" />
                    </IconButton>
                  </Box>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>
    );
  };
  
  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h6">
          LLM Model Management
        </Typography>
        <Box sx={{ display: 'flex', gap: 2 }}>
          <Button
            variant="outlined"
            startIcon={refreshing ? <CircularProgress size={16} /> : <RefreshIcon />}
            onClick={handleRefresh}
            disabled={refreshing}
          >
            Refresh
          </Button>
          <Button
            variant="contained"
            startIcon={<AddIcon />}
            onClick={handleOpenAddDialog}
            disabled={providers.length === 0}
          >
            Add Model
          </Button>
        </Box>
      </Box>
      
      {usingMockData && (
        <Alert severity="warning" sx={{ mb: 3 }}>
          Mock data is being displayed due to an issue with the API.
        </Alert>
      )}
      
      {/* Filters */}
      <Paper sx={{ p: 2, mb: 3 }}>
        <Grid container spacing={2} alignItems="center">
          <Grid item xs={12} sm={6} md={4}>
            <FormControl fullWidth size="small">
              <InputLabel>Filter by Provider</InputLabel>
              <Select
                value={filterProviderId}
                onChange={(e) => setFilterProviderId(e.target.value)}
                label="Filter by Provider"
              >
                <MenuItem value="">All Providers</MenuItem>
                {providers.map((provider) => (
                  <MenuItem key={provider.provider_id} value={provider.provider_id}>
                    {provider.display_name || provider.provider_id}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </Grid>
          <Grid item xs={12} sm={6} md={4}>
            <FormControl fullWidth size="small">
              <InputLabel>Filter by Type</InputLabel>
              <Select
                value={filterModelType}
                onChange={(e) => setFilterModelType(e.target.value)}
                label="Filter by Type"
              >
                <MenuItem value="">All Types</MenuItem>
                {modelTypes.map((type) => (
                  <MenuItem key={type.value} value={type.value}>
                    {type.label}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </Grid>
          <Grid item xs={12} sm={6} md={4}>
            <Button 
              variant="outlined" 
              onClick={() => {
                setFilterProviderId('');
                setFilterModelType('');
              }}
            >
              Clear Filters
            </Button>
          </Grid>
        </Grid>
      </Paper>
      
      {loading ? (
        <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
          <CircularProgress />
        </Box>
      ) : models.length === 0 ? (
        <Paper sx={{ p: 3, textAlign: 'center' }}>
          <Typography variant="body1" gutterBottom>
            No LLM models found.
          </Typography>
          {providers.length === 0 ? (
            <Alert severity="info" sx={{ mt: 2, mb: 2 }}>
              You need to add at least one provider before adding models.
            </Alert>
          ) : (
            <Button
              variant="contained"
              startIcon={<AddIcon />}
              onClick={handleOpenAddDialog}
              sx={{ mt: 2 }}
            >
              Add Model
            </Button>
          )}
        </Paper>
      ) : getFilteredModels().length === 0 ? (
        <Paper sx={{ p: 3, textAlign: 'center' }}>
          <Typography variant="body1" gutterBottom>
            No models match the current filters.
          </Typography>
          <Button
            variant="outlined"
            onClick={() => {
              setFilterProviderId('');
              setFilterModelType('');
            }}
            sx={{ mt: 2 }}
          >
            Clear Filters
          </Button>
        </Paper>
      ) : (
        <Box sx={{ mb: 4 }}>
          {renderModelTable()}
        </Box>
      )}
      
      {/* Model Form Dialog */}
      <Dialog open={dialogOpen} onClose={handleDialogClose} maxWidth="md" fullWidth>
        <DialogTitle>
          {dialogMode === 'add' ? 'Add LLM Model' : `Edit Model: ${selectedModel?.display_name || selectedModel?.model_id}`}
        </DialogTitle>
        <DialogContent dividers>
          {usingMockData && (
            <Alert severity="warning" sx={{ mb: 3 }}>
              You are currently working with mock data. Changes made will not be saved to the backend server.
            </Alert>
          )}
          <Grid container spacing={3}>
            <Grid item xs={12} sm={6}>
              <TextField
                name="model_id"
                label="Model ID"
                value={formValues.model_id}
                onChange={handleInputChange}
                fullWidth
                required
                margin="normal"
                disabled={dialogMode === 'edit'}
                helperText="Unique identifier for this model"
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                name="display_name"
                label="Display Name"
                value={formValues.display_name}
                onChange={handleInputChange}
                fullWidth
                required
                margin="normal"
                helperText="Human-readable name for this model"
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <FormControl fullWidth margin="normal">
                <InputLabel>Provider</InputLabel>
                <Select
                  name="provider_id"
                  value={formValues.provider_id}
                  onChange={handleInputChange}
                  label="Provider"
                  disabled={dialogMode === 'edit'}
                  required
                >
                  {providers.map((provider) => (
                    <MenuItem key={provider.provider_id} value={provider.provider_id}>
                      {provider.display_name || provider.provider_id}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} sm={6}>
              <FormControl fullWidth margin="normal">
                <InputLabel>Model Type</InputLabel>
                <Select
                  name="model_type"
                  value={formValues.model_type}
                  onChange={handleInputChange}
                  label="Model Type"
                  required
                >
                  {modelTypes.map((type) => (
                    <MenuItem key={type.value} value={type.value}>
                      {type.label}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                name="context_window"
                label="Context Window (tokens)"
                value={formValues.context_window === null ? '' : formValues.context_window}
                onChange={handleNumberInputChange}
                fullWidth
                margin="normal"
                type="number"
                helperText="Maximum context window size in tokens"
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                name="max_output_tokens"
                label="Max Output Tokens"
                value={formValues.max_output_tokens === null ? '' : formValues.max_output_tokens}
                onChange={handleNumberInputChange}
                fullWidth
                margin="normal"
                type="number"
                helperText="Maximum number of output tokens"
              />
            </Grid>
            <Grid item xs={12}>
              <Autocomplete
                multiple
                options={capabilityOptions}
                getOptionLabel={(option) => typeof option === 'string' ? option : option.label}
                value={formValues.capabilities.map(cap => {
                  const option = capabilityOptions.find(opt => opt.value === cap);
                  return option || cap;
                })}
                onChange={handleCapabilitiesChange}
                isOptionEqualToValue={(option, value) => {
                  return option.value === value || option.value === value.value;
                }}
                renderInput={(params) => (
                  <TextField
                    {...params}
                    label="Capabilities"
                    margin="normal"
                    helperText="Special capabilities of this model"
                  />
                )}
              />
            </Grid>
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button onClick={handleDialogClose}>Cancel</Button>
          <Button onClick={handleSubmit} variant="contained">
            {dialogMode === 'add' ? 'Add Model' : 'Save Changes'}
          </Button>
        </DialogActions>
      </Dialog>
      
      {/* Delete Confirmation Dialog */}
      <Dialog open={deleteConfirmOpen} onClose={() => setDeleteConfirmOpen(false)}>
        <DialogTitle>Confirm Delete</DialogTitle>
        <DialogContent>
          <Typography variant="body1">
            Are you sure you want to delete the model "{modelToDelete?.display_name || modelToDelete?.model_id}"?
          </Typography>
          <Typography variant="body2" color="error" sx={{ mt: 2 }}>
            This action cannot be undone.
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDeleteConfirmOpen(false)}>Cancel</Button>
          <Button onClick={handleDeleteConfirm} color="error" variant="contained">
            Delete Model
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default ModelManagement;
