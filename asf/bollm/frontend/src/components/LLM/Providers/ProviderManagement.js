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
  FormHelperText
} from '@mui/material';
import {
  Add as AddIcon,
  Edit as EditIcon,
  Delete as DeleteIcon,
  Refresh as RefreshIcon,
  Check as CheckIcon,
  Error as ErrorIcon,
  WarningAmber as WarningIcon,
  Info as InfoIcon
} from '@mui/icons-material';

import { useNotification } from '../../../context/NotificationContext';
import apiService from '../../../services/api';

/**
 * Provider management component
 * Allows adding, editing, and removing LLM providers
 */
const ProviderManagement = () => {
  const { showSuccess, showError } = useNotification();
  
  // State
  const [providers, setProviders] = useState([]);
  const [loading, setLoading] = useState(true);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [dialogMode, setDialogMode] = useState('add'); // 'add' or 'edit'
  const [selectedProvider, setSelectedProvider] = useState(null);
  const [formValues, setFormValues] = useState({
    provider_id: '',
    provider_type: 'openai',
    display_name: '',
    description: '',
    enabled: true,
    connection_params: {
      api_key: '',
      base_url: '',
      auth_type: 'api_key',
    },
    request_settings: {
      timeout_seconds: 30,
      retry_attempts: 3,
      rate_limit_rpm: 10,
    },
    models: {}
  });
  // Add validation state
  const [formErrors, setFormErrors] = useState({
    provider_id: '',
    display_name: '',
    connection_params: {
      api_key: '',
      base_url: ''
    },
    request_settings: {
      timeout_seconds: '',
      retry_attempts: '',
      rate_limit_rpm: ''
    }
  });
  const [testingProvider, setTestingProvider] = useState(null);
  const [refreshing, setRefreshing] = useState(false);
  const [deleteConfirmOpen, setDeleteConfirmOpen] = useState(false);
  const [providerToDelete, setProviderToDelete] = useState(null);
  
  // Provider types
  const providerTypes = [
    { value: 'openai', label: 'OpenAI' },
    { value: 'anthropic', label: 'Anthropic' },
    { value: 'local', label: 'Local (Ollama)' },
    { value: 'google', label: 'Google AI' },
    { value: 'cohere', label: 'Cohere' },
    { value: 'mistral', label: 'Mistral AI' },
    { value: 'azure_openai', label: 'Azure OpenAI' },
    { value: 'huggingface', label: 'HuggingFace' },
    { value: 'custom', label: 'Custom' }
  ];

  // Auth types
  const authTypes = [
    { value: 'api_key', label: 'API Key' },
    { value: 'oauth', label: 'OAuth 2.0' },
    { value: 'bearer', label: 'Bearer Token' },
    { value: 'basic', label: 'Basic Auth' },
    { value: 'none', label: 'No Auth' }
  ];
  
  // Validate form fields
  const validateForm = () => {
    let isValid = true;
    const errors = {
      provider_id: '',
      display_name: '',
      connection_params: {
        api_key: '',
        base_url: ''
      },
      request_settings: {
        timeout_seconds: '',
        retry_attempts: '',
        rate_limit_rpm: ''
      }
    };

    // Validate provider_id
    if (!formValues.provider_id) {
      errors.provider_id = 'Provider ID is required';
      isValid = false;
    } else if (!/^[a-z0-9_-]+$/.test(formValues.provider_id)) {
      errors.provider_id = 'Provider ID can only contain lowercase letters, numbers, hyphens, and underscores';
      isValid = false;
    }

    // Validate display_name
    if (!formValues.display_name) {
      errors.display_name = 'Display name is required';
      isValid = false;
    }

    // Validate API key if auth type requires it
    if (['api_key', 'bearer'].includes(formValues.connection_params.auth_type)) {
      if (!formValues.connection_params.api_key || formValues.connection_params.api_key === '************') {
        if (dialogMode === 'add') {
          errors.connection_params.api_key = 'API key is required';
          isValid = false;
        }
      }
    }

    // Validate base_url if provided
    if (formValues.connection_params.base_url && !formValues.connection_params.base_url.startsWith('http')) {
      errors.connection_params.base_url = 'Base URL must start with http:// or https://';
      isValid = false;
    }

    // Validate timeout_seconds
    const timeout = parseInt(formValues.request_settings.timeout_seconds);
    if (isNaN(timeout) || timeout < 1 || timeout > 300) {
      errors.request_settings.timeout_seconds = 'Timeout must be between 1 and 300 seconds';
      isValid = false;
    }

    // Validate retry_attempts
    const retries = parseInt(formValues.request_settings.retry_attempts);
    if (isNaN(retries) || retries < 0 || retries > 10) {
      errors.request_settings.retry_attempts = 'Retry attempts must be between 0 and 10';
      isValid = false;
    }

    // Validate rate_limit_rpm
    const rateLimit = parseInt(formValues.request_settings.rate_limit_rpm);
    if (isNaN(rateLimit) || rateLimit < 0) {
      errors.request_settings.rate_limit_rpm = 'Rate limit must be 0 or greater';
      isValid = false;
    }

    setFormErrors(errors);
    return isValid;
  };
  
  // Load providers on mount
  useEffect(() => {
    loadProviders();
  }, []);
  
  // Load providers
  const loadProviders = async () => {
    setLoading(true);
    
    try {
      const result = await apiService.llm.getProviders();
      
      if (result.success) {
        setProviders(result.data);
      } else {
        console.warn('Failed to load providers from API. Using mock data instead.');
        
      const mockProviders = [
        {
          provider_id: 'openai',
          provider_type: 'openai',
          display_name: 'OpenAI',
          status: 'operational',
          checked_at: new Date().toISOString(),
          models: []
        },
        {
          provider_id: 'anthropic',
          provider_type: 'anthropic',
          display_name: 'Anthropic',
          status: 'operational',
          checked_at: new Date().toISOString(),
          models: []
        },
        {
          provider_id: 'mistral',
          provider_type: 'mistral',
          display_name: 'Mistral AI',
          status: 'operational',
          checked_at: new Date().toISOString(),
          models: []
        }
      ];
        
        setProviders(mockProviders);
      }
    } catch (error) {
      console.error('Error loading providers:', error);
      
      console.warn('Failed to connect to the API. Using mock data instead.');
      
      const mockProviders = [
        {
          provider_id: 'openai',
          provider_type: 'openai',
          display_name: 'OpenAI',
          status: 'operational',
          checked_at: new Date().toISOString(),
          models: []
        },
        {
          provider_id: 'anthropic',
          provider_type: 'anthropic',
          display_name: 'Anthropic',
          status: 'operational',
          checked_at: new Date().toISOString(),
          models: []
        },
        {
          provider_id: 'mistral',
          provider_type: 'mistral',
          display_name: 'Mistral AI',
          status: 'operational',
          checked_at: new Date().toISOString(),
          models: []
        }
      ];
      
      setProviders(mockProviders);
      
      if (showError && typeof showError === 'function') {
        showError(`Error connecting to the API. Using mock data for demonstration.`);
      }
    } finally {
      setLoading(false);
    }
  };
  
  // Refresh providers
  const handleRefresh = async () => {
    setRefreshing(true);
    await loadProviders();
    setRefreshing(false);
  };
  
  // Open add dialog
  const handleOpenAddDialog = () => {
    setDialogMode('add');
    setSelectedProvider(null);
    // Reset form values
    setFormValues({
      provider_id: '',
      provider_type: 'openai',
      display_name: '',
      description: '',
      enabled: true,
      connection_params: {
        api_key: '',
        base_url: '',
        auth_type: 'api_key',
      },
      request_settings: {
        timeout_seconds: 30,
        retry_attempts: 3,
        rate_limit_rpm: 10,
      },
      models: {}
    });
    // Reset form errors
    setFormErrors({
      provider_id: '',
      display_name: '',
      connection_params: {
        api_key: '',
        base_url: ''
      },
      request_settings: {
        timeout_seconds: '',
        retry_attempts: '',
        rate_limit_rpm: ''
      }
    });
    setDialogOpen(true);
  };
  
  // Open edit dialog
  const handleOpenEditDialog = (provider) => {
    setDialogMode('edit');
    setSelectedProvider(provider);
    
    // Transform provider data to form values with enhanced fields
    setFormValues({
      provider_id: provider.provider_id,
      provider_type: provider.provider_type,
      display_name: provider.display_name || provider.provider_id,
      description: provider.description || '',
      enabled: provider.enabled !== false, // Default to true if not specified
      connection_params: {
        api_key: '************', // Don't show actual key
        base_url: provider.connection_params?.base_url || '',
        auth_type: provider.connection_params?.auth_type || 'api_key',
      },
      request_settings: {
        timeout_seconds: provider.request_settings?.timeout_seconds || 30,
        retry_attempts: provider.request_settings?.retry_attempts || 3,
        rate_limit_rpm: provider.request_settings?.rate_limit_rpm || 10,
      },
      models: provider.models || {}
    });
    
    // Reset form errors
    setFormErrors({
      provider_id: '',
      display_name: '',
      connection_params: {
        api_key: '',
        base_url: ''
      },
      request_settings: {
        timeout_seconds: '',
        retry_attempts: '',
        rate_limit_rpm: ''
      }
    });
    
    setDialogOpen(true);
  };
  
  // Handle dialog close
  const handleDialogClose = () => {
    setDialogOpen(false);
    // Reset form values
    setFormValues({
      provider_id: '',
      provider_type: 'openai',
      display_name: '',
      description: '',
      enabled: true,
      connection_params: {
        api_key: '',
        base_url: '',
        auth_type: 'api_key',
      },
      request_settings: {
        timeout_seconds: 30,
        retry_attempts: 3,
        rate_limit_rpm: 10,
      },
      models: {}
    });
    // Reset form errors
    setFormErrors({
      provider_id: '',
      display_name: '',
      connection_params: {
        api_key: '',
        base_url: ''
      },
      request_settings: {
        timeout_seconds: '',
        retry_attempts: '',
        rate_limit_rpm: ''
      }
    });
  };
  
  // Handle form input change with validation
  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormValues((prev) => ({
      ...prev,
      [name]: value
    }));
    
    // Clear error when typing
    setFormErrors(prev => ({
      ...prev,
      [name]: ''
    }));
  };
  
  // Handle connection params change with validation
  const handleConnectionParamChange = (e) => {
    const { name, value } = e.target;
    setFormValues((prev) => ({
      ...prev,
      connection_params: {
        ...prev.connection_params,
        [name]: value
      }
    }));
    
    // Clear error when typing
    setFormErrors(prev => ({
      ...prev,
      connection_params: {
        ...prev.connection_params,
        [name]: ''
      }
    }));
  };

  // Handle request settings change with validation
  const handleRequestSettingChange = (e) => {
    const { name, value } = e.target;
    setFormValues((prev) => ({
      ...prev,
      request_settings: {
        ...prev.request_settings,
        [name]: value
      }
    }));
    
    // Clear error when typing
    setFormErrors(prev => ({
      ...prev,
      request_settings: {
        ...prev.request_settings,
        [name]: ''
      }
    }));
  };
  
  // Handle form submit with enhanced validation
  const handleSubmit = async () => {
    // Validate all form fields
    if (!validateForm()) {
      // Form has errors
      showError('Please correct the errors in the form');
      return;
    }
    
    try {
      if (dialogMode === 'add') {
        // Format request data - let the backend handle model creation
        const createData = {
          ...formValues,
          // Empty models object - the backend will populate this based on the provider type
          models: {}
        };
        
        // Remove placeholder API key if unchanged
        if (createData.connection_params.api_key === '************') {
          delete createData.connection_params.api_key;
        }
        
        // Convert numeric fields to numbers
        createData.request_settings.timeout_seconds = parseInt(createData.request_settings.timeout_seconds);
        createData.request_settings.retry_attempts = parseInt(createData.request_settings.retry_attempts);
        createData.request_settings.rate_limit_rpm = parseInt(createData.request_settings.rate_limit_rpm);
        
        console.log('Creating provider with data:', createData);
        
        const result = await apiService.llm.registerProvider(createData);
        
        if (result.success) {
          showSuccess(`Provider "${formValues.display_name}" created successfully`);
          handleDialogClose();
          loadProviders();
        } else {
          console.error('Failed to create provider:', result);
          showError(`Failed to create provider: ${result.error}`);
        }
      } else {
        // Format update data
        const updateData = {
          display_name: formValues.display_name,
          description: formValues.description,
          enabled: formValues.enabled,
          connection_params: {...formValues.connection_params},
          request_settings: {...formValues.request_settings}
        };
        
        // Remove placeholder API key if unchanged
        if (updateData.connection_params.api_key === '************') {
          delete updateData.connection_params.api_key;
        }
        
        // Convert numeric fields to numbers
        updateData.request_settings.timeout_seconds = parseInt(updateData.request_settings.timeout_seconds);
        updateData.request_settings.retry_attempts = parseInt(updateData.request_settings.retry_attempts);
        updateData.request_settings.rate_limit_rpm = parseInt(updateData.request_settings.rate_limit_rpm);
        
        const result = await apiService.llm.updateProvider(
          formValues.provider_id,
          updateData
        );
        
        if (result.success) {
          showSuccess(`Provider "${formValues.display_name}" updated successfully`);
          handleDialogClose();
          loadProviders();
        } else {
          showError(`Failed to update provider: ${result.error}`);
        }
      }
    } catch (error) {
      console.error('Error saving provider:', error);
      showError(`Error saving provider: ${error.message}`);
    }
  };
  
  // Test provider with enhanced error handling
  const handleTestProvider = async (providerId) => {
    setTestingProvider(providerId);
    
    try {
      const result = await apiService.llm.testProvider(providerId);
      
      if (result.success) {
        showSuccess(`Provider "${providerId}" test successful`);
        
        // Update the provider status in the list without a full reload
        setProviders(prevProviders => 
          prevProviders.map(provider => 
            provider.provider_id === providerId 
              ? { 
                  ...provider, 
                  status: 'operational',
                  checked_at: new Date().toISOString() 
                }
              : provider
          )
        );
      } else {
        showError(`Provider "${providerId}" test failed: ${result.error}`);
        
        // Update the provider status to error
        setProviders(prevProviders => 
          prevProviders.map(provider => 
            provider.provider_id === providerId 
              ? { 
                  ...provider, 
                  status: 'error',
                  checked_at: new Date().toISOString() 
                }
              : provider
          )
        );
      }
    } catch (error) {
      console.error('Error testing provider:', error);
      showError(`Error testing provider: ${error.message}`);
      
      // Update the provider status to error
      setProviders(prevProviders => 
        prevProviders.map(provider => 
          provider.provider_id === providerId 
            ? { 
                ...provider, 
                status: 'error',
                checked_at: new Date().toISOString() 
              }
            : provider
        )
      );
    } finally {
      setTestingProvider(null);
    }
  };

  // Open delete confirmation dialog
  const handleOpenDeleteConfirm = (provider) => {
    setProviderToDelete(provider);
    setDeleteConfirmOpen(true);
  };

  // Handle delete confirmation
  const handleDeleteConfirm = async () => {
    try {
      if (!providerToDelete) return;
      
      const result = await apiService.llm.deleteProvider(providerToDelete.provider_id);
      
      if (result.success) {
        showSuccess(`Provider "${providerToDelete.display_name || providerToDelete.provider_id}" deleted successfully`);
        // Close the dialog
        setDeleteConfirmOpen(false);
        // Clear the provider to delete
        setProviderToDelete(null);
        // Reload the providers list
        loadProviders();
      } else {
        showError(`Failed to delete provider: ${result.error}`);
      }
    } catch (error) {
      console.error('Error deleting provider:', error);
      showError(`Error deleting provider: ${error.message}`);
    }
  };
  
  // Get status color
  const getStatusColor = (status) => {
    switch (status) {
      case 'operational':
        return 'success';
      case 'degraded':
        return 'warning';
      case 'error':
        return 'error';
      default:
        return 'default';
    }
  };
  
  // Get status icon
  const getStatusIcon = (status) => {
    switch (status) {
      case 'operational':
        return <CheckIcon fontSize="small" />;
      case 'degraded':
        return <WarningIcon fontSize="small" />;
      case 'error':
        return <ErrorIcon fontSize="small" />;
      default:
        return null;
    }
  };
  
  // Render provider cards
  const renderProviderCards = () => {
    return (
      <Grid container spacing={3}>
        {providers.map((provider) => (
          <Grid item xs={12} md={6} lg={4} key={provider.provider_id}>
            <Card sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
              <CardHeader
                title={provider.display_name || provider.provider_id}
                subheader={`ID: ${provider.provider_id}`}
                action={
                  <Chip
                    label={provider.status}
                    color={getStatusColor(provider.status)}
                    icon={getStatusIcon(provider.status)}
                    size="small"
                  />
                }
              />
              <CardContent sx={{ flexGrow: 1 }}>
                <Typography variant="body2" color="text.secondary" gutterBottom>
                  Provider Type: {provider.provider_type}
                </Typography>
                <Typography variant="body2" color="text.secondary" gutterBottom>
                  Last Checked: {new Date(provider.checked_at).toLocaleString()}
                </Typography>
                
                <Typography variant="subtitle2" gutterBottom sx={{ mt: 2 }}>
                  Models:
                </Typography>
                <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                  {provider.models.map((model) => (
                    <Chip key={model} label={model} size="small" variant="outlined" />
                  ))}
                </Box>
              </CardContent>
              <CardActions>
                <Button
                  size="small"
                  startIcon={<EditIcon />}
                  onClick={() => handleOpenEditDialog(provider)}
                >
                  Edit
                </Button>
                <Button
                  size="small"
                  color="error"
                  startIcon={<DeleteIcon />}
                  onClick={() => handleOpenDeleteConfirm(provider)}
                >
                  Delete
                </Button>
                <Button
                  size="small"
                  color="primary"
                  startIcon={testingProvider === provider.provider_id ? <CircularProgress size={16} /> : <RefreshIcon />}
                  onClick={() => handleTestProvider(provider.provider_id)}
                  disabled={testingProvider === provider.provider_id}
                >
                  Test
                </Button>
              </CardActions>
            </Card>
          </Grid>
        ))}
      </Grid>
    );
  };
  
  // Render provider table
  const renderProviderTable = () => {
    return (
      <TableContainer component={Paper}>
        <Table>
          <TableHead>
            <TableRow>
              <TableCell>ID</TableCell>
              <TableCell>Name</TableCell>
              <TableCell>Type</TableCell>
              <TableCell>Status</TableCell>
              <TableCell>Models</TableCell>
              <TableCell>Actions</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {providers.map((provider) => (
              <TableRow key={provider.provider_id}>
                <TableCell>{provider.provider_id}</TableCell>
                <TableCell>{provider.display_name || provider.provider_id}</TableCell>
                <TableCell>{provider.provider_type}</TableCell>
                <TableCell>
                  <Chip
                    label={provider.status}
                    color={getStatusColor(provider.status)}
                    icon={getStatusIcon(provider.status)}
                    size="small"
                  />
                </TableCell>
                <TableCell>
                  {provider.models.length > 0 ? (
                    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                      {provider.models.slice(0, 2).map((model) => (
                        <Chip key={model} label={model} size="small" variant="outlined" />
                      ))}
                      {provider.models.length > 2 && (
                        <Tooltip title={provider.models.slice(2).join(', ')}>
                          <Chip label={`+${provider.models.length - 2}`} size="small" />
                        </Tooltip>
                      )}
                    </Box>
                  ) : (
                    <Typography variant="body2" color="text.secondary">
                      No models
                    </Typography>
                  )}
                </TableCell>
                <TableCell>
                  <Box sx={{ display: 'flex', gap: 1 }}>
                    <IconButton
                      size="small"
                      onClick={() => handleOpenEditDialog(provider)}
                      title="Edit provider"
                    >
                      <EditIcon fontSize="small" />
                    </IconButton>
                    <IconButton
                      size="small"
                      color="error"
                      onClick={() => handleOpenDeleteConfirm(provider)}
                      title="Delete provider"
                    >
                      <DeleteIcon fontSize="small" />
                    </IconButton>
                    <IconButton
                      size="small"
                      onClick={() => handleTestProvider(provider.provider_id)}
                      disabled={testingProvider === provider.provider_id}
                      title="Test provider"
                    >
                      {testingProvider === provider.provider_id ? (
                        <CircularProgress size={16} />
                      ) : (
                        <RefreshIcon fontSize="small" />
                      )}
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
          LLM Provider Management
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
          >
            Add Provider
          </Button>
        </Box>
      </Box>
      
      {loading ? (
        <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
          <CircularProgress />
        </Box>
      ) : providers.length === 0 ? (
        <Paper sx={{ p: 3, textAlign: 'center' }}>
          <Typography variant="body1" gutterBottom>
            No LLM providers found.
          </Typography>
          <Button
            variant="contained"
            startIcon={<AddIcon />}
            onClick={handleOpenAddDialog}
            sx={{ mt: 2 }}
          >
            Add Provider
          </Button>
        </Paper>
      ) : (
        <Box sx={{ mb: 4 }}>
          {renderProviderTable()}
        </Box>
      )}
      
      {/* Provider Form Dialog */}
      <Dialog 
        open={dialogOpen} 
        onClose={handleDialogClose} 
        maxWidth="md" 
        fullWidth
        scroll="paper"
      >
        <DialogTitle>
          {dialogMode === 'add' ? 'Add LLM Provider' : `Edit Provider: ${selectedProvider?.display_name || selectedProvider?.provider_id}`}
        </DialogTitle>
        <DialogContent dividers>
          <Grid container spacing={3}>
            {/* Basic Information */}
            <Grid item xs={12}>
              <Typography variant="h6">Basic Information</Typography>
              <Divider sx={{ mt: 1, mb: 2 }} />
            </Grid>
            
            <Grid item xs={12} sm={6}>
              <TextField
                name="provider_id"
                label="Provider ID"
                value={formValues.provider_id}
                onChange={handleInputChange}
                fullWidth
                required
                margin="normal"
                disabled={dialogMode === 'edit'}
                helperText={formErrors.provider_id || "Unique identifier for this provider (lowercase, no spaces)"}
                error={!!formErrors.provider_id}
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
                helperText={formErrors.display_name || "Human-readable name for this provider"}
                error={!!formErrors.display_name}
              />
            </Grid>
            
            <Grid item xs={12} sm={6}>
              <TextField
                name="provider_type"
                label="Provider Type"
                value={formValues.provider_type}
                onChange={handleInputChange}
                select
                fullWidth
                margin="normal"
                helperText="Type of LLM provider"
              >
                {providerTypes.map((option) => (
                  <MenuItem key={option.value} value={option.value}>
                    {option.label}
                  </MenuItem>
                ))}
              </TextField>
            </Grid>
            
            <Grid item xs={12} sm={6}>
              <TextField
                name="description"
                label="Description"
                value={formValues.description}
                onChange={handleInputChange}
                fullWidth
                margin="normal"
                helperText="Optional description for this provider"
              />
            </Grid>
            
            {/* Connection Settings */}
            <Grid item xs={12}>
              <Typography variant="h6">Connection Settings</Typography>
              <Divider sx={{ mt: 1, mb: 2 }} />
            </Grid>
            
            <Grid item xs={12} sm={6}>
              <FormControl fullWidth margin="normal">
                <InputLabel>Authentication Type</InputLabel>
                <Select
                  name="auth_type"
                  value={formValues.connection_params.auth_type}
                  onChange={handleConnectionParamChange}
                  label="Authentication Type"
                >
                  {authTypes.map((option) => (
                    <MenuItem key={option.value} value={option.value}>
                      {option.label}
                    </MenuItem>
                  ))}
                </Select>
                <FormHelperText>Type of authentication to use</FormHelperText>
              </FormControl>
            </Grid>
            
            <Grid item xs={12} sm={6}>
              <TextField
                name="api_key"
                label="API Key"
                value={formValues.connection_params.api_key}
                onChange={handleConnectionParamChange}
                fullWidth
                margin="normal"
                type="password"
                helperText={formErrors.connection_params.api_key || "API key for authentication"}
                error={!!formErrors.connection_params.api_key}
              />
            </Grid>
            
            <Grid item xs={12} sm={6}>
              <TextField
                name="base_url"
                label="Base URL"
                value={formValues.connection_params.base_url}
                onChange={handleConnectionParamChange}
                fullWidth
                margin="normal"
                helperText={formErrors.connection_params.base_url || "Optional API endpoint URL"}
                error={!!formErrors.connection_params.base_url}
              />
            </Grid>
            
            {/* Request Settings */}
            <Grid item xs={12}>
              <Typography variant="h6">Request Settings</Typography>
              <Divider sx={{ mt: 1, mb: 2 }} />
            </Grid>
            
            <Grid item xs={12} sm={4}>
              <TextField
                name="timeout_seconds"
                label="Timeout (seconds)"
                value={formValues.request_settings.timeout_seconds}
                onChange={handleRequestSettingChange}
                fullWidth
                margin="normal"
                type="number"
                inputProps={{ min: 1, max: 300 }}
                helperText={formErrors.request_settings.timeout_seconds || "Request timeout in seconds"}
                error={!!formErrors.request_settings.timeout_seconds}
              />
            </Grid>
            
            <Grid item xs={12} sm={4}>
              <TextField
                name="retry_attempts"
                label="Retry Attempts"
                value={formValues.request_settings.retry_attempts}
                onChange={handleRequestSettingChange}
                fullWidth
                margin="normal"
                type="number"
                inputProps={{ min: 0, max: 10 }}
                helperText={formErrors.request_settings.retry_attempts || "Number of retry attempts on failure"}
                error={!!formErrors.request_settings.retry_attempts}
              />
            </Grid>
            
            <Grid item xs={12} sm={4}>
              <TextField
                name="rate_limit_rpm"
                label="Rate Limit (rpm)"
                value={formValues.request_settings.rate_limit_rpm}
                onChange={handleRequestSettingChange}
                fullWidth
                margin="normal"
                type="number"
                inputProps={{ min: 0 }}
                helperText={formErrors.request_settings.rate_limit_rpm || "Rate limit (requests per minute)"}
                error={!!formErrors.request_settings.rate_limit_rpm}
              />
            </Grid>
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button onClick={handleDialogClose}>Cancel</Button>
          <Button onClick={handleSubmit} variant="contained" color="primary">
            {dialogMode === 'add' ? 'Add Provider' : 'Save Changes'}
          </Button>
        </DialogActions>
      </Dialog>
      
      {/* Delete Confirmation Dialog */}
      <Dialog
        open={deleteConfirmOpen}
        onClose={() => setDeleteConfirmOpen(false)}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>Confirm Delete</DialogTitle>
        <DialogContent>
          <Typography variant="body1">
            Are you sure you want to delete the provider "{providerToDelete?.display_name || providerToDelete?.provider_id}"?
          </Typography>
          <Typography variant="body2" color="error" sx={{ mt: 2 }}>
            This action cannot be undone.
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDeleteConfirmOpen(false)}>Cancel</Button>
          <Button onClick={handleDeleteConfirm} variant="contained" color="error">
            Delete
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default ProviderManagement;
