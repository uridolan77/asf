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
  TableCell
} from '@mui/material';
import {
  Add as AddIcon,
  Edit as EditIcon,
  Delete as DeleteIcon,
  Refresh as RefreshIcon,
  Check as CheckIcon,
  Error as ErrorIcon,
  WarningAmber as WarningIcon
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
    enabled: true,
    connection_params: {
      api_key: '',
      base_url: '',
    },
    models: {}
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
    { value: 'custom', label: 'Custom' }
  ];
  
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
        showError(`Failed to load providers: ${result.error}`);
      }
    } catch (error) {
      console.error('Error loading providers:', error);
      showError(`Error loading providers: ${error.message}`);
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
    setFormValues({
      provider_id: '',
      provider_type: 'openai',
      display_name: '',
      enabled: true,
      connection_params: {
        api_key: '',
        base_url: '',
      },
      models: {}
    });
    setDialogOpen(true);
  };
  
  // Open edit dialog
  const handleOpenEditDialog = (provider) => {
    setDialogMode('edit');
    setSelectedProvider(provider);
    
    // Transform provider data to form values
    setFormValues({
      provider_id: provider.provider_id,
      provider_type: provider.provider_type,
      display_name: provider.display_name || provider.provider_id,
      enabled: true, // Assume it's enabled if it's in the list
      connection_params: {
        api_key: '************', // Don't show actual key
        base_url: '', // This needs to be fetched from the provider config
      },
      models: {} // This needs to be fetched from the provider config
    });
    
    setDialogOpen(true);
  };
  
  // Handle dialog close
  const handleDialogClose = () => {
    setDialogOpen(false);
    setFormValues({
      provider_id: '',
      provider_type: 'openai',
      display_name: '',
      enabled: true,
      connection_params: {
        api_key: '',
        base_url: '',
      },
      models: {}
    });
  };
  
  // Handle form input change
  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormValues((prev) => ({
      ...prev,
      [name]: value
    }));
  };
  
  // Handle connection params change
  const handleConnectionParamChange = (e) => {
    const { name, value } = e.target;
    setFormValues((prev) => ({
      ...prev,
      connection_params: {
        ...prev.connection_params,
        [name]: value
      }
    }));
  };
  
  // Handle form submit
  const handleSubmit = async () => {
    // Validate form
    if (!formValues.provider_id) {
      showError('Provider ID is required');
      return;
    }
    
    if (!formValues.display_name) {
      showError('Display name is required');
      return;
    }
    
    try {
      if (dialogMode === 'add') {
        // Create default models based on provider type
        let defaultModels = {};
        
        if (formValues.provider_type === 'openai') {
          defaultModels = {
            "gpt-4-turbo": {
              "display_name": "GPT-4 Turbo",
              "type": "chat",
              "capabilities": ["function_calling", "json_mode", "vision"],
              "context_window": 128000,
              "max_output_tokens": 4096
            },
            "gpt-3.5-turbo": {
              "display_name": "GPT-3.5 Turbo",
              "type": "chat",
              "capabilities": ["function_calling", "json_mode"],
              "context_window": 16385,
              "max_output_tokens": 4096
            }
          };
        } else if (formValues.provider_type === 'anthropic') {
          defaultModels = {
            "claude-3-opus-20240229": {
              "display_name": "Claude 3 Opus",
              "type": "chat",
              "capabilities": ["function_calling", "vision"],
              "context_window": 200000,
              "max_output_tokens": 4096
            },
            "claude-3-sonnet-20240229": {
              "display_name": "Claude 3 Sonnet",
              "type": "chat",
              "capabilities": ["function_calling", "vision"],
              "context_window": 200000,
              "max_output_tokens": 4096
            }
          };
        }
        
        // Create provider with default models or empty if not a known provider
        const createData = {
          ...formValues,
          models: defaultModels
        };
        
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
        // Update provider
        const result = await apiService.llm.updateProvider(
          formValues.provider_id,
          {
            display_name: formValues.display_name,
            connection_params: formValues.connection_params,
            enabled: formValues.enabled
          }
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
  
  // Test provider
  const handleTestProvider = async (providerId) => {
    setTestingProvider(providerId);
    
    try {
      const result = await apiService.llm.testProvider(providerId);
      
      if (result.success) {
        showSuccess(`Provider "${providerId}" test successful`);
      } else {
        showError(`Provider "${providerId}" test failed: ${result.error}`);
      }
    } catch (error) {
      console.error('Error testing provider:', error);
      showError(`Error testing provider: ${error.message}`);
    } finally {
      setTestingProvider(null);
    }
  };
  
  // Open delete confirm dialog
  const handleOpenDeleteConfirm = (provider) => {
    setProviderToDelete(provider);
    setDeleteConfirmOpen(true);
  };
  
  // Handle delete confirm
  const handleDeleteConfirm = async () => {
    if (!providerToDelete) return;
    
    try {
      const result = await apiService.llm.deleteProvider(providerToDelete.provider_id);
      
      if (result.success) {
        showSuccess(`Provider "${providerToDelete.display_name}" deleted successfully`);
        setDeleteConfirmOpen(false);
        setProviderToDelete(null);
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
      <Dialog open={dialogOpen} onClose={handleDialogClose} maxWidth="md" fullWidth>
        <DialogTitle>
          {dialogMode === 'add' ? 'Add LLM Provider' : `Edit Provider: ${selectedProvider?.display_name || selectedProvider?.provider_id}`}
        </DialogTitle>
        <DialogContent dividers>
          <Grid container spacing={3}>
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
                helperText="Unique identifier for this provider"
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
                helperText="Human-readable name for this provider"
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
            <Grid item xs={12}>
              <Divider sx={{ my: 2 }}>
                <Chip label="Connection Parameters" />
              </Divider>
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
                helperText="API key for authentication"
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
                helperText="Base URL for API calls (leave empty for default)"
              />
            </Grid>
            <Grid item xs={12}>
              <Alert severity="info" sx={{ mt: 2 }}>
                You can add models to this provider after saving it.
              </Alert>
            </Grid>
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button onClick={handleDialogClose}>Cancel</Button>
          <Button onClick={handleSubmit} variant="contained">
            {dialogMode === 'add' ? 'Add Provider' : 'Save Changes'}
          </Button>
        </DialogActions>
      </Dialog>
      
      {/* Delete Confirmation Dialog */}
      <Dialog open={deleteConfirmOpen} onClose={() => setDeleteConfirmOpen(false)}>
        <DialogTitle>Confirm Delete</DialogTitle>
        <DialogContent>
          <Typography variant="body1">
            Are you sure you want to delete the provider "{providerToDelete?.display_name || providerToDelete?.provider_id}"?
          </Typography>
          <Typography variant="body2" color="error" sx={{ mt: 2 }}>
            This action cannot be undone. All models associated with this provider will also be deleted.
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDeleteConfirmOpen(false)}>Cancel</Button>
          <Button onClick={handleDeleteConfirm} color="error" variant="contained">
            Delete Provider
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default ProviderManagement;