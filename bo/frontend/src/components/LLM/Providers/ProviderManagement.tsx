import React, { useState } from 'react';
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
  FormControlLabel,
  Switch,
  FormControl,
  InputLabel,
  Select,
  SelectChangeEvent
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
import { useLLMProviders, LLMProvider, CreateProviderParams, UpdateProviderParams } from '../../../hooks/useLLMProviders';
import { useFeatureFlags } from '../../../context/FeatureFlagContext';
import { ButtonLoader } from '../../UI/LoadingIndicators';

/**
 * Provider management component
 * Allows adding, editing, and removing LLM providers
 */
const ProviderManagement: React.FC = () => {
  const { showSuccess, showError } = useNotification();
  const { isEnabled } = useFeatureFlags();
  const useMockData = isEnabled('useMockData');
  
  // State
  const [dialogOpen, setDialogOpen] = useState<boolean>(false);
  const [dialogMode, setDialogMode] = useState<'add' | 'edit'>('add');
  const [selectedProvider, setSelectedProvider] = useState<LLMProvider | null>(null);
  const [formValues, setFormValues] = useState<{
    provider_id: string;
    provider_type: string;
    display_name: string;
    enabled: boolean;
    connection_params: {
      api_key: string;
      base_url: string;
      [key: string]: string;
    };
  }>({
    provider_id: '',
    provider_type: 'openai',
    display_name: '',
    enabled: true,
    connection_params: {
      api_key: '',
      base_url: '',
    }
  });
  const [deleteConfirmOpen, setDeleteConfirmOpen] = useState<boolean>(false);
  const [providerToDelete, setProviderToDelete] = useState<LLMProvider | null>(null);
  
  // Use the LLM providers hook
  const {
    providers,
    isLoadingProviders,
    isErrorProviders,
    errorProviders,
    refetchProviders,
    createProvider,
    updateProvider,
    deleteProvider,
    testProvider,
    providerTypes
  } = useLLMProviders();

  // Get the mutation functions
  const { mutate: createMutate, isPending: isCreating } = createProvider();
  const { mutate: updateMutate, isPending: isUpdating } = updateProvider();
  const { mutate: deleteMutate, isPending: isDeleting } = deleteProvider();
  const { mutate: testMutate, isPending: isTesting } = testProvider();
  
  // Provider type options
  const providerTypeOptions = providerTypes.length > 0 
    ? providerTypes.map(type => ({ value: type.id, label: type.name }))
    : [
        { value: 'openai', label: 'OpenAI' },
        { value: 'anthropic', label: 'Anthropic' },
        { value: 'local', label: 'Local (Ollama)' },
        { value: 'google', label: 'Google AI' },
        { value: 'cohere', label: 'Cohere' },
        { value: 'mistral', label: 'Mistral AI' },
        { value: 'azure_openai', label: 'Azure OpenAI' },
        { value: 'custom', label: 'Custom' }
      ];
  
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
      }
    });
    setDialogOpen(true);
  };
  
  // Open edit dialog
  const handleOpenEditDialog = (provider: LLMProvider) => {
    setDialogMode('edit');
    setSelectedProvider(provider);
    
    // Transform provider data to form values
    setFormValues({
      provider_id: provider.id,
      provider_type: provider.provider_type,
      display_name: provider.display_name || provider.name,
      enabled: provider.enabled,
      connection_params: {
        api_key: '************', // Don't show actual key
        base_url: provider.connection_params.base_url || '',
        ...provider.connection_params
      }
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
      }
    });
  };
  
  // Handle form input change
  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setFormValues((prev) => ({
      ...prev,
      [name]: value
    }));
  };

  // Handle select change
  const handleSelectChange = (e: SelectChangeEvent) => {
    const { name, value } = e.target;
    setFormValues((prev) => ({
      ...prev,
      [name]: value
    }));
  };

  // Handle switch change
  const handleSwitchChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, checked } = e.target;
    setFormValues((prev) => ({
      ...prev,
      [name]: checked
    }));
  };
  
  // Handle connection params change
  const handleConnectionParamChange = (e: React.ChangeEvent<HTMLInputElement>) => {
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
  const handleSubmit = () => {
    // Validate form
    if (!formValues.provider_id) {
      showError('Provider ID is required');
      return;
    }
    
    if (!formValues.display_name) {
      showError('Display name is required');
      return;
    }
    
    if (dialogMode === 'add') {
      // Create provider
      const createData: CreateProviderParams = {
        provider_type: formValues.provider_type,
        display_name: formValues.display_name,
        connection_params: formValues.connection_params,
        enabled: formValues.enabled
      };
      
      createMutate(createData, {
        onSuccess: () => {
          handleDialogClose();
        }
      });
    } else {
      // Update provider
      const updateData: UpdateProviderParams = {
        display_name: formValues.display_name,
        connection_params: formValues.connection_params,
        enabled: formValues.enabled
      };
      
      updateMutate({ providerId: formValues.provider_id, params: updateData }, {
        onSuccess: () => {
          handleDialogClose();
        }
      });
    }
  };
  
  // Test provider
  const handleTestProvider = (providerId: string) => {
    testMutate({ provider_id: providerId });
  };
  
  // Open delete confirm dialog
  const handleOpenDeleteConfirm = (provider: LLMProvider) => {
    setProviderToDelete(provider);
    setDeleteConfirmOpen(true);
  };
  
  // Handle delete confirm
  const handleDeleteConfirm = () => {
    if (!providerToDelete) return;
    
    deleteMutate(providerToDelete.id, {
      onSuccess: () => {
        setDeleteConfirmOpen(false);
        setProviderToDelete(null);
      }
    });
  };
  
  // Get status color
  const getStatusColor = (status?: string) => {
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
  const getStatusIcon = (status?: string) => {
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
          <Grid item xs={12} md={6} lg={4} key={provider.id}>
            <Card sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
              <CardHeader
                title={provider.display_name || provider.name}
                subheader={`ID: ${provider.id}`}
                action={
                  <Chip
                    label={provider.status || 'Unknown'}
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
                  Last Updated: {new Date(provider.updated_at).toLocaleString()}
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
                  startIcon={isTesting ? <ButtonLoader size={16} /> : <RefreshIcon />}
                  onClick={() => handleTestProvider(provider.id)}
                  disabled={isTesting}
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
              <TableRow key={provider.id}>
                <TableCell>{provider.id}</TableCell>
                <TableCell>{provider.display_name || provider.name}</TableCell>
                <TableCell>{provider.provider_type}</TableCell>
                <TableCell>
                  <Chip
                    label={provider.status || 'Unknown'}
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
                      onClick={() => handleTestProvider(provider.id)}
                      title="Test provider"
                      disabled={isTesting}
                    >
                      {isTesting ? <CircularProgress size={16} /> : <RefreshIcon fontSize="small" />}
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
      {useMockData && (
        <Alert severity="info" sx={{ mb: 2 }}>
          Using mock data. Toggle the "Use Mock Data" feature flag to use real API data.
        </Alert>
      )}
      
      <Box sx={{ mb: 3, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Typography variant="h5">LLM Providers</Typography>
        <Box sx={{ display: 'flex', gap: 1 }}>
          <Button
            variant="outlined"
            startIcon={isLoadingProviders ? <ButtonLoader size={20} /> : <RefreshIcon />}
            onClick={() => refetchProviders()}
            disabled={isLoadingProviders}
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
      
      {isErrorProviders && (
        <Alert severity="error" sx={{ mb: 3 }}>
          Error loading providers: {errorProviders?.message}
        </Alert>
      )}
      
      {isLoadingProviders ? (
        <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
          <CircularProgress />
        </Box>
      ) : providers.length === 0 ? (
        <Paper sx={{ p: 4, textAlign: 'center' }}>
          <Typography variant="h6" gutterBottom>No providers found</Typography>
          <Typography variant="body2" color="text.secondary" paragraph>
            Add a provider to get started with LLM integration.
          </Typography>
          <Button
            variant="contained"
            startIcon={<AddIcon />}
            onClick={handleOpenAddDialog}
          >
            Add Provider
          </Button>
        </Paper>
      ) : (
        renderProviderCards()
      )}
      
      {/* Add/Edit Provider Dialog */}
      <Dialog open={dialogOpen} onClose={handleDialogClose} maxWidth="md" fullWidth>
        <DialogTitle>
          {dialogMode === 'add' ? 'Add Provider' : 'Edit Provider'}
        </DialogTitle>
        <DialogContent dividers>
          <Grid container spacing={3}>
            {dialogMode === 'add' && (
              <Grid item xs={12} md={6}>
                <TextField
                  fullWidth
                  label="Provider ID"
                  name="provider_id"
                  value={formValues.provider_id}
                  onChange={handleInputChange}
                  required
                  helperText="Unique identifier for this provider"
                />
              </Grid>
            )}
            
            <Grid item xs={12} md={dialogMode === 'add' ? 6 : 12}>
              <TextField
                fullWidth
                label="Display Name"
                name="display_name"
                value={formValues.display_name}
                onChange={handleInputChange}
                required
                helperText="Human-readable name for this provider"
              />
            </Grid>
            
            {dialogMode === 'add' && (
              <Grid item xs={12}>
                <FormControl fullWidth>
                  <InputLabel id="provider-type-label">Provider Type</InputLabel>
                  <Select
                    labelId="provider-type-label"
                    name="provider_type"
                    value={formValues.provider_type}
                    onChange={handleSelectChange}
                    label="Provider Type"
                  >
                    {providerTypeOptions.map((option) => (
                      <MenuItem key={option.value} value={option.value}>
                        {option.label}
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>
              </Grid>
            )}
            
            <Grid item xs={12}>
              <Typography variant="subtitle1" gutterBottom>
                Connection Parameters
              </Typography>
              <Divider sx={{ mb: 2 }} />
              
              <Grid container spacing={2}>
                <Grid item xs={12}>
                  <TextField
                    fullWidth
                    label="API Key"
                    name="api_key"
                    value={formValues.connection_params.api_key}
                    onChange={handleConnectionParamChange}
                    type="password"
                    required
                    helperText="API key for authentication (stored securely)"
                  />
                </Grid>
                
                <Grid item xs={12}>
                  <TextField
                    fullWidth
                    label="Base URL (Optional)"
                    name="base_url"
                    value={formValues.connection_params.base_url}
                    onChange={handleConnectionParamChange}
                    helperText="Custom base URL for API requests (leave empty for default)"
                  />
                </Grid>
              </Grid>
            </Grid>
            
            <Grid item xs={12}>
              <FormControlLabel
                control={
                  <Switch
                    checked={formValues.enabled}
                    onChange={handleSwitchChange}
                    name="enabled"
                  />
                }
                label="Enable Provider"
              />
            </Grid>
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button onClick={handleDialogClose}>Cancel</Button>
          <Button
            variant="contained"
            onClick={handleSubmit}
            disabled={isCreating || isUpdating}
            startIcon={isCreating || isUpdating ? <ButtonLoader size={20} /> : null}
          >
            {dialogMode === 'add' ? 'Add Provider' : 'Update Provider'}
          </Button>
        </DialogActions>
      </Dialog>
      
      {/* Delete Confirmation Dialog */}
      <Dialog open={deleteConfirmOpen} onClose={() => setDeleteConfirmOpen(false)}>
        <DialogTitle>Confirm Delete</DialogTitle>
        <DialogContent>
          <Typography>
            Are you sure you want to delete the provider "{providerToDelete?.display_name || providerToDelete?.name}"?
            This action cannot be undone.
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDeleteConfirmOpen(false)}>Cancel</Button>
          <Button
            color="error"
            onClick={handleDeleteConfirm}
            disabled={isDeleting}
            startIcon={isDeleting ? <ButtonLoader size={20} /> : null}
          >
            Delete
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default ProviderManagement;
