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
      console.log('Fetching LLM providers and models...');
      // Load providers first
      const providersResult = await apiService.llm.getProviders();

      if (providersResult.success) {
        console.log('Providers loaded successfully:', providersResult.data);
        setProviders(providersResult.data);

        // Then load models
        const modelsResult = await apiService.llm.getModels();

        if (modelsResult.success) {
          console.log('Models loaded successfully:', modelsResult.data);
          setModels(modelsResult.data || []);
        } else {
          console.error('Failed to load models:', modelsResult.error);
          showError(`Failed to load models: ${modelsResult.error}`);
          // Set empty models array to avoid undefined errors
          setModels([]);
        }
      } else {
        console.error('Failed to load providers:', providersResult.error);
        showError(`Failed to load providers: ${providersResult.error}`);
        // Set empty providers array to avoid undefined errors
        setProviders([]);
        setModels([]);
      }
    } catch (error) {
      console.error('Error loading data:', error);
      showError(`Error loading data: ${error.message}`);
      // Set empty arrays to avoid undefined errors
      setProviders([]);
      setModels([]);
    } finally {
      setLoading(false);
    }
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

    if (filteredModels.length === 0) {
      return (
        <Paper sx={{ p: 3, textAlign: 'center' }}>
          <Typography variant="body1" color="text.secondary">
            {models.length === 0 ?
              'No models found. Click "Add Model" to create a new model.' :
              'No models match the current filters. Try changing or clearing the filters.'}
          </Typography>
          {models.length === 0 && providers.length > 0 && (
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
      );
    }

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

      {providers.length === 0 && !loading && (
        <Alert severity="info" sx={{ mb: 3 }}>
          No LLM providers found. Please add a provider in the Providers tab before adding models.
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