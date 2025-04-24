import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Container,
  Paper,
  Breadcrumbs,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Button,
  IconButton,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Chip,
  Switch,
  FormControlLabel,
  Tooltip,
  CircularProgress,
  Grid,
  Divider,
  Alert
} from '@mui/material';
import {
  Link as RouterLink,
  useNavigate
} from 'react-router-dom';
import NavigateNextIcon from '@mui/icons-material/NavigateNext';
import HomeIcon from '@mui/icons-material/Home';
import AddIcon from '@mui/icons-material/Add';
import EditIcon from '@mui/icons-material/Edit';
import DeleteIcon from '@mui/icons-material/Delete';
import RefreshIcon from '@mui/icons-material/Refresh';
import VisibilityIcon from '@mui/icons-material/Visibility';
import VisibilityOffIcon from '@mui/icons-material/VisibilityOff';
import ContentCopyIcon from '@mui/icons-material/ContentCopy';

// For production use, you would use your actual API service
// import apiService from '../../../services/api';
import { useNotification } from '../../../context/NotificationContext';

/**
 * Provider API Keys Management Page
 * For managing LLM provider API keys and credentials
 */
const ProviderKeysPage = () => {
  const navigate = useNavigate();
  const { showSuccess, showError } = useNotification();
  
  // State
  const [loading, setLoading] = useState(true);
  const [apiKeys, setApiKeys] = useState([]);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [dialogMode, setDialogMode] = useState('add'); // 'add' or 'edit'
  const [selectedKey, setSelectedKey] = useState(null);
  const [refreshing, setRefreshing] = useState(false);
  const [visibleKeyIds, setVisibleKeyIds] = useState([]);
  const [formValues, setFormValues] = useState({
    id: '',
    name: '',
    provider_id: '',
    key_type: 'api_key',
    key_value: '',
    is_default: false,
    usage_limit: 0,
    expires_at: ''
  });

  // Provider options (would be fetched from API)
  const providerOptions = [
    { id: 'openai', name: 'OpenAI' },
    { id: 'anthropic', name: 'Anthropic' },
    { id: 'cohere', name: 'Cohere' },
    { id: 'google', name: 'Google AI' },
    { id: 'mistral', name: 'Mistral AI' },
    { id: 'azure_openai', name: 'Azure OpenAI' }
  ];

  // Key type options
  const keyTypeOptions = [
    { value: 'api_key', label: 'API Key' },
    { value: 'oauth_token', label: 'OAuth Token' },
    { value: 'bearer_token', label: 'Bearer Token' },
    { value: 'basic_auth', label: 'Basic Auth' },
    { value: 'custom', label: 'Custom' }
  ];

  // Sample API keys data - would be fetched from API
  const sampleApiKeys = [
    {
      id: 'key1',
      name: 'OpenAI Production',
      provider_id: 'openai',
      key_type: 'api_key',
      key_value: 'sk-xxxxxxxxxxxxxxxxxxxxxxxx',
      is_default: true,
      created_at: '2023-09-15T12:00:00Z',
      expires_at: '2024-09-15T12:00:00Z',
      usage_limit: 1000000,
      usage_count: 456789
    },
    {
      id: 'key2',
      name: 'OpenAI Development',
      provider_id: 'openai',
      key_type: 'api_key',
      key_value: 'sk-yyyyyyyyyyyyyyyyyyyyyyyy',
      is_default: false,
      created_at: '2023-10-20T12:00:00Z',
      expires_at: null,
      usage_limit: 500000,
      usage_count: 123456
    },
    {
      id: 'key3',
      name: 'Anthropic Production',
      provider_id: 'anthropic',
      key_type: 'api_key',
      key_value: 'sk-ant-xxxxxxxxxxxxxxxxxx',
      is_default: true,
      created_at: '2023-11-05T12:00:00Z',
      expires_at: '2024-11-05T12:00:00Z',
      usage_limit: 750000,
      usage_count: 234567
    },
    {
      id: 'key4',
      name: 'Google AI',
      provider_id: 'google',
      key_type: 'oauth_token',
      key_value: 'ya29.xxxxxxxxxxxxxxxxxxx',
      is_default: true,
      created_at: '2023-12-10T12:00:00Z',
      expires_at: '2024-01-10T12:00:00Z',
      usage_limit: 0,
      usage_count: 56789
    }
  ];

  // Load API keys
  useEffect(() => {
    const loadApiKeys = async () => {
      setLoading(true);
      
      try {
        // In production, use an actual API call
        // const result = await apiService.llm.getApiKeys();
        
        // For development, use sample data
        setTimeout(() => {
          setApiKeys(sampleApiKeys);
          setLoading(false);
        }, 500);
        
      } catch (error) {
        console.error('Error loading API keys:', error);
        showError('Failed to load API keys');
        setLoading(false);
      }
    };
    
    loadApiKeys();
  }, []);

  // Refresh API keys
  const handleRefresh = async () => {
    setRefreshing(true);
    
    try {
      // In production, use an actual API call
      // const result = await apiService.llm.getApiKeys();
      
      // For development, use sample data
      setTimeout(() => {
        setApiKeys(sampleApiKeys);
        setRefreshing(false);
      }, 500);
      
    } catch (error) {
      console.error('Error refreshing API keys:', error);
      showError('Failed to refresh API keys');
      setRefreshing(false);
    }
  };

  // Toggle key visibility
  const toggleKeyVisibility = (keyId) => {
    if (visibleKeyIds.includes(keyId)) {
      setVisibleKeyIds(visibleKeyIds.filter(id => id !== keyId));
    } else {
      setVisibleKeyIds([...visibleKeyIds, keyId]);
    }
  };

  // Format key for display
  const formatKeyForDisplay = (key, isVisible) => {
    if (isVisible) {
      return key.key_value;
    }
    return key.key_value.substring(0, 4) + '•'.repeat(Math.max(0, key.key_value.length - 8)) + key.key_value.substring(key.key_value.length - 4);
  };

  // Copy key to clipboard
  const copyKeyToClipboard = (key) => {
    navigator.clipboard.writeText(key.key_value)
      .then(() => {
        showSuccess('API Key copied to clipboard');
      })
      .catch((err) => {
        console.error('Failed to copy key:', err);
        showError('Failed to copy API key to clipboard');
      });
  };

  // Open add dialog
  const handleOpenAddDialog = () => {
    setDialogMode('add');
    setSelectedKey(null);
    setFormValues({
      id: '',
      name: '',
      provider_id: '',
      key_type: 'api_key',
      key_value: '',
      is_default: false,
      usage_limit: 0,
      expires_at: ''
    });
    setDialogOpen(true);
  };

  // Open edit dialog
  const handleOpenEditDialog = (key) => {
    setDialogMode('edit');
    setSelectedKey(key);
    
    // Transform key data to form values
    setFormValues({
      id: key.id,
      name: key.name,
      provider_id: key.provider_id,
      key_type: key.key_type,
      key_value: key.key_value,
      is_default: key.is_default,
      usage_limit: key.usage_limit || 0,
      expires_at: key.expires_at ? new Date(key.expires_at).toISOString().split('T')[0] : ''
    });
    
    setDialogOpen(true);
  };

  // Open delete dialog
  const handleOpenDeleteDialog = (key) => {
    setSelectedKey(key);
    setDeleteDialogOpen(true);
  };

  // Handle form input change
  const handleInputChange = (e) => {
    const { name, value, checked } = e.target;
    setFormValues(prev => ({
      ...prev,
      [name]: name === 'is_default' ? checked : value
    }));
  };

  // Handle dialog close
  const handleDialogClose = () => {
    setDialogOpen(false);
    setDeleteDialogOpen(false);
    setSelectedKey(null);
  };

  // Handle form submit
  const handleSubmit = async () => {
    // Validate form
    if (!formValues.name) {
      showError('Name is required');
      return;
    }
    
    if (!formValues.provider_id) {
      showError('Provider is required');
      return;
    }
    
    if (!formValues.key_value) {
      showError('API Key value is required');
      return;
    }
    
    try {
      if (dialogMode === 'add') {
        // In production, use an actual API call
        // const result = await apiService.llm.createApiKey(formValues);
        
        // For development, simulate API call
        setTimeout(() => {
          // Add new key to state with generated ID
          const newKey = {
            ...formValues,
            id: `key${Date.now()}`,
            created_at: new Date().toISOString(),
            usage_count: 0
          };
          
          setApiKeys([...apiKeys, newKey]);
          showSuccess(`API Key "${formValues.name}" created successfully`);
          handleDialogClose();
        }, 500);
        
      } else {
        // In production, use an actual API call
        // const result = await apiService.llm.updateApiKey(formValues.id, formValues);
        
        // For development, simulate API call
        setTimeout(() => {
          // Update key in state
          setApiKeys(apiKeys.map(key => key.id === formValues.id ? { ...key, ...formValues } : key));
          showSuccess(`API Key "${formValues.name}" updated successfully`);
          handleDialogClose();
        }, 500);
      }
    } catch (error) {
      console.error('Error saving API key:', error);
      showError(`Error saving API key: ${error.message}`);
    }
  };

  // Handle delete confirm
  const handleDeleteConfirm = async () => {
    if (!selectedKey) return;
    
    try {
      // In production, use an actual API call
      // const result = await apiService.llm.deleteApiKey(selectedKey.id);
      
      // For development, simulate API call
      setTimeout(() => {
        // Remove key from state
        setApiKeys(apiKeys.filter(key => key.id !== selectedKey.id));
        showSuccess(`API Key "${selectedKey.name}" deleted successfully`);
        handleDialogClose();
      }, 500);
      
    } catch (error) {
      console.error('Error deleting API key:', error);
      showError(`Error deleting API key: ${error.message}`);
    }
  };

  // Get provider name by ID
  const getProviderName = (providerId) => {
    const provider = providerOptions.find(p => p.id === providerId);
    return provider ? provider.name : providerId;
  };

  // Format date for display
  const formatDate = (dateString) => {
    if (!dateString) return 'Never';
    return new Date(dateString).toLocaleDateString();
  };

  // Get usage percentage
  const getUsagePercentage = (key) => {
    if (!key.usage_limit || key.usage_limit === 0) return 0;
    return Math.min(100, Math.round((key.usage_count / key.usage_limit) * 100));
  };

  // Get usage color
  const getUsageColor = (percentage) => {
    if (percentage >= 90) return 'error';
    if (percentage >= 70) return 'warning';
    return 'success';
  };

  return (
    <Container maxWidth="xl">
      <Box sx={{ pt: 2, pb: 4 }}>
        {/* Breadcrumbs navigation */}
        <Breadcrumbs 
          separator={<NavigateNextIcon fontSize="small" />} 
          aria-label="breadcrumb"
          sx={{ mb: 2 }}
        >
          <RouterLink 
            to="/dashboard" 
            style={{ display: 'flex', alignItems: 'center', textDecoration: 'none', color: 'inherit' }}
          >
            <HomeIcon sx={{ mr: 0.5 }} fontSize="small" />
            Dashboard
          </RouterLink>
          <RouterLink 
            to="/llm/dashboard" 
            style={{ textDecoration: 'none', color: 'inherit' }}
          >
            LLM Management
          </RouterLink>
          <RouterLink 
            to="/llm/providers" 
            style={{ textDecoration: 'none', color: 'inherit' }}
          >
            Providers
          </RouterLink>
          <Typography color="text.primary">API Keys</Typography>
        </Breadcrumbs>
        
        {/* Page header */}
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
          <Box>
            <Typography variant="h4" component="h1" gutterBottom>
              Provider API Keys
            </Typography>
            <Typography variant="body1" color="text.secondary">
              Manage API keys and authentication credentials for your LLM providers
            </Typography>
          </Box>
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
              color="primary"
              startIcon={<AddIcon />}
              onClick={handleOpenAddDialog}
            >
              Add API Key
            </Button>
          </Box>
        </Box>
        
        {/* Main content */}
        <Paper sx={{ mt: 3 }}>
          {loading ? (
            <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
              <CircularProgress />
            </Box>
          ) : apiKeys.length === 0 ? (
            <Box sx={{ p: 4, textAlign: 'center' }}>
              <Typography variant="body1" gutterBottom>
                No API keys found. Add a new API key to get started.
              </Typography>
              <Button
                variant="contained"
                startIcon={<AddIcon />}
                onClick={handleOpenAddDialog}
                sx={{ mt: 2 }}
              >
                Add API Key
              </Button>
            </Box>
          ) : (
            <TableContainer>
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell>Name</TableCell>
                    <TableCell>Provider</TableCell>
                    <TableCell>API Key</TableCell>
                    <TableCell>Type</TableCell>
                    <TableCell>Created</TableCell>
                    <TableCell>Expires</TableCell>
                    <TableCell>Usage</TableCell>
                    <TableCell width="150">Actions</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {apiKeys.map((key) => {
                    const isVisible = visibleKeyIds.includes(key.id);
                    const usagePercentage = getUsagePercentage(key);
                    
                    return (
                      <TableRow key={key.id}>
                        <TableCell>
                          <Box sx={{ display: 'flex', alignItems: 'center' }}>
                            <Typography variant="body2">{key.name}</Typography>
                            {key.is_default && (
                              <Chip
                                label="Default"
                                size="small"
                                color="primary"
                                sx={{ ml: 1 }}
                              />
                            )}
                          </Box>
                        </TableCell>
                        <TableCell>{getProviderName(key.provider_id)}</TableCell>
                        <TableCell>
                          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                            <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                              {formatKeyForDisplay(key, isVisible)}
                            </Typography>
                            <Tooltip title={isVisible ? "Hide API Key" : "Show API Key"}>
                              <IconButton
                                size="small"
                                onClick={() => toggleKeyVisibility(key.id)}
                              >
                                {isVisible ? <VisibilityOffIcon fontSize="small" /> : <VisibilityIcon fontSize="small" />}
                              </IconButton>
                            </Tooltip>
                            <Tooltip title="Copy API Key">
                              <IconButton
                                size="small"
                                onClick={() => copyKeyToClipboard(key)}
                              >
                                <ContentCopyIcon fontSize="small" />
                              </IconButton>
                            </Tooltip>
                          </Box>
                        </TableCell>
                        <TableCell>
                          <Chip
                            label={key.key_type.replace('_', ' ')}
                            size="small"
                            variant="outlined"
                          />
                        </TableCell>
                        <TableCell>{formatDate(key.created_at)}</TableCell>
                        <TableCell>
                          {key.expires_at ? (
                            <Tooltip title={new Date(key.expires_at).toLocaleString()}>
                              <Typography variant="body2">{formatDate(key.expires_at)}</Typography>
                            </Tooltip>
                          ) : (
                            'Never'
                          )}
                        </TableCell>
                        <TableCell>
                          {key.usage_limit > 0 ? (
                            <Tooltip title={`${key.usage_count} / ${key.usage_limit} (${usagePercentage}%)`}>
                              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                <Box sx={{ width: '100%', mr: 1 }}>
                                  <LinearProgress
                                    variant="determinate"
                                    value={usagePercentage}
                                    color={getUsageColor(usagePercentage)}
                                  />
                                </Box>
                                <Typography variant="body2" color="text.secondary">
                                  {usagePercentage}%
                                </Typography>
                              </Box>
                            </Tooltip>
                          ) : (
                            <Typography variant="body2">
                              {key.usage_count} / Unlimited
                            </Typography>
                          )}
                        </TableCell>
                        <TableCell>
                          <Box sx={{ display: 'flex', gap: 1 }}>
                            <IconButton
                              size="small"
                              onClick={() => handleOpenEditDialog(key)}
                              title="Edit API key"
                            >
                              <EditIcon fontSize="small" />
                            </IconButton>
                            <IconButton
                              size="small"
                              color="error"
                              onClick={() => handleOpenDeleteDialog(key)}
                              title="Delete API key"
                            >
                              <DeleteIcon fontSize="small" />
                            </IconButton>
                          </Box>
                        </TableCell>
                      </TableRow>
                    );
                  })}
                </TableBody>
              </Table>
            </TableContainer>
          )}
        </Paper>
        
        {/* Add/Edit API Key Dialog */}
        <Dialog open={dialogOpen} onClose={handleDialogClose} maxWidth="md" fullWidth>
          <DialogTitle>
            {dialogMode === 'add' ? 'Add API Key' : `Edit API Key: ${selectedKey?.name}`}
          </DialogTitle>
          <DialogContent dividers>
            <Grid container spacing={3}>
              {/* Basic Info */}
              <Grid item xs={12}>
                <Typography variant="h6">Key Information</Typography>
                <Divider sx={{ mt: 1, mb: 2 }} />
              </Grid>
              
              <Grid item xs={12} sm={6}>
                <TextField
                  name="name"
                  label="Key Name"
                  value={formValues.name}
                  onChange={handleInputChange}
                  fullWidth
                  required
                  margin="normal"
                  helperText="A descriptive name for this API key"
                />
              </Grid>
              
              <Grid item xs={12} sm={6}>
                <FormControl fullWidth margin="normal" required>
                  <InputLabel>Provider</InputLabel>
                  <Select
                    name="provider_id"
                    value={formValues.provider_id}
                    onChange={handleInputChange}
                    label="Provider"
                  >
                    {providerOptions.map((provider) => (
                      <MenuItem key={provider.id} value={provider.id}>
                        {provider.name}
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>
              </Grid>
              
              <Grid item xs={12} sm={6}>
                <FormControl fullWidth margin="normal">
                  <InputLabel>Key Type</InputLabel>
                  <Select
                    name="key_type"
                    value={formValues.key_type}
                    onChange={handleInputChange}
                    label="Key Type"
                  >
                    {keyTypeOptions.map((option) => (
                      <MenuItem key={option.value} value={option.value}>
                        {option.label}
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>
              </Grid>
              
              <Grid item xs={12} sm={6}>
                <TextField
                  name="key_value"
                  label="API Key Value"
                  value={formValues.key_value}
                  onChange={handleInputChange}
                  fullWidth
                  required
                  margin="normal"
                  type="password"
                  helperText="The actual API key or token value"
                />
              </Grid>
              
              <Grid item xs={12} sm={6}>
                <TextField
                  name="usage_limit"
                  label="Usage Limit"
                  value={formValues.usage_limit}
                  onChange={handleInputChange}
                  fullWidth
                  type="number"
                  margin="normal"
                  inputProps={{ min: 0 }}
                  helperText="Maximum number of requests (0 for unlimited)"
                />
              </Grid>
              
              <Grid item xs={12} sm={6}>
                <TextField
                  name="expires_at"
                  label="Expiration Date"
                  value={formValues.expires_at}
                  onChange={handleInputChange}
                  fullWidth
                  type="date"
                  margin="normal"
                  InputLabelProps={{ shrink: true }}
                  helperText="Leave empty for no expiration"
                />
              </Grid>
              
              <Grid item xs={12}>
                <FormControlLabel
                  control={
                    <Switch
                      name="is_default"
                      checked={formValues.is_default}
                      onChange={handleInputChange}
                      color="primary"
                    />
                  }
                  label="Set as default key for this provider"
                />
              </Grid>
              
              <Grid item xs={12}>
                <Alert severity="warning" sx={{ mt: 2 }}>
                  API keys grant access to LLM services with potential usage costs. 
                  Store them securely and monitor for unauthorized use.
                </Alert>
              </Grid>
            </Grid>
          </DialogContent>
          <DialogActions>
            <Button onClick={handleDialogClose}>Cancel</Button>
            <Button onClick={handleSubmit} variant="contained" color="primary">
              {dialogMode === 'add' ? 'Add Key' : 'Save Changes'}
            </Button>
          </DialogActions>
        </Dialog>
        
        {/* Delete Confirmation Dialog */}
        <Dialog open={deleteDialogOpen} onClose={handleDialogClose}>
          <DialogTitle>Confirm Delete</DialogTitle>
          <DialogContent>
            <Typography variant="body1">
              Are you sure you want to delete the API key "{selectedKey?.name}"?
            </Typography>
            <Typography variant="body2" color="error" sx={{ mt: 2 }}>
              This action cannot be undone. Any services using this key will no longer function.
            </Typography>
          </DialogContent>
          <DialogActions>
            <Button onClick={handleDialogClose}>Cancel</Button>
            <Button onClick={handleDeleteConfirm} color="error" variant="contained">
              Delete Key
            </Button>
          </DialogActions>
        </Dialog>
      </Box>
    </Container>
  );
};

export default ProviderKeysPage;