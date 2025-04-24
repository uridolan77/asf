import React, { useState, useEffect } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  FormControlLabel,
  Switch,
  Box,
  Typography,
  Divider,
  Alert,
  CircularProgress,
  Tabs,
  Tab
} from '@mui/material';

import { useNotification } from '../../../context/NotificationContext';
import apiService from '../../../services/api';
import StdioConfig from './TransportConfig/StdioConfig';
import GrpcConfig from './TransportConfig/GrpcConfig';
import HttpConfig from './TransportConfig/HttpConfig';

/**
 * MCP Provider Configuration Dialog
 * 
 * This component provides a dialog for adding or editing MCP providers,
 * with support for different transport types (stdio, gRPC, HTTP).
 */
const MCPConfigDialog = ({ open, mode = 'add', provider = null, onClose, onSave }) => {
  const { showSuccess, showError } = useNotification();
  
  // State
  const [formValues, setFormValues] = useState({
    provider_id: '',
    display_name: '',
    transport_type: 'stdio',
    enable_streaming: true,
    timeout_seconds: 60,
    max_retries: 3,
    models: {},
    enabled: true,
    transport_config: {
      stdio: {
        command: 'npx',
        args: ['@anthropic/mcp-starter', '--no-color'],
        env: {},
        cwd: null
      },
      grpc: {
        endpoint: 'localhost:50051',
        use_tls: false,
        ca_cert: null,
        client_cert: null,
        client_key: null,
        metadata: {}
      },
      http: {
        base_url: 'https://api.example.com/mcp',
        headers: {
          'Content-Type': 'application/json'
        },
        verify_ssl: true
      }
    }
  });
  
  const [activeTab, setActiveTab] = useState(0);
  const [saving, setSaving] = useState(false);
  const [validationErrors, setValidationErrors] = useState({});
  
  // Initialize form values when provider changes
  useEffect(() => {
    if (provider && mode === 'edit') {
      // Deep copy to avoid modifying the original object
      const providerCopy = JSON.parse(JSON.stringify(provider));
      
      setFormValues({
        ...formValues,
        provider_id: providerCopy.provider_id,
        display_name: providerCopy.display_name || '',
        transport_type: providerCopy.transport_type || 'stdio',
        enable_streaming: providerCopy.enable_streaming !== false,
        timeout_seconds: providerCopy.timeout_seconds || 60,
        max_retries: providerCopy.max_retries || 3,
        models: providerCopy.models || {},
        enabled: providerCopy.enabled !== false,
        transport_config: {
          ...formValues.transport_config,
          ...(providerCopy.transport_config || {})
        }
      });
      
      // Set active tab based on transport type
      switch (providerCopy.transport_type) {
        case 'stdio':
          setActiveTab(0);
          break;
        case 'grpc':
          setActiveTab(1);
          break;
        case 'http':
          setActiveTab(2);
          break;
        default:
          setActiveTab(0);
      }
    } else {
      // Reset form for new provider
      setFormValues({
        provider_id: '',
        display_name: '',
        transport_type: 'stdio',
        enable_streaming: true,
        timeout_seconds: 60,
        max_retries: 3,
        models: {},
        enabled: true,
        transport_config: {
          stdio: {
            command: 'npx',
            args: ['@anthropic/mcp-starter', '--no-color'],
            env: {},
            cwd: null
          },
          grpc: {
            endpoint: 'localhost:50051',
            use_tls: false,
            ca_cert: null,
            client_cert: null,
            client_key: null,
            metadata: {}
          },
          http: {
            base_url: 'https://api.example.com/mcp',
            headers: {
              'Content-Type': 'application/json'
            },
            verify_ssl: true
          }
        }
      });
      setActiveTab(0);
    }
    
    // Clear validation errors
    setValidationErrors({});
  }, [provider, mode, open]);
  
  // Handle input change
  const handleInputChange = (e) => {
    const { name, value, type, checked } = e.target;
    
    setFormValues({
      ...formValues,
      [name]: type === 'checkbox' ? checked : value
    });
    
    // Clear validation error for this field
    if (validationErrors[name]) {
      setValidationErrors({
        ...validationErrors,
        [name]: null
      });
    }
  };
  
  // Handle transport type change
  const handleTransportTypeChange = (e) => {
    const transportType = e.target.value;
    
    setFormValues({
      ...formValues,
      transport_type: transportType
    });
    
    // Set active tab based on transport type
    switch (transportType) {
      case 'stdio':
        setActiveTab(0);
        break;
      case 'grpc':
        setActiveTab(1);
        break;
      case 'http':
        setActiveTab(2);
        break;
      default:
        setActiveTab(0);
    }
  };
  
  // Handle tab change
  const handleTabChange = (event, newValue) => {
    setActiveTab(newValue);
    
    // Update transport type based on tab
    let transportType = 'stdio';
    switch (newValue) {
      case 0:
        transportType = 'stdio';
        break;
      case 1:
        transportType = 'grpc';
        break;
      case 2:
        transportType = 'http';
        break;
      default:
        transportType = 'stdio';
    }
    
    setFormValues({
      ...formValues,
      transport_type: transportType
    });
  };
  
  // Update transport config
  const updateTransportConfig = (type, config) => {
    setFormValues({
      ...formValues,
      transport_config: {
        ...formValues.transport_config,
        [type]: config
      }
    });
  };
  
  // Validate form
  const validateForm = () => {
    const errors = {};
    
    if (!formValues.provider_id) {
      errors.provider_id = 'Provider ID is required';
    } else if (!/^[a-z0-9_-]+$/i.test(formValues.provider_id)) {
      errors.provider_id = 'Provider ID can only contain letters, numbers, underscores, and hyphens';
    }
    
    if (!formValues.display_name) {
      errors.display_name = 'Display name is required';
    }
    
    if (formValues.timeout_seconds <= 0) {
      errors.timeout_seconds = 'Timeout must be greater than 0';
    }
    
    if (formValues.max_retries < 0) {
      errors.max_retries = 'Max retries cannot be negative';
    }
    
    // Validate transport-specific config
    switch (formValues.transport_type) {
      case 'stdio':
        if (!formValues.transport_config.stdio.command) {
          errors.stdio_command = 'Command is required';
        }
        break;
      case 'grpc':
        if (!formValues.transport_config.grpc.endpoint) {
          errors.grpc_endpoint = 'Endpoint is required';
        }
        break;
      case 'http':
        if (!formValues.transport_config.http.base_url) {
          errors.http_base_url = 'Base URL is required';
        } else if (!/^https?:\/\/.+/.test(formValues.transport_config.http.base_url)) {
          errors.http_base_url = 'Base URL must be a valid URL starting with http:// or https://';
        }
        break;
      default:
        break;
    }
    
    setValidationErrors(errors);
    return Object.keys(errors).length === 0;
  };
  
  // Handle save
  const handleSave = async () => {
    if (!validateForm()) {
      return;
    }
    
    setSaving(true);
    
    try {
      // Prepare transport config based on selected type
      const transportConfig = formValues.transport_config[formValues.transport_type];
      
      // Prepare request data
      const requestData = {
        provider_id: formValues.provider_id,
        display_name: formValues.display_name,
        transport_config: {
          transport_type: formValues.transport_type,
          ...transportConfig
        },
        enable_streaming: formValues.enable_streaming,
        timeout_seconds: parseInt(formValues.timeout_seconds, 10),
        max_retries: parseInt(formValues.max_retries, 10),
        models: formValues.models,
        enabled: formValues.enabled
      };
      
      let result;
      
      if (mode === 'add') {
        // Create new provider
        result = await apiService.llm.registerMCPProvider(requestData);
      } else {
        // Update existing provider
        result = await apiService.llm.updateMCPProvider(formValues.provider_id, {
          display_name: formValues.display_name,
          transport_config: {
            transport_type: formValues.transport_type,
            ...transportConfig
          },
          enable_streaming: formValues.enable_streaming,
          timeout_seconds: parseInt(formValues.timeout_seconds, 10),
          max_retries: parseInt(formValues.max_retries, 10),
          enabled: formValues.enabled
        });
      }
      
      if (result.success) {
        showSuccess(`Provider ${mode === 'add' ? 'created' : 'updated'} successfully`);
        
        if (onSave) {
          onSave(result.data);
        }
      } else {
        showError(`Failed to ${mode === 'add' ? 'create' : 'update'} provider: ${result.error}`);
      }
    } catch (error) {
      console.error(`Error ${mode === 'add' ? 'creating' : 'updating'} provider:`, error);
      showError(`Error ${mode === 'add' ? 'creating' : 'updating'} provider: ${error.message}`);
    } finally {
      setSaving(false);
    }
  };
  
  return (
    <Dialog
      open={open}
      onClose={onClose}
      maxWidth="md"
      fullWidth
    >
      <DialogTitle>
        {mode === 'add' ? 'Add MCP Provider' : 'Edit MCP Provider'}
      </DialogTitle>
      
      <DialogContent>
        <Box sx={{ mt: 2 }}>
          {/* Basic Information */}
          <Typography variant="h6" gutterBottom>Basic Information</Typography>
          
          <Box sx={{ display: 'flex', gap: 2, mb: 3 }}>
            <TextField
              name="provider_id"
              label="Provider ID"
              value={formValues.provider_id}
              onChange={handleInputChange}
              fullWidth
              required
              disabled={mode === 'edit'}
              error={!!validationErrors.provider_id}
              helperText={validationErrors.provider_id}
            />
            
            <TextField
              name="display_name"
              label="Display Name"
              value={formValues.display_name}
              onChange={handleInputChange}
              fullWidth
              required
              error={!!validationErrors.display_name}
              helperText={validationErrors.display_name}
            />
          </Box>
          
          <Box sx={{ display: 'flex', gap: 2, mb: 3 }}>
            <FormControl fullWidth>
              <InputLabel id="transport-type-label">Transport Type</InputLabel>
              <Select
                labelId="transport-type-label"
                name="transport_type"
                value={formValues.transport_type}
                onChange={handleTransportTypeChange}
                label="Transport Type"
              >
                <MenuItem value="stdio">stdio (Subprocess)</MenuItem>
                <MenuItem value="grpc">gRPC</MenuItem>
                <MenuItem value="http">HTTP/REST</MenuItem>
              </Select>
            </FormControl>
            
            <TextField
              name="timeout_seconds"
              label="Timeout (seconds)"
              type="number"
              value={formValues.timeout_seconds}
              onChange={handleInputChange}
              fullWidth
              inputProps={{ min: 1 }}
              error={!!validationErrors.timeout_seconds}
              helperText={validationErrors.timeout_seconds}
            />
            
            <TextField
              name="max_retries"
              label="Max Retries"
              type="number"
              value={formValues.max_retries}
              onChange={handleInputChange}
              fullWidth
              inputProps={{ min: 0 }}
              error={!!validationErrors.max_retries}
              helperText={validationErrors.max_retries}
            />
          </Box>
          
          <Box sx={{ display: 'flex', gap: 2, mb: 3 }}>
            <FormControlLabel
              control={
                <Switch
                  name="enable_streaming"
                  checked={formValues.enable_streaming}
                  onChange={handleInputChange}
                />
              }
              label="Enable Streaming"
            />
            
            <FormControlLabel
              control={
                <Switch
                  name="enabled"
                  checked={formValues.enabled}
                  onChange={handleInputChange}
                />
              }
              label="Enabled"
            />
          </Box>
          
          <Divider sx={{ my: 3 }} />
          
          {/* Transport Configuration */}
          <Typography variant="h6" gutterBottom>Transport Configuration</Typography>
          
          <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 2 }}>
            <Tabs
              value={activeTab}
              onChange={handleTabChange}
              aria-label="transport configuration tabs"
            >
              <Tab label="stdio" id="tab-0" aria-controls="tabpanel-0" />
              <Tab label="gRPC" id="tab-1" aria-controls="tabpanel-1" />
              <Tab label="HTTP/REST" id="tab-2" aria-controls="tabpanel-2" />
            </Tabs>
          </Box>
          
          {/* stdio Config */}
          <Box role="tabpanel" hidden={activeTab !== 0} id="tabpanel-0" aria-labelledby="tab-0">
            {activeTab === 0 && (
              <StdioConfig
                config={formValues.transport_config.stdio}
                onChange={(config) => updateTransportConfig('stdio', config)}
                errors={validationErrors}
              />
            )}
          </Box>
          
          {/* gRPC Config */}
          <Box role="tabpanel" hidden={activeTab !== 1} id="tabpanel-1" aria-labelledby="tab-1">
            {activeTab === 1 && (
              <GrpcConfig
                config={formValues.transport_config.grpc}
                onChange={(config) => updateTransportConfig('grpc', config)}
                errors={validationErrors}
              />
            )}
          </Box>
          
          {/* HTTP Config */}
          <Box role="tabpanel" hidden={activeTab !== 2} id="tabpanel-2" aria-labelledby="tab-2">
            {activeTab === 2 && (
              <HttpConfig
                config={formValues.transport_config.http}
                onChange={(config) => updateTransportConfig('http', config)}
                errors={validationErrors}
              />
            )}
          </Box>
        </Box>
      </DialogContent>
      
      <DialogActions>
        <Button onClick={onClose}>Cancel</Button>
        <Button
          onClick={handleSave}
          variant="contained"
          color="primary"
          disabled={saving}
          startIcon={saving ? <CircularProgress size={20} /> : null}
        >
          {saving ? 'Saving...' : 'Save'}
        </Button>
      </DialogActions>
    </Dialog>
  );
};

export default MCPConfigDialog;
