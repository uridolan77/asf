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
  Tab,
  FormHelperText
} from '@mui/material';
import { Controller } from 'react-hook-form';

import { useNotification } from '../../../context/NotificationContext';
import { useMCPConfigForm } from '../../../hooks/useMCPConfigForm';
import StdioConfig from './TransportConfig/StdioConfig';
import GrpcConfig from './TransportConfig/GrpcConfig';
import HttpConfig from './TransportConfig/HttpConfig';

/**
 * MCP Provider Configuration Dialog
 * 
 * This component provides a dialog for adding or editing MCP providers,
 * with support for different transport types (stdio, gRPC, HTTP).
 * Uses React Hook Form for form handling and validation.
 */
const MCPConfigDialog = ({ open, mode = 'add', provider = null, onClose, onSave }) => {
  const { showError } = useNotification();
  
  // Form handling
  const { form, defaultValues } = useMCPConfigForm(provider);
  const { control, handleSubmit, setValue, watch, reset, formState: { errors, isSubmitting } } = form;
  
  // State
  const [activeTab, setActiveTab] = useState(0);
  const transportType = watch('transport_type');
  
  // Reset form when dialog opens/closes or provider changes
  useEffect(() => {
    if (open) {
      if (mode === 'edit' && provider) {
        // Map provider data to form values
        reset({
          provider_id: provider.provider_id,
          display_name: provider.display_name || '',
          transport_type: provider.transport_type || 'stdio',
          enable_streaming: provider.enable_streaming !== false,
          timeout_seconds: provider.timeout_seconds || 60,
          max_retries: provider.max_retries || 3,
          enabled: provider.enabled !== false,
          transport_config: {
            stdio: provider.transport_config?.stdio || defaultValues.transport_config.stdio,
            grpc: provider.transport_config?.grpc || defaultValues.transport_config.grpc,
            http: provider.transport_config?.http || defaultValues.transport_config.http
          }
        });
        
        // Set active tab based on transport type
        setActiveTab(
          provider.transport_type === 'grpc' ? 1 :
          provider.transport_type === 'http' ? 2 : 0
        );
      } else {
        // Reset to default values for new provider
        reset(defaultValues);
        setActiveTab(0);
      }
    }
  }, [open, mode, provider, reset, defaultValues]);
  
  // Handle tab change
  const handleTabChange = (_, newValue) => {
    setActiveTab(newValue);
    
    // Update transport type based on tab
    const transportTypes = ['stdio', 'grpc', 'http'];
    setValue('transport_type', transportTypes[newValue]);
  };
  
  // Handle form submission
  const onSubmit = async (data) => {
    try {
      // Prepare transport config based on selected type
      const transportConfig = {
        [data.transport_type]: data.transport_config[data.transport_type]
      };
      
      // Prepare request data
      const requestData = {
        provider_id: data.provider_id,
        display_name: data.display_name,
        transport_type: data.transport_type,
        transport_config: transportConfig,
        enable_streaming: data.enable_streaming,
        timeout_seconds: parseInt(data.timeout_seconds, 10),
        max_retries: parseInt(data.max_retries, 10),
        enabled: data.enabled
      };
      
      // Call onSave with the prepared data
      onSave(requestData);
    } catch (error) {
      console.error('Error saving MCP provider:', error);
      showError(`Error saving MCP provider: ${error.message}`);
    }
  };
  
  return (
    <Dialog
      open={open}
      onClose={onClose}
      maxWidth="md"
      fullWidth
      PaperProps={{
        sx: { minHeight: '60vh' }
      }}
    >
      <form onSubmit={handleSubmit(onSubmit)}>
        <DialogTitle>
          {mode === 'add' ? 'Add MCP Provider' : 'Edit MCP Provider'}
        </DialogTitle>
        
        <DialogContent>
          <Box sx={{ mt: 2 }}>
            {/* Basic Information */}
            <Typography variant="h6" gutterBottom>Basic Information</Typography>
            
            <Box sx={{ display: 'flex', gap: 2, mb: 3 }}>
              <Controller
                name="provider_id"
                control={control}
                render={({ field }) => (
                  <TextField
                    {...field}
                    label="Provider ID"
                    fullWidth
                    required
                    disabled={mode === 'edit'}
                    error={!!errors.provider_id}
                    helperText={errors.provider_id?.message}
                  />
                )}
              />
              
              <Controller
                name="display_name"
                control={control}
                render={({ field }) => (
                  <TextField
                    {...field}
                    label="Display Name"
                    fullWidth
                    required
                    error={!!errors.display_name}
                    helperText={errors.display_name?.message}
                  />
                )}
              />
            </Box>
            
            {/* General Settings */}
            <Typography variant="h6" gutterBottom>General Settings</Typography>
            
            <Box sx={{ display: 'flex', gap: 2, mb: 3 }}>
              <Controller
                name="timeout_seconds"
                control={control}
                render={({ field }) => (
                  <TextField
                    {...field}
                    label="Timeout (seconds)"
                    type="number"
                    fullWidth
                    required
                    inputProps={{ min: 1, max: 300 }}
                    error={!!errors.timeout_seconds}
                    helperText={errors.timeout_seconds?.message}
                  />
                )}
              />
              
              <Controller
                name="max_retries"
                control={control}
                render={({ field }) => (
                  <TextField
                    {...field}
                    label="Max Retries"
                    type="number"
                    fullWidth
                    required
                    inputProps={{ min: 0, max: 10 }}
                    error={!!errors.max_retries}
                    helperText={errors.max_retries?.message}
                  />
                )}
              />
            </Box>
            
            <Box sx={{ display: 'flex', gap: 2, mb: 3 }}>
              <Controller
                name="enable_streaming"
                control={control}
                render={({ field: { value, onChange } }) => (
                  <FormControlLabel
                    control={
                      <Switch
                        checked={value}
                        onChange={(e) => onChange(e.target.checked)}
                      />
                    }
                    label="Enable Streaming"
                  />
                )}
              />
              
              <Controller
                name="enabled"
                control={control}
                render={({ field: { value, onChange } }) => (
                  <FormControlLabel
                    control={
                      <Switch
                        checked={value}
                        onChange={(e) => onChange(e.target.checked)}
                      />
                    }
                    label="Enabled"
                  />
                )}
              />
            </Box>
            
            <Divider sx={{ my: 3 }} />
            
            {/* Transport Configuration */}
            <Typography variant="h6" gutterBottom>Transport Configuration</Typography>
            
            <Controller
              name="transport_type"
              control={control}
              render={({ field }) => (
                <FormControl fullWidth sx={{ mb: 2 }} error={!!errors.transport_type}>
                  <InputLabel id="transport-type-label">Transport Type</InputLabel>
                  <Select
                    {...field}
                    labelId="transport-type-label"
                    label="Transport Type"
                  >
                    <MenuItem value="stdio">Standard IO (subprocess)</MenuItem>
                    <MenuItem value="grpc">gRPC</MenuItem>
                    <MenuItem value="http">HTTP/REST</MenuItem>
                  </Select>
                  {errors.transport_type && (
                    <FormHelperText>{errors.transport_type.message}</FormHelperText>
                  )}
                </FormControl>
              )}
            />
            
            <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
              <Tabs
                value={activeTab}
                onChange={handleTabChange}
                aria-label="Transport configuration tabs"
              >
                <Tab label="Standard IO" id="tab-0" aria-controls="tabpanel-0" />
                <Tab label="gRPC" id="tab-1" aria-controls="tabpanel-1" />
                <Tab label="HTTP/REST" id="tab-2" aria-controls="tabpanel-2" />
              </Tabs>
            </Box>
            
            {/* stdio Config */}
            <Box role="tabpanel" hidden={activeTab !== 0} id="tabpanel-0" aria-labelledby="tab-0" sx={{ py: 2 }}>
              {activeTab === 0 && (
                <StdioConfig
                  control={control}
                  errors={errors}
                  transportType={transportType}
                />
              )}
            </Box>
            
            {/* gRPC Config */}
            <Box role="tabpanel" hidden={activeTab !== 1} id="tabpanel-1" aria-labelledby="tab-1" sx={{ py: 2 }}>
              {activeTab === 1 && (
                <GrpcConfig
                  control={control}
                  errors={errors}
                  transportType={transportType}
                />
              )}
            </Box>
            
            {/* HTTP Config */}
            <Box role="tabpanel" hidden={activeTab !== 2} id="tabpanel-2" aria-labelledby="tab-2" sx={{ py: 2 }}>
              {activeTab === 2 && (
                <HttpConfig
                  control={control}
                  errors={errors}
                  transportType={transportType}
                />
              )}
            </Box>
            
            {/* Validation Errors */}
            {Object.keys(errors).length > 0 && (
              <Alert severity="error" sx={{ mt: 2 }}>
                Please fix the validation errors before saving.
              </Alert>
            )}
          </Box>
        </DialogContent>
        
        <DialogActions>
          <Button onClick={onClose}>Cancel</Button>
          <Button
            type="submit"
            variant="contained"
            color="primary"
            disabled={isSubmitting}
            startIcon={isSubmitting ? <CircularProgress size={20} /> : null}
          >
            {isSubmitting ? 'Saving...' : 'Save'}
          </Button>
        </DialogActions>
      </form>
    </Dialog>
  );
};

export default MCPConfigDialog;
