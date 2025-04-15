import React, { useState, useEffect } from 'react';
import {
  Dialog, DialogTitle, DialogContent, DialogActions,
  Button, TextField, FormControlLabel, Switch, Typography,
  Box, Grid, Divider, InputAdornment, IconButton, Alert,
  FormControl, InputLabel, Select, MenuItem
} from '@mui/material';
import {
  Visibility as VisibilityIcon,
  VisibilityOff as VisibilityOffIcon,
  Save as SaveIcon,
  Cancel as CancelIcon
} from '@mui/icons-material';

/**
 * Client Configuration Dialog
 * 
 * This component provides a dialog for configuring medical clients.
 */
const ClientConfigDialog = ({ open, onClose, client, onUpdate }) => {
  // Form state
  const [config, setConfig] = useState({});
  const [showApiKey, setShowApiKey] = useState(false);
  const [errors, setErrors] = useState({});
  
  // Initialize form when client changes
  useEffect(() => {
    if (client) {
      setConfig({ ...client.config });
      setErrors({});
    }
  }, [client]);
  
  // Handle form change
  const handleChange = (field, value) => {
    setConfig({
      ...config,
      [field]: value
    });
    
    // Clear error for this field
    if (errors[field]) {
      setErrors({
        ...errors,
        [field]: null
      });
    }
  };
  
  // Handle form submission
  const handleSubmit = (e) => {
    e.preventDefault();
    
    // Validate form
    const newErrors = {};
    
    // Email validation
    if (config.email && !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(config.email)) {
      newErrors.email = 'Please enter a valid email address';
    }
    
    // URL validation
    if (config.base_url && !/^https?:\/\//.test(config.base_url)) {
      newErrors.base_url = 'Please enter a valid URL (starting with http:// or https://)';
    }
    
    // Numeric validation
    if (config.timeout && (isNaN(config.timeout) || config.timeout <= 0)) {
      newErrors.timeout = 'Please enter a positive number';
    }
    
    if (config.max_retries && (isNaN(config.max_retries) || config.max_retries <= 0)) {
      newErrors.max_retries = 'Please enter a positive number';
    }
    
    if (config.cache_ttl && (isNaN(config.cache_ttl) || config.cache_ttl <= 0)) {
      newErrors.cache_ttl = 'Please enter a positive number';
    }
    
    // Check if there are any errors
    if (Object.keys(newErrors).length > 0) {
      setErrors(newErrors);
      return;
    }
    
    // Submit form
    onUpdate(client.client_id, config);
    onClose();
  };
  
  // Render client-specific fields
  const renderClientSpecificFields = () => {
    if (!client) return null;
    
    switch (client.client_id) {
      case 'ncbi':
        return (
          <>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Email"
                value={config.email || ''}
                onChange={(e) => handleChange('email', e.target.value)}
                error={!!errors.email}
                helperText={errors.email || 'Email address for NCBI API requests'}
                required
              />
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="API Key"
                type={showApiKey ? 'text' : 'password'}
                value={config.api_key || ''}
                onChange={(e) => handleChange('api_key', e.target.value)}
                helperText="API key for NCBI E-utilities (optional but recommended)"
                InputProps={{
                  endAdornment: (
                    <InputAdornment position="end">
                      <IconButton
                        onClick={() => setShowApiKey(!showApiKey)}
                        edge="end"
                      >
                        {showApiKey ? <VisibilityOffIcon /> : <VisibilityIcon />}
                      </IconButton>
                    </InputAdornment>
                  )
                }}
              />
            </Grid>
          </>
        );
        
      case 'umls':
        return (
          <Grid item xs={12}>
            <TextField
              fullWidth
              label="API Key"
              type={showApiKey ? 'text' : 'password'}
              value={config.api_key || ''}
              onChange={(e) => handleChange('api_key', e.target.value)}
              error={!config.api_key}
              helperText={!config.api_key ? 'API key is required for UMLS' : 'API key for UMLS authentication'}
              required
              InputProps={{
                endAdornment: (
                  <InputAdornment position="end">
                    <IconButton
                      onClick={() => setShowApiKey(!showApiKey)}
                      edge="end"
                    >
                      {showApiKey ? <VisibilityOffIcon /> : <VisibilityIcon />}
                    </IconButton>
                  </InputAdornment>
                )
              }}
            />
          </Grid>
        );
        
      case 'clinical_trials':
        return (
          <Grid item xs={12}>
            <Alert severity="info">
              ClinicalTrials.gov API does not require authentication.
            </Alert>
          </Grid>
        );
        
      case 'cochrane':
        return (
          <Grid item xs={12}>
            <TextField
              fullWidth
              label="API Key"
              type={showApiKey ? 'text' : 'password'}
              value={config.api_key || ''}
              onChange={(e) => handleChange('api_key', e.target.value)}
              helperText="API key for Cochrane Library (if available)"
              InputProps={{
                endAdornment: (
                  <InputAdornment position="end">
                    <IconButton
                      onClick={() => setShowApiKey(!showApiKey)}
                      edge="end"
                    >
                      {showApiKey ? <VisibilityOffIcon /> : <VisibilityIcon />}
                    </IconButton>
                  </InputAdornment>
                )
              }}
            />
          </Grid>
        );
        
      case 'crossref':
        return (
          <>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Email"
                value={config.email || ''}
                onChange={(e) => handleChange('email', e.target.value)}
                error={!!errors.email}
                helperText={errors.email || 'Email address for Crossref API requests'}
                required
              />
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Plus API Token"
                type={showApiKey ? 'text' : 'password'}
                value={config.plus_api_token || ''}
                onChange={(e) => handleChange('plus_api_token', e.target.value)}
                helperText="Crossref Plus API token (optional)"
                InputProps={{
                  endAdornment: (
                    <InputAdornment position="end">
                      <IconButton
                        onClick={() => setShowApiKey(!showApiKey)}
                        edge="end"
                      >
                        {showApiKey ? <VisibilityOffIcon /> : <VisibilityIcon />}
                      </IconButton>
                    </InputAdornment>
                  )
                }}
              />
            </Grid>
          </>
        );
        
      case 'snomed':
        return (
          <>
            <Grid item xs={12}>
              <FormControl fullWidth>
                <InputLabel id="access-mode-label">Access Mode</InputLabel>
                <Select
                  labelId="access-mode-label"
                  value={config.access_mode || 'umls'}
                  label="Access Mode"
                  onChange={(e) => handleChange('access_mode', e.target.value)}
                >
                  <MenuItem value="umls">UMLS API</MenuItem>
                  <MenuItem value="api">SNOMED CT API</MenuItem>
                  <MenuItem value="local">Local Files</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            
            {config.access_mode === 'umls' && (
              <Grid item xs={12}>
                <TextField
                  fullWidth
                  label="API Key"
                  type={showApiKey ? 'text' : 'password'}
                  value={config.api_key || ''}
                  onChange={(e) => handleChange('api_key', e.target.value)}
                  error={!config.api_key && config.access_mode === 'umls'}
                  helperText={!config.api_key && config.access_mode === 'umls' ? 
                    'API key is required for UMLS access mode' : 'API key for UMLS authentication'}
                  required={config.access_mode === 'umls'}
                  InputProps={{
                    endAdornment: (
                      <InputAdornment position="end">
                        <IconButton
                          onClick={() => setShowApiKey(!showApiKey)}
                          edge="end"
                        >
                          {showApiKey ? <VisibilityOffIcon /> : <VisibilityIcon />}
                        </IconButton>
                      </InputAdornment>
                    )
                  }}
                />
              </Grid>
            )}
            
            {config.access_mode === 'api' && (
              <Grid item xs={12}>
                <TextField
                  fullWidth
                  label="API URL"
                  value={config.api_url || ''}
                  onChange={(e) => handleChange('api_url', e.target.value)}
                  error={!config.api_url && config.access_mode === 'api'}
                  helperText={!config.api_url && config.access_mode === 'api' ? 
                    'API URL is required for API access mode' : 'URL for SNOMED CT API'}
                  required={config.access_mode === 'api'}
                />
              </Grid>
            )}
            
            {config.access_mode === 'local' && (
              <Grid item xs={12}>
                <TextField
                  fullWidth
                  label="Local Data Path"
                  value={config.local_data_path || ''}
                  onChange={(e) => handleChange('local_data_path', e.target.value)}
                  error={!config.local_data_path && config.access_mode === 'local'}
                  helperText={!config.local_data_path && config.access_mode === 'local' ? 
                    'Local data path is required for local access mode' : 'Path to local SNOMED CT files'}
                  required={config.access_mode === 'local'}
                />
              </Grid>
            )}
            
            <Grid item xs={12}>
              <FormControl fullWidth>
                <InputLabel id="edition-label">Edition</InputLabel>
                <Select
                  labelId="edition-label"
                  value={config.edition || 'US'}
                  label="Edition"
                  onChange={(e) => handleChange('edition', e.target.value)}
                >
                  <MenuItem value="US">US Edition</MenuItem>
                  <MenuItem value="INT">International Edition</MenuItem>
                  <MenuItem value="UK">UK Edition</MenuItem>
                  <MenuItem value="AU">Australian Edition</MenuItem>
                </Select>
              </FormControl>
            </Grid>
          </>
        );
        
      default:
        return null;
    }
  };
  
  if (!client) return null;
  
  return (
    <Dialog
      open={open}
      onClose={onClose}
      maxWidth="md"
      fullWidth
    >
      <DialogTitle>Configure {client.name}</DialogTitle>
      <form onSubmit={handleSubmit}>
        <DialogContent>
          <Typography variant="subtitle2" gutterBottom>
            Client ID: {client.client_id}
          </Typography>
          
          <Divider sx={{ my: 2 }} />
          
          <Grid container spacing={2}>
            {/* Client-specific fields */}
            {renderClientSpecificFields()}
            
            {/* Common fields */}
            <Grid item xs={12}>
              <Typography variant="subtitle1" sx={{ mt: 2, mb: 1 }}>
                Common Settings
              </Typography>
            </Grid>
            
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Base URL"
                value={config.base_url || ''}
                onChange={(e) => handleChange('base_url', e.target.value)}
                error={!!errors.base_url}
                helperText={errors.base_url || 'Base URL for API requests'}
              />
            </Grid>
            
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Timeout (seconds)"
                type="number"
                value={config.timeout || ''}
                onChange={(e) => handleChange('timeout', e.target.value)}
                error={!!errors.timeout}
                helperText={errors.timeout || 'Request timeout in seconds'}
                inputProps={{ min: 1, step: 1 }}
              />
            </Grid>
            
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Max Retries"
                type="number"
                value={config.max_retries || ''}
                onChange={(e) => handleChange('max_retries', e.target.value)}
                error={!!errors.max_retries}
                helperText={errors.max_retries || 'Maximum number of retries for failed requests'}
                inputProps={{ min: 0, step: 1 }}
              />
            </Grid>
            
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Cache TTL (seconds)"
                type="number"
                value={config.cache_ttl || ''}
                onChange={(e) => handleChange('cache_ttl', e.target.value)}
                error={!!errors.cache_ttl}
                helperText={errors.cache_ttl || 'Cache time-to-live in seconds'}
                inputProps={{ min: 0, step: 1 }}
              />
            </Grid>
            
            <Grid item xs={12}>
              <FormControlLabel
                control={
                  <Switch
                    checked={config.use_cache === true}
                    onChange={(e) => handleChange('use_cache', e.target.checked)}
                  />
                }
                label="Use Cache"
              />
            </Grid>
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button 
            onClick={onClose}
            startIcon={<CancelIcon />}
          >
            Cancel
          </Button>
          <Button 
            type="submit"
            variant="contained" 
            color="primary"
            startIcon={<SaveIcon />}
          >
            Save
          </Button>
        </DialogActions>
      </form>
    </Dialog>
  );
};

export default ClientConfigDialog;
