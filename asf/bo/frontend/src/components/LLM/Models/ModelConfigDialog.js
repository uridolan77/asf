import React, { useState } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  TextField,
  Grid,
  Typography,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Switch,
  FormControlLabel,
  Divider,
  Box,
  IconButton,
  Tooltip,
  Alert
} from '@mui/material';
import {
  Info as InfoIcon,
  Add as AddIcon,
  Delete as DeleteIcon
} from '@mui/icons-material';

/**
 * Dialog component for configuring LLM models
 * 
 * @param {Object} props
 * @param {boolean} props.open - Dialog open state
 * @param {Function} props.onClose - Function to call when dialog is closed
 * @param {Object} props.config - Configuration object
 * @param {Function} props.onSave - Function to call when configuration is saved
 * @param {string} props.title - Dialog title
 */
const ModelConfigDialog = ({ open, onClose, config, onSave, title = 'Model Configuration' }) => {
  const [currentConfig, setCurrentConfig] = useState(config || {});
  const [error, setError] = useState('');
  
  // Handle form field changes
  const handleChange = (section, field, value) => {
    if (section) {
      setCurrentConfig(prev => ({
        ...prev,
        [section]: {
          ...(prev[section] || {}),
          [field]: value
        }
      }));
    } else {
      setCurrentConfig(prev => ({
        ...prev,
        [field]: value
      }));
    }
  };
  
  // Handle array field changes
  const handleArrayChange = (field, index, value) => {
    setCurrentConfig(prev => {
      const newArray = [...(prev[field] || [])];
      newArray[index] = value;
      return {
        ...prev,
        [field]: newArray
      };
    });
  };
  
  // Add item to array field
  const addArrayItem = (field, defaultValue = '') => {
    setCurrentConfig(prev => ({
      ...prev,
      [field]: [...(prev[field] || []), defaultValue]
    }));
  };
  
  // Remove item from array field
  const removeArrayItem = (field, index) => {
    setCurrentConfig(prev => {
      const newArray = [...(prev[field] || [])];
      newArray.splice(index, 1);
      return {
        ...prev,
        [field]: newArray
      };
    });
  };
  
  // Handle save
  const handleSave = () => {
    // Validate required fields
    if (!currentConfig.name) {
      setError('Name is required');
      return;
    }
    
    onSave(currentConfig);
  };
  
  return (
    <Dialog 
      open={open} 
      onClose={onClose}
      fullWidth
      maxWidth="md"
    >
      <DialogTitle>{title}</DialogTitle>
      <DialogContent dividers>
        {error && (
          <Alert severity="error" sx={{ mb: 2 }}>
            {error}
          </Alert>
        )}
        
        <Typography variant="subtitle2" gutterBottom>
          Basic Configuration
        </Typography>
        
        <Grid container spacing={2} sx={{ mb: 3 }}>
          <Grid item xs={12} sm={6}>
            <TextField
              label="Name"
              fullWidth
              value={currentConfig.name || ''}
              onChange={(e) => handleChange(null, 'name', e.target.value)}
              required
            />
          </Grid>
          <Grid item xs={12} sm={6}>
            <TextField
              label="Version"
              fullWidth
              value={currentConfig.version || ''}
              onChange={(e) => handleChange(null, 'version', e.target.value)}
            />
          </Grid>
          <Grid item xs={12}>
            <TextField
              label="Description"
              fullWidth
              multiline
              rows={2}
              value={currentConfig.description || ''}
              onChange={(e) => handleChange(null, 'description', e.target.value)}
            />
          </Grid>
          <Grid item xs={12} sm={6}>
            <FormControl fullWidth>
              <InputLabel>Status</InputLabel>
              <Select
                value={currentConfig.status || 'active'}
                label="Status"
                onChange={(e) => handleChange(null, 'status', e.target.value)}
              >
                <MenuItem value="active">Active</MenuItem>
                <MenuItem value="inactive">Inactive</MenuItem>
                <MenuItem value="development">Development</MenuItem>
                <MenuItem value="deprecated">Deprecated</MenuItem>
              </Select>
            </FormControl>
          </Grid>
          <Grid item xs={12} sm={6}>
            <FormControlLabel
              control={
                <Switch
                  checked={currentConfig.is_default || false}
                  onChange={(e) => handleChange(null, 'is_default', e.target.checked)}
                />
              }
              label="Set as Default Model"
            />
          </Grid>
        </Grid>
        
        <Divider sx={{ my: 2 }} />
        
        <Typography variant="subtitle2" gutterBottom>
          Model Parameters
        </Typography>
        
        <Grid container spacing={2} sx={{ mb: 3 }}>
          <Grid item xs={12} sm={6} md={4}>
            <TextField
              label="Context Window"
              type="number"
              fullWidth
              value={currentConfig.context_window || 4096}
              onChange={(e) => handleChange(null, 'context_window', parseInt(e.target.value))}
              InputProps={{ inputProps: { min: 0 } }}
            />
          </Grid>
          <Grid item xs={12} sm={6} md={4}>
            <TextField
              label="Default Temperature"
              type="number"
              fullWidth
              value={currentConfig.default_temperature || 0.7}
              onChange={(e) => handleChange(null, 'default_temperature', parseFloat(e.target.value))}
              InputProps={{ inputProps: { min: 0, max: 2, step: 0.1 } }}
            />
          </Grid>
          <Grid item xs={12} sm={6} md={4}>
            <TextField
              label="Max Output Tokens"
              type="number"
              fullWidth
              value={currentConfig.max_output_tokens || 1000}
              onChange={(e) => handleChange(null, 'max_output_tokens', parseInt(e.target.value))}
              InputProps={{ inputProps: { min: 0 } }}
            />
          </Grid>
          <Grid item xs={12} sm={6} md={4}>
            <FormControl fullWidth>
              <InputLabel>Precision</InputLabel>
              <Select
                value={currentConfig.precision || 'fp16'}
                label="Precision"
                onChange={(e) => handleChange(null, 'precision', e.target.value)}
              >
                <MenuItem value="fp16">FP16</MenuItem>
                <MenuItem value="fp32">FP32</MenuItem>
                <MenuItem value="int8">INT8</MenuItem>
                <MenuItem value="int4">INT4</MenuItem>
              </Select>
            </FormControl>
          </Grid>
          <Grid item xs={12} sm={6} md={4}>
            <FormControlLabel
              control={
                <Switch
                  checked={currentConfig.supports_streaming || false}
                  onChange={(e) => handleChange(null, 'supports_streaming', e.target.checked)}
                />
              }
              label="Supports Streaming"
            />
          </Grid>
          <Grid item xs={12} sm={6} md={4}>
            <FormControlLabel
              control={
                <Switch
                  checked={currentConfig.supports_function_calling || false}
                  onChange={(e) => handleChange(null, 'supports_function_calling', e.target.checked)}
                />
              }
              label="Supports Function Calling"
            />
          </Grid>
        </Grid>
        
        <Divider sx={{ my: 2 }} />
        
        <Typography variant="subtitle2" gutterBottom>
          Tags and Categories
        </Typography>
        
        <Box sx={{ mb: 3 }}>
          <Typography variant="body2" gutterBottom>
            Tags (used for filtering and categorization)
          </Typography>
          
          {(currentConfig.tags || []).map((tag, index) => (
            <Box key={index} sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
              <TextField
                fullWidth
                size="small"
                value={tag}
                onChange={(e) => handleArrayChange('tags', index, e.target.value)}
                placeholder="Enter tag"
              />
              <IconButton color="error" onClick={() => removeArrayItem('tags', index)}>
                <DeleteIcon />
              </IconButton>
            </Box>
          ))}
          
          <Button 
            startIcon={<AddIcon />}
            onClick={() => addArrayItem('tags')}
            size="small"
            sx={{ mt: 1 }}
          >
            Add Tag
          </Button>
        </Box>
        
        <Divider sx={{ my: 2 }} />
        
        <Typography variant="subtitle2" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
          Advanced Settings
          <Tooltip title="These settings are provider-specific and may not apply to all models">
            <IconButton size="small">
              <InfoIcon fontSize="small" />
            </IconButton>
          </Tooltip>
        </Typography>
        
        <Grid container spacing={2}>
          <Grid item xs={12} sm={6}>
            <TextField
              label="API Endpoint"
              fullWidth
              value={currentConfig.api_endpoint || ''}
              onChange={(e) => handleChange(null, 'api_endpoint', e.target.value)}
              placeholder="https://api.example.com/v1/completions"
            />
          </Grid>
          <Grid item xs={12} sm={6}>
            <TextField
              label="Model ID"
              fullWidth
              value={currentConfig.model_id || ''}
              onChange={(e) => handleChange(null, 'model_id', e.target.value)}
              placeholder="The provider-specific model ID"
            />
          </Grid>
          <Grid item xs={12} sm={6}>
            <TextField
              label="Tokenizer Path"
              fullWidth
              value={currentConfig.tokenizer_path || ''}
              onChange={(e) => handleChange(null, 'tokenizer_path', e.target.value)}
              placeholder="Path to custom tokenizer (if applicable)"
            />
          </Grid>
          <Grid item xs={12} sm={6}>
            <TextField
              label="Documentation URL"
              fullWidth
              value={currentConfig.documentation_url || ''}
              onChange={(e) => handleChange(null, 'documentation_url', e.target.value)}
              placeholder="https://docs.example.com/model"
            />
          </Grid>
        </Grid>
        
        {/* Provider-specific configuration */}
        {currentConfig.provider_config && (
          <>
            <Divider sx={{ my: 2 }} />
            <Typography variant="subtitle2" gutterBottom>
              Provider-Specific Configuration
            </Typography>
            
            <Alert severity="info" sx={{ mb: 2 }}>
              These settings are specific to the provider and model type.
            </Alert>
            
            <Grid container spacing={2}>
              {Object.entries(currentConfig.provider_config).map(([key, value]) => (
                <Grid item xs={12} sm={6} key={key}>
                  <TextField
                    label={key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                    fullWidth
                    value={value}
                    onChange={(e) => handleChange('provider_config', key, e.target.value)}
                  />
                </Grid>
              ))}
            </Grid>
          </>
        )}
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>Cancel</Button>
        <Button 
          variant="contained" 
          onClick={handleSave}
        >
          Save
        </Button>
      </DialogActions>
    </Dialog>
  );
};

export default ModelConfigDialog;