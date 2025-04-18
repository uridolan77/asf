import React, { useState } from 'react';
import {
  Box,
  TextField,
  Typography,
  Button,
  FormControlLabel,
  Switch,
  Paper,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Alert,
  IconButton
} from '@mui/material';
import {
  Add as AddIcon,
  Delete as DeleteIcon
} from '@mui/icons-material';

/**
 * HTTP Transport Configuration Component
 * 
 * This component provides a form for configuring the HTTP transport
 * for MCP providers, including base URL, headers, and SSL settings.
 */
const HttpConfig = ({ config, onChange, errors = {} }) => {
  const [headerKey, setHeaderKey] = useState('');
  const [headerValue, setHeaderValue] = useState('');
  const [headerDialogOpen, setHeaderDialogOpen] = useState(false);
  
  // Handle base URL change
  const handleBaseUrlChange = (e) => {
    onChange({
      ...config,
      base_url: e.target.value
    });
  };
  
  // Handle SSL verification toggle
  const handleSslToggle = (e) => {
    onChange({
      ...config,
      verify_ssl: e.target.checked
    });
  };
  
  // Handle certificate changes
  const handleCertChange = (field) => (e) => {
    onChange({
      ...config,
      [field]: e.target.value || null
    });
  };
  
  // Add header
  const handleAddHeader = () => {
    if (!headerKey.trim()) return;
    
    const updatedHeaders = {
      ...config.headers,
      [headerKey]: headerValue
    };
    
    onChange({
      ...config,
      headers: updatedHeaders
    });
    
    setHeaderKey('');
    setHeaderValue('');
    setHeaderDialogOpen(false);
  };
  
  // Remove header
  const handleRemoveHeader = (key) => {
    const updatedHeaders = { ...config.headers };
    delete updatedHeaders[key];
    
    onChange({
      ...config,
      headers: updatedHeaders
    });
  };
  
  return (
    <Box>
      <Alert severity="info" sx={{ mb: 3 }}>
        The HTTP transport communicates with an MCP server using HTTP/REST APIs.
        This is useful for cloud-based MCP services or when using a REST API gateway.
      </Alert>
      
      <TextField
        label="Base URL"
        value={config.base_url}
        onChange={handleBaseUrlChange}
        fullWidth
        required
        placeholder="https://api.example.com/mcp"
        error={!!errors.http_base_url}
        helperText={errors.http_base_url}
        sx={{ mb: 3 }}
      />
      
      {/* SSL Settings */}
      <Typography variant="subtitle1" gutterBottom>SSL Settings</Typography>
      
      <Paper variant="outlined" sx={{ p: 2, mb: 3 }}>
        <FormControlLabel
          control={
            <Switch
              checked={config.verify_ssl}
              onChange={handleSslToggle}
            />
          }
          label="Verify SSL Certificates"
          sx={{ mb: 2 }}
        />
        
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
          <TextField
            label="CA Certificate Path (optional)"
            value={config.ca_cert || ''}
            onChange={handleCertChange('ca_cert')}
            fullWidth
            placeholder="/path/to/ca.crt"
          />
          
          <TextField
            label="Client Certificate Path (optional)"
            value={config.client_cert || ''}
            onChange={handleCertChange('client_cert')}
            fullWidth
            placeholder="/path/to/client.crt"
          />
          
          <TextField
            label="Client Key Path (optional)"
            value={config.client_key || ''}
            onChange={handleCertChange('client_key')}
            fullWidth
            placeholder="/path/to/client.key"
          />
        </Box>
      </Paper>
      
      {/* Headers */}
      <Typography variant="subtitle1" gutterBottom>HTTP Headers</Typography>
      
      <Paper variant="outlined" sx={{ p: 2, mb: 3 }}>
        {Object.keys(config.headers).length > 0 ? (
          <Box sx={{ mb: 2 }}>
            {Object.entries(config.headers).map(([key, value]) => (
              <Box
                key={key}
                sx={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  p: 1,
                  borderBottom: '1px solid',
                  borderColor: 'divider'
                }}
              >
                <Box>
                  <Typography variant="body2" fontWeight="bold">{key}</Typography>
                  <Typography variant="body2" color="text.secondary">
                    {value.length > 30 ? `${value.substring(0, 30)}...` : value}
                  </Typography>
                </Box>
                
                <IconButton
                  size="small"
                  color="error"
                  onClick={() => handleRemoveHeader(key)}
                >
                  <DeleteIcon />
                </IconButton>
              </Box>
            ))}
          </Box>
        ) : (
          <Typography color="text.secondary" sx={{ mb: 2 }}>
            No headers added
          </Typography>
        )}
        
        <Button
          startIcon={<AddIcon />}
          variant="outlined"
          size="small"
          onClick={() => setHeaderDialogOpen(true)}
        >
          Add Header
        </Button>
      </Paper>
      
      {/* Add Header Dialog */}
      <Dialog
        open={headerDialogOpen}
        onClose={() => setHeaderDialogOpen(false)}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>Add HTTP Header</DialogTitle>
        <DialogContent>
          <TextField
            label="Header Name"
            value={headerKey}
            onChange={(e) => setHeaderKey(e.target.value)}
            fullWidth
            autoFocus
            margin="dense"
            placeholder="Content-Type"
          />
          <TextField
            label="Header Value"
            value={headerValue}
            onChange={(e) => setHeaderValue(e.target.value)}
            fullWidth
            margin="dense"
            placeholder="application/json"
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setHeaderDialogOpen(false)}>Cancel</Button>
          <Button onClick={handleAddHeader} variant="contained" color="primary">
            Add
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default HttpConfig;
