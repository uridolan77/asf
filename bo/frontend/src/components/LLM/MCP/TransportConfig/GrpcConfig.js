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
 * gRPC Transport Configuration Component
 * 
 * This component provides a form for configuring the gRPC transport
 * for MCP providers, including endpoint, TLS settings, and metadata.
 */
const GrpcConfig = ({ config, onChange, errors = {} }) => {
  const [metadataKey, setMetadataKey] = useState('');
  const [metadataValue, setMetadataValue] = useState('');
  const [metadataDialogOpen, setMetadataDialogOpen] = useState(false);
  
  // Handle endpoint change
  const handleEndpointChange = (e) => {
    onChange({
      ...config,
      endpoint: e.target.value
    });
  };
  
  // Handle TLS toggle
  const handleTlsToggle = (e) => {
    onChange({
      ...config,
      use_tls: e.target.checked
    });
  };
  
  // Handle certificate changes
  const handleCertChange = (field) => (e) => {
    onChange({
      ...config,
      [field]: e.target.value || null
    });
  };
  
  // Add metadata
  const handleAddMetadata = () => {
    if (!metadataKey.trim()) return;
    
    const updatedMetadata = {
      ...config.metadata,
      [metadataKey]: metadataValue
    };
    
    onChange({
      ...config,
      metadata: updatedMetadata
    });
    
    setMetadataKey('');
    setMetadataValue('');
    setMetadataDialogOpen(false);
  };
  
  // Remove metadata
  const handleRemoveMetadata = (key) => {
    const updatedMetadata = { ...config.metadata };
    delete updatedMetadata[key];
    
    onChange({
      ...config,
      metadata: updatedMetadata
    });
  };
  
  return (
    <Box>
      <Alert severity="info" sx={{ mb: 3 }}>
        The gRPC transport provides high-performance bi-directional streaming communication
        with an MCP server using the gRPC protocol.
      </Alert>
      
      <TextField
        label="Endpoint"
        value={config.endpoint}
        onChange={handleEndpointChange}
        fullWidth
        required
        placeholder="localhost:50051"
        error={!!errors.grpc_endpoint}
        helperText={errors.grpc_endpoint || "Format: host:port"}
        sx={{ mb: 3 }}
      />
      
      {/* TLS Settings */}
      <Typography variant="subtitle1" gutterBottom>TLS Settings</Typography>
      
      <Paper variant="outlined" sx={{ p: 2, mb: 3 }}>
        <FormControlLabel
          control={
            <Switch
              checked={config.use_tls}
              onChange={handleTlsToggle}
            />
          }
          label="Use TLS"
          sx={{ mb: 2 }}
        />
        
        {config.use_tls && (
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
        )}
      </Paper>
      
      {/* Metadata */}
      <Typography variant="subtitle1" gutterBottom>Metadata</Typography>
      
      <Paper variant="outlined" sx={{ p: 2, mb: 3 }}>
        {Object.keys(config.metadata).length > 0 ? (
          <Box sx={{ mb: 2 }}>
            {Object.entries(config.metadata).map(([key, value]) => (
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
                  onClick={() => handleRemoveMetadata(key)}
                >
                  <DeleteIcon />
                </IconButton>
              </Box>
            ))}
          </Box>
        ) : (
          <Typography color="text.secondary" sx={{ mb: 2 }}>
            No metadata added
          </Typography>
        )}
        
        <Button
          startIcon={<AddIcon />}
          variant="outlined"
          size="small"
          onClick={() => setMetadataDialogOpen(true)}
        >
          Add Metadata
        </Button>
      </Paper>
      
      {/* Add Metadata Dialog */}
      <Dialog
        open={metadataDialogOpen}
        onClose={() => setMetadataDialogOpen(false)}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>Add Metadata</DialogTitle>
        <DialogContent>
          <TextField
            label="Key"
            value={metadataKey}
            onChange={(e) => setMetadataKey(e.target.value)}
            fullWidth
            autoFocus
            margin="dense"
          />
          <TextField
            label="Value"
            value={metadataValue}
            onChange={(e) => setMetadataValue(e.target.value)}
            fullWidth
            margin="dense"
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setMetadataDialogOpen(false)}>Cancel</Button>
          <Button onClick={handleAddMetadata} variant="contained" color="primary">
            Add
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default GrpcConfig;
