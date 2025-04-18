import React, { useState } from 'react';
import {
  Box,
  TextField,
  Typography,
  Button,
  Chip,
  IconButton,
  Paper,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Alert
} from '@mui/material';
import {
  Add as AddIcon,
  Delete as DeleteIcon
} from '@mui/icons-material';

/**
 * stdio Transport Configuration Component
 * 
 * This component provides a form for configuring the stdio transport
 * for MCP providers, including command, arguments, environment variables,
 * and working directory.
 */
const StdioConfig = ({ config, onChange, errors = {} }) => {
  const [envKey, setEnvKey] = useState('');
  const [envValue, setEnvValue] = useState('');
  const [envDialogOpen, setEnvDialogOpen] = useState(false);
  const [argValue, setArgValue] = useState('');
  const [argDialogOpen, setArgDialogOpen] = useState(false);
  
  // Handle command change
  const handleCommandChange = (e) => {
    onChange({
      ...config,
      command: e.target.value
    });
  };
  
  // Handle working directory change
  const handleCwdChange = (e) => {
    onChange({
      ...config,
      cwd: e.target.value || null
    });
  };
  
  // Add environment variable
  const handleAddEnv = () => {
    if (!envKey.trim()) return;
    
    const updatedEnv = {
      ...config.env,
      [envKey]: envValue
    };
    
    onChange({
      ...config,
      env: updatedEnv
    });
    
    setEnvKey('');
    setEnvValue('');
    setEnvDialogOpen(false);
  };
  
  // Remove environment variable
  const handleRemoveEnv = (key) => {
    const updatedEnv = { ...config.env };
    delete updatedEnv[key];
    
    onChange({
      ...config,
      env: updatedEnv
    });
  };
  
  // Add argument
  const handleAddArg = () => {
    if (!argValue.trim()) return;
    
    const updatedArgs = [...config.args, argValue];
    
    onChange({
      ...config,
      args: updatedArgs
    });
    
    setArgValue('');
    setArgDialogOpen(false);
  };
  
  // Remove argument
  const handleRemoveArg = (index) => {
    const updatedArgs = [...config.args];
    updatedArgs.splice(index, 1);
    
    onChange({
      ...config,
      args: updatedArgs
    });
  };
  
  return (
    <Box>
      <Alert severity="info" sx={{ mb: 3 }}>
        The stdio transport uses a subprocess to communicate with the MCP server.
        This is useful for local development or when using Node.js-based MCP implementations.
      </Alert>
      
      <Box sx={{ display: 'flex', gap: 2, mb: 3 }}>
        <TextField
          label="Command"
          value={config.command}
          onChange={handleCommandChange}
          fullWidth
          required
          error={!!errors.stdio_command}
          helperText={errors.stdio_command}
        />
        
        <TextField
          label="Working Directory (optional)"
          value={config.cwd || ''}
          onChange={handleCwdChange}
          fullWidth
          placeholder="/path/to/working/directory"
        />
      </Box>
      
      {/* Arguments */}
      <Typography variant="subtitle1" gutterBottom>Command Arguments</Typography>
      
      <Paper variant="outlined" sx={{ p: 2, mb: 3 }}>
        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mb: 2 }}>
          {config.args.length > 0 ? (
            config.args.map((arg, index) => (
              <Chip
                key={index}
                label={arg}
                onDelete={() => handleRemoveArg(index)}
                deleteIcon={<DeleteIcon />}
              />
            ))
          ) : (
            <Typography color="text.secondary">No arguments added</Typography>
          )}
        </Box>
        
        <Button
          startIcon={<AddIcon />}
          variant="outlined"
          size="small"
          onClick={() => setArgDialogOpen(true)}
        >
          Add Argument
        </Button>
      </Paper>
      
      {/* Environment Variables */}
      <Typography variant="subtitle1" gutterBottom>Environment Variables</Typography>
      
      <Paper variant="outlined" sx={{ p: 2, mb: 3 }}>
        {Object.keys(config.env).length > 0 ? (
          <Box sx={{ mb: 2 }}>
            {Object.entries(config.env).map(([key, value]) => (
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
                  onClick={() => handleRemoveEnv(key)}
                >
                  <DeleteIcon />
                </IconButton>
              </Box>
            ))}
          </Box>
        ) : (
          <Typography color="text.secondary" sx={{ mb: 2 }}>
            No environment variables added
          </Typography>
        )}
        
        <Button
          startIcon={<AddIcon />}
          variant="outlined"
          size="small"
          onClick={() => setEnvDialogOpen(true)}
        >
          Add Environment Variable
        </Button>
      </Paper>
      
      {/* Add Argument Dialog */}
      <Dialog
        open={argDialogOpen}
        onClose={() => setArgDialogOpen(false)}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>Add Command Argument</DialogTitle>
        <DialogContent>
          <TextField
            label="Argument"
            value={argValue}
            onChange={(e) => setArgValue(e.target.value)}
            fullWidth
            autoFocus
            margin="dense"
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setArgDialogOpen(false)}>Cancel</Button>
          <Button onClick={handleAddArg} variant="contained" color="primary">
            Add
          </Button>
        </DialogActions>
      </Dialog>
      
      {/* Add Environment Variable Dialog */}
      <Dialog
        open={envDialogOpen}
        onClose={() => setEnvDialogOpen(false)}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>Add Environment Variable</DialogTitle>
        <DialogContent>
          <TextField
            label="Key"
            value={envKey}
            onChange={(e) => setEnvKey(e.target.value)}
            fullWidth
            autoFocus
            margin="dense"
          />
          <TextField
            label="Value"
            value={envValue}
            onChange={(e) => setEnvValue(e.target.value)}
            fullWidth
            margin="dense"
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setEnvDialogOpen(false)}>Cancel</Button>
          <Button onClick={handleAddEnv} variant="contained" color="primary">
            Add
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default StdioConfig;
