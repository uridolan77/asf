import React from 'react';
import {
  Box,
  TextField,
  Typography,
  FormHelperText,
  Chip,
  Paper
} from '@mui/material';
import { Controller } from 'react-hook-form';

/**
 * StdioConfig Component
 * 
 * This component provides form fields for configuring the stdio transport
 * for MCP providers. It uses React Hook Form for form handling.
 */
const StdioConfig = ({ control, errors, transportType }) => {
  // Check if this transport type is active
  const isActive = transportType === 'stdio';
  
  return (
    <Box sx={{ opacity: isActive ? 1 : 0.5, pointerEvents: isActive ? 'auto' : 'none' }}>
      <Typography variant="subtitle1" gutterBottom>
        Standard IO Configuration
      </Typography>
      
      <Typography variant="body2" color="text.secondary" paragraph>
        This transport launches a subprocess and communicates with it via standard input/output.
        Typically used for local MCP servers or Node.js implementations.
      </Typography>
      
      <Box sx={{ display: 'flex', gap: 2, mb: 3 }}>
        <Controller
          name="transport_config.stdio.command"
          control={control}
          render={({ field }) => (
            <TextField
              {...field}
              label="Command"
              fullWidth
              required={isActive}
              placeholder="npx"
              error={!!errors.transport_config?.stdio?.command}
              helperText={errors.transport_config?.stdio?.command?.message}
            />
          )}
        />
      </Box>
      
      <Box sx={{ mb: 3 }}>
        <Controller
          name="transport_config.stdio.args"
          control={control}
          render={({ field: { value, onChange } }) => (
            <>
              <TextField
                label="Arguments (comma-separated)"
                fullWidth
                value={Array.isArray(value) ? value.join(', ') : ''}
                onChange={(e) => {
                  const argsString = e.target.value;
                  const args = argsString.split(',').map(arg => arg.trim()).filter(Boolean);
                  onChange(args);
                }}
                placeholder="@anthropic/mcp-starter, --no-color"
                helperText="Enter command arguments separated by commas"
              />
              
              {Array.isArray(value) && value.length > 0 && (
                <Paper variant="outlined" sx={{ mt: 1, p: 1 }}>
                  <Typography variant="caption" display="block" gutterBottom>
                    Current Arguments:
                  </Typography>
                  <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                    {value.map((arg, index) => (
                      <Chip
                        key={index}
                        label={arg}
                        size="small"
                        onDelete={() => {
                          const newArgs = [...value];
                          newArgs.splice(index, 1);
                          onChange(newArgs);
                        }}
                      />
                    ))}
                  </Box>
                </Paper>
              )}
            </>
          )}
        />
      </Box>
      
      <Box sx={{ mb: 3 }}>
        <Controller
          name="transport_config.stdio.cwd"
          control={control}
          render={({ field }) => (
            <TextField
              {...field}
              label="Working Directory (optional)"
              fullWidth
              placeholder="/path/to/working/directory"
              helperText="Leave empty to use the current working directory"
              value={field.value || ''}
            />
          )}
        />
      </Box>
      
      <Box sx={{ mb: 3 }}>
        <Typography variant="subtitle2" gutterBottom>
          Environment Variables
        </Typography>
        
        <Typography variant="body2" color="text.secondary" paragraph>
          Environment variables will be passed to the subprocess.
          API keys and other sensitive information should be configured here.
        </Typography>
        
        <Controller
          name="transport_config.stdio.env"
          control={control}
          render={({ field: { value, onChange } }) => (
            <TextField
              label="Environment Variables (JSON)"
              fullWidth
              multiline
              rows={4}
              value={JSON.stringify(value || {}, null, 2)}
              onChange={(e) => {
                try {
                  const envJson = JSON.parse(e.target.value);
                  onChange(envJson);
                } catch (error) {
                  // Don't update if JSON is invalid
                  console.error('Invalid JSON for environment variables:', error);
                }
              }}
              placeholder='{\n  "API_KEY": "your-api-key",\n  "DEBUG": "true"\n}'
              helperText="Enter environment variables as JSON object"
              error={!!errors.transport_config?.stdio?.env}
            />
          )}
        />
        {errors.transport_config?.stdio?.env && (
          <FormHelperText error>
            {errors.transport_config.stdio.env.message}
          </FormHelperText>
        )}
      </Box>
    </Box>
  );
};

export default StdioConfig;
