import React from 'react';
import {
  Box,
  TextField,
  Typography,
  FormControlLabel,
  Switch,
  FormHelperText
} from '@mui/material';
import { Controller } from 'react-hook-form';

/**
 * HttpConfig Component
 * 
 * This component provides form fields for configuring the HTTP transport
 * for MCP providers. It uses React Hook Form for form handling.
 */
const HttpConfig = ({ control, errors, transportType }) => {
  // Check if this transport type is active
  const isActive = transportType === 'http';
  
  return (
    <Box sx={{ opacity: isActive ? 1 : 0.5, pointerEvents: isActive ? 'auto' : 'none' }}>
      <Typography variant="subtitle1" gutterBottom>
        HTTP/REST Configuration
      </Typography>
      
      <Typography variant="body2" color="text.secondary" paragraph>
        This transport connects to an HTTP/REST API endpoint.
        Suitable for cloud-based MCP providers or services with REST APIs.
      </Typography>
      
      <Box sx={{ display: 'flex', gap: 2, mb: 3 }}>
        <Controller
          name="transport_config.http.base_url"
          control={control}
          render={({ field }) => (
            <TextField
              {...field}
              label="Base URL"
              fullWidth
              required={isActive}
              placeholder="https://api.example.com/mcp"
              error={!!errors.transport_config?.http?.base_url}
              helperText={errors.transport_config?.http?.base_url?.message || "Base URL for the HTTP API"}
            />
          )}
        />
      </Box>
      
      <Box sx={{ mb: 3 }}>
        <Controller
          name="transport_config.http.verify_ssl"
          control={control}
          render={({ field: { value, onChange } }) => (
            <FormControlLabel
              control={
                <Switch
                  checked={value}
                  onChange={(e) => onChange(e.target.checked)}
                />
              }
              label="Verify SSL Certificates"
            />
          )}
        />
      </Box>
      
      <Box sx={{ mb: 3 }}>
        <Typography variant="subtitle2" gutterBottom>
          Headers
        </Typography>
        
        <Typography variant="body2" color="text.secondary" paragraph>
          Headers will be sent with each HTTP request.
          API keys and other authentication information should be configured here.
        </Typography>
        
        <Controller
          name="transport_config.http.headers"
          control={control}
          render={({ field: { value, onChange } }) => (
            <TextField
              label="Headers (JSON)"
              fullWidth
              multiline
              rows={4}
              value={JSON.stringify(value || {}, null, 2)}
              onChange={(e) => {
                try {
                  const headersJson = JSON.parse(e.target.value);
                  onChange(headersJson);
                } catch (error) {
                  // Don't update if JSON is invalid
                  console.error('Invalid JSON for headers:', error);
                }
              }}
              placeholder='{\n  "Content-Type": "application/json",\n  "Authorization": "Bearer your-token",\n  "x-api-key": "your-api-key"\n}'
              helperText="Enter headers as JSON object"
              error={!!errors.transport_config?.http?.headers}
            />
          )}
        />
        {errors.transport_config?.http?.headers && (
          <FormHelperText error>
            {errors.transport_config.http.headers.message}
          </FormHelperText>
        )}
      </Box>
      
      <Box sx={{ mb: 3 }}>
        <Typography variant="subtitle2" gutterBottom>
          Advanced HTTP Settings
        </Typography>
        
        <Typography variant="body2" color="text.secondary" paragraph>
          These settings control the behavior of the HTTP client.
        </Typography>
        
        <Box sx={{ display: 'flex', gap: 2 }}>
          <Controller
            name="transport_config.http.timeout_ms"
            control={control}
            render={({ field }) => (
              <TextField
                {...field}
                label="Request Timeout (ms)"
                type="number"
                fullWidth
                placeholder="30000"
                helperText="Request timeout in milliseconds (default: 30000)"
                value={field.value || ''}
                inputProps={{ min: 1000, step: 1000 }}
              />
            )}
          />
          
          <Controller
            name="transport_config.http.max_retries"
            control={control}
            render={({ field }) => (
              <TextField
                {...field}
                label="Max Retries"
                type="number"
                fullWidth
                placeholder="3"
                helperText="Maximum number of retries for failed requests"
                value={field.value || ''}
                inputProps={{ min: 0, max: 10 }}
              />
            )}
          />
        </Box>
      </Box>
    </Box>
  );
};

export default HttpConfig;
