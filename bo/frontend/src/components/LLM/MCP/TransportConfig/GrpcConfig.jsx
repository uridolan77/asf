import React from 'react';
import {
  Box,
  TextField,
  Typography,
  FormControlLabel,
  Switch,
  FormHelperText,
  Paper
} from '@mui/material';
import { Controller } from 'react-hook-form';

/**
 * GrpcConfig Component
 * 
 * This component provides form fields for configuring the gRPC transport
 * for MCP providers. It uses React Hook Form for form handling.
 */
const GrpcConfig = ({ control, errors, transportType }) => {
  // Check if this transport type is active
  const isActive = transportType === 'grpc';
  
  return (
    <Box sx={{ opacity: isActive ? 1 : 0.5, pointerEvents: isActive ? 'auto' : 'none' }}>
      <Typography variant="subtitle1" gutterBottom>
        gRPC Configuration
      </Typography>
      
      <Typography variant="body2" color="text.secondary" paragraph>
        This transport connects to a gRPC server for high-performance, bi-directional streaming.
        Suitable for production deployments with high throughput requirements.
      </Typography>
      
      <Box sx={{ display: 'flex', gap: 2, mb: 3 }}>
        <Controller
          name="transport_config.grpc.endpoint"
          control={control}
          render={({ field }) => (
            <TextField
              {...field}
              label="Endpoint"
              fullWidth
              required={isActive}
              placeholder="localhost:50051"
              error={!!errors.transport_config?.grpc?.endpoint}
              helperText={errors.transport_config?.grpc?.endpoint?.message || "Host and port of the gRPC server (e.g., localhost:50051)"}
            />
          )}
        />
      </Box>
      
      <Box sx={{ mb: 3 }}>
        <Controller
          name="transport_config.grpc.use_tls"
          control={control}
          render={({ field: { value, onChange } }) => (
            <FormControlLabel
              control={
                <Switch
                  checked={value}
                  onChange={(e) => onChange(e.target.checked)}
                />
              }
              label="Use TLS"
            />
          )}
        />
      </Box>
      
      <Box sx={{ mb: 3 }}>
        <Typography variant="subtitle2" gutterBottom>
          TLS Configuration
        </Typography>
        
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
          <Controller
            name="transport_config.grpc.ca_cert"
            control={control}
            render={({ field }) => (
              <TextField
                {...field}
                label="CA Certificate Path (optional)"
                fullWidth
                placeholder="/path/to/ca.crt"
                helperText="Path to CA certificate for TLS verification"
                value={field.value || ''}
              />
            )}
          />
          
          <Controller
            name="transport_config.grpc.client_cert"
            control={control}
            render={({ field }) => (
              <TextField
                {...field}
                label="Client Certificate Path (optional)"
                fullWidth
                placeholder="/path/to/client.crt"
                helperText="Path to client certificate for mutual TLS"
                value={field.value || ''}
              />
            )}
          />
          
          <Controller
            name="transport_config.grpc.client_key"
            control={control}
            render={({ field }) => (
              <TextField
                {...field}
                label="Client Key Path (optional)"
                fullWidth
                placeholder="/path/to/client.key"
                helperText="Path to client key for mutual TLS"
                value={field.value || ''}
              />
            )}
          />
        </Box>
      </Box>
      
      <Box sx={{ mb: 3 }}>
        <Typography variant="subtitle2" gutterBottom>
          Metadata
        </Typography>
        
        <Typography variant="body2" color="text.secondary" paragraph>
          Metadata will be sent with each gRPC request.
          API keys and other authentication information should be configured here.
        </Typography>
        
        <Controller
          name="transport_config.grpc.metadata"
          control={control}
          render={({ field: { value, onChange } }) => (
            <TextField
              label="Metadata (JSON)"
              fullWidth
              multiline
              rows={4}
              value={JSON.stringify(value || {}, null, 2)}
              onChange={(e) => {
                try {
                  const metadataJson = JSON.parse(e.target.value);
                  onChange(metadataJson);
                } catch (error) {
                  // Don't update if JSON is invalid
                  console.error('Invalid JSON for metadata:', error);
                }
              }}
              placeholder='{\n  "x-api-key": "your-api-key",\n  "authorization": "Bearer token"\n}'
              helperText="Enter metadata as JSON object"
              error={!!errors.transport_config?.grpc?.metadata}
            />
          )}
        />
        {errors.transport_config?.grpc?.metadata && (
          <FormHelperText error>
            {errors.transport_config.grpc.metadata.message}
          </FormHelperText>
        )}
      </Box>
    </Box>
  );
};

export default GrpcConfig;
