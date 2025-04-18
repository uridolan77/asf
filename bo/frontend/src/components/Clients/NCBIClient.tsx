import React, { useState, useEffect } from 'react';
import {
  Box, Card, CardContent, CardHeader, CardActions,
  Typography, Button, TextField, CircularProgress,
  Divider, Alert, Chip, Grid, Paper, IconButton,
  InputAdornment, Tooltip
} from '@mui/material';
import {
  Refresh as RefreshIcon,
  Settings as SettingsIcon,
  CheckCircle as CheckCircleIcon,
  Error as ErrorIcon,
  Warning as WarningIcon,
  Info as InfoIcon,
  Visibility as VisibilityIcon,
  VisibilityOff as VisibilityOffIcon,
  Search as SearchIcon
} from '@mui/icons-material';

import { useMedicalClients, MedicalClient, ClientConfig, TestResult } from '../../hooks/useMedicalClients';
import { useNotification } from '../../context/NotificationContext';
import { ButtonLoader } from '../UI/LoadingIndicators';

interface NCBIClientProps {
  client: MedicalClient | null;
  onRefresh: () => void;
  onConfigUpdate: (client: MedicalClient) => void;
}

/**
 * NCBI Client Component
 *
 * This component provides functionality for configuring and testing
 * the NCBI (National Center for Biotechnology Information) client.
 */
const NCBIClient: React.FC<NCBIClientProps> = ({ client, onRefresh, onConfigUpdate }) => {
  const { showSuccess, showError } = useNotification();
  const [testResults, setTestResults] = useState<TestResult | null>(null);
  const [showApiKey, setShowApiKey] = useState<boolean>(false);
  const [config, setConfig] = useState<ClientConfig>({
    email: '',
    api_key: '',
    base_url: '',
    timeout: 30,
    max_retries: 3,
    cache_ttl: 3600,
    use_cache: true
  });

  // Use the medical clients hook
  const {
    updateClientConfig,
    testClientConnection,
    useMockData
  } = useMedicalClients();

  // Get the mutation functions
  const {
    mutate: updateConfig,
    isPending: isUpdating
  } = updateClientConfig(client?.id || '');

  const {
    mutate: testConnection,
    isPending: isTesting
  } = testClientConnection(client?.id || '');

  // Initialize config from client data
  useEffect(() => {
    if (client && client.config) {
      setConfig(client.config);
    }
  }, [client]);

  // Handle config change
  const handleConfigChange = (field: string, value: any) => {
    setConfig({
      ...config,
      [field]: value
    });
  };

  // Handle form submission
  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!client) return;

    updateConfig({
      client_id: client.id,
      config
    }, {
      onSuccess: (data) => {
        showSuccess('NCBI client configuration updated successfully');
        if (onConfigUpdate) {
          onConfigUpdate(data);
        }
      }
    });
  };

  // Handle test connection
  const handleTestConnection = () => {
    if (!client) return;

    setTestResults(null);

    testConnection({
      client_id: client.id
    }, {
      onSuccess: (data) => {
        setTestResults(data);
        
        if (data.success) {
          showSuccess('NCBI connection test successful');
        } else {
          showError(`NCBI connection test failed: ${data.message}`);
        }
      }
    });
  };

  // Get status color
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'connected':
        return 'success';
      case 'disconnected':
        return 'error';
      case 'error':
        return 'error';
      default:
        return 'default';
    }
  };

  // Get status icon
  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'connected':
        return <CheckCircleIcon />;
      case 'disconnected':
        return <ErrorIcon />;
      case 'error':
        return <WarningIcon />;
      default:
        return <InfoIcon />;
    }
  };

  if (!client) {
    return (
      <Alert severity="warning">
        NCBI client not found. Please refresh the page.
      </Alert>
    );
  }

  return (
    <Box>
      {useMockData && (
        <Alert severity="info" sx={{ mb: 2 }}>
          Using mock data. Toggle the "Use Mock Data" feature flag to use real API data.
        </Alert>
      )}
      
      <Card variant="outlined">
        <CardHeader
          title="NCBI Client Configuration"
          subheader="Configure the National Center for Biotechnology Information client"
          action={
            <Chip
              label={client.status}
              color={getStatusColor(client.status)}
              icon={getStatusIcon(client.status)}
              size="small"
            />
          }
        />
        <CardContent>
          <form onSubmit={handleSubmit}>
            <Grid container spacing={2}>
              <Grid item xs={12} md={6}>
                <TextField
                  fullWidth
                  label="Email"
                  value={config.email || ''}
                  onChange={(e) => handleConfigChange('email', e.target.value)}
                  helperText="Email address for NCBI API requests"
                  required
                />
              </Grid>
              <Grid item xs={12} md={6}>
                <TextField
                  fullWidth
                  label="API Key"
                  type={showApiKey ? 'text' : 'password'}
                  value={config.api_key || ''}
                  onChange={(e) => handleConfigChange('api_key', e.target.value)}
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
              <Grid item xs={12}>
                <TextField
                  fullWidth
                  label="Base URL"
                  value={config.base_url || ''}
                  onChange={(e) => handleConfigChange('base_url', e.target.value)}
                  helperText="Base URL for NCBI API requests (e.g., https://eutils.ncbi.nlm.nih.gov/entrez/eutils/)"
                />
              </Grid>
              <Grid item xs={12} md={4}>
                <TextField
                  fullWidth
                  label="Timeout (seconds)"
                  type="number"
                  value={config.timeout || ''}
                  onChange={(e) => handleConfigChange('timeout', e.target.value)}
                  helperText="Request timeout in seconds"
                  inputProps={{ min: 1, step: 1 }}
                />
              </Grid>
              <Grid item xs={12} md={4}>
                <TextField
                  fullWidth
                  label="Max Retries"
                  type="number"
                  value={config.max_retries || ''}
                  onChange={(e) => handleConfigChange('max_retries', e.target.value)}
                  helperText="Maximum number of retries for failed requests"
                  inputProps={{ min: 0, step: 1 }}
                />
              </Grid>
              <Grid item xs={12} md={4}>
                <TextField
                  fullWidth
                  label="Cache TTL (seconds)"
                  type="number"
                  value={config.cache_ttl || ''}
                  onChange={(e) => handleConfigChange('cache_ttl', e.target.value)}
                  helperText="Cache time-to-live in seconds"
                  inputProps={{ min: 0, step: 1 }}
                />
              </Grid>
            </Grid>

            <Box sx={{ mt: 3, display: 'flex', justifyContent: 'flex-end' }}>
              <Button
                type="submit"
                variant="contained"
                color="primary"
                disabled={isUpdating}
                startIcon={isUpdating ? <ButtonLoader size={20} /> : <SettingsIcon />}
              >
                {isUpdating ? 'Saving...' : 'Save Configuration'}
              </Button>
            </Box>
          </form>

          <Divider sx={{ my: 3 }} />

          <Typography variant="h6" gutterBottom>
            Connection Test
          </Typography>
          <Button
            variant="outlined"
            color="primary"
            startIcon={isTesting ? <ButtonLoader size={20} /> : <RefreshIcon />}
            onClick={handleTestConnection}
            disabled={isTesting}
          >
            {isTesting ? 'Testing...' : 'Test Connection'}
          </Button>

          {testResults && (
            <Paper sx={{ mt: 2, p: 2 }}>
              <Typography variant="subtitle1" gutterBottom>
                Test Results:
              </Typography>
              <Alert severity={testResults.success ? "success" : "error"} sx={{ mb: 2 }}>
                {testResults.message}
              </Alert>
              {testResults.details?.api_version && (
                <Typography variant="body2">
                  API Version: {testResults.details.api_version}
                </Typography>
              )}
              {testResults.response_time && (
                <Typography variant="body2">
                  Response Time: {testResults.response_time.toFixed(2)}s
                </Typography>
              )}
              {testResults.details && Object.keys(testResults.details).length > 0 && (
                <Box sx={{ mt: 2 }}>
                  <Typography variant="subtitle2" gutterBottom>
                    Details:
                  </Typography>
                  <pre style={{
                    backgroundColor: '#f5f5f5',
                    padding: '8px',
                    borderRadius: '4px',
                    overflow: 'auto',
                    maxHeight: '200px',
                    fontSize: '0.8rem'
                  }}>
                    {JSON.stringify(testResults.details, null, 2)}
                  </pre>
                </Box>
              )}
            </Paper>
          )}
        </CardContent>
        <CardActions>
          <Button
            size="small"
            startIcon={<SearchIcon />}
            onClick={() => window.open('https://www.ncbi.nlm.nih.gov/', '_blank')}
          >
            Visit NCBI Website
          </Button>
          <Button
            size="small"
            startIcon={<InfoIcon />}
            onClick={() => window.open('https://www.ncbi.nlm.nih.gov/books/NBK25500/', '_blank')}
          >
            API Documentation
          </Button>
        </CardActions>
      </Card>
    </Box>
  );
};

export default NCBIClient;
