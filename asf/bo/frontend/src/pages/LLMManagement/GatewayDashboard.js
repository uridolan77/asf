import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Paper,
  Grid,
  Card,
  CardHeader,
  CardContent,
  CardActions,
  Button,
  Chip,
  Divider,
  TextField,
  CircularProgress,
  Alert,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow
} from '@mui/material';
import {
  ExpandMore as ExpandMoreIcon,
  Settings as SettingsIcon,
  Refresh as RefreshIcon,
  Check as CheckIcon,
  Error as ErrorIcon,
  Warning as WarningIcon
} from '@mui/icons-material';

import { useNotification } from '../../context/NotificationContext';
import apiService from '../../services/api';
import { ContentLoader } from '../../components/UI/LoadingIndicators';

/**
 * LLM Gateway Dashboard component
 */
const GatewayDashboard = ({ status, onRefresh }) => {
  const { showSuccess, showError } = useNotification();
  
  const [providers, setProviders] = useState([]);
  const [loading, setLoading] = useState(true);
  const [testingProvider, setTestingProvider] = useState(null);
  const [testResults, setTestResults] = useState(null);
  const [configDialogOpen, setConfigDialogOpen] = useState(false);
  const [selectedProvider, setSelectedProvider] = useState(null);
  
  // Load providers on mount
  useEffect(() => {
    loadProviders();
  }, []);
  
  // Load providers
  const loadProviders = async () => {
    setLoading(true);
    
    try {
      const result = await apiService.llm.getProviders();
      
      if (result.success) {
        setProviders(result.data);
      } else {
        showError(`Failed to load providers: ${result.error}`);
      }
    } catch (error) {
      console.error('Error loading providers:', error);
      showError(`Error loading providers: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };
  
  // Handle provider test
  const handleTestProvider = async (providerId) => {
    setTestingProvider(providerId);
    setTestResults(null);
    
    try {
      const result = await apiService.llm.testProvider(providerId);
      
      if (result.success) {
        setTestResults(result.data);
        showSuccess(`Provider ${providerId} test successful`);
      } else {
        setTestResults(result.data);
        showError(`Provider ${providerId} test failed: ${result.data.message}`);
      }
    } catch (error) {
      console.error(`Error testing provider ${providerId}:`, error);
      showError(`Error testing provider ${providerId}: ${error.message}`);
    } finally {
      setTestingProvider(null);
    }
  };
  
  // Handle provider configuration
  const handleConfigureProvider = (provider) => {
    setSelectedProvider(provider);
    setConfigDialogOpen(true);
  };
  
  // Get status color
  const getStatusColor = (status) => {
    switch (status) {
      case 'operational':
        return 'success';
      case 'degraded':
        return 'warning';
      case 'error':
        return 'error';
      default:
        return 'default';
    }
  };
  
  // Get status icon
  const getStatusIcon = (status) => {
    switch (status) {
      case 'operational':
        return <CheckIcon fontSize="small" />;
      case 'degraded':
        return <WarningIcon fontSize="small" />;
      case 'error':
        return <ErrorIcon fontSize="small" />;
      default:
        return null;
    }
  };
  
  // Render provider cards
  const renderProviderCards = () => {
    return (
      <Grid container spacing={3}>
        {providers.map((provider) => (
          <Grid item xs={12} md={6} lg={4} key={provider.provider_id}>
            <Card 
              sx={{ 
                height: '100%', 
                display: 'flex', 
                flexDirection: 'column',
                transition: 'all 0.3s ease',
                '&:hover': {
                  boxShadow: 6,
                  transform: 'translateY(-4px)'
                }
              }}
            >
              <CardHeader
                title={provider.display_name || provider.provider_id}
                subheader={`Type: ${provider.provider_type}`}
                action={
                  <Chip
                    label={provider.status}
                    color={getStatusColor(provider.status)}
                    icon={getStatusIcon(provider.status)}
                    size="small"
                  />
                }
              />
              <CardContent sx={{ flexGrow: 1 }}>
                <Typography variant="body2" color="text.secondary" gutterBottom>
                  Last checked: {new Date(provider.checked_at).toLocaleString()}
                </Typography>
                
                <Typography variant="subtitle2" gutterBottom>
                  Available Models:
                </Typography>
                <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5, mb: 2 }}>
                  {provider.models.map((model) => (
                    <Chip
                      key={model}
                      label={model}
                      size="small"
                      variant="outlined"
                    />
                  ))}
                </Box>
                
                {testResults && testResults.provider_id === provider.provider_id && (
                  <Alert 
                    severity={testResults.success ? "success" : "error"}
                    sx={{ mt: 1 }}
                  >
                    {testResults.message}
                    {testResults.success && testResults.response && (
                      <Typography variant="body2" sx={{ mt: 1 }}>
                        Response: {testResults.response.substring(0, 100)}
                        {testResults.response.length > 100 ? '...' : ''}
                      </Typography>
                    )}
                  </Alert>
                )}
              </CardContent>
              <CardActions>
                <Button
                  size="small"
                  startIcon={<SettingsIcon />}
                  onClick={() => handleConfigureProvider(provider)}
                >
                  Configure
                </Button>
                <Button
                  size="small"
                  startIcon={testingProvider === provider.provider_id ? <CircularProgress size={16} /> : <RefreshIcon />}
                  onClick={() => handleTestProvider(provider.provider_id)}
                  disabled={testingProvider === provider.provider_id}
                >
                  Test Connection
                </Button>
              </CardActions>
            </Card>
          </Grid>
        ))}
      </Grid>
    );
  };
  
  if (loading) {
    return <ContentLoader height={200} message="Loading providers..." />;
  }
  
  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        LLM Gateway Management
      </Typography>
      
      <Typography paragraph>
        Manage LLM providers and configurations, test connections, and execute direct LLM requests.
      </Typography>
      
      {/* Provider cards */}
      {providers.length > 0 ? (
        renderProviderCards()
      ) : (
        <Alert severity="info">
          No providers found. Please refresh to load providers.
        </Alert>
      )}
      
      {/* Direct LLM testing */}
      <Accordion sx={{ mt: 4 }}>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography variant="subtitle1">Direct LLM Testing</Typography>
        </AccordionSummary>
        <AccordionDetails>
          <Typography paragraph>
            Test LLM models directly by sending requests through the gateway.
          </Typography>
          
          <Typography variant="subtitle2" gutterBottom>
            Coming soon...
          </Typography>
        </AccordionDetails>
      </Accordion>
      
      {/* Gateway configuration */}
      <Accordion sx={{ mt: 2 }}>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography variant="subtitle1">Gateway Configuration</Typography>
        </AccordionSummary>
        <AccordionDetails>
          <Typography paragraph>
            Configure global gateway settings.
          </Typography>
          
          <Typography variant="subtitle2" gutterBottom>
            Coming soon...
          </Typography>
        </AccordionDetails>
      </Accordion>
    </Box>
  );
};

export default GatewayDashboard;
