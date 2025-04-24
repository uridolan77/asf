import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Container,
  Paper,
  Breadcrumbs,
  Grid,
  Card,
  CardHeader,
  CardContent,
  Button,
  IconButton,
  Chip,
  CircularProgress,
  LinearProgress,
  Divider,
  Tooltip,
  List,
  ListItem,
  ListItemText,
  Alert
} from '@mui/material';
import { Link as RouterLink } from 'react-router-dom';
import NavigateNextIcon from '@mui/icons-material/NavigateNext';
import HomeIcon from '@mui/icons-material/Home';
import RefreshIcon from '@mui/icons-material/Refresh';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import WarningIcon from '@mui/icons-material/Warning';
import ErrorIcon from '@mui/icons-material/Error';
import SignalCellularAltIcon from '@mui/icons-material/SignalCellularAlt';
import AccessTimeIcon from '@mui/icons-material/AccessTime';
import SpeedIcon from '@mui/icons-material/Speed';

// For production use, you would use your actual API service
// import apiService from '../../../services/api';
import { useNotification } from '../../../context/NotificationContext';

/**
 * Provider Health Monitoring Page
 * Shows health status, latency metrics, and uptime for each LLM provider
 */
const ProviderHealthPage = () => {
  const { showSuccess, showError } = useNotification();
  
  // State
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [providerHealth, setProviderHealth] = useState([]);
  const [lastUpdated, setLastUpdated] = useState(null);
  
  // Load provider health data
  useEffect(() => {
    loadHealthData();
  }, []);
  
  // Load health data
  const loadHealthData = async () => {
    setLoading(true);
    
    try {
      // In production, use an actual API call
      // const result = await apiService.llm.getProviderHealth();
      
      // For development, use sample data
      setTimeout(() => {
        const sampleHealthData = [
          {
            provider_id: 'openai',
            display_name: 'OpenAI',
            status: 'operational',
            latency: 285,
            uptime: 99.98,
            checked_at: new Date().toISOString(),
            models_status: {
              'gpt-4-turbo': { status: 'operational', latency: 310 },
              'gpt-3.5-turbo': { status: 'operational', latency: 260 }
            },
            incidents: []
          },
          {
            provider_id: 'anthropic',
            display_name: 'Anthropic',
            status: 'degraded',
            latency: 420,
            uptime: 98.5,
            checked_at: new Date().toISOString(),
            models_status: {
              'claude-3-opus': { status: 'operational', latency: 450 },
              'claude-3-sonnet': { status: 'degraded', latency: 620 },
              'claude-3-haiku': { status: 'operational', latency: 320 }
            },
            incidents: [
              {
                id: 'inc_123',
                title: 'Increased latency on Claude Sonnet',
                status: 'investigating',
                created_at: new Date(Date.now() - 3600000).toISOString(),
                updated_at: new Date().toISOString(),
                description: 'We are investigating increased latency on Claude 3 Sonnet. Other models are unaffected.'
              }
            ]
          },
          {
            provider_id: 'cohere',
            display_name: 'Cohere',
            status: 'operational',
            latency: 315,
            uptime: 99.7,
            checked_at: new Date().toISOString(),
            models_status: {
              'command-r': { status: 'operational', latency: 340 },
              'command-r-plus': { status: 'operational', latency: 380 }
            },
            incidents: []
          },
          {
            provider_id: 'google',
            display_name: 'Google AI',
            status: 'error',
            latency: 0,
            uptime: 97.2,
            checked_at: new Date().toISOString(),
            models_status: {},
            incidents: [
              {
                id: 'inc_456',
                title: 'API Connection Error',
                status: 'identified',
                created_at: new Date(Date.now() - 7200000).toISOString(),
                updated_at: new Date(Date.now() - 1800000).toISOString(),
                description: 'We are experiencing connection issues with the Google AI API. Our team is working to resolve this issue.'
              }
            ]
          }
        ];
        
        setProviderHealth(sampleHealthData);
        setLastUpdated(new Date());
        setLoading(false);
      }, 800);
      
    } catch (error) {
      console.error('Error loading provider health data:', error);
      showError('Failed to load provider health data');
      setLoading(false);
    }
  };
  
  // Refresh health data
  const handleRefresh = async () => {
    setRefreshing(true);
    
    try {
      await loadHealthData();
      showSuccess('Provider health data refreshed');
    } catch (error) {
      console.error('Error refreshing provider health data:', error);
      showError('Failed to refresh provider health data');
    } finally {
      setRefreshing(false);
    }
  };
  
  // Get status color
  const getStatusColor = (status) => {
    switch (status) {
      case 'operational': return 'success';
      case 'degraded': return 'warning';
      case 'error': return 'error';
      default: return 'default';
    }
  };
  
  // Get status icon
  const getStatusIcon = (status) => {
    switch (status) {
      case 'operational': return <CheckCircleIcon fontSize="small" />;
      case 'degraded': return <WarningIcon fontSize="small" />;
      case 'error': return <ErrorIcon fontSize="small" />;
      default: return null;
    }
  };
  
  // Get latency color
  const getLatencyColor = (latency) => {
    if (latency === 0) return 'error';
    if (latency < 300) return 'success';
    if (latency < 500) return 'info';
    if (latency < 800) return 'warning';
    return 'error';
  };
  
  // Format date for display
  const formatDateTime = (dateString) => {
    if (!dateString) return 'N/A';
    return new Date(dateString).toLocaleString();
  };
  
  // Get short relative time
  const getRelativeTime = (dateString) => {
    if (!dateString) return '';
    
    const now = new Date();
    const date = new Date(dateString);
    const diffSeconds = Math.floor((now - date) / 1000);
    
    if (diffSeconds < 60) return 'Just now';
    if (diffSeconds < 3600) return `${Math.floor(diffSeconds / 60)}m ago`;
    if (diffSeconds < 86400) return `${Math.floor(diffSeconds / 3600)}h ago`;
    return `${Math.floor(diffSeconds / 86400)}d ago`;
  };
  
  return (
    <Container maxWidth="xl">
      <Box sx={{ pt: 2, pb: 4 }}>
        {/* Breadcrumbs navigation */}
        <Breadcrumbs 
          separator={<NavigateNextIcon fontSize="small" />} 
          aria-label="breadcrumb"
          sx={{ mb: 2 }}
        >
          <RouterLink 
            to="/dashboard" 
            style={{ display: 'flex', alignItems: 'center', textDecoration: 'none', color: 'inherit' }}
          >
            <HomeIcon sx={{ mr: 0.5 }} fontSize="small" />
            Dashboard
          </RouterLink>
          <RouterLink 
            to="/llm/dashboard" 
            style={{ textDecoration: 'none', color: 'inherit' }}
          >
            LLM Management
          </RouterLink>
          <RouterLink 
            to="/llm/providers" 
            style={{ textDecoration: 'none', color: 'inherit' }}
          >
            Providers
          </RouterLink>
          <Typography color="text.primary">Health Status</Typography>
        </Breadcrumbs>
        
        {/* Page header */}
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
          <Box>
            <Typography variant="h4" component="h1" gutterBottom>
              Provider Health Status
            </Typography>
            <Typography variant="body1" color="text.secondary">
              Monitor the health and performance of your LLM providers and models
            </Typography>
          </Box>
          <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
            {lastUpdated && (
              <Typography variant="body2" color="text.secondary" sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                <AccessTimeIcon fontSize="small" />
                Last updated: {formatDateTime(lastUpdated)}
              </Typography>
            )}
            <Button
              variant="outlined"
              startIcon={refreshing ? <CircularProgress size={16} /> : <RefreshIcon />}
              onClick={handleRefresh}
              disabled={refreshing || loading}
            >
              Refresh
            </Button>
          </Box>
        </Box>
        
        {/* Health status overview */}
        {loading ? (
          <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
            <CircularProgress />
          </Box>
        ) : (
          <>
            {/* Health overview cards */}
            <Grid container spacing={3} sx={{ mb: 4 }}>
              {providerHealth.map((provider) => (
                <Grid item xs={12} md={6} lg={3} key={provider.provider_id}>
                  <Card 
                    sx={{ 
                      height: '100%',
                      borderLeft: 6,
                      borderColor: `${getStatusColor(provider.status)}.main`
                    }}
                  >
                    <CardHeader
                      title={provider.display_name}
                      subheader={
                        <Box sx={{ display: 'flex', alignItems: 'center', mt: 0.5 }}>
                          <Chip
                            label={provider.status}
                            color={getStatusColor(provider.status)}
                            icon={getStatusIcon(provider.status)}
                            size="small"
                          />
                          <Typography variant="body2" color="text.secondary" sx={{ ml: 1 }}>
                            {getRelativeTime(provider.checked_at)}
                          </Typography>
                        </Box>
                      }
                    />
                    <CardContent>
                      <Grid container spacing={2}>
                        <Grid item xs={6}>
                          <Tooltip title="Average response time">
                            <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                              <SpeedIcon fontSize="small" color={getLatencyColor(provider.latency)} sx={{ mr: 1 }} />
                              <Typography variant="body2">Latency:</Typography>
                            </Box>
                          </Tooltip>
                          <Typography variant="h6" sx={{ ml: 4 }}>
                            {provider.latency > 0 ? `${provider.latency} ms` : 'N/A'}
                          </Typography>
                        </Grid>
                        <Grid item xs={6}>
                          <Tooltip title="Percentage of successful API calls">
                            <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                              <SignalCellularAltIcon fontSize="small" color="primary" sx={{ mr: 1 }} />
                              <Typography variant="body2">Uptime:</Typography>
                            </Box>
                          </Tooltip>
                          <Typography variant="h6" sx={{ ml: 4 }}>
                            {provider.uptime.toFixed(2)}%
                          </Typography>
                        </Grid>
                      </Grid>
                      
                      {provider.incidents.length > 0 && (
                        <Alert severity="warning" sx={{ mt: 2 }}>
                          {provider.incidents.length} active incident{provider.incidents.length > 1 ? 's' : ''}
                        </Alert>
                      )}
                    </CardContent>
                  </Card>
                </Grid>
              ))}
            </Grid>
            
            {/* Detailed health status */}
            <Paper sx={{ p: 3, mb: 4 }}>
              <Typography variant="h6" gutterBottom>
                Detailed Health Status
              </Typography>
              
              {providerHealth.map((provider) => (
                <Box key={provider.provider_id} sx={{ mb: 4 }}>
                  <Typography variant="subtitle1" sx={{ mb: 1, display: 'flex', alignItems: 'center' }}>
                    {provider.display_name}
                    <Chip
                      label={provider.status}
                      color={getStatusColor(provider.status)}
                      size="small"
                      sx={{ ml: 2 }}
                    />
                  </Typography>
                  
                  <Divider sx={{ mb: 2 }} />
                  
                  <Grid container spacing={3}>
                    {/* Model status */}
                    <Grid item xs={12} md={6}>
                      <Typography variant="subtitle2" gutterBottom>
                        Models Status
                      </Typography>
                      
                      {Object.keys(provider.models_status).length > 0 ? (
                        <List disablePadding dense>
                          {Object.entries(provider.models_status).map(([modelId, modelStatus]) => (
                            <ListItem key={modelId} disablePadding sx={{ py: 0.5 }}>
                              <ListItemText
                                primary={
                                  <Box sx={{ display: 'flex', alignItems: 'center' }}>
                                    <Typography variant="body2">{modelId}</Typography>
                                    <Chip
                                      label={modelStatus.status}
                                      color={getStatusColor(modelStatus.status)}
                                      size="small"
                                      sx={{ ml: 1 }}
                                    />
                                  </Box>
                                }
                                secondary={
                                  <Box sx={{ display: 'flex', alignItems: 'center', mt: 0.5 }}>
                                    <Typography variant="body2" color="text.secondary">
                                      Latency: {modelStatus.latency} ms
                                    </Typography>
                                    <Box sx={{ ml: 2, flexGrow: 1, maxWidth: 100 }}>
                                      <LinearProgress
                                        variant="determinate"
                                        value={Math.min(100, modelStatus.latency / 10)}
                                        color={getLatencyColor(modelStatus.latency)}
                                      />
                                    </Box>
                                  </Box>
                                }
                              />
                            </ListItem>
                          ))}
                        </List>
                      ) : (
                        <Typography variant="body2" color="text.secondary">
                          No model status information available
                        </Typography>
                      )}
                    </Grid>
                    
                    {/* Incidents */}
                    <Grid item xs={12} md={6}>
                      <Typography variant="subtitle2" gutterBottom>
                        Active Incidents
                      </Typography>
                      
                      {provider.incidents.length > 0 ? (
                        <List disablePadding>
                          {provider.incidents.map((incident) => (
                            <ListItem key={incident.id} sx={{ px: 0, py: 1, display: 'block' }}>
                              <Typography variant="body2" fontWeight="medium">
                                {incident.title}
                              </Typography>
                              <Box sx={{ display: 'flex', alignItems: 'center', my: 0.5 }}>
                                <Chip
                                  label={incident.status}
                                  size="small"
                                  color={incident.status === 'resolved' ? 'success' : 'warning'}
                                />
                                <Typography variant="caption" color="text.secondary" sx={{ ml: 1 }}>
                                  Reported {getRelativeTime(incident.created_at)}
                                </Typography>
                              </Box>
                              <Typography variant="body2" color="text.secondary">
                                {incident.description}
                              </Typography>
                            </ListItem>
                          ))}
                        </List>
                      ) : (
                        <Typography variant="body2" color="text.secondary">
                          No active incidents
                        </Typography>
                      )}
                    </Grid>
                  </Grid>
                </Box>
              ))}
            </Paper>
          </>
        )}
      </Box>
    </Container>
  );
};

export default ProviderHealthPage;