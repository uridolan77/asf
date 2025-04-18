import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  Grid,
  Divider,
  Chip,
  Alert,
  CircularProgress,
  Tabs,
  Tab,
  IconButton,
  Tooltip,
  useTheme
} from '@mui/material';
import {
  Dashboard as DashboardIcon,
  Refresh as RefreshIcon,
  OpenInNew as OpenInNewIcon,
  Settings as SettingsIcon,
  Error as ErrorIcon,
  CheckCircle as CheckCircleIcon,
  Warning as WarningIcon
} from '@mui/icons-material';
import apiService from '../../../services/api';

/**
 * Grafana Dashboards component for MCP providers
 * 
 * Displays available Grafana dashboards and provides links to open them.
 */
const GrafanaDashboards = ({ providerId }) => {
  const theme = useTheme();
  const [dashboards, setDashboards] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [setupStatus, setSetupStatus] = useState(null);
  const [setupInProgress, setSetupInProgress] = useState(false);
  const [provisionStatus, setProvisionStatus] = useState(null);
  const [provisionInProgress, setProvisionInProgress] = useState(false);
  const [tabValue, setTabValue] = useState(0);
  
  // Fetch dashboards
  const fetchDashboards = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const response = await apiService.llm.getGrafanaDashboards();
      setDashboards(response.dashboards || []);
    } catch (error) {
      console.error('Error fetching Grafana dashboards:', error);
      setError('Failed to load Grafana dashboards');
    } finally {
      setLoading(false);
    }
  };
  
  // Fetch dashboards on mount
  useEffect(() => {
    fetchDashboards();
  }, []);
  
  // Handle tab change
  const handleTabChange = (event, newValue) => {
    setTabValue(newValue);
  };
  
  // Handle setup Grafana
  const handleSetupGrafana = async () => {
    try {
      setSetupInProgress(true);
      setSetupStatus(null);
      
      const response = await apiService.llm.setupGrafana();
      setSetupStatus({
        success: true,
        message: response.message || 'Grafana setup started successfully'
      });
      
      // Refresh dashboards after a delay
      setTimeout(() => {
        fetchDashboards();
      }, 5000);
    } catch (error) {
      console.error('Error setting up Grafana:', error);
      setSetupStatus({
        success: false,
        message: error.response?.data?.detail || 'Failed to set up Grafana'
      });
    } finally {
      setSetupInProgress(false);
    }
  };
  
  // Handle provision provider dashboard
  const handleProvisionDashboard = async () => {
    if (!providerId) return;
    
    try {
      setProvisionInProgress(true);
      setProvisionStatus(null);
      
      const response = await apiService.llm.provisionProviderDashboard(providerId);
      setProvisionStatus({
        success: true,
        message: response.message || 'Dashboard provisioned successfully',
        url: response.url
      });
      
      // Refresh dashboards
      fetchDashboards();
    } catch (error) {
      console.error('Error provisioning dashboard:', error);
      setProvisionStatus({
        success: false,
        message: error.response?.data?.detail || 'Failed to provision dashboard'
      });
    } finally {
      setProvisionInProgress(false);
    }
  };
  
  // Get dashboard by ID
  const getDashboard = (id) => {
    return dashboards.find(dashboard => dashboard.dashboard_id === id);
  };
  
  // Get provider dashboard
  const getProviderDashboard = () => {
    if (!providerId) return null;
    
    return dashboards.find(dashboard => 
      dashboard.dashboard_id === `mcp-provider-${providerId}` || 
      dashboard.dashboard_id.includes(providerId)
    );
  };
  
  // Render dashboard card
  const renderDashboardCard = (dashboardId, title, description) => {
    const dashboard = getDashboard(dashboardId);
    const hasUrl = dashboard && dashboard.url;
    
    return (
      <Card variant="outlined" sx={{ mb: 2 }}>
        <CardContent>
          <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
            <Box display="flex" alignItems="center">
              <DashboardIcon sx={{ mr: 1, color: theme.palette.primary.main }} />
              <Typography variant="h6">{title}</Typography>
            </Box>
            
            {hasUrl ? (
              <Tooltip title="Open in Grafana">
                <IconButton 
                  color="primary"
                  onClick={() => window.open(dashboard.url, '_blank')}
                >
                  <OpenInNewIcon />
                </IconButton>
              </Tooltip>
            ) : (
              <Chip 
                label="Not Available" 
                color="default" 
                size="small" 
                variant="outlined" 
              />
            )}
          </Box>
          
          <Typography variant="body2" color="text.secondary" mb={2}>
            {description}
          </Typography>
          
          <Box display="flex" alignItems="center">
            <Chip
              icon={hasUrl ? <CheckCircleIcon /> : <WarningIcon />}
              label={hasUrl ? 'Available' : 'Not Provisioned'}
              color={hasUrl ? 'success' : 'warning'}
              variant="outlined"
              size="small"
            />
          </Box>
        </CardContent>
      </Card>
    );
  };
  
  return (
    <Box>
      <Tabs
        value={tabValue}
        onChange={handleTabChange}
        indicatorColor="primary"
        textColor="primary"
        variant="fullWidth"
        sx={{ mb: 2 }}
      >
        <Tab label="Available Dashboards" />
        <Tab label="Setup & Configuration" />
      </Tabs>
      
      {/* Available Dashboards Tab */}
      {tabValue === 0 && (
        <Box>
          {loading ? (
            <Box display="flex" justifyContent="center" alignItems="center" height={200}>
              <CircularProgress />
            </Box>
          ) : error ? (
            <Alert severity="error" sx={{ mb: 2 }}>
              {error}
            </Alert>
          ) : (
            <>
              <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
                <Typography variant="h6">
                  Grafana Dashboards
                </Typography>
                
                <Button
                  startIcon={<RefreshIcon />}
                  onClick={fetchDashboards}
                  size="small"
                >
                  Refresh
                </Button>
              </Box>
              
              <Grid container spacing={2}>
                <Grid item xs={12} md={6}>
                  {renderDashboardCard(
                    'mcp_overview',
                    'MCP Overview',
                    'Overview of all MCP providers with key metrics and status information.'
                  )}
                  
                  {renderDashboardCard(
                    'mcp_performance',
                    'MCP Performance',
                    'Detailed performance metrics for all MCP providers including latency, throughput, and token usage.'
                  )}
                </Grid>
                
                <Grid item xs={12} md={6}>
                  {renderDashboardCard(
                    'mcp_errors',
                    'MCP Errors',
                    'Error analysis and circuit breaker status for all MCP providers.'
                  )}
                  
                  {providerId && renderDashboardCard(
                    `mcp-provider-${providerId}`,
                    `${providerId} Dashboard`,
                    `Detailed metrics and status for the ${providerId} provider.`
                  )}
                </Grid>
              </Grid>
              
              {providerId && !getProviderDashboard() && (
                <Box mt={2}>
                  <Alert 
                    severity="info" 
                    action={
                      <Button
                        color="inherit"
                        size="small"
                        onClick={handleProvisionDashboard}
                        disabled={provisionInProgress}
                      >
                        {provisionInProgress ? 'Provisioning...' : 'Provision Now'}
                      </Button>
                    }
                  >
                    No dashboard found for provider {providerId}. You can provision a new dashboard.
                  </Alert>
                </Box>
              )}
              
              {provisionStatus && (
                <Alert 
                  severity={provisionStatus.success ? 'success' : 'error'}
                  sx={{ mt: 2 }}
                  action={
                    provisionStatus.url && (
                      <Button
                        color="inherit"
                        size="small"
                        onClick={() => window.open(provisionStatus.url, '_blank')}
                      >
                        Open
                      </Button>
                    )
                  }
                >
                  {provisionStatus.message}
                </Alert>
              )}
            </>
          )}
        </Box>
      )}
      
      {/* Setup & Configuration Tab */}
      {tabValue === 1 && (
        <Box>
          <Typography variant="h6" gutterBottom>
            Grafana Setup & Configuration
          </Typography>
          
          <Card variant="outlined" sx={{ mb: 2 }}>
            <CardContent>
              <Typography variant="subtitle1" gutterBottom>
                Setup Grafana Integration
              </Typography>
              
              <Typography variant="body2" color="text.secondary" paragraph>
                This will set up Grafana with the necessary datasources, dashboards, and alert rules for MCP monitoring.
                The setup process runs in the background and may take a few minutes to complete.
              </Typography>
              
              <Box display="flex" justifyContent="space-between" alignItems="center">
                <Button
                  variant="contained"
                  startIcon={<SettingsIcon />}
                  onClick={handleSetupGrafana}
                  disabled={setupInProgress}
                >
                  {setupInProgress ? 'Setting up...' : 'Setup Grafana'}
                </Button>
                
                {setupStatus && (
                  <Chip
                    icon={setupStatus.success ? <CheckCircleIcon /> : <ErrorIcon />}
                    label={setupStatus.message}
                    color={setupStatus.success ? 'success' : 'error'}
                    variant="outlined"
                  />
                )}
              </Box>
            </CardContent>
          </Card>
          
          <Card variant="outlined">
            <CardContent>
              <Typography variant="subtitle1" gutterBottom>
                Grafana Configuration
              </Typography>
              
              <Grid container spacing={2}>
                <Grid item xs={12} md={6}>
                  <Typography variant="body2" color="text.secondary" gutterBottom>
                    <strong>Prometheus URL:</strong>
                  </Typography>
                  <Typography variant="body2" gutterBottom>
                    http://localhost:9090
                  </Typography>
                </Grid>
                
                <Grid item xs={12} md={6}>
                  <Typography variant="body2" color="text.secondary" gutterBottom>
                    <strong>Grafana URL:</strong>
                  </Typography>
                  <Typography variant="body2" gutterBottom>
                    http://localhost:3000
                  </Typography>
                </Grid>
                
                <Grid item xs={12}>
                  <Divider sx={{ my: 1 }} />
                  <Typography variant="body2" color="text.secondary" gutterBottom>
                    <strong>Default Credentials:</strong>
                  </Typography>
                  <Typography variant="body2" gutterBottom>
                    Username: admin<br />
                    Password: admin
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    Note: You should change the default password after the first login.
                  </Typography>
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Box>
      )}
    </Box>
  );
};

export default GrafanaDashboards;
