import React, { useState, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import {
  Box,
  Container,
  Grid,
  Paper,
  Typography,
  Tabs,
  Tab,
  Button,
  Divider,
  Chip,
  Alert,
  Skeleton,
  useTheme
} from '@mui/material';
import {
  Refresh as RefreshIcon,
  Settings as SettingsIcon,
  Storage as StorageIcon,
  Speed as SpeedIcon,
  BugReport as BugReportIcon,
  Insights as InsightsIcon
} from '@mui/icons-material';

import PageLayout from '../components/Layout/PageLayout';
import MCPUsageStats from '../components/LLM/MCP/MCPUsageStats';
import CircuitBreakerStatus from '../components/LLM/MCP/CircuitBreakerStatus';
import { useMCPWebSocket } from '../hooks/useMCPWebSocket';
import apiService from '../services/api';

// Tab panel component
function TabPanel(props) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`provider-tabpanel-${index}`}
      aria-labelledby={`provider-tab-${index}`}
      {...other}
    >
      {value === index && (
        <Box sx={{ p: 3 }}>
          {children}
        </Box>
      )}
    </div>
  );
}

/**
 * MCP Provider Dashboard
 * 
 * Comprehensive dashboard for monitoring and managing an MCP provider.
 */
const MCPProviderDashboard = () => {
  const theme = useTheme();
  const { providerId } = useParams();
  const [tabValue, setTabValue] = useState(0);
  const [provider, setProvider] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [connectionPoolStats, setConnectionPoolStats] = useState(null);
  const [loadingPoolStats, setLoadingPoolStats] = useState(false);
  
  // Get real-time status via WebSocket
  const {
    status,
    isConnected,
    requestStatus,
    connectionState
  } = useMCPWebSocket(providerId);
  
  // Fetch provider details
  useEffect(() => {
    const fetchProvider = async () => {
      try {
        setLoading(true);
        const data = await apiService.llm.getProvider(providerId);
        setProvider(data);
        setError(null);
      } catch (error) {
        console.error(`Error fetching provider ${providerId}:`, error);
        setError('Failed to load provider details');
      } finally {
        setLoading(false);
      }
    };
    
    fetchProvider();
  }, [providerId]);
  
  // Fetch connection pool stats
  const fetchConnectionPoolStats = async () => {
    try {
      setLoadingPoolStats(true);
      const data = await apiService.llm.getConnectionPoolStats(providerId);
      setConnectionPoolStats(data);
    } catch (error) {
      console.error(`Error fetching connection pool stats for ${providerId}:`, error);
    } finally {
      setLoadingPoolStats(false);
    }
  };
  
  // Fetch connection pool stats when tab changes to "Connections"
  useEffect(() => {
    if (tabValue === 2) {
      fetchConnectionPoolStats();
    }
  }, [tabValue, providerId]);
  
  // Handle tab change
  const handleTabChange = (event, newValue) => {
    setTabValue(newValue);
  };
  
  // Handle refresh
  const handleRefresh = () => {
    requestStatus();
    
    if (tabValue === 2) {
      fetchConnectionPoolStats();
    }
  };
  
  // Handle session refresh
  const handleSessionRefresh = async () => {
    try {
      await apiService.llm.refreshSessions(providerId);
      fetchConnectionPoolStats();
    } catch (error) {
      console.error(`Error refreshing sessions for ${providerId}:`, error);
    }
  };
  
  // Handle circuit breaker reset
  const handleCircuitBreakerReset = async () => {
    try {
      await apiService.llm.resetCircuitBreaker(providerId);
      requestStatus();
    } catch (error) {
      console.error(`Error resetting circuit breaker for ${providerId}:`, error);
    }
  };
  
  // Get status color
  const getStatusColor = () => {
    if (!status) return theme.palette.grey[500];
    
    switch (status.status) {
      case 'available':
        return theme.palette.success.main;
      case 'unavailable':
        return theme.palette.error.main;
      case 'unhealthy':
        return theme.palette.warning.main;
      default:
        return theme.palette.grey[500];
    }
  };
  
  // Get WebSocket status color
  const getWebSocketStatusColor = () => {
    if (!isConnected) return theme.palette.error.main;
    return theme.palette.success.main;
  };
  
  return (
    <PageLayout title={`MCP Provider: ${providerId}`}>
      <Container maxWidth="xl">
        {error && (
          <Alert severity="error" sx={{ mb: 2 }}>
            {error}
          </Alert>
        )}
        
        {/* Provider Header */}
        <Paper sx={{ p: 2, mb: 2 }}>
          <Box display="flex" justifyContent="space-between" alignItems="center">
            <Box>
              {loading ? (
                <Skeleton variant="text" width={300} height={40} />
              ) : (
                <Typography variant="h4" component="h1">
                  {provider?.display_name || providerId}
                </Typography>
              )}
              
              {loading ? (
                <Skeleton variant="text" width={200} />
              ) : (
                <Typography variant="body2" color="text.secondary">
                  {provider?.provider_type || 'MCP Provider'}
                </Typography>
              )}
            </Box>
            
            <Box>
              <Button
                variant="outlined"
                startIcon={<RefreshIcon />}
                onClick={handleRefresh}
                sx={{ mr: 1 }}
              >
                Refresh
              </Button>
              
              <Button
                variant="outlined"
                startIcon={<SettingsIcon />}
                href={`/mcp/providers/${providerId}/settings`}
              >
                Settings
              </Button>
            </Box>
          </Box>
          
          <Divider sx={{ my: 2 }} />
          
          <Grid container spacing={2}>
            <Grid item xs={12} md={4}>
              <Box display="flex" alignItems="center">
                <Box
                  sx={{
                    width: 12,
                    height: 12,
                    borderRadius: '50%',
                    bgcolor: getStatusColor(),
                    mr: 1
                  }}
                />
                <Typography variant="body1">
                  Status: <strong>{status?.status || 'Unknown'}</strong>
                </Typography>
              </Box>
              <Typography variant="body2" color="text.secondary">
                {status?.message || 'No status message available'}
              </Typography>
            </Grid>
            
            <Grid item xs={12} md={4}>
              <Box display="flex" alignItems="center">
                <Box
                  sx={{
                    width: 12,
                    height: 12,
                    borderRadius: '50%',
                    bgcolor: getWebSocketStatusColor(),
                    mr: 1
                  }}
                />
                <Typography variant="body1">
                  WebSocket: <strong>{isConnected ? 'Connected' : 'Disconnected'}</strong>
                </Typography>
              </Box>
              <Typography variant="body2" color="text.secondary">
                {connectionState || 'No connection state available'}
              </Typography>
            </Grid>
            
            <Grid item xs={12} md={4}>
              <Box display="flex" flexWrap="wrap" gap={1}>
                {loading ? (
                  <Skeleton variant="rectangular" width={300} height={32} />
                ) : (
                  <>
                    {provider?.models?.map(model => (
                      <Chip
                        key={model}
                        label={model}
                        size="small"
                        color="primary"
                        variant="outlined"
                      />
                    ))}
                    
                    {!provider?.models?.length && (
                      <Typography variant="body2" color="text.secondary">
                        No models available
                      </Typography>
                    )}
                  </>
                )}
              </Box>
            </Grid>
          </Grid>
        </Paper>
        
        {/* Tabs */}
        <Paper sx={{ mb: 2 }}>
          <Tabs
            value={tabValue}
            onChange={handleTabChange}
            indicatorColor="primary"
            textColor="primary"
            variant="scrollable"
            scrollButtons="auto"
          >
            <Tab label="Overview" icon={<InsightsIcon />} iconPosition="start" />
            <Tab label="Metrics" icon={<SpeedIcon />} iconPosition="start" />
            <Tab label="Connections" icon={<StorageIcon />} iconPosition="start" />
            <Tab label="Errors" icon={<BugReportIcon />} iconPosition="start" />
          </Tabs>
          
          {/* Overview Tab */}
          <TabPanel value={tabValue} index={0}>
            <Grid container spacing={3}>
              <Grid item xs={12} md={8}>
                <MCPUsageStats providerId={providerId} />
              </Grid>
              
              <Grid item xs={12} md={4}>
                <CircuitBreakerStatus providerId={providerId} />
                
                <Paper variant="outlined" sx={{ p: 2 }}>
                  <Typography variant="h6" gutterBottom>
                    Quick Actions
                  </Typography>
                  
                  <Box display="flex" flexDirection="column" gap={1}>
                    <Button
                      variant="outlined"
                      onClick={handleSessionRefresh}
                      startIcon={<RefreshIcon />}
                    >
                      Refresh Sessions
                    </Button>
                    
                    <Button
                      variant="outlined"
                      color="warning"
                      onClick={handleCircuitBreakerReset}
                      startIcon={<BugReportIcon />}
                    >
                      Reset Circuit Breaker
                    </Button>
                  </Box>
                </Paper>
              </Grid>
            </Grid>
          </TabPanel>
          
          {/* Metrics Tab */}
          <TabPanel value={tabValue} index={1}>
            <Typography variant="h6" gutterBottom>
              Performance Metrics
            </Typography>
            
            {/* Metrics content will go here */}
            <Alert severity="info">
              Detailed metrics visualization is under development.
            </Alert>
          </TabPanel>
          
          {/* Connections Tab */}
          <TabPanel value={tabValue} index={2}>
            <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
              <Typography variant="h6">
                Connection Pool
              </Typography>
              
              <Button
                variant="outlined"
                startIcon={<RefreshIcon />}
                onClick={fetchConnectionPoolStats}
                disabled={loadingPoolStats}
              >
                Refresh
              </Button>
            </Box>
            
            {loadingPoolStats ? (
              <Skeleton variant="rectangular" height={200} />
            ) : connectionPoolStats ? (
              <Grid container spacing={2}>
                <Grid item xs={12} md={6}>
                  <Paper variant="outlined" sx={{ p: 2 }}>
                    <Typography variant="subtitle1" gutterBottom>
                      Pool Statistics
                    </Typography>
                    
                    <Grid container spacing={2}>
                      <Grid item xs={6}>
                        <Typography variant="body2" color="text.secondary">
                          Pool Size
                        </Typography>
                        <Typography variant="h6">
                          {connectionPoolStats.pool_size} / {connectionPoolStats.max_size}
                        </Typography>
                      </Grid>
                      
                      <Grid item xs={6}>
                        <Typography variant="body2" color="text.secondary">
                          Active Connections
                        </Typography>
                        <Typography variant="h6">
                          {connectionPoolStats.active_connections || 0}
                        </Typography>
                      </Grid>
                      
                      <Grid item xs={6}>
                        <Typography variant="body2" color="text.secondary">
                          Transport Type
                        </Typography>
                        <Typography variant="h6">
                          {connectionPoolStats.transport_type || 'Unknown'}
                        </Typography>
                      </Grid>
                      
                      <Grid item xs={6}>
                        <Typography variant="body2" color="text.secondary">
                          Idle Timeout
                        </Typography>
                        <Typography variant="h6">
                          {connectionPoolStats.idle_timeout || 'N/A'} seconds
                        </Typography>
                      </Grid>
                    </Grid>
                  </Paper>
                </Grid>
                
                <Grid item xs={12} md={6}>
                  <Paper variant="outlined" sx={{ p: 2 }}>
                    <Typography variant="subtitle1" gutterBottom>
                      Connection Health
                    </Typography>
                    
                    <Grid container spacing={2}>
                      <Grid item xs={6}>
                        <Typography variant="body2" color="text.secondary">
                          Success Rate
                        </Typography>
                        <Typography variant="h6">
                          {connectionPoolStats.success_rate 
                            ? `${(connectionPoolStats.success_rate * 100).toFixed(1)}%` 
                            : 'N/A'}
                        </Typography>
                      </Grid>
                      
                      <Grid item xs={6}>
                        <Typography variant="body2" color="text.secondary">
                          Avg Response Time
                        </Typography>
                        <Typography variant="h6">
                          {connectionPoolStats.avg_response_time_ms 
                            ? `${connectionPoolStats.avg_response_time_ms.toFixed(1)} ms` 
                            : 'N/A'}
                        </Typography>
                      </Grid>
                      
                      <Grid item xs={6}>
                        <Typography variant="body2" color="text.secondary">
                          Total Requests
                        </Typography>
                        <Typography variant="h6">
                          {connectionPoolStats.total_requests || 0}
                        </Typography>
                      </Grid>
                      
                      <Grid item xs={6}>
                        <Typography variant="body2" color="text.secondary">
                          Error Count
                        </Typography>
                        <Typography variant="h6">
                          {connectionPoolStats.error_count || 0}
                        </Typography>
                      </Grid>
                    </Grid>
                  </Paper>
                </Grid>
                
                {connectionPoolStats.connections && connectionPoolStats.connections.length > 0 && (
                  <Grid item xs={12}>
                    <Paper variant="outlined" sx={{ p: 2 }}>
                      <Typography variant="subtitle1" gutterBottom>
                        Active Connections
                      </Typography>
                      
                      <Box sx={{ overflowX: 'auto' }}>
                        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                          <thead>
                            <tr>
                              <th style={{ textAlign: 'left', padding: '8px', borderBottom: '1px solid #ddd' }}>ID</th>
                              <th style={{ textAlign: 'left', padding: '8px', borderBottom: '1px solid #ddd' }}>Created</th>
                              <th style={{ textAlign: 'left', padding: '8px', borderBottom: '1px solid #ddd' }}>Last Used</th>
                              <th style={{ textAlign: 'right', padding: '8px', borderBottom: '1px solid #ddd' }}>Requests</th>
                              <th style={{ textAlign: 'right', padding: '8px', borderBottom: '1px solid #ddd' }}>Errors</th>
                              <th style={{ textAlign: 'right', padding: '8px', borderBottom: '1px solid #ddd' }}>Success Rate</th>
                              <th style={{ textAlign: 'right', padding: '8px', borderBottom: '1px solid #ddd' }}>Avg Time (ms)</th>
                            </tr>
                          </thead>
                          <tbody>
                            {connectionPoolStats.connections.map((conn, index) => (
                              <tr key={conn.id || index}>
                                <td style={{ padding: '8px', borderBottom: '1px solid #ddd' }}>{conn.id}</td>
                                <td style={{ padding: '8px', borderBottom: '1px solid #ddd' }}>
                                  {new Date(conn.created_at).toLocaleString()}
                                </td>
                                <td style={{ padding: '8px', borderBottom: '1px solid #ddd' }}>
                                  {new Date(conn.last_used_at).toLocaleString()}
                                </td>
                                <td style={{ textAlign: 'right', padding: '8px', borderBottom: '1px solid #ddd' }}>
                                  {conn.request_count}
                                </td>
                                <td style={{ textAlign: 'right', padding: '8px', borderBottom: '1px solid #ddd' }}>
                                  {conn.error_count}
                                </td>
                                <td style={{ textAlign: 'right', padding: '8px', borderBottom: '1px solid #ddd' }}>
                                  {(conn.success_rate * 100).toFixed(1)}%
                                </td>
                                <td style={{ textAlign: 'right', padding: '8px', borderBottom: '1px solid #ddd' }}>
                                  {conn.avg_response_time_ms.toFixed(1)}
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </Box>
                    </Paper>
                  </Grid>
                )}
              </Grid>
            ) : (
              <Alert severity="info">
                No connection pool statistics available.
              </Alert>
            )}
          </TabPanel>
          
          {/* Errors Tab */}
          <TabPanel value={tabValue} index={3}>
            <Typography variant="h6" gutterBottom>
              Error Analysis
            </Typography>
            
            {/* Errors content will go here */}
            <Alert severity="info">
              Error analysis visualization is under development.
            </Alert>
          </TabPanel>
        </Paper>
      </Container>
    </PageLayout>
  );
};

export default MCPProviderDashboard;
