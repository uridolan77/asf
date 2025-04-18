import React, { useState, useEffect } from 'react';
import {
  Box,
  Paper,
  Typography,
  Button,
  Grid,
  Card,
  CardHeader,
  CardContent,
  Alert,
  CircularProgress,
  Chip,
  Divider,
  List,
  ListItem,
  ListItemIcon,
  ListItemText
} from '@mui/material';
import {
  CloudQueue as CloudQueueIcon,
  Speed as SpeedIcon,
  Timer as TimerIcon,
  Refresh as RefreshIcon,
  Error as ErrorIcon,
  CheckCircle as CheckCircleIcon
} from '@mui/icons-material';

import { useNotification } from '../../../context/NotificationContext';
import apiService from '../../../services/api';
import { ContentLoader } from '../../UI/LoadingIndicators';

/**
 * MCP Status Monitor Component
 * 
 * This component displays real-time status information for MCP providers,
 * including connection status, circuit breaker state, and performance metrics.
 */
const MCPStatusMonitor = ({ providerId }) => {
  const { showSuccess, showError } = useNotification();
  
  // State
  const [status, setStatus] = useState(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [autoRefresh, setAutoRefresh] = useState(false);
  const [refreshInterval, setRefreshInterval] = useState(null);
  
  // Load status on mount and when providerId changes
  useEffect(() => {
    if (providerId) {
      loadStatus();
    }
    
    return () => {
      // Clean up interval on unmount
      if (refreshInterval) {
        clearInterval(refreshInterval);
      }
    };
  }, [providerId]);
  
  // Set up auto-refresh
  useEffect(() => {
    if (autoRefresh) {
      const interval = setInterval(() => {
        loadStatus(false);
      }, 10000); // Refresh every 10 seconds
      
      setRefreshInterval(interval);
      
      return () => clearInterval(interval);
    } else if (refreshInterval) {
      clearInterval(refreshInterval);
      setRefreshInterval(null);
    }
  }, [autoRefresh, providerId]);
  
  // Load provider status
  const loadStatus = async (showLoading = true) => {
    if (!providerId) return;
    
    if (showLoading) {
      setLoading(true);
    } else {
      setRefreshing(true);
    }
    
    try {
      const result = await apiService.llm.getMCPProviderStatus(providerId);
      
      if (result.success) {
        setStatus(result.data);
      } else {
        showError(`Failed to load provider status: ${result.error}`);
      }
    } catch (error) {
      console.error('Error loading provider status:', error);
      showError(`Error loading provider status: ${error.message}`);
    } finally {
      if (showLoading) {
        setLoading(false);
      } else {
        setRefreshing(false);
      }
    }
  };
  
  // Toggle auto-refresh
  const toggleAutoRefresh = () => {
    setAutoRefresh(!autoRefresh);
  };
  
  // Refresh status
  const refreshStatus = async () => {
    await loadStatus(false);
    showSuccess('Status refreshed successfully');
  };
  
  // Get status color
  const getStatusColor = (status) => {
    switch (status) {
      case 'operational':
      case 'available':
      case 'connected':
        return 'success';
      case 'degraded':
      case 'warning':
        return 'warning';
      case 'error':
      case 'disconnected':
        return 'error';
      default:
        return 'default';
    }
  };
  
  // Get circuit breaker color
  const getCircuitBreakerColor = (state) => {
    switch (state) {
      case 'closed':
        return 'success';
      case 'half-open':
        return 'warning';
      case 'open':
        return 'error';
      default:
        return 'default';
    }
  };
  
  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h6">Provider Status</Typography>
        
        <Box sx={{ display: 'flex', gap: 1 }}>
          <Button
            variant="outlined"
            onClick={refreshStatus}
            disabled={refreshing || loading}
            startIcon={refreshing ? <CircularProgress size={20} /> : <RefreshIcon />}
          >
            {refreshing ? 'Refreshing...' : 'Refresh'}
          </Button>
          
          <Button
            variant={autoRefresh ? 'contained' : 'outlined'}
            color={autoRefresh ? 'primary' : 'inherit'}
            onClick={toggleAutoRefresh}
            startIcon={<TimerIcon />}
          >
            {autoRefresh ? 'Auto-Refresh On' : 'Auto-Refresh Off'}
          </Button>
        </Box>
      </Box>
      
      {loading ? (
        <ContentLoader height={200} message="Loading provider status..." />
      ) : !status ? (
        <Alert severity="info">
          No status information available for this provider.
        </Alert>
      ) : (
        <Grid container spacing={3}>
          {/* Status Overview */}
          <Grid item xs={12} md={6}>
            <Card variant="outlined">
              <CardHeader
                title="Status Overview"
                avatar={<CloudQueueIcon color={getStatusColor(status.status)} />}
              />
              <CardContent>
                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <Typography variant="body1">Status:</Typography>
                    <Chip
                      label={status.status}
                      color={getStatusColor(status.status)}
                      size="small"
                    />
                  </Box>
                  
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <Typography variant="body1">Provider ID:</Typography>
                    <Typography variant="body2" fontFamily="monospace">
                      {status.provider_id}
                    </Typography>
                  </Box>
                  
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <Typography variant="body1">Display Name:</Typography>
                    <Typography variant="body2">
                      {status.display_name || '-'}
                    </Typography>
                  </Box>
                  
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <Typography variant="body1">Transport Type:</Typography>
                    <Typography variant="body2">
                      {status.transport_type}
                    </Typography>
                  </Box>
                  
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <Typography variant="body1">Last Checked:</Typography>
                    <Typography variant="body2">
                      {new Date(status.checked_at).toLocaleString()}
                    </Typography>
                  </Box>
                  
                  {status.message && (
                    <Alert
                      severity={
                        status.status === 'error' ? 'error' :
                        status.status === 'warning' || status.status === 'degraded' ? 'warning' : 'info'
                      }
                      sx={{ mt: 1 }}
                    >
                      {status.message}
                    </Alert>
                  )}
                </Box>
              </CardContent>
            </Card>
          </Grid>
          
          {/* Circuit Breaker */}
          <Grid item xs={12} md={6}>
            <Card variant="outlined">
              <CardHeader
                title="Circuit Breaker"
                avatar={<SpeedIcon color={status.circuit_breaker ? getCircuitBreakerColor(status.circuit_breaker.state) : 'default'} />}
              />
              <CardContent>
                {status.circuit_breaker ? (
                  <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <Typography variant="body1">State:</Typography>
                      <Chip
                        label={status.circuit_breaker.state}
                        color={getCircuitBreakerColor(status.circuit_breaker.state)}
                        size="small"
                      />
                    </Box>
                    
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <Typography variant="body1">Failure Count:</Typography>
                      <Typography variant="body2">
                        {status.circuit_breaker.failure_count}
                      </Typography>
                    </Box>
                    
                    {status.circuit_breaker.last_failure && (
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <Typography variant="body1">Last Failure:</Typography>
                        <Typography variant="body2">
                          {new Date(status.circuit_breaker.last_failure).toLocaleString()}
                        </Typography>
                      </Box>
                    )}
                    
                    {status.circuit_breaker.reset_timeout && (
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <Typography variant="body1">Reset Timeout:</Typography>
                        <Typography variant="body2">
                          {status.circuit_breaker.reset_timeout} seconds
                        </Typography>
                      </Box>
                    )}
                    
                    <Alert
                      severity={
                        status.circuit_breaker.state === 'open' ? 'error' :
                        status.circuit_breaker.state === 'half-open' ? 'warning' : 'success'
                      }
                      sx={{ mt: 1 }}
                    >
                      {status.circuit_breaker.state === 'open' ? (
                        'Circuit breaker is open. Requests are being rejected to prevent cascading failures.'
                      ) : status.circuit_breaker.state === 'half-open' ? (
                        'Circuit breaker is half-open. Limited requests are being allowed to test recovery.'
                      ) : (
                        'Circuit breaker is closed. Requests are being processed normally.'
                      )}
                    </Alert>
                  </Box>
                ) : (
                  <Alert severity="info">
                    No circuit breaker information available for this provider.
                  </Alert>
                )}
              </CardContent>
            </Card>
          </Grid>
          
          {/* Available Models */}
          <Grid item xs={12}>
            <Card variant="outlined">
              <CardHeader title="Available Models" />
              <CardContent>
                {status.models && status.models.length > 0 ? (
                  <Grid container spacing={1}>
                    {status.models.map((model, index) => (
                      <Grid item key={index} xs={12} sm={6} md={4} lg={3}>
                        <Chip
                          label={model}
                          variant="outlined"
                          sx={{ width: '100%' }}
                        />
                      </Grid>
                    ))}
                  </Grid>
                ) : (
                  <Alert severity="info">
                    No models available for this provider.
                  </Alert>
                )}
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      )}
    </Box>
  );
};

export default MCPStatusMonitor;
