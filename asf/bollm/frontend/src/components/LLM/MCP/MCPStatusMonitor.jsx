import React, { useState, useEffect } from 'react';
import {
  Box,
  Paper,
  Typography,
  Grid,
  Card,
  CardContent,
  CardHeader,
  Divider,
  Chip,
  CircularProgress,
  Button,
  Alert,
  LinearProgress,
  Tooltip,
  Badge
} from '@mui/material';
import {
  CheckCircle as CheckCircleIcon,
  Error as ErrorIcon,
  Warning as WarningIcon,
  Refresh as RefreshIcon,
  Speed as SpeedIcon,
  Memory as MemoryIcon,
  Storage as StorageIcon,
  Sync as SyncIcon,
  Wifi as WifiIcon,
  WifiOff as WifiOffIcon,
  Notifications as NotificationsIcon
} from '@mui/icons-material';
import { useQuery } from '@tanstack/react-query';

import apiService from '../../../services/api';
import { ContentLoader } from '../../UI/LoadingIndicators';
import { useMCPWebSocket } from '../../../hooks/useMCPWebSocket';

/**
 * MCPStatusMonitor Component
 *
 * This component provides real-time monitoring of MCP providers,
 * including status, health checks, and circuit breaker state.
 */
const MCPStatusMonitor = ({ providerId }) => {
  // Fetch provider status via REST API
  const {
    data: status,
    isLoading,
    isError,
    error,
    refetch
  } = useQuery({
    queryKey: ['mcpProviderStatus', providerId],
    queryFn: () => apiService.llm.getMCPProviderStatus(providerId),
    enabled: !!providerId,
    refetchInterval: 30000, // Refetch every 30 seconds
    refetchOnWindowFocus: true,
    onError: (err) => {
      console.error(`Error fetching status for provider ${providerId}:`, err);
    }
  });

  // Get real-time updates via WebSocket with fallback
  let wsHook;
  try {
    wsHook = useMCPWebSocket(providerId);
  } catch (error) {
    console.error('Error using WebSocket hook:', error);
    wsHook = {
      isConnected: false,
      statusUpdates: [],
      requestStatusUpdate: () => Promise.resolve(false)
    };
  }

  const {
    isConnected: wsConnected,
    statusUpdates,
    requestStatusUpdate
  } = wsHook;

  // Request status update when component mounts
  useEffect(() => {
    if (providerId && wsConnected) {
      requestStatusUpdate();
    }
  }, [providerId, wsConnected, requestStatusUpdate]);

  // Use the most recent status update from WebSocket if available
  const latestStatus = statusUpdates.length > 0 ? statusUpdates[0] : null;
  const displayStatus = latestStatus || status;

  // Get status color
  const getStatusColor = (status) => {
    if (!status) return 'default';

    return status === 'operational' ||
           status === 'available' ||
           status === 'connected' ? 'success' :
           status === 'degraded' || status === 'partial' ? 'warning' :
           status === 'error' || status === 'unavailable' ? 'error' : 'default';
  };

  // Get status icon
  const getStatusIcon = (status) => {
    if (!status) return <SyncIcon />;

    return status === 'operational' ||
           status === 'available' ||
           status === 'connected' ? <CheckCircleIcon color="success" /> :
           status === 'degraded' || status === 'partial' ? <WarningIcon color="warning" /> :
           status === 'error' || status === 'unavailable' ? <ErrorIcon color="error" /> : <SyncIcon />;
  };

  // Format date
  const formatDate = (dateString) => {
    if (!dateString) return 'N/A';

    try {
      return new Date(dateString).toLocaleString();
    } catch (error) {
      return dateString;
    }
  };

  // Calculate time since last check
  const getTimeSinceLastCheck = (checkedAt) => {
    if (!checkedAt) return 'N/A';

    try {
      const checkedDate = new Date(checkedAt);
      const now = new Date();
      const diffMs = now - checkedDate;
      const diffSec = Math.floor(diffMs / 1000);

      if (diffSec < 60) {
        return `${diffSec} seconds ago`;
      } else if (diffSec < 3600) {
        return `${Math.floor(diffSec / 60)} minutes ago`;
      } else if (diffSec < 86400) {
        return `${Math.floor(diffSec / 3600)} hours ago`;
      } else {
        return `${Math.floor(diffSec / 86400)} days ago`;
      }
    } catch (error) {
      return checkedAt;
    }
  };

  if (!providerId) {
    return (
      <Alert severity="info">
        Please select a provider to view its status.
      </Alert>
    );
  }

  if (isLoading) {
    return <ContentLoader height={200} message={`Loading status for provider ${providerId}...`} />;
  }

  if (isError) {
    return (
      <Alert
        severity="error"
        action={
          <Button color="inherit" size="small" onClick={() => refetch()}>
            Retry
          </Button>
        }
      >
        Error loading provider status: {error?.message || 'Unknown error'}
      </Alert>
    );
  }

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h6">Provider Status: {displayStatus?.display_name || providerId}</Typography>

        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          <Tooltip title={wsConnected ? 'WebSocket Connected' : 'WebSocket Disconnected'}>
            {wsConnected ? (
              <Badge color="success" variant="dot">
                <WifiIcon color="action" />
              </Badge>
            ) : (
              <WifiOffIcon color="disabled" />
            )}
          </Tooltip>

          <Tooltip title="Last updated">
            <Typography variant="body2" color="text.secondary">
              {getTimeSinceLastCheck(displayStatus?.checked_at)}
            </Typography>
          </Tooltip>

          <Button
            variant="outlined"
            size="small"
            startIcon={<RefreshIcon />}
            onClick={() => {
              refetch();
              if (wsConnected) {
                requestStatusUpdate();
              }
            }}
          >
            Refresh
          </Button>
        </Box>
      </Box>

      {statusUpdates.length > 0 && (
        <Alert severity="info" sx={{ mb: 3 }}>
          <Box sx={{ display: 'flex', alignItems: 'center' }}>
            <NotificationsIcon sx={{ mr: 1 }} />
            <Typography variant="body2">
              Receiving real-time updates via WebSocket
            </Typography>
          </Box>
        </Alert>
      )}

      <Grid container spacing={3}>
        {/* Status Card */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardHeader
              title="Provider Status"
              avatar={getStatusIcon(status?.status)}
              action={
                <Chip
                  label={status?.status || 'unknown'}
                  color={getStatusColor(status?.status)}
                />
              }
            />
            <Divider />
            <CardContent>
              <Grid container spacing={2}>
                <Grid item xs={6}>
                  <Typography variant="subtitle2">Provider ID:</Typography>
                  <Typography variant="body2">{displayStatus?.provider_id || 'N/A'}</Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="subtitle2">Transport Type:</Typography>
                  <Typography variant="body2">{displayStatus?.transport_type || 'N/A'}</Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="subtitle2">Last Checked:</Typography>
                  <Typography variant="body2">{formatDate(displayStatus?.checked_at)}</Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="subtitle2">Message:</Typography>
                  <Typography variant="body2">{displayStatus?.message || 'No message'}</Typography>
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>

        {/* Circuit Breaker Card */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardHeader
              title="Circuit Breaker"
              avatar={
                <Tooltip title="Circuit Breaker Status">
                  {displayStatus?.circuit_breaker?.state === 'open' ? (
                    <ErrorIcon color="error" />
                  ) : (
                    <CheckCircleIcon color="success" />
                  )}
                </Tooltip>
              }
              action={
                <Chip
                  label={displayStatus?.circuit_breaker?.state || 'unknown'}
                  color={displayStatus?.circuit_breaker?.state === 'open' ? 'error' : 'success'}
                />
              }
            />
            <Divider />
            <CardContent>
              <Grid container spacing={2}>
                <Grid item xs={12}>
                  <Typography variant="subtitle2">Failure Count:</Typography>
                  <Box sx={{ display: 'flex', alignItems: 'center', mt: 1 }}>
                    <Box sx={{ width: '100%', mr: 1 }}>
                      <LinearProgress
                        variant="determinate"
                        value={Math.min((displayStatus?.circuit_breaker?.failure_count || 0) * 20, 100)}
                        color={
                          (displayStatus?.circuit_breaker?.failure_count || 0) >= 5 ? 'error' :
                          (displayStatus?.circuit_breaker?.failure_count || 0) >= 3 ? 'warning' : 'primary'
                        }
                      />
                    </Box>
                    <Box sx={{ minWidth: 35 }}>
                      <Typography variant="body2" color="text.secondary">
                        {displayStatus?.circuit_breaker?.failure_count || 0}/5
                      </Typography>
                    </Box>
                  </Box>
                </Grid>

                {displayStatus?.circuit_breaker?.recovery_time && (
                  <Grid item xs={12}>
                    <Typography variant="subtitle2">Recovery Time:</Typography>
                    <Typography variant="body2">
                      {formatDate(displayStatus?.circuit_breaker?.recovery_time)}
                    </Typography>
                  </Grid>
                )}

                {displayStatus?.circuit_breaker?.last_failure && (
                  <Grid item xs={12}>
                    <Typography variant="subtitle2">Last Failure:</Typography>
                    <Typography variant="body2">
                      {formatDate(displayStatus?.circuit_breaker?.last_failure)}
                    </Typography>
                  </Grid>
                )}
              </Grid>
            </CardContent>
          </Card>
        </Grid>

        {/* Models Card */}
        <Grid item xs={12}>
          <Card>
            <CardHeader
              title="Supported Models"
              avatar={<StorageIcon />}
            />
            <Divider />
            <CardContent>
              {displayStatus?.models && displayStatus.models.length > 0 ? (
                <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                  {displayStatus.models.map((model) => (
                    <Chip
                      key={model}
                      label={model}
                      variant="outlined"
                      color="primary"
                    />
                  ))}
                </Box>
              ) : (
                <Typography variant="body2" color="text.secondary">
                  No models available for this provider.
                </Typography>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default MCPStatusMonitor;
