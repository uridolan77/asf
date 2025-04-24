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
  ToggleButtonGroup,
  ToggleButton,
  CircularProgress,
  Button,
  Alert,
  Tooltip,
  Badge
} from '@mui/material';
import {
  Refresh as RefreshIcon,
  TrendingUp as TrendingUpIcon,
  Speed as SpeedIcon,
  Token as TokenIcon,
  QueryStats as QueryStatsIcon,
  Wifi as WifiIcon,
  WifiOff as WifiOffIcon
} from '@mui/icons-material';
import { useQuery } from '@tanstack/react-query';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip,
  Legend, ResponsiveContainer, LineChart, Line, AreaChart, Area
} from 'recharts';

import apiService from '../../../services/api';
import { ContentLoader } from '../../UI/LoadingIndicators';
import { useMCPWebSocket } from '../../../hooks/useMCPWebSocket';

/**
 * MCPUsageStats Component
 *
 * This component provides visualizations of MCP provider usage statistics,
 * including request counts, token usage, and latency metrics.
 */
const MCPUsageStats = ({ providerId }) => {
  // State
  const [period, setPeriod] = useState('day');
  const [realtimeMetrics, setRealtimeMetrics] = useState(null);

  // Fetch provider usage stats via REST API
  const {
    data: usageStats,
    isLoading,
    isError,
    error,
    refetch
  } = useQuery({
    queryKey: ['mcpProviderUsage', providerId, period],
    queryFn: () => apiService.llm.getMCPProviderUsage(providerId, period),
    enabled: !!providerId,
    refetchInterval: 60000, // Refetch every minute
    refetchOnWindowFocus: true,
    onError: (err) => {
      console.error(`Error fetching usage stats for provider ${providerId}:`, err);
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
      metrics: null,
      requestMetrics: () => Promise.resolve(false)
    };
  }

  const {
    isConnected: wsConnected,
    metrics,
    requestMetrics
  } = wsHook;

  // Request metrics update when component mounts
  useEffect(() => {
    if (providerId && wsConnected) {
      requestMetrics();
    }
  }, [providerId, wsConnected, requestMetrics]);

  // Update metrics when new data arrives via WebSocket
  useEffect(() => {
    if (metrics) {
      setRealtimeMetrics(metrics);
    }
  }, [metrics]);

  // Combine REST API data with real-time WebSocket data
  const displayStats = realtimeMetrics || usageStats;

  // Handle period change
  const handlePeriodChange = (_, newPeriod) => {
    if (newPeriod) {
      setPeriod(newPeriod);
    }
  };

  // Format number with commas
  const formatNumber = (num) => {
    return num?.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",") || '0';
  };

  // Handle refresh
  const handleRefresh = () => {
    refetch();
    if (wsConnected) {
      requestMetrics();
    }
  };

  // Generate mock hourly data for visualization
  const generateHourlyData = (stats) => {
    if (!stats) return [];

    const now = new Date();
    const data = [];

    // Generate data for the last 24 hours
    for (let i = 23; i >= 0; i--) {
      const hour = new Date(now);
      hour.setHours(now.getHours() - i);

      // Generate some random data based on the total stats
      const factor = Math.random() * 0.5 + 0.5; // Random factor between 0.5 and 1
      const hourlyRequests = Math.floor((stats.total_requests / 24) * factor);
      const hourlyTokens = Math.floor((stats.total_tokens / 24) * factor);
      const hourlyLatency = stats.average_latency_ms * (Math.random() * 0.5 + 0.75); // Random factor between 0.75 and 1.25

      data.push({
        hour: hour.getHours(),
        requests: hourlyRequests,
        tokens: hourlyTokens,
        latency: hourlyLatency.toFixed(2)
      });
    }

    return data;
  };

  // Generate mock daily data for visualization
  const generateDailyData = (stats) => {
    if (!stats) return [];

    const now = new Date();
    const data = [];

    // Generate data for the last 7 days
    for (let i = 6; i >= 0; i--) {
      const day = new Date(now);
      day.setDate(now.getDate() - i);

      // Generate some random data based on the total stats
      const factor = Math.random() * 0.5 + 0.5; // Random factor between 0.5 and 1
      const dailyRequests = Math.floor((stats.total_requests / 7) * factor);
      const dailyTokens = Math.floor((stats.total_tokens / 7) * factor);
      const dailyLatency = stats.average_latency_ms * (Math.random() * 0.5 + 0.75); // Random factor between 0.75 and 1.25

      data.push({
        day: day.toLocaleDateString('en-US', { weekday: 'short' }),
        requests: dailyRequests,
        tokens: dailyTokens,
        latency: dailyLatency.toFixed(2)
      });
    }

    return data;
  };

  // Generate mock monthly data for visualization
  const generateMonthlyData = (stats) => {
    if (!stats) return [];

    const now = new Date();
    const data = [];

    // Generate data for the last 6 months
    for (let i = 5; i >= 0; i--) {
      const month = new Date(now);
      month.setMonth(now.getMonth() - i);

      // Generate some random data based on the total stats
      const factor = Math.random() * 0.5 + 0.5; // Random factor between 0.5 and 1
      const monthlyRequests = Math.floor((stats.total_requests / 6) * factor);
      const monthlyTokens = Math.floor((stats.total_tokens / 6) * factor);
      const monthlyLatency = stats.average_latency_ms * (Math.random() * 0.5 + 0.75); // Random factor between 0.75 and 1.25

      data.push({
        month: month.toLocaleDateString('en-US', { month: 'short' }),
        requests: monthlyRequests,
        tokens: monthlyTokens,
        latency: monthlyLatency.toFixed(2)
      });
    }

    return data;
  };

  // Get chart data based on period
  const getChartData = (stats) => {
    if (!stats) return [];

    switch (period) {
      case 'day':
        return generateHourlyData(stats);
      case 'week':
        return generateDailyData(stats);
      case 'month':
        return generateMonthlyData(stats);
      default:
        return [];
    }
  };

  // Get x-axis key based on period
  const getXAxisKey = () => {
    switch (period) {
      case 'day':
        return 'hour';
      case 'week':
        return 'day';
      case 'month':
        return 'month';
      default:
        return '';
    }
  };

  if (!providerId) {
    return (
      <Alert severity="info">
        Please select a provider to view its usage statistics.
      </Alert>
    );
  }

  if (isLoading) {
    return <ContentLoader height={200} message={`Loading usage statistics for provider ${providerId}...`} />;
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
        Error loading usage statistics: {error?.message || 'Unknown error'}
      </Alert>
    );
  }

  const chartData = getChartData(usageStats);
  const xAxisKey = getXAxisKey();

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h6">Usage Statistics: {usageStats?.provider_id}</Typography>

        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          <ToggleButtonGroup
            value={period}
            exclusive
            onChange={handlePeriodChange}
            size="small"
          >
            <ToggleButton value="day">Day</ToggleButton>
            <ToggleButton value="week">Week</ToggleButton>
            <ToggleButton value="month">Month</ToggleButton>
          </ToggleButtonGroup>

          <Tooltip title={wsConnected ? "Real-time updates active" : "Real-time updates inactive"}>
            {wsConnected ? (
              <WifiIcon color="success" />
            ) : (
              <WifiOffIcon color="disabled" />
            )}
          </Tooltip>

          <Button
            variant="outlined"
            size="small"
            startIcon={<RefreshIcon />}
            onClick={handleRefresh}
          >
            Refresh
          </Button>
        </Box>
      </Box>

      <Grid container spacing={3}>
        {/* Summary Cards */}
        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                <QueryStatsIcon color="primary" sx={{ mr: 1 }} />
                <Typography variant="h6">Total Requests</Typography>
              </Box>
              <Typography variant="h4">{formatNumber(displayStats?.total_requests)}</Typography>
              <Typography variant="body2" color="text.secondary">
                Success Rate: {displayStats ? ((displayStats.successful_requests / displayStats.total_requests) * 100).toFixed(1) : 0}%
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                <TokenIcon color="primary" sx={{ mr: 1 }} />
                <Typography variant="h6">Total Tokens</Typography>
              </Box>
              <Typography variant="h4">{formatNumber(displayStats?.total_tokens)}</Typography>
              <Typography variant="body2" color="text.secondary">
                Avg. Tokens/Request: {displayStats ? Math.round(displayStats.total_tokens / displayStats.total_requests) : 0}
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                <SpeedIcon color="primary" sx={{ mr: 1 }} />
                <Typography variant="h6">Avg. Latency</Typography>
              </Box>
              <Typography variant="h4">{displayStats?.average_latency_ms.toFixed(1)} ms</Typography>
              <Typography variant="body2" color="text.secondary">
                Period: {new Date(displayStats?.period_start).toLocaleDateString()} - {new Date(displayStats?.period_end).toLocaleDateString()}
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                <TrendingUpIcon color="primary" sx={{ mr: 1 }} />
                <Typography variant="h6">Success Rate</Typography>
              </Box>
              <Box sx={{ display: 'flex', alignItems: 'center' }}>
                <Box sx={{ position: 'relative', display: 'inline-flex', mr: 2 }}>
                  <CircularProgress
                    variant="determinate"
                    value={displayStats ? (displayStats.successful_requests / displayStats.total_requests) * 100 : 0}
                    size={60}
                    thickness={5}
                    color={
                      displayStats && (displayStats.successful_requests / displayStats.total_requests) >= 0.95 ? 'success' :
                      displayStats && (displayStats.successful_requests / displayStats.total_requests) >= 0.9 ? 'primary' :
                      displayStats && (displayStats.successful_requests / displayStats.total_requests) >= 0.8 ? 'warning' : 'error'
                    }
                  />
                  <Box
                    sx={{
                      top: 0,
                      left: 0,
                      bottom: 0,
                      right: 0,
                      position: 'absolute',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                    }}
                  >
                    <Typography variant="caption" component="div" color="text.secondary">
                      {displayStats ? ((displayStats.successful_requests / displayStats.total_requests) * 100).toFixed(0) : 0}%
                    </Typography>
                  </Box>
                </Box>
                <Box>
                  <Typography variant="body2">
                    {formatNumber(displayStats?.successful_requests)} / {formatNumber(displayStats?.total_requests)}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    successful requests
                  </Typography>
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Charts */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardHeader title="Request Volume" />
            <Divider />
            <CardContent sx={{ height: 300 }}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={chartData}
                  margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey={xAxisKey} />
                  <YAxis />
                  <RechartsTooltip />
                  <Legend />
                  <Bar dataKey="requests" name="Requests" fill="#8884d8" />
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={6}>
          <Card>
            <CardHeader title="Token Usage" />
            <Divider />
            <CardContent sx={{ height: 300 }}>
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart
                  data={chartData}
                  margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey={xAxisKey} />
                  <YAxis />
                  <RechartsTooltip />
                  <Legend />
                  <Area type="monotone" dataKey="tokens" name="Tokens" fill="#82ca9d" stroke="#82ca9d" />
                </AreaChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12}>
          <Card>
            <CardHeader title="Latency Metrics" />
            <Divider />
            <CardContent sx={{ height: 300 }}>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart
                  data={chartData}
                  margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey={xAxisKey} />
                  <YAxis />
                  <RechartsTooltip />
                  <Legend />
                  <Line type="monotone" dataKey="latency" name="Latency (ms)" stroke="#ff7300" />
                </LineChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default MCPUsageStats;
