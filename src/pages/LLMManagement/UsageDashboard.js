import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Paper,
  Grid,
  Card,
  CardHeader,
  CardContent,
  TextField,
  Button,
  CircularProgress
} from '@mui/material';

import { useNotification } from '../../context/NotificationContext';
import apiService from '../../services/api';
import { ContentLoader } from '../../components/UI/LoadingIndicators';

/**
 * Usage Dashboard component
 */
const UsageDashboard = ({ status, onRefresh }) => {
  const { showSuccess, showError } = useNotification();

  const [usageData, setUsageData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [dateRange, setDateRange] = useState({
    startDate: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000), // 30 days ago
    endDate: new Date()
  });

  // Load usage data on mount
  useEffect(() => {
    loadUsageData();
  }, []);

  // Load usage data
  const loadUsageData = async () => {
    setLoading(true);

    try {
      const result = await apiService.llm.getUsageStatistics(
        dateRange.startDate.toISOString(),
        dateRange.endDate.toISOString()
      );

      if (result.success) {
        setUsageData(result.data);
      } else {
        showError(`Failed to load usage data: ${result.error}`);
      }
    } catch (error) {
      console.error('Error loading usage data:', error);
      showError(`Error loading usage data: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  // Handle date range change
  const handleDateRangeChange = () => {
    loadUsageData();
  };

  if (loading) {
    return <ContentLoader height={200} message="Loading usage statistics..." />;
  }

  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        LLM Usage Statistics
      </Typography>

      <Typography paragraph>
        Monitor usage statistics for all LLM components, including token usage, request counts, and latency metrics.
      </Typography>

      {/* Date range selector */}
      <Paper sx={{ p: 3, mb: 3 }}>
        <Grid container spacing={3} alignItems="center">
          <Grid item xs={12} md={4}>
            <TextField
              label="Start Date"
              type="date"
              value={dateRange.startDate.toISOString().split('T')[0]}
              onChange={(e) => setDateRange({ ...dateRange, startDate: new Date(e.target.value) })}
              fullWidth
              InputLabelProps={{ shrink: true }}
            />
          </Grid>
          <Grid item xs={12} md={4}>
            <TextField
              label="End Date"
              type="date"
              value={dateRange.endDate.toISOString().split('T')[0]}
              onChange={(e) => setDateRange({ ...dateRange, endDate: new Date(e.target.value) })}
              fullWidth
              InputLabelProps={{ shrink: true }}
            />
          </Grid>
          <Grid item xs={12} md={4}>
            <Button
              variant="contained"
              onClick={handleDateRangeChange}
              fullWidth
            >
              Apply Date Range
            </Button>
          </Grid>
        </Grid>
      </Paper>

      {/* Summary cards */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} md={4}>
          <Card>
            <CardHeader title="Total Requests" />
            <CardContent>
              <Typography variant="h3" align="center">
                {usageData?.total_requests || 0}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={4}>
          <Card>
            <CardHeader title="Total Tokens" />
            <CardContent>
              <Typography variant="h3" align="center">
                {usageData?.components?.gateway?.total_tokens || 0}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={4}>
          <Card>
            <CardHeader title="Average Latency" />
            <CardContent>
              <Typography variant="h3" align="center">
                {usageData?.components?.gateway?.average_latency_ms?.toFixed(2) || 0} ms
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Component comparison */}
      <Paper sx={{ p: 3, mb: 3 }}>
        <Typography variant="h6" gutterBottom>
          Requests by Component
        </Typography>
        <Box sx={{ p: 2 }}>
          <Typography variant="body1">
            Gateway: {usageData?.components?.gateway?.total_requests || 0} requests
          </Typography>
          <Typography variant="body1">
            DSPy: {usageData?.components?.dspy?.total_requests || 0} requests
          </Typography>
          <Typography variant="body1">
            BiomedLM: {usageData?.components?.biomedlm?.total_requests || 0} requests
          </Typography>
        </Box>
      </Paper>

      {/* Provider and model info */}
      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Requests by Provider
            </Typography>
            <Box sx={{ p: 2 }}>
              {usageData?.components?.gateway?.providers ? (
                Object.entries(usageData.components.gateway.providers).map(([provider, data]) => (
                  <Typography key={provider} variant="body1">
                    {provider}: {data.requests} requests
                  </Typography>
                ))
              ) : (
                <Typography variant="body2" color="text.secondary">
                  No provider data available
                </Typography>
              )}
            </Box>
          </Paper>
        </Grid>
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Usage by Model
            </Typography>
            <Box sx={{ p: 2 }}>
              {usageData?.components?.gateway?.models ? (
                Object.entries(usageData.components.gateway.models).map(([model, data]) => (
                  <Typography key={model} variant="body1">
                    {model}: {data.requests} requests, {data.tokens} tokens
                  </Typography>
                ))
              ) : (
                <Typography variant="body2" color="text.secondary">
                  No model data available
                </Typography>
              )}
            </Box>
          </Paper>
        </Grid>
      </Grid>

      {/* DSPy and BiomedLM usage */}
      <Grid container spacing={3} sx={{ mt: 1 }}>
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              DSPy Module Usage
            </Typography>
            <Box sx={{ p: 2 }}>
              {usageData?.components?.dspy?.modules ? (
                Object.entries(usageData.components.dspy.modules).map(([module, count]) => (
                  <Typography key={module} variant="body1">
                    {module}: {count} requests
                  </Typography>
                ))
              ) : (
                <Typography variant="body2" color="text.secondary">
                  No DSPy module usage data available
                </Typography>
              )}
            </Box>
          </Paper>
        </Grid>
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              BiomedLM Model Usage
            </Typography>
            <Box sx={{ p: 2 }}>
              {usageData?.components?.biomedlm?.models ? (
                Object.entries(usageData.components.biomedlm.models).map(([model, count]) => (
                  <Typography key={model} variant="body1">
                    {model}: {count} requests
                  </Typography>
                ))
              ) : (
                <Typography variant="body2" color="text.secondary">
                  No BiomedLM model usage data available
                </Typography>
              )}
            </Box>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default UsageDashboard;
