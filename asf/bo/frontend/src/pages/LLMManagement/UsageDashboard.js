import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Paper,
  Grid,
  Card,
  CardHeader,
  CardContent,
  Divider,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  TextField,
  Button,
  CircularProgress
} from '@mui/material';
import { DatePicker } from '@mui/x-date-pickers/DatePicker';
import { AdapterDateFns } from '@mui/x-date-pickers/AdapterDateFns';
import { LocalizationProvider } from '@mui/x-date-pickers/LocalizationProvider';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell
} from 'recharts';

import { useNotification } from '../../context/NotificationContext';
import apiService from '../../services/api';
import { ContentLoader } from '../../components/UI/LoadingIndicators';

// Chart colors
const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8', '#82CA9D'];

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
  
  // Prepare data for provider pie chart
  const prepareProviderPieData = () => {
    if (!usageData || !usageData.components.gateway || !usageData.components.gateway.providers) {
      return [];
    }
    
    return Object.entries(usageData.components.gateway.providers).map(([provider, data]) => ({
      name: provider,
      value: data.requests
    }));
  };
  
  // Prepare data for model bar chart
  const prepareModelBarData = () => {
    if (!usageData || !usageData.components.gateway || !usageData.components.gateway.models) {
      return [];
    }
    
    return Object.entries(usageData.components.gateway.models).map(([model, data]) => ({
      name: model,
      requests: data.requests,
      tokens: data.tokens
    }));
  };
  
  // Prepare data for component comparison
  const prepareComponentData = () => {
    if (!usageData || !usageData.components) {
      return [];
    }
    
    return [
      {
        name: 'Gateway',
        requests: usageData.components.gateway?.total_requests || 0
      },
      {
        name: 'DSPy',
        requests: usageData.components.dspy?.total_requests || 0
      },
      {
        name: 'BiomedLM',
        requests: usageData.components.biomedlm?.total_requests || 0
      }
    ];
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
            <LocalizationProvider dateAdapter={AdapterDateFns}>
              <DatePicker
                label="Start Date"
                value={dateRange.startDate}
                onChange={(newValue) => setDateRange({ ...dateRange, startDate: newValue })}
                renderInput={(params) => <TextField {...params} fullWidth />}
              />
            </LocalizationProvider>
          </Grid>
          <Grid item xs={12} md={4}>
            <LocalizationProvider dateAdapter={AdapterDateFns}>
              <DatePicker
                label="End Date"
                value={dateRange.endDate}
                onChange={(newValue) => setDateRange({ ...dateRange, endDate: newValue })}
                renderInput={(params) => <TextField {...params} fullWidth />}
              />
            </LocalizationProvider>
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
      
      {/* Component comparison chart */}
      <Paper sx={{ p: 3, mb: 3 }}>
        <Typography variant="h6" gutterBottom>
          Requests by Component
        </Typography>
        <Box sx={{ height: 300 }}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={prepareComponentData()}
              margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
            >
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Bar dataKey="requests" fill="#8884d8" name="Requests" />
            </BarChart>
          </ResponsiveContainer>
        </Box>
      </Paper>
      
      {/* Provider and model charts */}
      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Requests by Provider
            </Typography>
            <Box sx={{ height: 300 }}>
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={prepareProviderPieData()}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                    outerRadius={80}
                    fill="#8884d8"
                    dataKey="value"
                  >
                    {prepareProviderPieData().map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                  <Legend />
                </PieChart>
              </ResponsiveContainer>
            </Box>
          </Paper>
        </Grid>
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Usage by Model
            </Typography>
            <Box sx={{ height: 300 }}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={prepareModelBarData()}
                  margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" />
                  <YAxis yAxisId="left" orientation="left" stroke="#8884d8" />
                  <YAxis yAxisId="right" orientation="right" stroke="#82ca9d" />
                  <Tooltip />
                  <Legend />
                  <Bar yAxisId="left" dataKey="requests" fill="#8884d8" name="Requests" />
                  <Bar yAxisId="right" dataKey="tokens" fill="#82ca9d" name="Tokens" />
                </BarChart>
              </ResponsiveContainer>
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
            {usageData?.components?.dspy?.modules ? (
              <Box sx={{ height: 300 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart
                    data={Object.entries(usageData.components.dspy.modules).map(([module, count]) => ({
                      name: module,
                      requests: count
                    }))}
                    margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="name" />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    <Bar dataKey="requests" fill="#00C49F" name="Requests" />
                  </BarChart>
                </ResponsiveContainer>
              </Box>
            ) : (
              <Typography variant="body2" color="text.secondary" align="center" sx={{ py: 5 }}>
                No DSPy module usage data available
              </Typography>
            )}
          </Paper>
        </Grid>
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              BiomedLM Model Usage
            </Typography>
            {usageData?.components?.biomedlm?.models ? (
              <Box sx={{ height: 300 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart
                    data={Object.entries(usageData.components.biomedlm.models).map(([model, count]) => ({
                      name: model,
                      requests: count
                    }))}
                    margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="name" />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    <Bar dataKey="requests" fill="#FFBB28" name="Requests" />
                  </BarChart>
                </ResponsiveContainer>
              </Box>
            ) : (
              <Typography variant="body2" color="text.secondary" align="center" sx={{ py: 5 }}>
                No BiomedLM model usage data available
              </Typography>
            )}
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default UsageDashboard;
