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
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Divider
} from '@mui/material';
import {
  BarChart as BarChartIcon,
  Refresh as RefreshIcon,
  Timeline as TimelineIcon,
  Speed as SpeedIcon,
  Token as TokenIcon
} from '@mui/icons-material';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  LineChart,
  Line,
  PieChart,
  Pie,
  Cell
} from 'recharts';

import { useNotification } from '../../../context/NotificationContext';
import apiService from '../../../services/api';
import { ContentLoader } from '../../UI/LoadingIndicators';

// Chart colors
const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8'];

/**
 * MCP Usage Statistics Component
 * 
 * This component displays usage statistics for MCP providers,
 * including request counts, token usage, and performance metrics.
 */
const MCPUsageStats = ({ providerId }) => {
  const { showSuccess, showError } = useNotification();
  
  // State
  const [usage, setUsage] = useState(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [period, setPeriod] = useState('day');
  
  // Load usage on mount and when providerId or period changes
  useEffect(() => {
    if (providerId) {
      loadUsage();
    }
  }, [providerId, period]);
  
  // Load provider usage
  const loadUsage = async () => {
    if (!providerId) return;
    
    setLoading(true);
    
    try {
      const result = await apiService.llm.getMCPProviderUsage(providerId, period);
      
      if (result.success) {
        setUsage(result.data);
      } else {
        showError(`Failed to load provider usage: ${result.error}`);
      }
    } catch (error) {
      console.error('Error loading provider usage:', error);
      showError(`Error loading provider usage: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };
  
  // Refresh usage
  const refreshUsage = async () => {
    setRefreshing(true);
    
    try {
      await loadUsage();
      showSuccess('Usage statistics refreshed successfully');
    } catch (error) {
      console.error('Error refreshing usage statistics:', error);
      showError(`Error refreshing usage statistics: ${error.message}`);
    } finally {
      setRefreshing(false);
    }
  };
  
  // Handle period change
  const handlePeriodChange = (e) => {
    setPeriod(e.target.value);
  };
  
  // Generate mock data for demonstration
  const generateMockData = () => {
    // Mock request data
    const requestData = [
      { name: 'Total', value: 120 },
      { name: 'Successful', value: 100 },
      { name: 'Failed', value: 20 }
    ];
    
    // Mock token usage data
    const tokenData = [
      { name: 'Prompt', value: 15000 },
      { name: 'Completion', value: 8000 }
    ];
    
    // Mock hourly data
    const hourlyData = Array.from({ length: 24 }, (_, i) => ({
      hour: `${i}:00`,
      requests: Math.floor(Math.random() * 20),
      tokens: Math.floor(Math.random() * 2000)
    }));
    
    // Mock model usage data
    const modelData = [
      { name: 'claude-3-opus', requests: 30, tokens: 8000 },
      { name: 'claude-3-sonnet', requests: 50, tokens: 10000 },
      { name: 'claude-3-haiku', requests: 40, tokens: 5000 }
    ];
    
    return {
      requestData,
      tokenData,
      hourlyData,
      modelData
    };
  };
  
  // Get mock data for demonstration
  const mockData = generateMockData();
  
  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h6">Usage Statistics</Typography>
        
        <Box sx={{ display: 'flex', gap: 2 }}>
          <FormControl sx={{ minWidth: 120 }}>
            <InputLabel id="period-select-label">Period</InputLabel>
            <Select
              labelId="period-select-label"
              value={period}
              label="Period"
              onChange={handlePeriodChange}
              size="small"
            >
              <MenuItem value="day">Day</MenuItem>
              <MenuItem value="week">Week</MenuItem>
              <MenuItem value="month">Month</MenuItem>
            </Select>
          </FormControl>
          
          <Button
            variant="outlined"
            onClick={refreshUsage}
            disabled={refreshing || loading}
            startIcon={refreshing ? <CircularProgress size={20} /> : <RefreshIcon />}
          >
            {refreshing ? 'Refreshing...' : 'Refresh'}
          </Button>
        </Box>
      </Box>
      
      {loading ? (
        <ContentLoader height={200} message="Loading usage statistics..." />
      ) : !usage ? (
        <Alert severity="info">
          No usage statistics available for this provider.
          <br />
          Showing mock data for demonstration purposes.
        </Alert>
      ) : (
        <Alert severity="info">
          Showing mock data for demonstration purposes.
        </Alert>
      )}
      
      <Grid container spacing={3} sx={{ mt: 1 }}>
        {/* Request Statistics */}
        <Grid item xs={12} md={6}>
          <Card variant="outlined">
            <CardHeader
              title="Request Statistics"
              avatar={<BarChartIcon color="primary" />}
            />
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart
                  data={mockData.requestData}
                  margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="value" name="Requests" fill="#8884d8" />
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid>
        
        {/* Token Usage */}
        <Grid item xs={12} md={6}>
          <Card variant="outlined">
            <CardHeader
              title="Token Usage"
              avatar={<TokenIcon color="primary" />}
            />
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie
                    data={mockData.tokenData}
                    cx="50%"
                    cy="50%"
                    labelLine={true}
                    label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                    outerRadius={80}
                    fill="#8884d8"
                    dataKey="value"
                  >
                    {mockData.tokenData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip formatter={(value) => `${value.toLocaleString()} tokens`} />
                  <Legend />
                </PieChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid>
        
        {/* Hourly Usage */}
        <Grid item xs={12}>
          <Card variant="outlined">
            <CardHeader
              title="Hourly Usage"
              avatar={<TimelineIcon color="primary" />}
            />
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart
                  data={mockData.hourlyData}
                  margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="hour" />
                  <YAxis yAxisId="left" orientation="left" stroke="#8884d8" />
                  <YAxis yAxisId="right" orientation="right" stroke="#82ca9d" />
                  <Tooltip />
                  <Legend />
                  <Line
                    yAxisId="left"
                    type="monotone"
                    dataKey="requests"
                    name="Requests"
                    stroke="#8884d8"
                    activeDot={{ r: 8 }}
                  />
                  <Line
                    yAxisId="right"
                    type="monotone"
                    dataKey="tokens"
                    name="Tokens"
                    stroke="#82ca9d"
                  />
                </LineChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid>
        
        {/* Model Usage */}
        <Grid item xs={12}>
          <Card variant="outlined">
            <CardHeader
              title="Model Usage"
              avatar={<BarChartIcon color="primary" />}
            />
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart
                  data={mockData.modelData}
                  margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" />
                  <YAxis yAxisId="left" orientation="left" stroke="#8884d8" />
                  <YAxis yAxisId="right" orientation="right" stroke="#82ca9d" />
                  <Tooltip />
                  <Legend />
                  <Bar
                    yAxisId="left"
                    dataKey="requests"
                    name="Requests"
                    fill="#8884d8"
                  />
                  <Bar
                    yAxisId="right"
                    dataKey="tokens"
                    name="Tokens"
                    fill="#82ca9d"
                  />
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid>
        
        {/* Performance Metrics */}
        <Grid item xs={12}>
          <Card variant="outlined">
            <CardHeader
              title="Performance Metrics"
              avatar={<SpeedIcon color="primary" />}
            />
            <CardContent>
              <Grid container spacing={2}>
                <Grid item xs={12} sm={6} md={3}>
                  <Paper
                    elevation={0}
                    variant="outlined"
                    sx={{ p: 2, textAlign: 'center', height: '100%' }}
                  >
                    <Typography variant="h6" color="text.secondary" gutterBottom>
                      Average Latency
                    </Typography>
                    <Typography variant="h4" color="primary">
                      245 ms
                    </Typography>
                  </Paper>
                </Grid>
                
                <Grid item xs={12} sm={6} md={3}>
                  <Paper
                    elevation={0}
                    variant="outlined"
                    sx={{ p: 2, textAlign: 'center', height: '100%' }}
                  >
                    <Typography variant="h6" color="text.secondary" gutterBottom>
                      Success Rate
                    </Typography>
                    <Typography variant="h4" color="primary">
                      98.2%
                    </Typography>
                  </Paper>
                </Grid>
                
                <Grid item xs={12} sm={6} md={3}>
                  <Paper
                    elevation={0}
                    variant="outlined"
                    sx={{ p: 2, textAlign: 'center', height: '100%' }}
                  >
                    <Typography variant="h6" color="text.secondary" gutterBottom>
                      Total Tokens
                    </Typography>
                    <Typography variant="h4" color="primary">
                      23,000
                    </Typography>
                  </Paper>
                </Grid>
                
                <Grid item xs={12} sm={6} md={3}>
                  <Paper
                    elevation={0}
                    variant="outlined"
                    sx={{ p: 2, textAlign: 'center', height: '100%' }}
                  >
                    <Typography variant="h6" color="text.secondary" gutterBottom>
                      Circuit Breaker Trips
                    </Typography>
                    <Typography variant="h4" color="primary">
                      0
                    </Typography>
                  </Paper>
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default MCPUsageStats;
