import React, { useState, useEffect } from 'react';
import {
  Box,
  Paper,
  Typography,
  Button,
  Grid,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Alert,
  Divider,
  CircularProgress,
  Tabs,
  Tab,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow
} from '@mui/material';
import {
  Refresh as RefreshIcon,
  BarChart as BarChartIcon,
  DateRange as DateRangeIcon
} from '@mui/icons-material';

// Import recharts components if available, otherwise use empty components
let BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  PieChart, Pie, Cell;

try {
  const recharts = require('recharts');
  BarChart = recharts.BarChart;
  Bar = recharts.Bar;
  XAxis = recharts.XAxis;
  YAxis = recharts.YAxis;
  CartesianGrid = recharts.CartesianGrid;
  Tooltip = recharts.Tooltip;
  Legend = recharts.Legend;
  ResponsiveContainer = recharts.ResponsiveContainer;
  PieChart = recharts.PieChart;
  Pie = recharts.Pie;
  Cell = recharts.Cell;
} catch (error) {
  // Create empty fallback components if recharts is not available
  console.warn('recharts library not found, using fallback components');
  const EmptyComponent = props => <div>{props.children || 'Chart not available'}</div>;
  BarChart = EmptyComponent;
  Bar = EmptyComponent;
  XAxis = EmptyComponent;
  YAxis = EmptyComponent;
  CartesianGrid = EmptyComponent;
  Tooltip = EmptyComponent;
  Legend = EmptyComponent;
  ResponsiveContainer = props => <div style={{ height: props.height || '300px' }}>{props.children}</div>;
  PieChart = EmptyComponent;
  Pie = EmptyComponent;
  Cell = EmptyComponent;
}

import apiService from '../../services/api';
import { useNotification } from '../../context/NotificationContext';

// Chart colors
const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#A28BFC', '#FF6B6B'];

/**
 * Dashboard for LLM usage statistics
 */
const UsageDashboard = ({ status, onRefresh }) => {
  const [loading, setLoading] = useState(false);
  const [usageData, setUsageData] = useState(null);
  const [timeRange, setTimeRange] = useState('week');
  const [activeTab, setActiveTab] = useState(0);
  
  const { showSuccess, showError } = useNotification();
  
  // Load usage data on mount and when timeRange changes
  useEffect(() => {
    loadUsageData();
  }, [timeRange]);
  
  // Load usage statistics
  const loadUsageData = async () => {
    setLoading(true);
    
    // Calculate date range based on selected time range
    const endDate = new Date();
    let startDate = new Date();
    
    switch (timeRange) {
      case 'day':
        startDate.setDate(startDate.getDate() - 1);
        break;
      case 'week':
        startDate.setDate(startDate.getDate() - 7);
        break;
      case 'month':
        startDate.setMonth(startDate.getMonth() - 1);
        break;
      case 'quarter':
        startDate.setMonth(startDate.getMonth() - 3);
        break;
      default:
        startDate.setDate(startDate.getDate() - 7);
    }
    
    try {
      const result = await apiService.llm.getUsageStatistics(
        startDate.toISOString(),
        endDate.toISOString()
      );
      
      if (result.success) {
        setUsageData(result.data);
        showSuccess('Usage statistics loaded successfully');
      } else {
        showError(`Failed to load usage statistics: ${result.error}`);
      }
    } catch (error) {
      console.error('Error loading usage statistics:', error);
      showError(`Error loading usage statistics: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };
  
  // Handle time range change
  const handleTimeRangeChange = (event) => {
    setTimeRange(event.target.value);
  };
  
  // Handle tab change
  const handleTabChange = (_, newValue) => {
    setActiveTab(newValue);
  };
  
  // Format large numbers with commas
  const formatNumber = (num) => {
    return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
  };
  
  // Format timestamps for charts
  const formatTimestamp = (timestamp) => {
    const date = new Date(timestamp);
    return timeRange === 'day' 
      ? date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      : date.toLocaleDateString([], { month: 'short', day: 'numeric' });
  };
  
  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h5">
          <BarChartIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
          LLM Usage Dashboard
        </Typography>
        <Box sx={{ display: 'flex', gap: 2 }}>
          <FormControl sx={{ minWidth: 120 }}>
            <InputLabel id="time-range-label">Time Range</InputLabel>
            <Select
              labelId="time-range-label"
              id="time-range-select"
              value={timeRange}
              label="Time Range"
              onChange={handleTimeRangeChange}
              disabled={loading}
              size="small"
            >
              <MenuItem value="day">Last 24 Hours</MenuItem>
              <MenuItem value="week">Last 7 Days</MenuItem>
              <MenuItem value="month">Last Month</MenuItem>
              <MenuItem value="quarter">Last Quarter</MenuItem>
            </Select>
          </FormControl>
          
          <Button 
            variant="outlined" 
            startIcon={loading ? <CircularProgress size={20} /> : <RefreshIcon />}
            onClick={loadUsageData}
            disabled={loading}
            size="small"
          >
            Refresh
          </Button>
        </Box>
      </Box>
      
      {status?.overall_status !== 'operational' && (
        <Alert severity="warning" sx={{ mb: 3 }}>
          Some LLM components are not fully operational, which may affect usage data.
        </Alert>
      )}
      
      {loading ? (
        <Box sx={{ display: 'flex', justifyContent: 'center', my: 5 }}>
          <CircularProgress />
        </Box>
      ) : !usageData ? (
        <Alert severity="info" sx={{ mb: 3 }}>
          No usage data available for the selected time period.
        </Alert>
      ) : (
        <Box>
          {/* Summary Cards */}
          <Grid container spacing={3} sx={{ mb: 3 }}>
            <Grid item xs={12} sm={6} md={3}>
              <Paper sx={{ p: 2 }}>
                <Typography variant="h6" gutterBottom>
                  Total Requests
                </Typography>
                <Typography variant="h3" sx={{ fontWeight: 'bold', color: 'primary.main' }}>
                  {formatNumber(usageData.summary.total_requests)}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Average: {formatNumber(Math.round(usageData.summary.avg_requests_per_day))} / day
                </Typography>
              </Paper>
            </Grid>
            
            <Grid item xs={12} sm={6} md={3}>
              <Paper sx={{ p: 2 }}>
                <Typography variant="h6" gutterBottom>
                  Total Tokens
                </Typography>
                <Typography variant="h3" sx={{ fontWeight: 'bold', color: 'secondary.main' }}>
                  {formatNumber(usageData.summary.total_tokens)}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Avg: {formatNumber(Math.round(usageData.summary.avg_tokens_per_request))} / request
                </Typography>
              </Paper>
            </Grid>
            
            <Grid item xs={12} sm={6} md={3}>
              <Paper sx={{ p: 2 }}>
                <Typography variant="h6" gutterBottom>
                  Avg Latency
                </Typography>
                <Typography variant="h3" sx={{ fontWeight: 'bold', color: 'success.main' }}>
                  {Math.round(usageData.summary.avg_latency_ms)}ms
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Max: {formatNumber(usageData.summary.max_latency_ms)}ms
                </Typography>
              </Paper>
            </Grid>
            
            <Grid item xs={12} sm={6} md={3}>
              <Paper sx={{ p: 2 }}>
                <Typography variant="h6" gutterBottom>
                  Success Rate
                </Typography>
                <Typography variant="h3" sx={{ fontWeight: 'bold', color: 'info.main' }}>
                  {(usageData.summary.success_rate * 100).toFixed(1)}%
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Errors: {formatNumber(usageData.summary.error_count)}
                </Typography>
              </Paper>
            </Grid>
          </Grid>
          
          {/* Tabs */}
          <Paper sx={{ mb: 3 }}>
            <Tabs 
              value={activeTab} 
              onChange={handleTabChange} 
              aria-label="Usage tabs"
              sx={{ borderBottom: 1, borderColor: 'divider' }}
            >
              <Tab label="Overview" id="tab-0" aria-controls="tabpanel-0" />
              <Tab label="By Provider" id="tab-1" aria-controls="tabpanel-1" />
              <Tab label="By Model" id="tab-2" aria-controls="tabpanel-2" />
              <Tab label="By Component" id="tab-3" aria-controls="tabpanel-3" />
            </Tabs>
            
            {/* Overview Tab */}
            <Box 
              role="tabpanel" 
              hidden={activeTab !== 0} 
              id="tabpanel-0" 
              aria-labelledby="tab-0"
              sx={{ p: 3 }}
            >
              {activeTab === 0 && (
                <Grid container spacing={3}>
                  <Grid item xs={12}>
                    <Typography variant="h6" gutterBottom>
                      Request Volume Over Time
                    </Typography>
                    <Box sx={{ height: 300 }}>
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart
                          data={usageData.time_series.map(item => ({
                            name: formatTimestamp(item.timestamp),
                            Requests: item.request_count,
                            Tokens: Math.round(item.total_tokens / 100) // Scale down for chart
                          }))}
                          margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                        >
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="name" />
                          <YAxis />
                          <Tooltip formatter={(value, name) => [
                            name === 'Tokens' ? formatNumber(value * 100) : formatNumber(value),
                            name
                          ]} />
                          <Legend />
                          <Bar dataKey="Requests" fill="#8884d8" />
                          <Bar dataKey="Tokens" fill="#82ca9d" />
                        </BarChart>
                      </ResponsiveContainer>
                    </Box>
                  </Grid>
                  
                  <Grid item xs={12} md={6}>
                    <Typography variant="h6" gutterBottom>
                      Request Distribution by Provider
                    </Typography>
                    <Box sx={{ height: 300 }}>
                      <ResponsiveContainer width="100%" height="100%">
                        <PieChart>
                          <Pie
                            data={usageData.by_provider.map((item, index) => ({
                              name: item.provider_name,
                              value: item.request_count
                            }))}
                            cx="50%"
                            cy="50%"
                            outerRadius={80}
                            fill="#8884d8"
                            dataKey="value"
                            label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                          >
                            {usageData.by_provider.map((_, index) => (
                              <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                            ))}
                          </Pie>
                          <Tooltip 
                            formatter={(value) => [formatNumber(value), 'Requests']} 
                          />
                        </PieChart>
                      </ResponsiveContainer>
                    </Box>
                  </Grid>
                  
                  <Grid item xs={12} md={6}>
                    <Typography variant="h6" gutterBottom>
                      Token Usage by Component
                    </Typography>
                    <Box sx={{ height: 300 }}>
                      <ResponsiveContainer width="100%" height="100%">
                        <PieChart>
                          <Pie
                            data={usageData.by_component.map((item, index) => ({
                              name: item.component_name,
                              value: item.total_tokens
                            }))}
                            cx="50%"
                            cy="50%"
                            outerRadius={80}
                            fill="#8884d8"
                            dataKey="value"
                            label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                          >
                            {usageData.by_component.map((_, index) => (
                              <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                            ))}
                          </Pie>
                          <Tooltip 
                            formatter={(value) => [formatNumber(value), 'Tokens']} 
                          />
                        </PieChart>
                      </ResponsiveContainer>
                    </Box>
                  </Grid>
                </Grid>
              )}
            </Box>
            
            {/* By Provider Tab */}
            <Box 
              role="tabpanel" 
              hidden={activeTab !== 1} 
              id="tabpanel-1" 
              aria-labelledby="tab-1"
              sx={{ p: 3 }}
            >
              {activeTab === 1 && (
                <TableContainer>
                  <Table>
                    <TableHead>
                      <TableRow>
                        <TableCell>Provider</TableCell>
                        <TableCell align="right">Requests</TableCell>
                        <TableCell align="right">Tokens</TableCell>
                        <TableCell align="right">Avg Tokens/Req</TableCell>
                        <TableCell align="right">Avg Latency (ms)</TableCell>
                        <TableCell align="right">Success Rate</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {usageData.by_provider.map((row) => (
                        <TableRow key={row.provider_id}>
                          <TableCell component="th" scope="row">
                            {row.provider_name}
                          </TableCell>
                          <TableCell align="right">{formatNumber(row.request_count)}</TableCell>
                          <TableCell align="right">{formatNumber(row.total_tokens)}</TableCell>
                          <TableCell align="right">
                            {formatNumber(Math.round(row.total_tokens / row.request_count))}
                          </TableCell>
                          <TableCell align="right">{Math.round(row.avg_latency_ms)}</TableCell>
                          <TableCell align="right">
                            {(row.success_rate * 100).toFixed(1)}%
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              )}
            </Box>
            
            {/* By Model Tab */}
            <Box 
              role="tabpanel" 
              hidden={activeTab !== 2} 
              id="tabpanel-2" 
              aria-labelledby="tab-2"
              sx={{ p: 3 }}
            >
              {activeTab === 2 && (
                <TableContainer>
                  <Table>
                    <TableHead>
                      <TableRow>
                        <TableCell>Model</TableCell>
                        <TableCell>Provider</TableCell>
                        <TableCell align="right">Requests</TableCell>
                        <TableCell align="right">Tokens</TableCell>
                        <TableCell align="right">Avg Tokens/Req</TableCell>
                        <TableCell align="right">Avg Latency (ms)</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {usageData.by_model.map((row) => (
                        <TableRow key={`${row.provider_id}-${row.model_id}`}>
                          <TableCell component="th" scope="row">
                            {row.model_name}
                          </TableCell>
                          <TableCell>{row.provider_name}</TableCell>
                          <TableCell align="right">{formatNumber(row.request_count)}</TableCell>
                          <TableCell align="right">{formatNumber(row.total_tokens)}</TableCell>
                          <TableCell align="right">
                            {formatNumber(Math.round(row.total_tokens / row.request_count))}
                          </TableCell>
                          <TableCell align="right">{Math.round(row.avg_latency_ms)}</TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              )}
            </Box>
            
            {/* By Component Tab */}
            <Box 
              role="tabpanel" 
              hidden={activeTab !== 3} 
              id="tabpanel-3" 
              aria-labelledby="tab-3"
              sx={{ p: 3 }}
            >
              {activeTab === 3 && (
                <TableContainer>
                  <Table>
                    <TableHead>
                      <TableRow>
                        <TableCell>Component</TableCell>
                        <TableCell align="right">Requests</TableCell>
                        <TableCell align="right">Tokens</TableCell>
                        <TableCell align="right">Avg Tokens/Req</TableCell>
                        <TableCell align="right">Avg Latency (ms)</TableCell>
                        <TableCell align="right">Features Used</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {usageData.by_component.map((row) => (
                        <TableRow key={row.component_id}>
                          <TableCell component="th" scope="row">
                            {row.component_name}
                          </TableCell>
                          <TableCell align="right">{formatNumber(row.request_count)}</TableCell>
                          <TableCell align="right">{formatNumber(row.total_tokens)}</TableCell>
                          <TableCell align="right">
                            {formatNumber(Math.round(row.total_tokens / row.request_count))}
                          </TableCell>
                          <TableCell align="right">{Math.round(row.avg_latency_ms)}</TableCell>
                          <TableCell align="right">{row.features_count || 'N/A'}</TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              )}
            </Box>
          </Paper>
          
          {/* Usage Breakdown */}
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Cost and Usage Breakdown
            </Typography>
            <Typography variant="body2" paragraph>
              This data represents estimated costs based on current provider pricing.
              Actual billing may vary based on your specific agreements with LLM providers.
            </Typography>
            
            <TableContainer>
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>Provider</TableCell>
                    <TableCell align="right">Input Tokens</TableCell>
                    <TableCell align="right">Output Tokens</TableCell>
                    <TableCell align="right">Total Tokens</TableCell>
                    <TableCell align="right">Est. Cost</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {usageData.cost_breakdown.map((row) => (
                    <TableRow key={row.provider_id}>
                      <TableCell component="th" scope="row">
                        {row.provider_name}
                      </TableCell>
                      <TableCell align="right">{formatNumber(row.input_tokens)}</TableCell>
                      <TableCell align="right">{formatNumber(row.output_tokens)}</TableCell>
                      <TableCell align="right">{formatNumber(row.total_tokens)}</TableCell>
                      <TableCell align="right">${row.estimated_cost.toFixed(2)}</TableCell>
                    </TableRow>
                  ))}
                  <TableRow>
                    <TableCell component="th" scope="row" sx={{ fontWeight: 'bold' }}>
                      TOTAL
                    </TableCell>
                    <TableCell align="right" sx={{ fontWeight: 'bold' }}>
                      {formatNumber(usageData.cost_breakdown.reduce((sum, row) => sum + row.input_tokens, 0))}
                    </TableCell>
                    <TableCell align="right" sx={{ fontWeight: 'bold' }}>
                      {formatNumber(usageData.cost_breakdown.reduce((sum, row) => sum + row.output_tokens, 0))}
                    </TableCell>
                    <TableCell align="right" sx={{ fontWeight: 'bold' }}>
                      {formatNumber(usageData.cost_breakdown.reduce((sum, row) => sum + row.total_tokens, 0))}
                    </TableCell>
                    <TableCell align="right" sx={{ fontWeight: 'bold' }}>
                      ${usageData.cost_breakdown.reduce((sum, row) => sum + row.estimated_cost, 0).toFixed(2)}
                    </TableCell>
                  </TableRow>
                </TableBody>
              </Table>
            </TableContainer>
          </Paper>
        </Box>
      )}
    </Box>
  );
};

export default UsageDashboard;
