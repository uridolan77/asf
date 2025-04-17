import React, { useState, useEffect } from 'react';
import {
  Box,
  Paper,
  Typography,
  Grid,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Button,
  Card,
  CardContent,
  LinearProgress,
  Alert,
  Tab,
  Tabs
} from '@mui/material';
import {
  Refresh as RefreshIcon,
  CalendarToday as CalendarIcon,
  Assessment as AssessmentIcon
} from '@mui/icons-material';
import { 
  LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, 
  Tooltip, Legend, ResponsiveContainer, PieChart, Pie, Cell 
} from 'recharts';
import ClientService from '../../services/ClientService';
import { TabPanel, a11yProps } from '../common/TabPanel';

interface DSPyMetricsVisualizationProps {
  clientId: string;
}

const DSPyMetricsVisualization: React.FC<DSPyMetricsVisualizationProps> = ({ clientId }) => {
  const [metrics, setMetrics] = useState<any>({});
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [timeRange, setTimeRange] = useState<string>('day');
  const [metricType, setMetricType] = useState<string>('latency');
  const [currentTab, setCurrentTab] = useState<number>(0);

  // Define colors for charts
  const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884d8', '#82ca9d'];

  useEffect(() => {
    loadMetrics();
  }, [clientId, timeRange, metricType]);

  const loadMetrics = async () => {
    try {
      setLoading(true);
      setError(null);
      
      // In a real implementation, we would fetch actual metrics from the API
      // const data = await ClientService.getDSPyMetrics(clientId, timeRange, metricType);
      // setMetrics(data);
      
      // For now, generate mock metrics
      setTimeout(() => {
        const mockData = generateMockMetrics();
        setMetrics(mockData);
        setLoading(false);
      }, 800);
    } catch (err) {
      console.error('Error loading metrics:', err);
      setError('Failed to load metrics. Please try again.');
      setLoading(false);
    }
  };

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setCurrentTab(newValue);
  };

  // Helper function to generate mock metrics
  const generateMockMetrics = () => {
    // Generate time series data based on the selected time range
    const now = new Date();
    const timeSeriesData = [];
    
    const formatLabel = (date: Date) => {
      if (timeRange === 'hour') {
        return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
      } else if (timeRange === 'day') {
        return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
      } else if (timeRange === 'week') {
        return date.toLocaleDateString([], { weekday: 'short' });
      } else if (timeRange === 'month') {
        return `${date.getDate()}/${date.getMonth() + 1}`;
      }
      return date.toLocaleDateString();
    };
    
    const points = timeRange === 'hour' ? 12 : // 5 minute intervals
                   timeRange === 'day' ? 24 :  // hourly
                   timeRange === 'week' ? 7 :  // daily
                   timeRange === 'month' ? 30 : 12; // monthly
    
    const interval = timeRange === 'hour' ? 5 * 60 * 1000 :
                     timeRange === 'day' ? 60 * 60 * 1000 :
                     timeRange === 'week' ? 24 * 60 * 60 * 1000 :
                     timeRange === 'month' ? 24 * 60 * 60 * 1000 : 30 * 24 * 60 * 60 * 1000;
    
    for (let i = points - 1; i >= 0; i--) {
      const date = new Date(now.getTime() - (i * interval));
      
      const baseLatency = Math.random() * 0.3 + 0.7; // Base latency between 0.7 and 1.0 seconds
      const jitter = Math.random() * 0.5 - 0.25; // Add some randomness
      
      const entry = {
        timestamp: date.toISOString(),
        label: formatLabel(date),
        latency: baseLatency + jitter + (i > points / 2 ? 0.2 : 0), // Slight improvement over time
        successRate: 95 + Math.random() * 5 - (i > points / 2 ? 2 : 0), // Success rate 93-100%
        cacheHitRate: 40 + Math.random() * 20 + (i > points / 2 ? -5 : 5), // Cache hit rate improving over time
        tokensUsed: Math.floor(800 + Math.random() * 400), // Tokens between 800-1200
        costEstimate: (800 + Math.random() * 400) * 0.00002
      };
      
      timeSeriesData.push(entry);
    }
    
    // Generate module performance data
    const modulePerformanceData = [
      { name: 'MedicalRAG', latency: 1.2, successRate: 96.5, cacheHitRate: 65, calls: 430 },
      { name: 'ContradictionDetection', latency: 0.9, successRate: 98.2, cacheHitRate: 45, calls: 310 },
      { name: 'EvidenceExtraction', latency: 1.5, successRate: 94.8, cacheHitRate: 38, calls: 280 },
      { name: 'MedicalSummarization', latency: 2.1, successRate: 92.7, cacheHitRate: 55, calls: 120 },
      { name: 'ClinicalQA', latency: 0.8, successRate: 97.5, cacheHitRate: 72, calls: 220 }
    ];
    
    // Generate model usage data
    const modelUsageData = [
      { name: 'GPT-4', tokens: 980000, cost: 19.60, calls: 560 },
      { name: 'GPT-3.5', tokens: 1240000, cost: 2.48, calls: 620 },
      { name: 'Claude 3', tokens: 450000, cost: 9.00, calls: 150 }
    ];

    // Generate pie charts data
    const errorTypeData = [
      { name: 'API Errors', value: 45 },
      { name: 'Timeout', value: 32 },
      { name: 'Validation', value: 18 },
      { name: 'Circuit Open', value: 5 }
    ];

    const requestTypeData = [
      { name: 'MODULE_CALL', value: 65 },
      { name: 'LLM_CALL', value: 30 },
      { name: 'CACHE_OPERATION', value: 5 }
    ];
    
    // Return all metrics
    return {
      timeSeriesData,
      modulePerformanceData,
      modelUsageData,
      errorTypeData,
      requestTypeData,
      summary: {
        totalCalls: 1360,
        avgLatency: 1.2,
        successRate: 96.8,
        cacheHitRate: 55.3,
        totalTokens: 2670000,
        totalCost: 31.08
      }
    };
  };

  return (
    <Box>
      <Paper elevation={2} sx={{ p: 3, mb: 3 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
          <Typography variant="h5">Performance Metrics</Typography>
          <Box sx={{ display: 'flex', gap: 2 }}>
            <FormControl size="small" sx={{ minWidth: 120 }}>
              <InputLabel id="time-range-label">Time Range</InputLabel>
              <Select
                labelId="time-range-label"
                value={timeRange}
                onChange={(e) => setTimeRange(e.target.value)}
                label="Time Range"
              >
                <MenuItem value="hour">Last Hour</MenuItem>
                <MenuItem value="day">Last 24 Hours</MenuItem>
                <MenuItem value="week">Last 7 Days</MenuItem>
                <MenuItem value="month">Last 30 Days</MenuItem>
              </Select>
            </FormControl>
            <Button
              variant="outlined"
              startIcon={<RefreshIcon />}
              onClick={loadMetrics}
              disabled={loading}
            >
              Refresh
            </Button>
          </Box>
        </Box>

        {loading ? (
          <LinearProgress />
        ) : error ? (
          <Alert severity="error" sx={{ mb: 3 }}>
            {error}
          </Alert>
        ) : metrics.summary ? (
          <>
            <Grid container spacing={2} sx={{ mb: 4 }}>
              <Grid item xs={12} sm={6} md={2}>
                <Card elevation={1}>
                  <CardContent>
                    <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                      Total Calls
                    </Typography>
                    <Typography variant="h4" color="primary">
                      {metrics.summary.totalCalls.toLocaleString()}
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
              <Grid item xs={12} sm={6} md={2}>
                <Card elevation={1}>
                  <CardContent>
                    <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                      Avg Latency
                    </Typography>
                    <Typography variant="h4" color="primary">
                      {metrics.summary.avgLatency.toFixed(2)}s
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
              <Grid item xs={12} sm={6} md={2}>
                <Card elevation={1}>
                  <CardContent>
                    <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                      Success Rate
                    </Typography>
                    <Typography variant="h4" color={metrics.summary.successRate > 95 ? 'success.main' : 'warning.main'}>
                      {metrics.summary.successRate.toFixed(1)}%
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
              <Grid item xs={12} sm={6} md={2}>
                <Card elevation={1}>
                  <CardContent>
                    <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                      Cache Hit Rate
                    </Typography>
                    <Typography variant="h4" color="primary">
                      {metrics.summary.cacheHitRate.toFixed(1)}%
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
              <Grid item xs={12} sm={6} md={2}>
                <Card elevation={1}>
                  <CardContent>
                    <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                      Total Tokens
                    </Typography>
                    <Typography variant="h4" color="primary">
                      {(metrics.summary.totalTokens / 1000000).toFixed(2)}M
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
              <Grid item xs={12} sm={6} md={2}>
                <Card elevation={1}>
                  <CardContent>
                    <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                      Total Cost
                    </Typography>
                    <Typography variant="h4" color="primary">
                      ${metrics.summary.totalCost.toFixed(2)}
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
            </Grid>

            <Box sx={{ width: '100%', mb: 4 }}>
              <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
                <Tabs value={currentTab} onChange={handleTabChange} aria-label="metrics tabs">
                  <Tab icon={<CalendarIcon />} label="Time Series" {...a11yProps(0)} />
                  <Tab icon={<AssessmentIcon />} label="Analysis" {...a11yProps(1)} />
                </Tabs>
              </Box>
              
              <TabPanel value={currentTab} index={0}>
                <FormControl size="small" sx={{ mb: 2, minWidth: 120 }}>
                  <InputLabel id="metric-type-label">Metric</InputLabel>
                  <Select
                    labelId="metric-type-label"
                    value={metricType}
                    onChange={(e) => setMetricType(e.target.value)}
                    label="Metric"
                  >
                    <MenuItem value="latency">Latency</MenuItem>
                    <MenuItem value="successRate">Success Rate</MenuItem>
                    <MenuItem value="cacheHitRate">Cache Hit Rate</MenuItem>
                    <MenuItem value="tokensUsed">Tokens Used</MenuItem>
                    <MenuItem value="costEstimate">Cost</MenuItem>
                  </Select>
                </FormControl>
                
                <Box sx={{ height: 400 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart
                      data={metrics.timeSeriesData}
                      margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="label" />
                      <YAxis />
                      <Tooltip />
                      <Legend />
                      <Line
                        type="monotone"
                        dataKey={metricType}
                        stroke="#8884d8"
                        activeDot={{ r: 8 }}
                        name={metricType === 'latency' ? 'Latency (s)' : 
                              metricType === 'successRate' ? 'Success Rate (%)' : 
                              metricType === 'cacheHitRate' ? 'Cache Hit Rate (%)' : 
                              metricType === 'tokensUsed' ? 'Tokens Used' : 
                              'Cost ($)'}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </Box>
              </TabPanel>
              
              <TabPanel value={currentTab} index={1}>
                <Grid container spacing={4}>
                  <Grid item xs={12} md={6}>
                    <Typography variant="h6" gutterBottom>Module Performance</Typography>
                    <Box sx={{ height: 300 }}>
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart
                          data={metrics.modulePerformanceData}
                          margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                        >
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="name" />
                          <YAxis />
                          <Tooltip />
                          <Legend />
                          <Bar dataKey="latency" name="Latency (s)" fill="#8884d8" />
                          <Bar dataKey="calls" name="Call Volume" fill="#82ca9d" />
                        </BarChart>
                      </ResponsiveContainer>
                    </Box>
                  </Grid>
                  
                  <Grid item xs={12} md={6}>
                    <Typography variant="h6" gutterBottom>Model Usage</Typography>
                    <Box sx={{ height: 300 }}>
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart
                          data={metrics.modelUsageData}
                          margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                        >
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="name" />
                          <YAxis yAxisId="left" orientation="left" stroke="#8884d8" />
                          <YAxis yAxisId="right" orientation="right" stroke="#82ca9d" />
                          <Tooltip />
                          <Legend />
                          <Bar yAxisId="left" dataKey="tokens" name="Tokens (1k)" fill="#8884d8" />
                          <Bar yAxisId="right" dataKey="cost" name="Cost ($)" fill="#82ca9d" />
                        </BarChart>
                      </ResponsiveContainer>
                    </Box>
                  </Grid>
                  
                  <Grid item xs={12} md={6}>
                    <Typography variant="h6" gutterBottom>Error Distribution</Typography>
                    <Box sx={{ height: 300 }}>
                      <ResponsiveContainer width="100%" height="100%">
                        <PieChart>
                          <Pie
                            data={metrics.errorTypeData}
                            cx="50%"
                            cy="50%"
                            labelLine={false}
                            label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                            outerRadius={80}
                            fill="#8884d8"
                            dataKey="value"
                          >
                            {metrics.errorTypeData.map((entry, index) => (
                              <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                            ))}
                          </Pie>
                          <Tooltip />
                          <Legend />
                        </PieChart>
                      </ResponsiveContainer>
                    </Box>
                  </Grid>
                  
                  <Grid item xs={12} md={6}>
                    <Typography variant="h6" gutterBottom>Request Type Distribution</Typography>
                    <Box sx={{ height: 300 }}>
                      <ResponsiveContainer width="100%" height="100%">
                        <PieChart>
                          <Pie
                            data={metrics.requestTypeData}
                            cx="50%"
                            cy="50%"
                            labelLine={false}
                            label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                            outerRadius={80}
                            fill="#8884d8"
                            dataKey="value"
                          >
                            {metrics.requestTypeData.map((entry, index) => (
                              <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                            ))}
                          </Pie>
                          <Tooltip />
                          <Legend />
                        </PieChart>
                      </ResponsiveContainer>
                    </Box>
                  </Grid>
                </Grid>
              </TabPanel>
            </Box>
          </>
        ) : (
          <Alert severity="info">No metrics data available.</Alert>
        )}
      </Paper>
    </Box>
  );
};

export default DSPyMetricsVisualization;