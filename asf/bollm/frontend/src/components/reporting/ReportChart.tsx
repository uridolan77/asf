import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Grid,
  Paper,
  SelectChangeEvent
} from '@mui/material';
import {
  BarChart,
  Bar,
  LineChart,
  Line,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer
} from 'recharts';
import { ReportResult, Dimension, Metric } from '../../types/reporting';

interface ReportChartProps {
  result: ReportResult | null;
  loading: boolean;
  dimensions: Dimension[];
  metrics: Metric[];
}

type ChartType = 'bar' | 'line' | 'pie';

const CHART_COLORS = [
  '#8884d8', '#82ca9d', '#ffc658', '#ff8042', '#0088fe', '#00c49f', '#ffbb28', '#ff8042',
  '#a4de6c', '#d0ed57', '#83a6ed', '#8dd1e1', '#a4262c', '#ca5010', '#8764b8', '#038387'
];

const ReportChart: React.FC<ReportChartProps> = ({
  result,
  loading,
  dimensions,
  metrics
}) => {
  const [chartType, setChartType] = useState<ChartType>('bar');
  const [dimensionField, setDimensionField] = useState<string>('');
  const [metricFields, setMetricFields] = useState<string[]>([]);
  const [chartData, setChartData] = useState<any[]>([]);
  
  useEffect(() => {
    if (result && result.data && !loading) {
      const { dimensions: dimensionIds, metrics: metricIds, rows } = result.data;
      
      // Set default dimension and metric if not already set
      if (!dimensionField && dimensionIds.length > 0) {
        setDimensionField(dimensionIds[0]);
      }
      
      if (metricFields.length === 0 && metricIds.length > 0) {
        setMetricFields([metricIds[0]]);
      }
      
      // Prepare chart data
      if (dimensionField && metricFields.length > 0) {
        prepareChartData();
      }
    }
  }, [result, dimensionField, metricFields, loading]);
  
  const prepareChartData = () => {
    if (!result || !result.data) return;
    
    const { rows } = result.data;
    
    // For pie charts, we can only use one metric
    const metricsToUse = chartType === 'pie' ? [metricFields[0]] : metricFields;
    
    // Group by dimension field
    const groupedData: Record<string, Record<string, number>> = {};
    
    rows.forEach(row => {
      const dimensionValue = String(row[dimensionField] || 'Unknown');
      
      if (!groupedData[dimensionValue]) {
        groupedData[dimensionValue] = {};
      }
      
      metricsToUse.forEach(metric => {
        groupedData[dimensionValue][metric] = row[metric] || 0;
      });
    });
    
    // Convert to chart data format
    const chartData = Object.entries(groupedData).map(([name, values]) => ({
      name,
      ...values
    }));
    
    // Sort data for better visualization
    chartData.sort((a, b) => {
      const metricField = metricsToUse[0];
      return (b[metricField] || 0) - (a[metricField] || 0);
    });
    
    // Limit to top 10 for better visualization
    const limitedData = chartData.slice(0, 10);
    
    setChartData(limitedData);
  };
  
  const handleChartTypeChange = (event: SelectChangeEvent) => {
    setChartType(event.target.value as ChartType);
    
    // For pie charts, limit to one metric
    if (event.target.value === 'pie' && metricFields.length > 1) {
      setMetricFields([metricFields[0]]);
    }
  };
  
  const handleDimensionChange = (event: SelectChangeEvent) => {
    setDimensionField(event.target.value);
  };
  
  const handleMetricChange = (event: SelectChangeEvent<string[]>) => {
    const value = event.target.value;
    setMetricFields(typeof value === 'string' ? [value] : value);
  };
  
  const getDimensionName = (id: string): string => {
    const dimension = dimensions.find(d => d.id === id);
    return dimension ? dimension.name : id;
  };
  
  const getMetricName = (id: string): string => {
    const metric = metrics.find(m => m.id === id);
    return metric ? metric.name : id;
  };
  
  if (!result || !result.data || loading) {
    return (
      <Box sx={{ p: 2, textAlign: 'center' }}>
        <Typography variant="body1" color="text.secondary">
          No data to display
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Run the report to see results
        </Typography>
      </Box>
    );
  }
  
  const { data } = result;
  const { dimensions: dimensionIds, metrics: metricIds } = data;
  
  const renderChart = () => {
    if (chartData.length === 0) {
      return (
        <Box sx={{ p: 2, textAlign: 'center' }}>
          <Typography variant="body1" color="text.secondary">
            No data to display for the selected dimension and metrics
          </Typography>
        </Box>
      );
    }
    
    switch (chartType) {
      case 'bar':
        return (
          <ResponsiveContainer width="100%" height={400}>
            <BarChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 70 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis 
                dataKey="name" 
                angle={-45} 
                textAnchor="end" 
                height={70} 
                interval={0}
                tick={{ fontSize: 12 }}
              />
              <YAxis />
              <Tooltip />
              <Legend />
              {metricFields.map((metric, index) => (
                <Bar 
                  key={metric} 
                  dataKey={metric} 
                  name={getMetricName(metric)} 
                  fill={CHART_COLORS[index % CHART_COLORS.length]} 
                />
              ))}
            </BarChart>
          </ResponsiveContainer>
        );
        
      case 'line':
        return (
          <ResponsiveContainer width="100%" height={400}>
            <LineChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 70 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis 
                dataKey="name" 
                angle={-45} 
                textAnchor="end" 
                height={70} 
                interval={0}
                tick={{ fontSize: 12 }}
              />
              <YAxis />
              <Tooltip />
              <Legend />
              {metricFields.map((metric, index) => (
                <Line 
                  key={metric} 
                  type="monotone" 
                  dataKey={metric} 
                  name={getMetricName(metric)} 
                  stroke={CHART_COLORS[index % CHART_COLORS.length]} 
                  activeDot={{ r: 8 }} 
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        );
        
      case 'pie':
        const metricField = metricFields[0];
        return (
          <ResponsiveContainer width="100%" height={400}>
            <PieChart margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
              <Pie
                data={chartData}
                dataKey={metricField}
                nameKey="name"
                cx="50%"
                cy="50%"
                outerRadius={150}
                fill="#8884d8"
                label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
              >
                {chartData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={CHART_COLORS[index % CHART_COLORS.length]} />
                ))}
              </Pie>
              <Tooltip formatter={(value) => new Intl.NumberFormat('en-US').format(value as number)} />
              <Legend />
            </PieChart>
          </ResponsiveContainer>
        );
        
      default:
        return null;
    }
  };
  
  return (
    <Box>
      <Grid container spacing={2} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={4}>
          <FormControl fullWidth size="small">
            <InputLabel>Chart Type</InputLabel>
            <Select
              value={chartType}
              onChange={handleChartTypeChange}
              label="Chart Type"
            >
              <MenuItem value="bar">Bar Chart</MenuItem>
              <MenuItem value="line">Line Chart</MenuItem>
              <MenuItem value="pie">Pie Chart</MenuItem>
            </Select>
          </FormControl>
        </Grid>
        
        <Grid item xs={12} sm={4}>
          <FormControl fullWidth size="small">
            <InputLabel>Dimension</InputLabel>
            <Select
              value={dimensionField}
              onChange={handleDimensionChange}
              label="Dimension"
            >
              {dimensionIds.map(id => (
                <MenuItem key={id} value={id}>
                  {getDimensionName(id)}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        </Grid>
        
        <Grid item xs={12} sm={4}>
          <FormControl fullWidth size="small">
            <InputLabel>Metrics</InputLabel>
            <Select
              multiple
              value={metricFields}
              onChange={handleMetricChange}
              label="Metrics"
              renderValue={(selected) => (
                <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                  {(selected as string[]).map((value) => (
                    <Typography key={value} variant="body2" noWrap>
                      {getMetricName(value)}
                      {(selected as string[]).length > 1 && ', '}
                    </Typography>
                  ))}
                </Box>
              )}
              disabled={chartType === 'pie' && metricFields.length > 0}
            >
              {metricIds.map(id => (
                <MenuItem key={id} value={id}>
                  {getMetricName(id)}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        </Grid>
      </Grid>
      
      <Paper sx={{ p: 2 }}>
        {renderChart()}
      </Paper>
    </Box>
  );
};

export default ReportChart;
