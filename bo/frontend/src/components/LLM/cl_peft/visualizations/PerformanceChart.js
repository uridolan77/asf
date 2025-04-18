import React, { useState } from 'react';
import {
  Box,
  Card,
  CardContent,
  CardHeader,
  Divider,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Typography,
  useTheme
} from '@mui/material';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  Filler
} from 'chart.js';
import { Line, Bar } from 'react-chartjs-2';
import { faker } from '@faker-js/faker';

// Register ChartJS components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

const PerformanceChart = ({ adapter, metrics = ['loss', 'accuracy', 'f1'], title = 'Performance Metrics' }) => {
  const theme = useTheme();
  const [chartType, setChartType] = useState('line');
  const [metricType, setMetricType] = useState('all');
  
  // Generate task labels from adapter task history
  const taskLabels = adapter?.task_history?.map(task => task.task_id) || [];
  
  // Generate mock data if no task history or metrics
  const generateMockData = () => {
    const labels = taskLabels.length > 0 ? taskLabels : ['Task 1', 'Task 2', 'Task 3', 'Task 4', 'Task 5'];
    
    const datasets = metrics.map((metric, index) => {
      // Generate colors based on index
      const hue = (index * 137) % 360; // Use golden ratio to spread colors
      const color = `hsl(${hue}, 70%, 60%)`;
      const backgroundColor = `hsla(${hue}, 70%, 60%, 0.2)`;
      
      // Generate data points
      const dataPoints = labels.map(() => {
        if (metric.includes('loss')) {
          // Loss should decrease over time
          return faker.number.float({ min: 0.2, max: 2.0, precision: 0.01 });
        } else {
          // Accuracy, F1, etc. should increase over time
          return faker.number.float({ min: 0.6, max: 0.95, precision: 0.01 });
        }
      });
      
      return {
        label: metric,
        data: dataPoints,
        borderColor: color,
        backgroundColor,
        tension: 0.3,
        fill: chartType === 'area'
      };
    });
    
    return {
      labels,
      datasets
    };
  };
  
  // Extract real data from adapter task history if available
  const extractRealData = () => {
    if (!adapter?.task_history || adapter.task_history.length === 0) {
      return generateMockData();
    }
    
    const labels = adapter.task_history.map(task => task.task_id);
    
    const datasets = metrics.map((metric, index) => {
      // Generate colors based on index
      const hue = (index * 137) % 360;
      const color = `hsl(${hue}, 70%, 60%)`;
      const backgroundColor = `hsla(${hue}, 70%, 60%, 0.2)`;
      
      // Extract metric values from task history
      const dataPoints = adapter.task_history.map(task => {
        const metricKey = metric.includes('eval_') ? metric : `eval_${metric}`;
        return task.eval_metrics?.[metricKey] || task.metrics?.[metric] || null;
      });
      
      return {
        label: metric,
        data: dataPoints,
        borderColor: color,
        backgroundColor,
        tension: 0.3,
        fill: chartType === 'area'
      };
    });
    
    return {
      labels,
      datasets
    };
  };
  
  // Get chart data
  const chartData = extractRealData();
  
  // Filter datasets based on metric type
  if (metricType !== 'all') {
    chartData.datasets = chartData.datasets.filter(dataset => 
      metricType === 'accuracy' ? 
        dataset.label.includes('acc') || dataset.label.includes('f1') || dataset.label.includes('precision') || dataset.label.includes('recall') : 
        dataset.label.includes('loss')
    );
  }
  
  // Chart options
  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top',
      },
      title: {
        display: true,
        text: title
      },
      tooltip: {
        callbacks: {
          label: function(context) {
            let label = context.dataset.label || '';
            if (label) {
              label += ': ';
            }
            if (context.parsed.y !== null) {
              label += context.parsed.y.toFixed(4);
            }
            return label;
          }
        }
      }
    },
    scales: {
      y: {
        beginAtZero: false,
        ticks: {
          callback: function(value) {
            return value.toFixed(2);
          }
        }
      }
    },
    interaction: {
      mode: 'index',
      intersect: false,
    },
    animation: {
      duration: 1000
    }
  };
  
  const handleChartTypeChange = (event) => {
    setChartType(event.target.value);
  };
  
  const handleMetricTypeChange = (event) => {
    setMetricType(event.target.value);
  };
  
  return (
    <Card variant="outlined">
      <CardHeader 
        title={title}
        action={
          <Box sx={{ display: 'flex', gap: 2 }}>
            <FormControl size="small" sx={{ minWidth: 120 }}>
              <InputLabel>Chart Type</InputLabel>
              <Select
                value={chartType}
                onChange={handleChartTypeChange}
                label="Chart Type"
              >
                <MenuItem value="line">Line</MenuItem>
                <MenuItem value="bar">Bar</MenuItem>
                <MenuItem value="area">Area</MenuItem>
              </Select>
            </FormControl>
            <FormControl size="small" sx={{ minWidth: 120 }}>
              <InputLabel>Metrics</InputLabel>
              <Select
                value={metricType}
                onChange={handleMetricTypeChange}
                label="Metrics"
              >
                <MenuItem value="all">All Metrics</MenuItem>
                <MenuItem value="accuracy">Accuracy Metrics</MenuItem>
                <MenuItem value="loss">Loss Metrics</MenuItem>
              </Select>
            </FormControl>
          </Box>
        }
      />
      <Divider />
      <CardContent>
        <Box sx={{ height: 400, position: 'relative' }}>
          {chartType === 'bar' ? (
            <Bar data={chartData} options={options} />
          ) : (
            <Line 
              data={chartData} 
              options={options} 
            />
          )}
        </Box>
      </CardContent>
    </Card>
  );
};

export default PerformanceChart;
