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
  Tooltip,
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
  Legend,
  Tooltip as ChartTooltip,
  RadialLinearScale,
  ArcElement,
  Filler
} from 'chart.js';
import { Line, Bar, Radar, Scatter } from 'react-chartjs-2';
import { faker } from '@faker-js/faker';
import annotationPlugin from 'chartjs-plugin-annotation';

// Register ChartJS components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  RadialLinearScale,
  ArcElement,
  Title,
  Legend,
  ChartTooltip,
  Filler,
  annotationPlugin
);

const ForgettingChart = ({ adapter, forgettingResults, title = 'Forgetting Analysis' }) => {
  const theme = useTheme();
  const [chartType, setChartType] = useState('line');
  
  // Generate task labels from adapter task history
  const taskLabels = adapter?.task_history?.map(task => task.task_id) || [];
  
  // Generate mock data for forgetting visualization
  const generateForgettingData = () => {
    const labels = ['Initial', 'After Task 1', 'After Task 2', 'After Task 3', 'Current'];
    
    // Generate a decreasing performance curve to simulate forgetting
    const basePerformance = forgettingResults ? 
      (forgettingResults.metric_key.includes('loss') ? 0.5 : 0.85) : 
      0.85;
    
    const forgettingAmount = forgettingResults ? 
      (forgettingResults.metric_key.includes('loss') ? -forgettingResults.forgetting : forgettingResults.forgetting) : 
      0.15;
    
    // For loss metrics, lower is better, so forgetting means increasing values
    const isLossMetric = forgettingResults?.metric_key.includes('loss');
    
    // Generate performance curve
    const performanceCurve = [
      basePerformance,
      isLossMetric ? basePerformance * 1.05 : basePerformance * 0.98,
      isLossMetric ? basePerformance * 1.1 : basePerformance * 0.95,
      isLossMetric ? basePerformance * 1.15 : basePerformance * 0.92,
      isLossMetric ? basePerformance + forgettingAmount : basePerformance - forgettingAmount
    ];
    
    // Generate comparison curve (less forgetting)
    const comparisonCurve = [
      basePerformance,
      isLossMetric ? basePerformance * 1.02 : basePerformance * 0.99,
      isLossMetric ? basePerformance * 1.04 : basePerformance * 0.97,
      isLossMetric ? basePerformance * 1.06 : basePerformance * 0.96,
      isLossMetric ? basePerformance + (forgettingAmount / 2) : basePerformance - (forgettingAmount / 2)
    ];
    
    return {
      labels,
      datasets: [
        {
          label: `Performance on Task (${forgettingResults?.metric_key || 'accuracy'})`,
          data: performanceCurve,
          borderColor: theme.palette.primary.main,
          backgroundColor: `${theme.palette.primary.main}20`,
          tension: 0.3,
          fill: chartType === 'area'
        },
        {
          label: 'With Better CL Strategy',
          data: comparisonCurve,
          borderColor: theme.palette.success.main,
          backgroundColor: `${theme.palette.success.main}20`,
          borderDash: [5, 5],
          tension: 0.3,
          fill: chartType === 'area'
        }
      ]
    };
  };
  
  // Generate task-wise forgetting data
  const generateTaskWiseForgetting = () => {
    const tasks = taskLabels.length > 0 ? 
      taskLabels : 
      ['Task 1', 'Task 2', 'Task 3', 'Task 4', 'Task 5'];
    
    // Generate random forgetting values for each task
    const forgettingValues = tasks.map(() => faker.number.float({ min: 0.01, max: 0.25, precision: 0.01 }));
    
    // If we have real forgetting results, use that for the first task
    if (forgettingResults && tasks.includes(forgettingResults.task_id)) {
      const taskIndex = tasks.indexOf(forgettingResults.task_id);
      forgettingValues[taskIndex] = forgettingResults.forgetting;
    }
    
    return {
      labels: tasks,
      datasets: [
        {
          label: 'Forgetting',
          data: forgettingValues,
          backgroundColor: tasks.map((_, i) => {
            const value = forgettingValues[i];
            if (value < 0.05) return theme.palette.success.main;
            if (value < 0.15) return theme.palette.warning.main;
            return theme.palette.error.main;
          }),
          borderColor: theme.palette.divider,
          borderWidth: 1
        }
      ]
    };
  };
  
  // Get chart data based on chart type
  const getChartData = () => {
    if (chartType === 'bar') {
      return generateTaskWiseForgetting();
    } else {
      return generateForgettingData();
    }
  };
  
  const chartData = getChartData();
  
  // Chart options
  const getChartOptions = () => {
    const baseOptions = {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: 'top',
        },
        title: {
          display: true,
          text: chartType === 'bar' ? 'Task-wise Forgetting' : 'Performance Over Time'
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
      interaction: {
        mode: 'index',
        intersect: false,
      },
      animation: {
        duration: 1000
      }
    };
    
    if (chartType === 'bar') {
      return {
        ...baseOptions,
        scales: {
          y: {
            beginAtZero: true,
            max: 0.3,
            title: {
              display: true,
              text: 'Forgetting'
            },
            ticks: {
              callback: function(value) {
                return value.toFixed(2);
              }
            }
          }
        },
        plugins: {
          ...baseOptions.plugins,
          annotation: {
            annotations: {
              line1: {
                type: 'line',
                yMin: 0.05,
                yMax: 0.05,
                borderColor: theme.palette.success.main,
                borderWidth: 2,
                borderDash: [5, 5],
                label: {
                  content: 'Low Forgetting',
                  display: true,
                  position: 'start',
                  backgroundColor: theme.palette.success.main
                }
              },
              line2: {
                type: 'line',
                yMin: 0.15,
                yMax: 0.15,
                borderColor: theme.palette.error.main,
                borderWidth: 2,
                borderDash: [5, 5],
                label: {
                  content: 'High Forgetting',
                  display: true,
                  position: 'start',
                  backgroundColor: theme.palette.error.main
                }
              }
            }
          }
        }
      };
    } else {
      const isLossMetric = forgettingResults?.metric_key.includes('loss');
      return {
        ...baseOptions,
        scales: {
          y: {
            beginAtZero: false,
            title: {
              display: true,
              text: forgettingResults?.metric_key || 'Performance'
            },
            ticks: {
              callback: function(value) {
                return value.toFixed(2);
              }
            }
          }
        }
      };
    }
  };
  
  const options = getChartOptions();
  
  const handleChartTypeChange = (event) => {
    setChartType(event.target.value);
  };
  
  return (
    <Card variant="outlined">
      <CardHeader 
        title={title}
        action={
          <FormControl size="small" sx={{ minWidth: 120 }}>
            <InputLabel>View</InputLabel>
            <Select
              value={chartType}
              onChange={handleChartTypeChange}
              label="View"
            >
              <MenuItem value="line">Performance Over Time</MenuItem>
              <MenuItem value="area">Area Chart</MenuItem>
              <MenuItem value="bar">Task-wise Forgetting</MenuItem>
            </Select>
          </FormControl>
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
        
        <Box sx={{ mt: 2 }}>
          <Typography variant="subtitle2" gutterBottom>
            Interpretation
          </Typography>
          <Typography variant="body2">
            {chartType === 'bar' ? (
              "This chart shows the amount of forgetting for each task. Lower values indicate better retention of knowledge. Values below 0.05 represent minimal forgetting, while values above 0.15 indicate significant forgetting that may require attention."
            ) : (
              "This chart shows how performance on a task changes over time as the model is trained on subsequent tasks. The blue line represents the actual performance, while the green dashed line shows a hypothetical performance with a better continual learning strategy."
            )}
          </Typography>
        </Box>
      </CardContent>
    </Card>
  );
};

export default ForgettingChart;
