import React, { useEffect, useRef } from 'react';
import { Box, Typography, Paper, useTheme } from '@mui/material';
import { Line } from 'react-chartjs-2';
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend, Filler } from 'chart.js';
import annotationPlugin from 'chartjs-plugin-annotation';

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler,
  annotationPlugin
);

/**
 * ForgettingChart component for visualizing forgetting curves
 */
const ForgettingChart = ({ 
  forgettingData, 
  metricKey = 'eval_loss', 
  title = 'Forgetting Curve',
  height = 300
}) => {
  const theme = useTheme();
  const chartRef = useRef(null);
  
  // Format data for chart
  const formatChartData = () => {
    if (!forgettingData || !forgettingData.forgetting_curve) {
      return {
        labels: [],
        datasets: []
      };
    }
    
    const forgettingCurve = forgettingData.forgetting_curve;
    const tasks = Object.keys(forgettingCurve);
    
    // For each task, create a dataset
    const datasets = tasks.map((task, index) => {
      // Get values for this task
      const values = Array.isArray(forgettingCurve[task]) 
        ? forgettingCurve[task] 
        : [forgettingCurve[task]];
      
      // Generate colors based on index
      const hue = (index * 137) % 360; // Golden angle approximation for good color distribution
      const color = `hsl(${hue}, 70%, 60%)`;
      const backgroundColor = `hsla(${hue}, 70%, 60%, 0.2)`;
      
      return {
        label: task,
        data: values,
        borderColor: color,
        backgroundColor: backgroundColor,
        tension: 0.3,
        fill: false,
        pointRadius: 4,
        pointHoverRadius: 6
      };
    });
    
    // Create labels based on number of tasks
    const maxLength = Math.max(...datasets.map(d => d.data.length));
    const labels = Array.from({ length: maxLength }, (_, i) => `Task ${i + 1}`);
    
    return {
      labels,
      datasets
    };
  };
  
  // Get chart options
  const getChartOptions = () => {
    const isLossMetric = metricKey.toLowerCase().includes('loss');
    
    return {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: 'top',
        },
        title: {
          display: true,
          text: title,
          font: {
            size: 16,
            weight: 'bold'
          }
        },
        tooltip: {
          mode: 'index',
          intersect: false,
          callbacks: {
            label: (context) => {
              const label = context.dataset.label || '';
              const value = context.parsed.y;
              return `${label}: ${value.toFixed(4)}`;
            }
          }
        },
        annotation: {
          annotations: {
            line1: {
              type: 'line',
              yMin: forgettingData?.forgetting || 0,
              yMax: forgettingData?.forgetting || 0,
              borderColor: theme.palette.error.main,
              borderWidth: 2,
              borderDash: [6, 6],
              label: {
                display: true,
                content: `Forgetting: ${forgettingData?.forgetting?.toFixed(4) || 'N/A'}`,
                position: 'end',
                backgroundColor: theme.palette.error.main
              }
            }
          }
        }
      },
      scales: {
        y: {
          beginAtZero: false,
          title: {
            display: true,
            text: getMetricName(metricKey)
          },
          ticks: {
            callback: (value) => value.toFixed(2)
          },
          // For loss metrics, lower is better, so invert the scale
          reverse: isLossMetric
        },
        x: {
          title: {
            display: true,
            text: 'Tasks'
          }
        }
      }
    };
  };
  
  // Get metric name from key
  const getMetricName = (key) => {
    const metricNames = {
      'eval_loss': 'Loss',
      'eval_accuracy': 'Accuracy',
      'eval_f1': 'F1 Score',
      'eval_precision': 'Precision',
      'eval_recall': 'Recall',
      'eval_perplexity': 'Perplexity'
    };
    
    return metricNames[key] || key;
  };
  
  if (!forgettingData) {
    return (
      <Paper 
        variant="outlined" 
        sx={{ 
          p: 2, 
          height, 
          display: 'flex', 
          alignItems: 'center', 
          justifyContent: 'center' 
        }}
      >
        <Typography variant="body1" color="textSecondary">
          No forgetting data available
        </Typography>
      </Paper>
    );
  }
  
  return (
    <Box sx={{ height }}>
      <Line 
        ref={chartRef}
        data={formatChartData()} 
        options={getChartOptions()} 
      />
    </Box>
  );
};

export default ForgettingChart;
