import React from 'react';
import { Box, Typography, Paper, useTheme } from '@mui/material';
import { Bar } from 'react-chartjs-2';
import { Chart as ChartJS, CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend } from 'chart.js';

// Register Chart.js components
ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

/**
 * ParameterImportanceChart component for visualizing parameter importance
 */
const ParameterImportanceChart = ({ 
  importanceData, 
  title = 'Parameter Importance',
  height = 300,
  maxParams = 10
}) => {
  const theme = useTheme();
  
  // Format data for chart
  const formatChartData = () => {
    if (!importanceData || !importanceData.importance) {
      return {
        labels: [],
        datasets: []
      };
    }
    
    const importance = importanceData.importance;
    
    // Sort parameters by importance
    const sortedParams = Object.entries(importance)
      .sort((a, b) => b[1] - a[1])
      .slice(0, maxParams);
    
    const labels = sortedParams.map(([param]) => {
      // Shorten parameter names if too long
      return param.length > 20 ? param.substring(0, 17) + '...' : param;
    });
    
    const values = sortedParams.map(([_, value]) => value);
    
    // Generate colors based on importance
    const colors = values.map(value => {
      const hue = 200; // Blue hue
      const saturation = 70;
      const lightness = 100 - (value * 50); // Higher importance = darker color
      
      return `hsl(${hue}, ${saturation}%, ${lightness}%)`;
    });
    
    return {
      labels,
      datasets: [
        {
          label: 'Importance',
          data: values,
          backgroundColor: colors,
          borderColor: colors.map(color => color.replace('%)', '%, 1)')),
          borderWidth: 1
        }
      ]
    };
  };
  
  // Get chart options
  const getChartOptions = () => {
    return {
      responsive: true,
      maintainAspectRatio: false,
      indexAxis: 'y', // Horizontal bar chart
      plugins: {
        legend: {
          display: false
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
          callbacks: {
            label: (context) => {
              const value = context.parsed.x;
              return `Importance: ${value.toFixed(4)}`;
            }
          }
        }
      },
      scales: {
        x: {
          beginAtZero: true,
          max: 1,
          title: {
            display: true,
            text: 'Importance'
          },
          ticks: {
            callback: (value) => value.toFixed(2)
          }
        },
        y: {
          title: {
            display: true,
            text: 'Parameter'
          }
        }
      }
    };
  };
  
  if (!importanceData || !importanceData.importance) {
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
          No parameter importance data available
        </Typography>
      </Paper>
    );
  }
  
  return (
    <Box sx={{ height }}>
      <Bar 
        data={formatChartData()} 
        options={getChartOptions()} 
      />
    </Box>
  );
};

export default ParameterImportanceChart;
