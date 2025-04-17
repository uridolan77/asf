import React from 'react';
import { Box, Typography, Paper, useTheme } from '@mui/material';
import { Scatter } from 'react-chartjs-2';
import { Chart as ChartJS, LinearScale, PointElement, LineElement, Tooltip, Legend } from 'chart.js';

// Register Chart.js components
ChartJS.register(LinearScale, PointElement, LineElement, Tooltip, Legend);

/**
 * TaskSimilarityChart component for visualizing task similarity in 2D space
 */
const TaskSimilarityChart = ({ 
  similarityData, 
  title = 'Task Similarity',
  height = 300
}) => {
  const theme = useTheme();
  
  // Format data for chart
  const formatChartData = () => {
    if (!similarityData || !similarityData.embeddings) {
      return {
        datasets: []
      };
    }
    
    const embeddings = similarityData.embeddings;
    const tasks = Object.keys(embeddings);
    
    // Create dataset with points for each task
    const data = tasks.map((task, index) => {
      // Get embedding for this task
      const embedding = embeddings[task];
      
      // Generate color based on index
      const hue = (index * 137) % 360; // Golden angle approximation for good color distribution
      const color = `hsl(${hue}, 70%, 60%)`;
      
      return {
        x: embedding[0],
        y: embedding[1],
        task
      };
    });
    
    return {
      datasets: [
        {
          label: 'Tasks',
          data,
          backgroundColor: data.map((_, index) => {
            const hue = (index * 137) % 360;
            return `hsl(${hue}, 70%, 60%)`;
          }),
          pointRadius: 8,
          pointHoverRadius: 12
        }
      ]
    };
  };
  
  // Get chart options
  const getChartOptions = () => {
    return {
      responsive: true,
      maintainAspectRatio: false,
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
              const point = context.raw;
              return `Task: ${point.task}`;
            }
          }
        }
      },
      scales: {
        x: {
          title: {
            display: true,
            text: 'Dimension 1'
          },
          ticks: {
            display: false
          }
        },
        y: {
          title: {
            display: true,
            text: 'Dimension 2'
          },
          ticks: {
            display: false
          }
        }
      }
    };
  };
  
  if (!similarityData || !similarityData.embeddings) {
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
          No similarity data available
        </Typography>
      </Paper>
    );
  }
  
  return (
    <Box sx={{ height }}>
      <Scatter 
        data={formatChartData()} 
        options={getChartOptions()} 
      />
    </Box>
  );
};

export default TaskSimilarityChart;
