import React, { useState, useEffect, useRef } from 'react';
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
  RadialLinearScale,
  PointElement,
  LineElement,
  Filler,
  Tooltip as ChartTooltip,
  Legend
} from 'chart.js';
import { Radar } from 'react-chartjs-2';
import { faker } from '@faker-js/faker';

// Register ChartJS components
ChartJS.register(
  RadialLinearScale,
  PointElement,
  LineElement,
  Filler,
  ChartTooltip,
  Legend
);

const TaskRelationshipChart = ({ adapter, title = 'Task Relationships' }) => {
  const theme = useTheme();
  const [metricType, setMetricType] = useState('similarity');
  
  // Generate task labels from adapter task history
  const taskLabels = adapter?.task_history?.map(task => task.task_id) || [];
  const tasks = taskLabels.length > 0 ? 
    taskLabels : 
    ['Task 1', 'Task 2', 'Task 3', 'Task 4', 'Task 5'];
  
  // Generate similarity data
  const generateSimilarityData = () => {
    return {
      labels: tasks,
      datasets: tasks.map((task, index) => {
        // Generate colors based on index
        const hue = (index * 137) % 360;
        const color = `hsl(${hue}, 70%, 60%)`;
        const backgroundColor = `hsla(${hue}, 70%, 60%, 0.2)`;
        
        // Generate similarity values for this task with all other tasks
        const similarities = tasks.map((_, i) => {
          if (i === index) return 1; // Same task has perfect similarity
          
          // Generate a random similarity value, but make adjacent tasks more similar
          const distance = Math.abs(i - index);
          const baseSimilarity = 0.3 + (0.7 / (distance + 1));
          return faker.number.float({ min: baseSimilarity - 0.2, max: baseSimilarity + 0.2, precision: 0.01 });
        });
        
        return {
          label: task,
          data: similarities,
          backgroundColor,
          borderColor: color,
          borderWidth: 1
        };
      })
    };
  };
  
  // Generate transfer data
  const generateTransferData = () => {
    return {
      labels: tasks,
      datasets: tasks.map((task, index) => {
        // Generate colors based on index
        const hue = (index * 137) % 360;
        const color = `hsl(${hue}, 70%, 60%)`;
        const backgroundColor = `hsla(${hue}, 70%, 60%, 0.2)`;
        
        // Generate transfer values for this task with all other tasks
        const transfers = tasks.map((_, i) => {
          if (i === index) return 0; // No transfer to self
          
          // Generate a random transfer value
          // Positive values indicate positive transfer, negative values indicate negative transfer
          const distance = Math.abs(i - index);
          const baseTransfer = 0.2 - (0.1 * distance);
          return faker.number.float({ min: baseTransfer - 0.1, max: baseTransfer + 0.1, precision: 0.01 });
        });
        
        return {
          label: task,
          data: transfers,
          backgroundColor,
          borderColor: color,
          borderWidth: 1
        };
      })
    };
  };
  
  // Generate forgetting data
  const generateForgettingData = () => {
    return {
      labels: tasks,
      datasets: tasks.map((task, index) => {
        // Generate colors based on index
        const hue = (index * 137) % 360;
        const color = `hsl(${hue}, 70%, 60%)`;
        const backgroundColor = `hsla(${hue}, 70%, 60%, 0.2)`;
        
        // Generate forgetting values for this task with all other tasks
        const forgetting = tasks.map((_, i) => {
          if (i === index) return 0; // No forgetting of self
          
          // Generate a random forgetting value
          const distance = Math.abs(i - index);
          const baseForgetting = 0.05 + (0.05 * distance);
          return faker.number.float({ min: baseForgetting - 0.03, max: baseForgetting + 0.03, precision: 0.01 });
        });
        
        return {
          label: task,
          data: forgetting,
          backgroundColor,
          borderColor: color,
          borderWidth: 1
        };
      })
    };
  };
  
  // Get chart data based on metric type
  const getChartData = () => {
    switch (metricType) {
      case 'similarity':
        return generateSimilarityData();
      case 'transfer':
        return generateTransferData();
      case 'forgetting':
        return generateForgettingData();
      default:
        return generateSimilarityData();
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
          text: `Task ${metricType === 'similarity' ? 'Similarity' : metricType === 'transfer' ? 'Transfer' : 'Forgetting'}`
        },
        tooltip: {
          callbacks: {
            label: function(context) {
              let label = context.dataset.label || '';
              if (label) {
                label += ' to ' + context.chart.data.labels[context.dataIndex] + ': ';
              }
              if (context.parsed.r !== null) {
                label += context.parsed.r.toFixed(2);
              }
              return label;
            }
          }
        }
      },
      scales: {
        r: {
          min: metricType === 'transfer' ? -0.3 : 0,
          max: metricType === 'similarity' ? 1 : 0.3,
          ticks: {
            stepSize: metricType === 'similarity' ? 0.2 : 0.1,
            callback: function(value) {
              return value.toFixed(1);
            }
          },
          pointLabels: {
            font: {
              size: 12
            }
          }
        }
      }
    };
    
    return baseOptions;
  };
  
  const options = getChartOptions();
  
  const handleMetricTypeChange = (event) => {
    setMetricType(event.target.value);
  };
  
  return (
    <Card variant="outlined">
      <CardHeader 
        title={title}
        action={
          <FormControl size="small" sx={{ minWidth: 120 }}>
            <InputLabel>Metric</InputLabel>
            <Select
              value={metricType}
              onChange={handleMetricTypeChange}
              label="Metric"
            >
              <MenuItem value="similarity">Task Similarity</MenuItem>
              <MenuItem value="transfer">Knowledge Transfer</MenuItem>
              <MenuItem value="forgetting">Forgetting</MenuItem>
            </Select>
          </FormControl>
        }
      />
      <Divider />
      <CardContent>
        <Box sx={{ height: 400, position: 'relative' }}>
          <Radar data={chartData} options={options} />
        </Box>
        
        <Box sx={{ mt: 2 }}>
          <Typography variant="subtitle2" gutterBottom>
            Interpretation
          </Typography>
          <Typography variant="body2">
            {metricType === 'similarity' ? (
              "This chart shows the similarity between tasks. Higher values indicate greater similarity. Tasks that are more similar tend to have more knowledge transfer and potentially more interference."
            ) : metricType === 'transfer' ? (
              "This chart shows knowledge transfer between tasks. Positive values indicate positive transfer (learning one task helps with another), while negative values indicate negative transfer (learning one task hinders performance on another)."
            ) : (
              "This chart shows forgetting between tasks. Higher values indicate more forgetting when learning a new task. Tasks that cause more forgetting may benefit from stronger continual learning mechanisms."
            )}
          </Typography>
        </Box>
      </CardContent>
    </Card>
  );
};

export default TaskRelationshipChart;
