import React, { useState, useEffect, useRef } from 'react';
import {
  Box,
  Typography,
  Paper,
  Grid,
  Button,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  TextField,
  CircularProgress,
  Alert,
  Card,
  CardContent,
  CardHeader,
  Divider,
  Chip,
  Tooltip,
  IconButton,
  useTheme
} from '@mui/material';
import {
  PlayArrow as PlayArrowIcon,
  Refresh as RefreshIcon,
  Download as DownloadIcon,
  Info as InfoIcon,
  BarChart as BarChartIcon
} from '@mui/icons-material';
import { Chart, registerables } from 'chart.js';
import { useSnackbar } from 'notistack';

import { evaluateAdapter, computeForgetting } from '../../../services/cl_peft_service';

// Register Chart.js components
Chart.register(...registerables);

/**
 * Evaluation component for CL-PEFT adapters
 */
const Evaluation = ({ adapter, onEvaluationComplete }) => {
  const theme = useTheme();
  const { enqueueSnackbar } = useSnackbar();
  const chartRef = useRef(null);
  const chartInstance = useRef(null);
  
  const [loading, setLoading] = useState(false);
  const [evaluating, setEvaluating] = useState(false);
  const [evaluationResults, setEvaluationResults] = useState(null);
  const [forgettingResults, setForgettingResults] = useState(null);
  const [selectedTask, setSelectedTask] = useState('');
  const [selectedMetric, setSelectedMetric] = useState('eval_loss');
  
  // Available metrics
  const metrics = [
    { key: 'eval_loss', name: 'Loss', description: 'Evaluation loss' },
    { key: 'eval_accuracy', name: 'Accuracy', description: 'Evaluation accuracy' },
    { key: 'eval_f1', name: 'F1 Score', description: 'Evaluation F1 score' },
    { key: 'eval_precision', name: 'Precision', description: 'Evaluation precision' },
    { key: 'eval_recall', name: 'Recall', description: 'Evaluation recall' }
  ];
  
  // Get task history from adapter
  const taskHistory = adapter?.task_history || [];
  
  // Initialize selected task
  useEffect(() => {
    if (taskHistory.length > 0 && !selectedTask) {
      setSelectedTask(taskHistory[0].task_id);
    }
  }, [taskHistory, selectedTask]);
  
  // Handle task selection change
  const handleTaskChange = (event) => {
    setSelectedTask(event.target.value);
    setEvaluationResults(null);
    setForgettingResults(null);
  };
  
  // Handle metric selection change
  const handleMetricChange = (event) => {
    setSelectedMetric(event.target.value);
    
    // Update chart if forgetting results exist
    if (forgettingResults) {
      updateChart(forgettingResults, event.target.value);
    }
  };
  
  // Run evaluation
  const handleEvaluate = async () => {
    if (!selectedTask) {
      enqueueSnackbar('Please select a task to evaluate', { variant: 'warning' });
      return;
    }
    
    setEvaluating(true);
    
    try {
      const results = await evaluateAdapter(adapter.adapter_id, {
        task_id: selectedTask
      });
      
      setEvaluationResults(results);
      enqueueSnackbar('Evaluation completed successfully', { variant: 'success' });
      
      if (onEvaluationComplete) {
        onEvaluationComplete();
      }
    } catch (error) {
      console.error('Error evaluating adapter:', error);
      enqueueSnackbar('Failed to evaluate adapter', { variant: 'error' });
    } finally {
      setEvaluating(false);
    }
  };
  
  // Compute forgetting
  const handleComputeForgetting = async () => {
    if (!selectedTask) {
      enqueueSnackbar('Please select a task to compute forgetting', { variant: 'warning' });
      return;
    }
    
    setLoading(true);
    
    try {
      const results = await computeForgetting(adapter.adapter_id, {
        task_id: selectedTask,
        metric_key: selectedMetric
      });
      
      setForgettingResults(results);
      updateChart(results, selectedMetric);
      
      enqueueSnackbar('Forgetting computation completed', { variant: 'success' });
    } catch (error) {
      console.error('Error computing forgetting:', error);
      enqueueSnackbar('Failed to compute forgetting', { variant: 'error' });
    } finally {
      setLoading(false);
    }
  };
  
  // Update chart with forgetting results
  const updateChart = (results, metricKey) => {
    if (!chartRef.current) return;
    
    // Destroy existing chart
    if (chartInstance.current) {
      chartInstance.current.destroy();
    }
    
    // Create new chart
    const ctx = chartRef.current.getContext('2d');
    
    // Sample data for visualization
    // In a real implementation, this would use actual data from the forgetting results
    const labels = ['Task 1', 'Task 2', 'Task 3', 'Task 4'];
    const data = [0.95, 0.92, 0.88, 0.82];
    
    // Create chart
    chartInstance.current = new Chart(ctx, {
      type: 'line',
      data: {
        labels: labels,
        datasets: [{
          label: `${getMetricName(metricKey)} Over Tasks`,
          data: data,
          borderColor: theme.palette.primary.main,
          backgroundColor: theme.palette.primary.light,
          tension: 0.1,
          fill: true
        }]
      },
      options: {
        responsive: true,
        plugins: {
          legend: {
            position: 'top',
          },
          title: {
            display: true,
            text: 'Forgetting Curve'
          },
          tooltip: {
            mode: 'index',
            intersect: false,
          }
        },
        scales: {
          y: {
            beginAtZero: false,
            title: {
              display: true,
              text: getMetricName(metricKey)
            }
          },
          x: {
            title: {
              display: true,
              text: 'Tasks'
            }
          }
        }
      }
    });
  };
  
  // Get metric name from key
  const getMetricName = (key) => {
    const metric = metrics.find(m => m.key === key);
    return metric ? metric.name : key;
  };
  
  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        Evaluation
      </Typography>
      <Typography variant="body2" color="textSecondary" paragraph>
        Evaluate the adapter on specific tasks and compute forgetting metrics.
      </Typography>
      
      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Card variant="outlined">
            <CardHeader title="Run Evaluation" />
            <Divider />
            <CardContent>
              <FormControl fullWidth sx={{ mb: 2 }}>
                <InputLabel id="task-select-label">Select Task</InputLabel>
                <Select
                  labelId="task-select-label"
                  id="task-select"
                  value={selectedTask}
                  onChange={handleTaskChange}
                  label="Select Task"
                >
                  {taskHistory.map((task) => (
                    <MenuItem key={task.task_id} value={task.task_id}>
                      {task.task_id}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
              
              <Button
                variant="contained"
                color="primary"
                startIcon={evaluating ? <CircularProgress size={20} /> : <PlayArrowIcon />}
                onClick={handleEvaluate}
                disabled={evaluating || !selectedTask}
                fullWidth
              >
                {evaluating ? 'Evaluating...' : 'Run Evaluation'}
              </Button>
              
              {evaluationResults && (
                <Box sx={{ mt: 3 }}>
                  <Typography variant="subtitle2" gutterBottom>
                    Evaluation Results
                  </Typography>
                  <Paper variant="outlined" sx={{ p: 2 }}>
                    {Object.entries(evaluationResults.results).map(([key, value]) => (
                      <Box key={key} sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                        <Typography variant="body2">{key}:</Typography>
                        <Typography variant="body2" fontWeight="bold">
                          {typeof value === 'number' ? value.toFixed(4) : value}
                        </Typography>
                      </Box>
                    ))}
                  </Paper>
                </Box>
              )}
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} md={6}>
          <Card variant="outlined">
            <CardHeader title="Compute Forgetting" />
            <Divider />
            <CardContent>
              <FormControl fullWidth sx={{ mb: 2 }}>
                <InputLabel id="metric-select-label">Select Metric</InputLabel>
                <Select
                  labelId="metric-select-label"
                  id="metric-select"
                  value={selectedMetric}
                  onChange={handleMetricChange}
                  label="Select Metric"
                >
                  {metrics.map((metric) => (
                    <MenuItem key={metric.key} value={metric.key}>
                      {metric.name}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
              
              <Button
                variant="contained"
                color="secondary"
                startIcon={loading ? <CircularProgress size={20} /> : <BarChartIcon />}
                onClick={handleComputeForgetting}
                disabled={loading || !selectedTask}
                fullWidth
              >
                {loading ? 'Computing...' : 'Compute Forgetting'}
              </Button>
              
              {forgettingResults && (
                <Box sx={{ mt: 3 }}>
                  <Typography variant="subtitle2" gutterBottom>
                    Forgetting Results
                  </Typography>
                  <Paper variant="outlined" sx={{ p: 2 }}>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                      <Typography variant="body2">Forgetting:</Typography>
                      <Typography variant="body2" fontWeight="bold">
                        {forgettingResults.forgetting.toFixed(4)}
                      </Typography>
                    </Box>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                      <Typography variant="body2">Metric:</Typography>
                      <Typography variant="body2" fontWeight="bold">
                        {getMetricName(forgettingResults.metric_key)}
                      </Typography>
                    </Box>
                  </Paper>
                </Box>
              )}
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12}>
          <Card variant="outlined">
            <CardHeader title="Forgetting Visualization" />
            <Divider />
            <CardContent>
              {forgettingResults ? (
                <Box sx={{ height: 300 }}>
                  <canvas ref={chartRef} />
                </Box>
              ) : (
                <Alert severity="info">
                  Compute forgetting to see visualization.
                </Alert>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default Evaluation;
