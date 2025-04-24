import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Paper,
  Grid,
  TextField,
  Button,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  FormHelperText,
  Divider,
  Card,
  CardContent,
  CardHeader,
  CircularProgress,
  Alert,
  Chip,
  Tabs,
  Tab,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  useTheme
} from '@mui/material';
import {
  Assessment as AssessmentIcon,
  BarChart as BarChartIcon,
  Refresh as RefreshIcon,
  CompareArrows as CompareArrowsIcon
} from '@mui/icons-material';
import { useSnackbar } from 'notistack';
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
  RadialLinearScale,
  ArcElement
} from 'chart.js';
import { Line, Bar, Radar } from 'react-chartjs-2';
import { PerformanceChart, ForgettingChart, TaskRelationshipChart } from './visualizations';

import { evaluateAdapter, computeForgetting } from '../../../services/cl_peft_service';

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
  Tooltip,
  Legend
);

// Mock datasets for demonstration
const MOCK_DATASETS = [
  { id: 'dataset_1', name: 'Medical QA Dataset', description: 'Question-answering dataset for medical domain', size: '1.2 GB', examples: 50000 },
  { id: 'dataset_2', name: 'Clinical Notes', description: 'Clinical notes for medical summarization', size: '800 MB', examples: 30000 },
  { id: 'dataset_3', name: 'Medical Literature', description: 'Medical literature for knowledge extraction', size: '2.5 GB', examples: 100000 },
  { id: 'dataset_4', name: 'Patient Records', description: 'Anonymized patient records for information extraction', size: '1.5 GB', examples: 45000 },
  { id: 'dataset_5', name: 'Medical Terminology', description: 'Medical terminology dataset for entity recognition', size: '500 MB', examples: 20000 }
];

// Mock tasks for demonstration
const MOCK_TASKS = [
  { id: 'task_1', name: 'Medical QA', description: 'Question answering for medical domain' },
  { id: 'task_2', name: 'Clinical Summarization', description: 'Summarization of clinical notes' },
  { id: 'task_3', name: 'Entity Recognition', description: 'Medical entity recognition' },
  { id: 'task_4', name: 'Relation Extraction', description: 'Extract relations between medical entities' },
  { id: 'task_5', name: 'Text Classification', description: 'Classification of medical texts' }
];

function TabPanel(props) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`evaluation-tabpanel-${index}`}
      aria-labelledby={`evaluation-tab-${index}`}
      {...other}
    >
      {value === index && (
        <Box sx={{ p: 3 }}>
          {children}
        </Box>
      )}
    </div>
  );
}

function a11yProps(index) {
  return {
    id: `evaluation-tab-${index}`,
    'aria-controls': `evaluation-tabpanel-${index}`,
  };
}

const Evaluation = ({ adapter, onRefresh }) => {
  const theme = useTheme();
  const { enqueueSnackbar } = useSnackbar();

  const [tabValue, setTabValue] = useState(0);
  const [loading, setLoading] = useState(false);
  const [evaluationResults, setEvaluationResults] = useState(null);
  const [forgettingResults, setForgettingResults] = useState(null);

  const [formData, setFormData] = useState({
    task_id: '',
    dataset_id: '',
    metric_key: 'eval_loss'
  });

  const [errors, setErrors] = useState({});

  const handleTabChange = (event, newValue) => {
    setTabValue(newValue);
  };

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData({
      ...formData,
      [name]: value
    });

    // Clear error for this field
    if (errors[name]) {
      setErrors({
        ...errors,
        [name]: null
      });
    }
  };

  const validateForm = () => {
    const newErrors = {};

    if (!formData.task_id.trim()) {
      newErrors.task_id = 'Task ID is required';
    }

    if (!formData.dataset_id) {
      newErrors.dataset_id = 'Evaluation dataset is required';
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleEvaluate = async () => {
    if (!validateForm()) {
      return;
    }

    setLoading(true);
    try {
      const results = await evaluateAdapter(adapter.adapter_id, {
        task_id: formData.task_id,
        dataset_id: formData.dataset_id
      });

      // For demo purposes, generate mock results
      const mockResults = {
        adapter_id: adapter.adapter_id,
        task_id: formData.task_id,
        results: {
          eval_loss: Math.random() * 0.5 + 0.5,
          eval_accuracy: Math.random() * 0.3 + 0.7,
          eval_f1: Math.random() * 0.3 + 0.6,
          eval_precision: Math.random() * 0.3 + 0.6,
          eval_recall: Math.random() * 0.3 + 0.6,
          eval_runtime: Math.random() * 100 + 50,
          eval_samples_per_second: Math.random() * 50 + 10,
          eval_steps_per_second: Math.random() * 5 + 1
        }
      };

      setEvaluationResults(mockResults);
      enqueueSnackbar('Evaluation completed successfully', { variant: 'success' });
    } catch (error) {
      console.error('Error evaluating adapter:', error);
      enqueueSnackbar('Failed to evaluate adapter', { variant: 'error' });
    } finally {
      setLoading(false);
    }
  };

  const handleComputeForgetting = async () => {
    if (!validateForm()) {
      return;
    }

    setLoading(true);
    try {
      const results = await computeForgetting(adapter.adapter_id, {
        task_id: formData.task_id,
        dataset_id: formData.dataset_id,
        metric_key: formData.metric_key
      });

      // For demo purposes, generate mock results
      const mockResults = {
        adapter_id: adapter.adapter_id,
        task_id: formData.task_id,
        forgetting: Math.random() * 0.2,
        metric_key: formData.metric_key
      };

      setForgettingResults(mockResults);
      enqueueSnackbar('Forgetting computation completed successfully', { variant: 'success' });
    } catch (error) {
      console.error('Error computing forgetting:', error);
      enqueueSnackbar('Failed to compute forgetting', { variant: 'error' });
    } finally {
      setLoading(false);
    }
  };

  const formatMetric = (value) => {
    if (typeof value === 'number') {
      return value.toFixed(4);
    }
    return value;
  };

  if (!adapter) {
    return null;
  }

  // Generate chart data for evaluation results
  const getChartData = () => {
    if (!evaluationResults) return null;

    const metrics = Object.entries(evaluationResults.results)
      .filter(([key]) => !key.includes('runtime') && !key.includes('samples_per_second') && !key.includes('steps_per_second'))
      .map(([key, value]) => ({ key, value }));

    return {
      labels: metrics.map(m => m.key.replace('eval_', '')),
      datasets: [
        {
          label: 'Evaluation Metrics',
          data: metrics.map(m => m.value),
          backgroundColor: 'rgba(75, 192, 192, 0.2)',
          borderColor: 'rgba(75, 192, 192, 1)',
          borderWidth: 1
        }
      ]
    };
  };

  // Generate radar chart data for comparison
  const getRadarChartData = () => {
    if (!evaluationResults) return null;

    const metrics = Object.entries(evaluationResults.results)
      .filter(([key]) => !key.includes('runtime') && !key.includes('samples_per_second') && !key.includes('steps_per_second'))
      .map(([key, value]) => ({ key, value }));

    // Generate some mock data for comparison
    const mockComparisonData = metrics.map(m => Math.random() * 0.3 + 0.6);

    return {
      labels: metrics.map(m => m.key.replace('eval_', '')),
      datasets: [
        {
          label: 'Current Evaluation',
          data: metrics.map(m => m.value),
          backgroundColor: 'rgba(75, 192, 192, 0.2)',
          borderColor: 'rgba(75, 192, 192, 1)',
          borderWidth: 1
        },
        {
          label: 'Previous Evaluation',
          data: mockComparisonData,
          backgroundColor: 'rgba(255, 99, 132, 0.2)',
          borderColor: 'rgba(255, 99, 132, 1)',
          borderWidth: 1
        }
      ]
    };
  };

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h5" component="h2">
          Evaluate Adapter
        </Typography>
        <Button
          variant="outlined"
          size="small"
          startIcon={<RefreshIcon />}
          onClick={onRefresh}
        >
          Refresh
        </Button>
      </Box>

      <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 3 }}>
        <Tabs value={tabValue} onChange={handleTabChange} aria-label="evaluation tabs">
          <Tab label="Evaluate" icon={<AssessmentIcon />} iconPosition="start" {...a11yProps(0)} />
          <Tab label="Results" icon={<BarChartIcon />} iconPosition="start" {...a11yProps(1)} />
          <Tab label="Forgetting" icon={<CompareArrowsIcon />} iconPosition="start" {...a11yProps(2)} />
        </Tabs>
      </Box>

      <TabPanel value={tabValue} index={0}>
        <Card variant="outlined">
          <CardHeader title="Evaluation Configuration" />
          <Divider />
          <CardContent>
            <Grid container spacing={3}>
              <Grid item xs={12} md={6}>
                <FormControl fullWidth required error={!!errors.task_id}>
                  <InputLabel>Task</InputLabel>
                  <Select
                    name="task_id"
                    value={formData.task_id}
                    onChange={handleChange}
                    label="Task"
                  >
                    {adapter.task_history && adapter.task_history.length > 0 ? (
                      adapter.task_history.map((task) => (
                        <MenuItem key={task.task_id} value={task.task_id}>
                          {task.task_id}
                        </MenuItem>
                      ))
                    ) : (
                      <MenuItem value="">
                        <em>No tasks available</em>
                      </MenuItem>
                    )}
                    {MOCK_TASKS.map((task) => (
                      <MenuItem key={task.id} value={task.id}>
                        {task.name}
                      </MenuItem>
                    ))}
                  </Select>
                  <FormHelperText>
                    {errors.task_id || "Select the task to evaluate"}
                  </FormHelperText>
                </FormControl>
              </Grid>

              <Grid item xs={12} md={6}>
                <FormControl fullWidth required error={!!errors.dataset_id}>
                  <InputLabel>Evaluation Dataset</InputLabel>
                  <Select
                    name="dataset_id"
                    value={formData.dataset_id}
                    onChange={handleChange}
                    label="Evaluation Dataset"
                  >
                    {MOCK_DATASETS.map((dataset) => (
                      <MenuItem key={dataset.id} value={dataset.id}>
                        {dataset.name}
                      </MenuItem>
                    ))}
                  </Select>
                  <FormHelperText>
                    {errors.dataset_id || "Select the dataset to use for evaluation"}
                  </FormHelperText>
                </FormControl>
              </Grid>

              {formData.dataset_id && (
                <Grid item xs={12}>
                  <Paper variant="outlined" sx={{ p: 2 }}>
                    <Typography variant="subtitle2" gutterBottom>
                      Dataset Information
                    </Typography>
                    {MOCK_DATASETS.filter(d => d.id === formData.dataset_id).map((dataset) => (
                      <Box key={dataset.id}>
                        <Typography variant="body2" paragraph>
                          {dataset.description}
                        </Typography>
                        <Box sx={{ display: 'flex', gap: 2 }}>
                          <Chip
                            label={`${dataset.examples.toLocaleString()} examples`}
                            size="small"
                            variant="outlined"
                          />
                          <Chip
                            label={dataset.size}
                            size="small"
                            variant="outlined"
                          />
                        </Box>
                      </Box>
                    ))}
                  </Paper>
                </Grid>
              )}

              <Grid item xs={12}>
                <Box sx={{ display: 'flex', justifyContent: 'flex-end', mt: 2 }}>
                  <Button
                    variant="contained"
                    onClick={handleEvaluate}
                    disabled={loading}
                    startIcon={loading ? <CircularProgress size={20} /> : <AssessmentIcon />}
                  >
                    {loading ? 'Evaluating...' : 'Evaluate'}
                  </Button>
                </Box>
              </Grid>
            </Grid>
          </CardContent>
        </Card>
      </TabPanel>

      <TabPanel value={tabValue} index={1}>
        {evaluationResults ? (
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Card variant="outlined">
                <CardHeader title="Evaluation Results" />
                <Divider />
                <CardContent>
                  <TableContainer>
                    <Table size="small">
                      <TableHead>
                        <TableRow>
                          <TableCell>Metric</TableCell>
                          <TableCell align="right">Value</TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {Object.entries(evaluationResults.results).map(([key, value]) => (
                          <TableRow key={key}>
                            <TableCell component="th" scope="row">
                              {key}
                            </TableCell>
                            <TableCell align="right">{formatMetric(value)}</TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </TableContainer>
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12} md={6}>
              <PerformanceChart
                adapter={adapter}
                metrics={Object.keys(evaluationResults.results).filter(key =>
                  !key.includes('runtime') && !key.includes('samples_per_second') && !key.includes('steps_per_second')
                )}
                title="Evaluation Metrics"
              />
            </Grid>

            <Grid item xs={12}>
              <TaskRelationshipChart adapter={adapter} />
            </Grid>
          </Grid>
        ) : (
          <Alert severity="info">
            No evaluation results available. Use the Evaluate tab to evaluate the adapter.
          </Alert>
        )}
      </TabPanel>

      <TabPanel value={tabValue} index={2}>
        <Card variant="outlined" sx={{ mb: 3 }}>
          <CardHeader title="Compute Forgetting" />
          <Divider />
          <CardContent>
            <Grid container spacing={3}>
              <Grid item xs={12} md={4}>
                <FormControl fullWidth required error={!!errors.task_id}>
                  <InputLabel>Task</InputLabel>
                  <Select
                    name="task_id"
                    value={formData.task_id}
                    onChange={handleChange}
                    label="Task"
                  >
                    {adapter.task_history && adapter.task_history.length > 0 ? (
                      adapter.task_history.map((task) => (
                        <MenuItem key={task.task_id} value={task.task_id}>
                          {task.task_id}
                        </MenuItem>
                      ))
                    ) : (
                      <MenuItem value="">
                        <em>No tasks available</em>
                      </MenuItem>
                    )}
                    {MOCK_TASKS.map((task) => (
                      <MenuItem key={task.id} value={task.id}>
                        {task.name}
                      </MenuItem>
                    ))}
                  </Select>
                  <FormHelperText>
                    {errors.task_id || "Select the task to compute forgetting for"}
                  </FormHelperText>
                </FormControl>
              </Grid>

              <Grid item xs={12} md={4}>
                <FormControl fullWidth required error={!!errors.dataset_id}>
                  <InputLabel>Evaluation Dataset</InputLabel>
                  <Select
                    name="dataset_id"
                    value={formData.dataset_id}
                    onChange={handleChange}
                    label="Evaluation Dataset"
                  >
                    {MOCK_DATASETS.map((dataset) => (
                      <MenuItem key={dataset.id} value={dataset.id}>
                        {dataset.name}
                      </MenuItem>
                    ))}
                  </Select>
                  <FormHelperText>
                    {errors.dataset_id || "Select the dataset to use for evaluation"}
                  </FormHelperText>
                </FormControl>
              </Grid>

              <Grid item xs={12} md={4}>
                <FormControl fullWidth>
                  <InputLabel>Metric</InputLabel>
                  <Select
                    name="metric_key"
                    value={formData.metric_key}
                    onChange={handleChange}
                    label="Metric"
                  >
                    <MenuItem value="eval_loss">Loss</MenuItem>
                    <MenuItem value="eval_accuracy">Accuracy</MenuItem>
                    <MenuItem value="eval_f1">F1 Score</MenuItem>
                    <MenuItem value="eval_precision">Precision</MenuItem>
                    <MenuItem value="eval_recall">Recall</MenuItem>
                  </Select>
                  <FormHelperText>
                    Select the metric to use for forgetting calculation
                  </FormHelperText>
                </FormControl>
              </Grid>

              <Grid item xs={12}>
                <Box sx={{ display: 'flex', justifyContent: 'flex-end', mt: 2 }}>
                  <Button
                    variant="contained"
                    onClick={handleComputeForgetting}
                    disabled={loading}
                    startIcon={loading ? <CircularProgress size={20} /> : <CompareArrowsIcon />}
                  >
                    {loading ? 'Computing...' : 'Compute Forgetting'}
                  </Button>
                </Box>
              </Grid>
            </Grid>
          </CardContent>
        </Card>

        {forgettingResults ? (
          <Card variant="outlined">
            <CardHeader title="Forgetting Results" />
            <Divider />
            <CardContent>
              <Grid container spacing={3}>
                <Grid item xs={12} md={6}>
                  <Paper variant="outlined" sx={{ p: 3, textAlign: 'center' }}>
                    <Typography variant="h6" gutterBottom>
                      Forgetting Metric
                    </Typography>
                    <Typography variant="h3" color={theme.palette.primary.main}>
                      {forgettingResults.forgetting.toFixed(4)}
                    </Typography>
                    <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                      Metric: {forgettingResults.metric_key}
                    </Typography>
                  </Paper>
                </Grid>

                <Grid item xs={12} md={6}>
                  <Paper variant="outlined" sx={{ p: 3 }}>
                    <Typography variant="subtitle1" gutterBottom>
                      Interpretation
                    </Typography>
                    <Typography variant="body2" paragraph>
                      {forgettingResults.forgetting < 0.05 ? (
                        "The model shows minimal forgetting on this task, indicating good retention of previously learned knowledge."
                      ) : forgettingResults.forgetting < 0.15 ? (
                        "The model shows moderate forgetting on this task. This is typical for sequential learning without strong continual learning mechanisms."
                      ) : (
                        "The model shows significant forgetting on this task. Consider using stronger continual learning strategies or revisiting the task."
                      )}
                    </Typography>
                    <Alert severity={
                      forgettingResults.forgetting < 0.05 ? "success" :
                      forgettingResults.forgetting < 0.15 ? "warning" : "error"
                    }>
                      {forgettingResults.forgetting < 0.05 ? (
                        "Low forgetting - Good retention"
                      ) : forgettingResults.forgetting < 0.15 ? (
                        "Moderate forgetting - Acceptable for most applications"
                      ) : (
                        "High forgetting - Consider retraining or using different CL strategies"
                      )}
                    </Alert>
                  </Paper>
                </Grid>

                <Grid item xs={12}>
                  <ForgettingChart adapter={adapter} forgettingResults={forgettingResults} />
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        ) : (
          <Alert severity="info">
            No forgetting results available. Use the form above to compute forgetting.
          </Alert>
        )}
      </TabPanel>
    </Box>
  );
};

export default Evaluation;
