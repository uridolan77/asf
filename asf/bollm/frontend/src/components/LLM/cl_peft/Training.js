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
  Stepper,
  Step,
  StepLabel,
  CircularProgress,
  Alert,
  Chip,
  Autocomplete,
  Slider,
  Switch,
  FormControlLabel,
  useTheme
} from '@mui/material';
import {
  Send as SendIcon,
  Save as SaveIcon,
  Dataset as DatasetIcon,
  Settings as SettingsIcon,
  PlayArrow as PlayArrowIcon,
  Stop as StopIcon
} from '@mui/icons-material';
import { useSnackbar } from 'notistack';

import { trainAdapter } from '../../../services/cl_peft_service';

// Mock datasets for demonstration
const MOCK_DATASETS = [
  { id: 'dataset_1', name: 'Medical QA Dataset', description: 'Question-answering dataset for medical domain', size: '1.2 GB', examples: 50000 },
  { id: 'dataset_2', name: 'Clinical Notes', description: 'Clinical notes for medical summarization', size: '800 MB', examples: 30000 },
  { id: 'dataset_3', name: 'Medical Literature', description: 'Medical literature for knowledge extraction', size: '2.5 GB', examples: 100000 },
  { id: 'dataset_4', name: 'Patient Records', description: 'Anonymized patient records for information extraction', size: '1.5 GB', examples: 45000 },
  { id: 'dataset_5', name: 'Medical Terminology', description: 'Medical terminology dataset for entity recognition', size: '500 MB', examples: 20000 }
];

const Training = ({ adapter, onRefresh }) => {
  const theme = useTheme();
  const { enqueueSnackbar } = useSnackbar();
  
  const [activeStep, setActiveStep] = useState(0);
  const [loading, setLoading] = useState(false);
  const [trainingStarted, setTrainingStarted] = useState(false);
  const [trainingProgress, setTrainingProgress] = useState(0);
  const [trainingLogs, setTrainingLogs] = useState([]);
  
  const [formData, setFormData] = useState({
    task_id: '',
    dataset_id: '',
    eval_dataset_id: '',
    num_train_epochs: 3,
    per_device_train_batch_size: 4,
    learning_rate: 5e-5,
    weight_decay: 0.01,
    max_grad_norm: 1.0,
    warmup_steps: 500,
    logging_steps: 10,
    save_steps: 500,
    evaluation_strategy: 'epoch',
    advanced_settings_enabled: false
  });
  
  const [errors, setErrors] = useState({});
  
  // Simulate training progress
  useEffect(() => {
    let interval;
    if (trainingStarted && trainingProgress < 100) {
      interval = setInterval(() => {
        setTrainingProgress(prev => {
          const newProgress = prev + 1;
          if (newProgress >= 100) {
            clearInterval(interval);
            setTrainingStarted(false);
            enqueueSnackbar('Training completed successfully', { variant: 'success' });
            onRefresh();
          }
          return newProgress;
        });
        
        // Add a log message every ~10%
        if (trainingProgress % 10 === 0) {
          const epoch = Math.floor(trainingProgress / (100 / formData.num_train_epochs));
          const step = Math.floor((trainingProgress % (100 / formData.num_train_epochs)) * 10);
          setTrainingLogs(prev => [
            ...prev,
            {
              timestamp: new Date().toISOString(),
              message: `Epoch ${epoch}/${formData.num_train_epochs}, Step ${step}: loss=${(1.5 - trainingProgress/100).toFixed(4)}, lr=${(formData.learning_rate * (1 - trainingProgress/100)).toExponential(2)}`
            }
          ]);
        }
      }, 300);
    }
    return () => clearInterval(interval);
  }, [trainingStarted, trainingProgress, formData.num_train_epochs, formData.learning_rate, enqueueSnackbar, onRefresh]);
  
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
  
  const handleSliderChange = (name) => (event, newValue) => {
    setFormData({
      ...formData,
      [name]: newValue
    });
  };
  
  const handleAdvancedSettingsToggle = () => {
    setFormData({
      ...formData,
      advanced_settings_enabled: !formData.advanced_settings_enabled
    });
  };
  
  const validateStep = (step) => {
    const newErrors = {};
    
    if (step === 0) {
      if (!formData.task_id.trim()) {
        newErrors.task_id = 'Task ID is required';
      }
      
      if (!formData.dataset_id) {
        newErrors.dataset_id = 'Training dataset is required';
      }
    }
    
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };
  
  const handleNext = () => {
    if (validateStep(activeStep)) {
      setActiveStep((prevStep) => prevStep + 1);
    }
  };
  
  const handleBack = () => {
    setActiveStep((prevStep) => prevStep - 1);
  };
  
  const handleStartTraining = async () => {
    if (!validateStep(activeStep)) {
      return;
    }
    
    setLoading(true);
    try {
      await trainAdapter(adapter.adapter_id, formData);
      setTrainingStarted(true);
      setTrainingProgress(0);
      setTrainingLogs([
        {
          timestamp: new Date().toISOString(),
          message: `Starting training for task ${formData.task_id} with dataset ${formData.dataset_id}`
        }
      ]);
      enqueueSnackbar('Training started successfully', { variant: 'success' });
    } catch (error) {
      console.error('Error starting training:', error);
      enqueueSnackbar('Failed to start training', { variant: 'error' });
      setTrainingStarted(false);
    } finally {
      setLoading(false);
    }
  };
  
  const handleStopTraining = () => {
    setTrainingStarted(false);
    enqueueSnackbar('Training stopped', { variant: 'warning' });
  };
  
  const steps = ['Task Configuration', 'Training Parameters', 'Review & Start'];
  
  if (!adapter) {
    return null;
  }
  
  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h5" component="h2">
          Train Adapter
        </Typography>
      </Box>
      
      {trainingStarted ? (
        <Box>
          <Alert severity="info" sx={{ mb: 3 }}>
            Training in progress. Please do not navigate away from this page.
          </Alert>
          
          <Card variant="outlined" sx={{ mb: 3 }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Training Progress
              </Typography>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <Box sx={{ width: '100%', mr: 1 }}>
                  <LinearProgressWithLabel value={trainingProgress} />
                </Box>
                <Box sx={{ minWidth: 35 }}>
                  <Typography variant="body2" color="text.secondary">
                    {`${Math.round(trainingProgress)}%`}
                  </Typography>
                </Box>
              </Box>
              <Button
                variant="contained"
                color="error"
                startIcon={<StopIcon />}
                onClick={handleStopTraining}
              >
                Stop Training
              </Button>
            </CardContent>
          </Card>
          
          <Card variant="outlined">
            <CardHeader title="Training Logs" />
            <Divider />
            <CardContent sx={{ maxHeight: 300, overflow: 'auto' }}>
              {trainingLogs.map((log, index) => (
                <Box key={index} sx={{ mb: 1 }}>
                  <Typography variant="caption" color="text.secondary">
                    {new Date(log.timestamp).toLocaleTimeString()}
                  </Typography>
                  <Typography variant="body2">
                    {log.message}
                  </Typography>
                </Box>
              ))}
            </CardContent>
          </Card>
        </Box>
      ) : (
        <Box>
          <Stepper activeStep={activeStep} sx={{ mb: 4 }}>
            {steps.map((label) => (
              <Step key={label}>
                <StepLabel>{label}</StepLabel>
              </Step>
            ))}
          </Stepper>
          
          {activeStep === 0 && (
            <Card variant="outlined">
              <CardHeader title="Task Configuration" />
              <Divider />
              <CardContent>
                <Grid container spacing={3}>
                  <Grid item xs={12}>
                    <TextField
                      name="task_id"
                      label="Task ID"
                      fullWidth
                      required
                      value={formData.task_id}
                      onChange={handleChange}
                      error={!!errors.task_id}
                      helperText={errors.task_id || "A unique identifier for this training task"}
                    />
                  </Grid>
                  
                  <Grid item xs={12}>
                    <FormControl fullWidth required error={!!errors.dataset_id}>
                      <InputLabel>Training Dataset</InputLabel>
                      <Select
                        name="dataset_id"
                        value={formData.dataset_id}
                        onChange={handleChange}
                        label="Training Dataset"
                      >
                        {MOCK_DATASETS.map((dataset) => (
                          <MenuItem key={dataset.id} value={dataset.id}>
                            {dataset.name}
                          </MenuItem>
                        ))}
                      </Select>
                      <FormHelperText>
                        {errors.dataset_id || "Select the dataset to use for training"}
                      </FormHelperText>
                    </FormControl>
                  </Grid>
                  
                  <Grid item xs={12}>
                    <FormControl fullWidth>
                      <InputLabel>Evaluation Dataset (Optional)</InputLabel>
                      <Select
                        name="eval_dataset_id"
                        value={formData.eval_dataset_id}
                        onChange={handleChange}
                        label="Evaluation Dataset (Optional)"
                      >
                        <MenuItem value="">
                          <em>None</em>
                        </MenuItem>
                        {MOCK_DATASETS.map((dataset) => (
                          <MenuItem key={dataset.id} value={dataset.id}>
                            {dataset.name}
                          </MenuItem>
                        ))}
                      </Select>
                      <FormHelperText>
                        Select a dataset for evaluation during training (optional)
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
                </Grid>
              </CardContent>
            </Card>
          )}
          
          {activeStep === 1 && (
            <Card variant="outlined">
              <CardHeader 
                title="Training Parameters" 
                action={
                  <FormControlLabel
                    control={
                      <Switch
                        checked={formData.advanced_settings_enabled}
                        onChange={handleAdvancedSettingsToggle}
                      />
                    }
                    label="Advanced Settings"
                  />
                }
              />
              <Divider />
              <CardContent>
                <Grid container spacing={3}>
                  <Grid item xs={12} md={6}>
                    <Typography gutterBottom>
                      Number of Training Epochs: {formData.num_train_epochs}
                    </Typography>
                    <Slider
                      value={formData.num_train_epochs}
                      onChange={handleSliderChange('num_train_epochs')}
                      step={1}
                      marks
                      min={1}
                      max={10}
                      valueLabelDisplay="auto"
                    />
                  </Grid>
                  
                  <Grid item xs={12} md={6}>
                    <Typography gutterBottom>
                      Learning Rate: {formData.learning_rate.toExponential(2)}
                    </Typography>
                    <Slider
                      value={Math.log10(formData.learning_rate) + 5}
                      onChange={(e, value) => {
                        setFormData({
                          ...formData,
                          learning_rate: Math.pow(10, value - 5)
                        });
                      }}
                      step={0.1}
                      marks={[
                        { value: 0, label: '1e-5' },
                        { value: 1, label: '1e-4' },
                        { value: 2, label: '1e-3' }
                      ]}
                      min={0}
                      max={2}
                      valueLabelDisplay="auto"
                      valueLabelFormat={(value) => Math.pow(10, value - 5).toExponential(2)}
                    />
                  </Grid>
                  
                  <Grid item xs={12} md={6}>
                    <Typography gutterBottom>
                      Batch Size: {formData.per_device_train_batch_size}
                    </Typography>
                    <Slider
                      value={formData.per_device_train_batch_size}
                      onChange={handleSliderChange('per_device_train_batch_size')}
                      step={1}
                      marks={[
                        { value: 1, label: '1' },
                        { value: 4, label: '4' },
                        { value: 8, label: '8' },
                        { value: 16, label: '16' }
                      ]}
                      min={1}
                      max={16}
                      valueLabelDisplay="auto"
                    />
                  </Grid>
                  
                  {formData.advanced_settings_enabled && (
                    <>
                      <Grid item xs={12}>
                        <Divider>Advanced Settings</Divider>
                      </Grid>
                      
                      <Grid item xs={12} md={6}>
                        <Typography gutterBottom>
                          Weight Decay: {formData.weight_decay}
                        </Typography>
                        <Slider
                          value={formData.weight_decay}
                          onChange={handleSliderChange('weight_decay')}
                          step={0.01}
                          min={0}
                          max={0.1}
                          valueLabelDisplay="auto"
                        />
                      </Grid>
                      
                      <Grid item xs={12} md={6}>
                        <Typography gutterBottom>
                          Max Gradient Norm: {formData.max_grad_norm}
                        </Typography>
                        <Slider
                          value={formData.max_grad_norm}
                          onChange={handleSliderChange('max_grad_norm')}
                          step={0.1}
                          min={0.1}
                          max={5}
                          valueLabelDisplay="auto"
                        />
                      </Grid>
                      
                      <Grid item xs={12} md={6}>
                        <TextField
                          name="warmup_steps"
                          label="Warmup Steps"
                          type="number"
                          fullWidth
                          value={formData.warmup_steps}
                          onChange={handleChange}
                          InputProps={{ inputProps: { min: 0 } }}
                        />
                      </Grid>
                      
                      <Grid item xs={12} md={6}>
                        <TextField
                          name="logging_steps"
                          label="Logging Steps"
                          type="number"
                          fullWidth
                          value={formData.logging_steps}
                          onChange={handleChange}
                          InputProps={{ inputProps: { min: 1 } }}
                        />
                      </Grid>
                      
                      <Grid item xs={12} md={6}>
                        <TextField
                          name="save_steps"
                          label="Save Steps"
                          type="number"
                          fullWidth
                          value={formData.save_steps}
                          onChange={handleChange}
                          InputProps={{ inputProps: { min: 1 } }}
                        />
                      </Grid>
                      
                      <Grid item xs={12} md={6}>
                        <FormControl fullWidth>
                          <InputLabel>Evaluation Strategy</InputLabel>
                          <Select
                            name="evaluation_strategy"
                            value={formData.evaluation_strategy}
                            onChange={handleChange}
                            label="Evaluation Strategy"
                          >
                            <MenuItem value="no">No</MenuItem>
                            <MenuItem value="steps">Steps</MenuItem>
                            <MenuItem value="epoch">Epoch</MenuItem>
                          </Select>
                        </FormControl>
                      </Grid>
                    </>
                  )}
                </Grid>
              </CardContent>
            </Card>
          )}
          
          {activeStep === 2 && (
            <Card variant="outlined">
              <CardHeader title="Review & Start Training" />
              <Divider />
              <CardContent>
                <Grid container spacing={3}>
                  <Grid item xs={12} md={6}>
                    <Typography variant="subtitle1" gutterBottom>
                      Task Configuration
                    </Typography>
                    <Paper variant="outlined" sx={{ p: 2 }}>
                      <Box sx={{ mb: 2 }}>
                        <Typography variant="subtitle2">Task ID</Typography>
                        <Typography variant="body2">{formData.task_id}</Typography>
                      </Box>
                      
                      <Box sx={{ mb: 2 }}>
                        <Typography variant="subtitle2">Training Dataset</Typography>
                        <Typography variant="body2">
                          {MOCK_DATASETS.find(d => d.id === formData.dataset_id)?.name || 'None'}
                        </Typography>
                      </Box>
                      
                      <Box>
                        <Typography variant="subtitle2">Evaluation Dataset</Typography>
                        <Typography variant="body2">
                          {formData.eval_dataset_id 
                            ? MOCK_DATASETS.find(d => d.id === formData.eval_dataset_id)?.name 
                            : 'None'}
                        </Typography>
                      </Box>
                    </Paper>
                  </Grid>
                  
                  <Grid item xs={12} md={6}>
                    <Typography variant="subtitle1" gutterBottom>
                      Training Parameters
                    </Typography>
                    <Paper variant="outlined" sx={{ p: 2 }}>
                      <Grid container spacing={2}>
                        <Grid item xs={6}>
                          <Typography variant="subtitle2">Epochs</Typography>
                          <Typography variant="body2">{formData.num_train_epochs}</Typography>
                        </Grid>
                        
                        <Grid item xs={6}>
                          <Typography variant="subtitle2">Batch Size</Typography>
                          <Typography variant="body2">{formData.per_device_train_batch_size}</Typography>
                        </Grid>
                        
                        <Grid item xs={6}>
                          <Typography variant="subtitle2">Learning Rate</Typography>
                          <Typography variant="body2">{formData.learning_rate.toExponential(2)}</Typography>
                        </Grid>
                        
                        <Grid item xs={6}>
                          <Typography variant="subtitle2">Weight Decay</Typography>
                          <Typography variant="body2">{formData.weight_decay}</Typography>
                        </Grid>
                        
                        {formData.advanced_settings_enabled && (
                          <>
                            <Grid item xs={6}>
                              <Typography variant="subtitle2">Max Gradient Norm</Typography>
                              <Typography variant="body2">{formData.max_grad_norm}</Typography>
                            </Grid>
                            
                            <Grid item xs={6}>
                              <Typography variant="subtitle2">Warmup Steps</Typography>
                              <Typography variant="body2">{formData.warmup_steps}</Typography>
                            </Grid>
                            
                            <Grid item xs={6}>
                              <Typography variant="subtitle2">Logging Steps</Typography>
                              <Typography variant="body2">{formData.logging_steps}</Typography>
                            </Grid>
                            
                            <Grid item xs={6}>
                              <Typography variant="subtitle2">Save Steps</Typography>
                              <Typography variant="body2">{formData.save_steps}</Typography>
                            </Grid>
                            
                            <Grid item xs={6}>
                              <Typography variant="subtitle2">Evaluation Strategy</Typography>
                              <Typography variant="body2">{formData.evaluation_strategy}</Typography>
                            </Grid>
                          </>
                        )}
                      </Grid>
                    </Paper>
                  </Grid>
                  
                  <Grid item xs={12}>
                    <Alert severity="info">
                      Training this adapter will take approximately {formData.num_train_epochs * 5} minutes. 
                      You can navigate away from this page and the training will continue in the background.
                    </Alert>
                  </Grid>
                </Grid>
              </CardContent>
            </Card>
          )}
          
          <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 3 }}>
            <Button
              disabled={activeStep === 0}
              onClick={handleBack}
            >
              Back
            </Button>
            <Box>
              {activeStep === steps.length - 1 ? (
                <Button
                  variant="contained"
                  color="primary"
                  onClick={handleStartTraining}
                  disabled={loading}
                  startIcon={loading ? <CircularProgress size={20} /> : <PlayArrowIcon />}
                >
                  {loading ? 'Starting...' : 'Start Training'}
                </Button>
              ) : (
                <Button
                  variant="contained"
                  onClick={handleNext}
                >
                  Next
                </Button>
              )}
            </Box>
          </Box>
        </Box>
      )}
    </Box>
  );
};

// Helper component for the progress bar
function LinearProgressWithLabel(props) {
  return (
    <Box sx={{ display: 'flex', alignItems: 'center', width: '100%' }}>
      <Box sx={{ width: '100%', mr: 1 }}>
        <LinearProgress variant="determinate" {...props} />
      </Box>
    </Box>
  );
}

// Custom LinearProgress component
function LinearProgress(props) {
  const theme = useTheme();
  
  return (
    <Box
      sx={{
        height: 10,
        borderRadius: 5,
        bgcolor: theme.palette.grey[theme.palette.mode === 'light' ? 200 : 800],
        position: 'relative',
        overflow: 'hidden'
      }}
    >
      <Box
        sx={{
          position: 'absolute',
          left: 0,
          top: 0,
          bottom: 0,
          width: `${props.value}%`,
          bgcolor: theme.palette.primary.main,
          transition: 'width 0.3s ease'
        }}
      />
    </Box>
  );
}

export default Training;
