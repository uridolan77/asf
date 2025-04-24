import React, { useState, useEffect } from 'react';
import {
  Box,
  Paper,
  Typography,
  Button,
  Grid,
  Card,
  CardHeader,
  CardContent,
  CardActions,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Alert,
  Chip,
  Divider,
  CircularProgress,
  IconButton,
  Tooltip,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Tabs,
  Tab,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Snackbar
} from '@mui/material';
import {
  Refresh as RefreshIcon,
  ExpandMore as ExpandMoreIcon,
  Psychology as PsychologyIcon,
  Add as AddIcon,
  Delete as DeleteIcon,
  Edit as EditIcon,
  PlayArrow as PlayArrowIcon,
  Code as CodeIcon,
  Settings as SettingsIcon,
  BugReport as BugReportIcon,
  AutoFixHigh as AutoFixHighIcon,
  LineWeight as LineWeightIcon
} from '@mui/icons-material';

import apiService from '../../services/api';
import { useNotification } from '../../context/NotificationContext';

/**
 * Dashboard for DSPy operations and module management
 */
const DSPyDashboard = ({ status, onRefresh }) => {
  const [loading, setLoading] = useState(false);
  const [modules, setModules] = useState([]);
  const [selectedModule, setSelectedModule] = useState(null);
  const [selectedModuleId, setSelectedModuleId] = useState('');
  const [input, setInput] = useState('');
  const [parameters, setParameters] = useState({});
  const [outputResult, setOutputResult] = useState(null);
  const [executing, setExecuting] = useState(false);
  const [activeTab, setActiveTab] = useState(0);
  const [optimizing, setOptimizing] = useState(false);
  const [optimizationConfig, setOptimizationConfig] = useState({
    metric: 'accuracy',
    max_iterations: 5,
    dataset_size: 10
  });
  const [configDialogOpen, setConfigDialogOpen] = useState(false);
  const [optimizationResult, setOptimizationResult] = useState(null);
  
  const { showSuccess, showError } = useNotification();
  
  // Load modules on mount
  useEffect(() => {
    if (status?.status === 'available') {
      loadModules();
    }
  }, [status]);
  
  // Load available DSPy modules
  const loadModules = async () => {
    setLoading(true);
    
    try {
      const result = await apiService.llm.getDspyModules();
      
      if (result.success) {
        setModules(result.data);
        showSuccess('DSPy modules loaded successfully');
        
        // Set first module as selected if none is selected
        if (result.data.length > 0 && !selectedModuleId) {
          setSelectedModuleId(result.data[0].id);
          setSelectedModule(result.data[0]);
          
          // Initialize parameters with default values
          if (result.data[0].parameters) {
            const defaultParams = {};
            result.data[0].parameters.forEach(param => {
              defaultParams[param.name] = param.default_value || '';
            });
            setParameters(defaultParams);
          }
        }
      } else {
        showError(`Failed to load DSPy modules: ${result.error}`);
      }
    } catch (error) {
      console.error('Error loading DSPy modules:', error);
      showError(`Error loading DSPy modules: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };
  
  // Handle module change
  const handleModuleChange = (event) => {
    const moduleId = event.target.value;
    setSelectedModuleId(moduleId);
    
    // Find the module by ID
    const module = modules.find(m => m.id === moduleId);
    setSelectedModule(module);
    
    // Reset input and output
    setInput('');
    setOutputResult(null);
    
    // Initialize parameters with default values
    if (module && module.parameters) {
      const defaultParams = {};
      module.parameters.forEach(param => {
        defaultParams[param.name] = param.default_value || '';
      });
      setParameters(defaultParams);
    } else {
      setParameters({});
    }
  };
  
  // Handle parameter change
  const handleParameterChange = (paramName, value) => {
    setParameters(prevParams => ({
      ...prevParams,
      [paramName]: value
    }));
  };
  
  // Handle input change
  const handleInputChange = (event) => {
    setInput(event.target.value);
  };
  
  // Handle tab change
  const handleTabChange = (_, newValue) => {
    setActiveTab(newValue);
  };
  
  // Handle optimization config change
  const handleOptimizationConfigChange = (field, value) => {
    setOptimizationConfig(prev => ({
      ...prev,
      [field]: value
    }));
  };
  
  // Execute DSPy module
  const executeModule = async () => {
    if (!selectedModule || !input.trim()) {
      showError('Please select a module and enter input text');
      return;
    }
    
    setExecuting(true);
    setOutputResult(null);
    
    try {
      const result = await apiService.llm.executeDspyModule({
        module_id: selectedModuleId,
        input: input,
        parameters: parameters
      });
      
      if (result.success) {
        setOutputResult(result.data);
        showSuccess('DSPy module executed successfully');
      } else {
        showError(`Failed to execute DSPy module: ${result.error}`);
      }
    } catch (error) {
      console.error('Error executing DSPy module:', error);
      showError(`Error executing DSPy module: ${error.message}`);
    } finally {
      setExecuting(false);
    }
  };
  
  // Optimize DSPy module
  const optimizeModule = async () => {
    if (!selectedModule) {
      showError('Please select a module to optimize');
      return;
    }
    
    setOptimizing(true);
    setOptimizationResult(null);
    
    try {
      const result = await apiService.llm.optimizeDspyModule({
        module_id: selectedModuleId,
        config: optimizationConfig
      });
      
      if (result.success) {
        setOptimizationResult(result.data);
        showSuccess('DSPy module optimized successfully');
      } else {
        showError(`Failed to optimize DSPy module: ${result.error}`);
      }
    } catch (error) {
      console.error('Error optimizing DSPy module:', error);
      showError(`Error optimizing DSPy module: ${error.message}`);
    } finally {
      setOptimizing(false);
      setConfigDialogOpen(false);
    }
  };
  
  // Get parameter field based on type
  const getParameterField = (param) => {
    const value = parameters[param.name] || '';
    
    if (param.type === 'boolean') {
      return (
        <FormControl fullWidth key={param.name} sx={{ mb: 2 }}>
          <InputLabel>{param.display_name || param.name}</InputLabel>
          <Select
            value={value.toString()}
            label={param.display_name || param.name}
            onChange={(e) => handleParameterChange(param.name, e.target.value === 'true')}
          >
            <MenuItem value="true">True</MenuItem>
            <MenuItem value="false">False</MenuItem>
          </Select>
          {param.description && (
            <Typography variant="caption" color="text.secondary" sx={{ mt: 0.5 }}>
              {param.description}
            </Typography>
          )}
        </FormControl>
      );
    } else if (param.type === 'select' && param.options) {
      return (
        <FormControl fullWidth key={param.name} sx={{ mb: 2 }}>
          <InputLabel>{param.display_name || param.name}</InputLabel>
          <Select
            value={value}
            label={param.display_name || param.name}
            onChange={(e) => handleParameterChange(param.name, e.target.value)}
          >
            {param.options.map(option => (
              <MenuItem key={option.value} value={option.value}>
                {option.label}
              </MenuItem>
            ))}
          </Select>
          {param.description && (
            <Typography variant="caption" color="text.secondary" sx={{ mt: 0.5 }}>
              {param.description}
            </Typography>
          )}
        </FormControl>
      );
    } else if (param.type === 'number') {
      return (
        <TextField
          key={param.name}
          fullWidth
          label={param.display_name || param.name}
          type="number"
          value={value}
          onChange={(e) => handleParameterChange(param.name, Number(e.target.value))}
          sx={{ mb: 2 }}
          helperText={param.description}
          InputProps={{
            inputProps: { 
              min: param.min, 
              max: param.max, 
              step: param.step || 1 
            }
          }}
        />
      );
    } else {
      return (
        <TextField
          key={param.name}
          fullWidth
          label={param.display_name || param.name}
          value={value}
          onChange={(e) => handleParameterChange(param.name, e.target.value)}
          sx={{ mb: 2 }}
          helperText={param.description}
          multiline={param.multiline}
          rows={param.multiline ? 3 : 1}
        />
      );
    }
  };
  
  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h5">
          <PsychologyIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
          DSPy Dashboard
        </Typography>
        <Box>
          <Button 
            variant="outlined" 
            startIcon={loading ? <CircularProgress size={20} /> : <RefreshIcon />}
            onClick={loadModules}
            disabled={loading}
          >
            Refresh
          </Button>
        </Box>
      </Box>
      
      {status?.status !== 'available' && (
        <Alert severity="error" sx={{ mb: 3 }}>
          DSPy service is currently unavailable. Please check the server status.
        </Alert>
      )}
      
      <Tabs 
        value={activeTab} 
        onChange={handleTabChange} 
        aria-label="DSPy tabs"
        sx={{ mb: 2, borderBottom: 1, borderColor: 'divider' }}
      >
        <Tab label="Execute" id="tab-0" aria-controls="tabpanel-0" />
        <Tab label="Optimize" id="tab-1" aria-controls="tabpanel-1" />
        <Tab label="Modules" id="tab-2" aria-controls="tabpanel-2" />
      </Tabs>
      
      {/* Execute Module Panel */}
      <Box role="tabpanel" hidden={activeTab !== 0} id="tabpanel-0" aria-labelledby="tab-0">
        {activeTab === 0 && (
          <Grid container spacing={3}>
            <Grid item xs={12}>
              <Paper sx={{ p: 2 }}>
                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                  <FormControl fullWidth>
                    <InputLabel id="module-select-label">DSPy Module</InputLabel>
                    <Select
                      labelId="module-select-label"
                      id="module-select"
                      value={selectedModuleId}
                      label="DSPy Module"
                      onChange={handleModuleChange}
                      disabled={loading || executing || modules.length === 0}
                    >
                      {modules.map((module) => (
                        <MenuItem key={module.id} value={module.id}>
                          {module.name}
                        </MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                  
                  {selectedModule && (
                    <>
                      <Typography variant="body2" color="text.secondary" gutterBottom>
                        {selectedModule.description}
                      </Typography>
                      
                      {selectedModule.tags && selectedModule.tags.length > 0 && (
                        <Box sx={{ mb: 2 }}>
                          {selectedModule.tags.map(tag => (
                            <Chip 
                              key={tag} 
                              label={tag} 
                              size="small" 
                              sx={{ mr: 0.5, mb: 0.5 }}
                            />
                          ))}
                        </Box>
                      )}
                      
                      {selectedModule.parameters && selectedModule.parameters.length > 0 && (
                        <Accordion>
                          <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                            <Typography>Parameters</Typography>
                          </AccordionSummary>
                          <AccordionDetails>
                            {selectedModule.parameters.map(param => getParameterField(param))}
                          </AccordionDetails>
                        </Accordion>
                      )}
                      
                      <TextField
                        label="Input"
                        multiline
                        rows={4}
                        fullWidth
                        variant="outlined"
                        value={input}
                        onChange={handleInputChange}
                        placeholder={selectedModule.input_placeholder || "Enter your input here..."}
                        disabled={executing}
                      />
                      
                      <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                        <Button
                          variant="contained"
                          startIcon={executing ? <CircularProgress size={20} /> : <PlayArrowIcon />}
                          onClick={executeModule}
                          disabled={loading || executing || !selectedModule || !input.trim()}
                        >
                          {executing ? 'Executing...' : 'Execute'}
                        </Button>
                        
                        <Button
                          variant="outlined"
                          onClick={() => {
                            setInput('');
                            setOutputResult(null);
                          }}
                          disabled={executing}
                        >
                          Clear
                        </Button>
                      </Box>
                      
                      <Divider sx={{ my: 2 }} />
                      
                      <Typography variant="h6" gutterBottom>
                        Output
                      </Typography>
                      
                      {executing ? (
                        <Box sx={{ display: 'flex', justifyContent: 'center', my: 3 }}>
                          <CircularProgress />
                        </Box>
                      ) : outputResult ? (
                        <Box>
                          <Paper
                            variant="outlined"
                            sx={{
                              p: 2,
                              mb: 2,
                              backgroundColor: 'grey.50',
                              maxHeight: '300px',
                              overflowY: 'auto',
                              whiteSpace: 'pre-wrap'
                            }}
                          >
                            {outputResult.output}
                          </Paper>
                          
                          {outputResult.metadata && (
                            <Accordion>
                              <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                                <Typography>Metadata and Trace</Typography>
                              </AccordionSummary>
                              <AccordionDetails>
                                <Typography variant="subtitle2" gutterBottom>
                                  Execution Time:
                                </Typography>
                                <Typography variant="body2" paragraph>
                                  {outputResult.metadata.execution_time}ms
                                </Typography>
                                
                                <Typography variant="subtitle2" gutterBottom>
                                  Token Usage:
                                </Typography>
                                <Typography variant="body2" paragraph>
                                  Input: {outputResult.metadata.token_usage?.input_tokens || 0} | 
                                  Output: {outputResult.metadata.token_usage?.output_tokens || 0} | 
                                  Total: {outputResult.metadata.token_usage?.total_tokens || 0}
                                </Typography>
                                
                                {outputResult.trace && (
                                  <>
                                    <Typography variant="subtitle2" gutterBottom>
                                      Execution Trace:
                                    </Typography>
                                    <Paper
                                      variant="outlined"
                                      sx={{
                                        p: 2,
                                        backgroundColor: 'grey.900',
                                        color: 'grey.100',
                                        maxHeight: '200px',
                                        overflowY: 'auto',
                                        fontFamily: 'monospace',
                                        fontSize: '0.85rem',
                                        whiteSpace: 'pre-wrap'
                                      }}
                                    >
                                      {outputResult.trace}
                                    </Paper>
                                  </>
                                )}
                              </AccordionDetails>
                            </Accordion>
                          )}
                        </Box>
                      ) : (
                        <Alert severity="info">
                          Output will appear here after execution
                        </Alert>
                      )}
                    </>
                  )}
                </Box>
              </Paper>
            </Grid>
          </Grid>
        )}
      </Box>
      
      {/* Optimize Module Panel */}
      <Box role="tabpanel" hidden={activeTab !== 1} id="tabpanel-1" aria-labelledby="tab-1">
        {activeTab === 1 && (
          <Grid container spacing={3}>
            <Grid item xs={12}>
              <Paper sx={{ p: 2 }}>
                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                  <FormControl fullWidth>
                    <InputLabel id="optimize-module-select-label">Select Module to Optimize</InputLabel>
                    <Select
                      labelId="optimize-module-select-label"
                      id="optimize-module-select"
                      value={selectedModuleId}
                      label="Select Module to Optimize"
                      onChange={handleModuleChange}
                      disabled={loading || optimizing || modules.length === 0}
                    >
                      {modules.map((module) => (
                        <MenuItem key={module.id} value={module.id}>
                          {module.name}
                        </MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                  
                  {selectedModule && (
                    <>
                      <Alert severity="info" sx={{ mb: 2 }}>
                        Optimization uses DSPy's Teleprompter to improve the module's performance by automatically 
                        refining prompts based on example inputs and outputs.
                      </Alert>
                      
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
                        <Button
                          variant="contained"
                          startIcon={<SettingsIcon />}
                          onClick={() => setConfigDialogOpen(true)}
                          disabled={optimizing}
                        >
                          Configure Optimization
                        </Button>
                        
                        <Button
                          variant="contained"
                          color="primary"
                          startIcon={optimizing ? <CircularProgress size={20} /> : <AutoFixHighIcon />}
                          onClick={optimizeModule}
                          disabled={loading || optimizing || !selectedModule}
                        >
                          {optimizing ? 'Optimizing...' : 'Optimize Module'}
                        </Button>
                      </Box>
                      
                      <Divider sx={{ my: 2 }} />
                      
                      <Typography variant="h6" gutterBottom>
                        Optimization Results
                      </Typography>
                      
                      {optimizing ? (
                        <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', my: 3 }}>
                          <CircularProgress sx={{ mb: 2 }} />
                          <Typography variant="body2" color="text.secondary">
                            Optimization can take several minutes depending on the complexity...
                          </Typography>
                        </Box>
                      ) : optimizationResult ? (
                        <Box>
                          <Alert 
                            severity="success" 
                            sx={{ mb: 2 }}
                          >
                            Module optimization complete. Performance improved from {optimizationResult.initial_metric}% to {optimizationResult.final_metric}% on {optimizationConfig.metric}.
                          </Alert>
                          
                          <Accordion>
                            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                              <Typography>Optimization Details</Typography>
                            </AccordionSummary>
                            <AccordionDetails>
                              <Typography variant="subtitle2" gutterBottom>
                                Iterations Completed:
                              </Typography>
                              <Typography variant="body2" paragraph>
                                {optimizationResult.iterations_completed} of {optimizationConfig.max_iterations}
                              </Typography>
                              
                              <Typography variant="subtitle2" gutterBottom>
                                Total Optimization Time:
                              </Typography>
                              <Typography variant="body2" paragraph>
                                {optimizationResult.total_time}s
                              </Typography>
                              
                              <Typography variant="subtitle2" gutterBottom>
                                Improvement By Iteration:
                              </Typography>
                              <Box sx={{ pl: 2 }}>
                                {optimizationResult.iteration_metrics.map((metric, idx) => (
                                  <Typography key={idx} variant="body2">
                                    Iteration {idx + 1}: {metric}% {idx === optimizationResult.best_iteration ? "(Best)" : ""}
                                  </Typography>
                                ))}
                              </Box>
                            </AccordionDetails>
                          </Accordion>
                          
                          <Accordion>
                            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                              <Typography>Prompt Changes</Typography>
                            </AccordionSummary>
                            <AccordionDetails>
                              <Typography variant="subtitle2" gutterBottom>
                                Original Prompt:
                              </Typography>
                              <Paper
                                variant="outlined"
                                sx={{
                                  p: 2,
                                  mb: 2,
                                  backgroundColor: 'grey.50',
                                  maxHeight: '200px',
                                  overflowY: 'auto',
                                  fontFamily: 'monospace',
                                  fontSize: '0.85rem',
                                  whiteSpace: 'pre-wrap'
                                }}
                              >
                                {optimizationResult.original_prompt}
                              </Paper>
                              
                              <Typography variant="subtitle2" gutterBottom>
                                Optimized Prompt:
                              </Typography>
                              <Paper
                                variant="outlined"
                                sx={{
                                  p: 2,
                                  backgroundColor: 'grey.50',
                                  maxHeight: '200px',
                                  overflowY: 'auto',
                                  fontFamily: 'monospace',
                                  fontSize: '0.85rem',
                                  whiteSpace: 'pre-wrap'
                                }}
                              >
                                {optimizationResult.optimized_prompt}
                              </Paper>
                            </AccordionDetails>
                          </Accordion>
                        </Box>
                      ) : (
                        <Alert severity="info">
                          Optimization results will appear here after running the optimization process
                        </Alert>
                      )}
                    </>
                  )}
                </Box>
              </Paper>
            </Grid>
          </Grid>
        )}
      </Box>
      
      {/* Modules List Panel */}
      <Box role="tabpanel" hidden={activeTab !== 2} id="tabpanel-2" aria-labelledby="tab-2">
        {activeTab === 2 && (
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Available DSPy Modules
            </Typography>
            
            {loading ? (
              <Box sx={{ display: 'flex', justifyContent: 'center', my: 3 }}>
                <CircularProgress />
              </Box>
            ) : modules.length === 0 ? (
              <Alert severity="info">No DSPy modules available</Alert>
            ) : (
              <Grid container spacing={2}>
                {modules.map((module) => (
                  <Grid item xs={12} md={6} key={module.id}>
                    <Card variant="outlined">
                      <CardHeader
                        title={module.name}
                        subheader={`Type: ${module.module_type || 'Unknown'}`}
                        avatar={
                          <PsychologyIcon color={module.optimized ? 'success' : 'action'} />
                        }
                        action={
                          module.optimized && (
                            <Tooltip title="This module has been optimized">
                              <Chip 
                                label="Optimized" 
                                color="success"
                                size="small"
                                icon={<AutoFixHighIcon />}
                              />
                            </Tooltip>
                          )
                        }
                      />
                      <CardContent>
                        <Typography variant="body2" color="text.secondary" gutterBottom>
                          {module.description || 'No description available'}
                        </Typography>
                        
                        {module.tags && module.tags.length > 0 && (
                          <Box sx={{ mt: 1 }}>
                            {module.tags.map(tag => (
                              <Chip 
                                key={tag} 
                                label={tag} 
                                size="small" 
                                sx={{ mr: 0.5, mb: 0.5 }}
                              />
                            ))}
                          </Box>
                        )}
                        
                        {module.metrics && (
                          <Box sx={{ mt: 2 }}>
                            <Typography variant="caption" color="text.secondary">
                              Performance Metrics:
                            </Typography>
                            <List dense>
                              {Object.entries(module.metrics).map(([key, value]) => (
                                <ListItem key={key} sx={{ py: 0 }}>
                                  <ListItemText
                                    primary={`${key}: ${typeof value === 'number' ? value.toFixed(2) : value}`}
                                  />
                                </ListItem>
                              ))}
                            </List>
                          </Box>
                        )}
                      </CardContent>
                      <CardActions>
                        <Button 
                          size="small" 
                          startIcon={<PlayArrowIcon />}
                          onClick={() => {
                            setSelectedModuleId(module.id);
                            setSelectedModule(module);
                            setActiveTab(0);
                          }}
                        >
                          Use Module
                        </Button>
                        <Button 
                          size="small"
                          startIcon={<AutoFixHighIcon />}
                          onClick={() => {
                            setSelectedModuleId(module.id);
                            setSelectedModule(module);
                            setActiveTab(1);
                          }}
                        >
                          Optimize
                        </Button>
                      </CardActions>
                    </Card>
                  </Grid>
                ))}
              </Grid>
            )}
          </Paper>
        )}
      </Box>
      
      {/* Optimization Config Dialog */}
      <Dialog
        open={configDialogOpen}
        onClose={() => setConfigDialogOpen(false)}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>Optimization Configuration</DialogTitle>
        <DialogContent dividers>
          <FormControl fullWidth sx={{ mb: 2 }}>
            <InputLabel id="metric-select-label">Optimization Metric</InputLabel>
            <Select
              labelId="metric-select-label"
              id="metric-select"
              value={optimizationConfig.metric}
              label="Optimization Metric"
              onChange={(e) => handleOptimizationConfigChange('metric', e.target.value)}
            >
              <MenuItem value="accuracy">Accuracy</MenuItem>
              <MenuItem value="f1">F1 Score</MenuItem>
              <MenuItem value="precision">Precision</MenuItem>
              <MenuItem value="recall">Recall</MenuItem>
              <MenuItem value="rouge">ROUGE</MenuItem>
              <MenuItem value="bleu">BLEU</MenuItem>
            </Select>
          </FormControl>
          
          <TextField
            fullWidth
            label="Maximum Iterations"
            type="number"
            value={optimizationConfig.max_iterations}
            onChange={(e) => handleOptimizationConfigChange('max_iterations', parseInt(e.target.value))}
            sx={{ mb: 2 }}
            InputProps={{ inputProps: { min: 1, max: 20 } }}
            helperText="More iterations may improve results but take longer (1-20)"
          />
          
          <TextField
            fullWidth
            label="Dataset Size"
            type="number"
            value={optimizationConfig.dataset_size}
            onChange={(e) => handleOptimizationConfigChange('dataset_size', parseInt(e.target.value))}
            sx={{ mb: 2 }}
            InputProps={{ inputProps: { min: 5, max: 100 } }}
            helperText="Number of examples to use for optimization (5-100)"
          />
          
          <Alert severity="warning" sx={{ mt: 2 }}>
            Optimization can take several minutes to complete depending on the configuration settings.
            Higher values for iterations and dataset size will result in longer processing times.
          </Alert>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setConfigDialogOpen(false)}>Cancel</Button>
          <Button 
            variant="contained" 
            onClick={optimizeModule}
            disabled={optimizing}
          >
            Start Optimization
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default DSPyDashboard;
