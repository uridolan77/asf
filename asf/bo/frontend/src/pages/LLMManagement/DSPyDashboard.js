import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Paper,
  Grid,
  Card,
  CardHeader,
  CardContent,
  CardActions,
  Button,
  Chip,
  Divider,
  TextField,
  CircularProgress,
  Alert,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow
} from '@mui/material';
import {
  ExpandMore as ExpandMoreIcon,
  Settings as SettingsIcon,
  PlayArrow as PlayArrowIcon,
  Add as AddIcon,
  Delete as DeleteIcon,
  Tune as TuneIcon
} from '@mui/icons-material';

import { useNotification } from '../../context/NotificationContext';
import apiService from '../../services/api';
import { ContentLoader } from '../../components/UI/LoadingIndicators';

/**
 * DSPy Dashboard component
 */
const DSPyDashboard = ({ status, onRefresh }) => {
  const { showSuccess, showError } = useNotification();
  
  const [modules, setModules] = useState([]);
  const [loading, setLoading] = useState(true);
  const [executingModule, setExecutingModule] = useState(null);
  const [executionResults, setExecutionResults] = useState(null);
  
  // Load modules on mount
  useEffect(() => {
    loadModules();
  }, []);
  
  // Load modules
  const loadModules = async () => {
    setLoading(true);
    
    try {
      const result = await apiService.llm.getDspyModules();
      
      if (result.success) {
        setModules(result.data);
      } else {
        showError(`Failed to load modules: ${result.error}`);
      }
    } catch (error) {
      console.error('Error loading modules:', error);
      showError(`Error loading modules: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };
  
  // Handle module execution
  const handleExecuteModule = async (moduleName) => {
    setExecutingModule(moduleName);
    setExecutionResults(null);
    
    try {
      const result = await apiService.llm.executeDspyModule(moduleName, {
        question: "What are the common side effects of statins?"
      });
      
      if (result.success) {
        setExecutionResults(result.data);
        showSuccess(`Module ${moduleName} executed successfully`);
      } else {
        showError(`Module ${moduleName} execution failed: ${result.error}`);
      }
    } catch (error) {
      console.error(`Error executing module ${moduleName}:`, error);
      showError(`Error executing module ${moduleName}: ${error.message}`);
    } finally {
      setExecutingModule(null);
    }
  };
  
  // Handle module optimization
  const handleOptimizeModule = (moduleName) => {
    showSuccess(`Module optimization for ${moduleName} is not yet implemented`);
  };
  
  // Render module cards
  const renderModuleCards = () => {
    return (
      <Grid container spacing={3}>
        {modules.map((module) => (
          <Grid item xs={12} md={6} key={module.name}>
            <Card 
              sx={{ 
                height: '100%', 
                display: 'flex', 
                flexDirection: 'column',
                transition: 'all 0.3s ease',
                '&:hover': {
                  boxShadow: 6,
                  transform: 'translateY(-4px)'
                }
              }}
            >
              <CardHeader
                title={module.name}
                subheader={`Usage: ${module.usage_count} times`}
              />
              <CardContent sx={{ flexGrow: 1 }}>
                <Typography variant="body2" color="text.secondary" gutterBottom>
                  {module.description || "No description available"}
                </Typography>
                
                <Typography variant="subtitle2" gutterBottom sx={{ mt: 2 }}>
                  Signature:
                </Typography>
                <Typography variant="body2" sx={{ fontFamily: 'monospace', bgcolor: 'grey.100', p: 1, borderRadius: 1 }}>
                  {module.signature || "No signature available"}
                </Typography>
                
                {module.parameters && Object.keys(module.parameters).length > 0 && (
                  <>
                    <Typography variant="subtitle2" gutterBottom sx={{ mt: 2 }}>
                      Parameters:
                    </Typography>
                    <TableContainer component={Paper} variant="outlined" sx={{ mt: 1 }}>
                      <Table size="small">
                        <TableHead>
                          <TableRow>
                            <TableCell>Name</TableCell>
                            <TableCell>Value</TableCell>
                          </TableRow>
                        </TableHead>
                        <TableBody>
                          {Object.entries(module.parameters).map(([key, value]) => (
                            <TableRow key={key}>
                              <TableCell>{key}</TableCell>
                              <TableCell>{JSON.stringify(value)}</TableCell>
                            </TableRow>
                          ))}
                        </TableBody>
                      </Table>
                    </TableContainer>
                  </>
                )}
                
                {executionResults && executionResults.module_name === module.name && (
                  <Box sx={{ mt: 2 }}>
                    <Typography variant="subtitle2" gutterBottom>
                      Execution Results:
                    </Typography>
                    <Paper variant="outlined" sx={{ p: 1.5, bgcolor: 'success.light', color: 'white' }}>
                      <Typography variant="body2">
                        {executionResults.outputs.answer || JSON.stringify(executionResults.outputs)}
                      </Typography>
                      <Typography variant="caption" sx={{ mt: 1, display: 'block' }}>
                        Execution time: {executionResults.execution_time_ms.toFixed(2)}ms
                      </Typography>
                    </Paper>
                  </Box>
                )}
              </CardContent>
              <CardActions>
                <Button
                  size="small"
                  startIcon={executingModule === module.name ? <CircularProgress size={16} /> : <PlayArrowIcon />}
                  onClick={() => handleExecuteModule(module.name)}
                  disabled={executingModule === module.name}
                >
                  Execute
                </Button>
                <Button
                  size="small"
                  startIcon={<TuneIcon />}
                  onClick={() => handleOptimizeModule(module.name)}
                >
                  Optimize
                </Button>
                <Button
                  size="small"
                  startIcon={<DeleteIcon />}
                  color="error"
                >
                  Delete
                </Button>
              </CardActions>
            </Card>
          </Grid>
        ))}
        
        {/* Add new module card */}
        <Grid item xs={12} md={6}>
          <Card 
            sx={{ 
              height: '100%', 
              display: 'flex', 
              flexDirection: 'column',
              justifyContent: 'center',
              alignItems: 'center',
              p: 3,
              bgcolor: 'grey.100',
              border: '2px dashed',
              borderColor: 'grey.300',
              cursor: 'pointer',
              transition: 'all 0.3s ease',
              '&:hover': {
                borderColor: 'primary.main',
                bgcolor: 'grey.200'
              }
            }}
          >
            <AddIcon sx={{ fontSize: 48, color: 'grey.500', mb: 2 }} />
            <Typography variant="h6" color="text.secondary">
              Register New Module
            </Typography>
          </Card>
        </Grid>
      </Grid>
    );
  };
  
  if (loading) {
    return <ContentLoader height={200} message="Loading DSPy modules..." />;
  }
  
  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        DSPy Module Management
      </Typography>
      
      <Typography paragraph>
        Manage DSPy modules for advanced LLM programming. DSPy provides a framework for building
        complex LLM applications with optimized prompts.
      </Typography>
      
      {/* Module cards */}
      {modules.length > 0 ? (
        renderModuleCards()
      ) : (
        <Alert severity="info" sx={{ mb: 3 }}>
          No DSPy modules found. Register a new module to get started.
        </Alert>
      )}
      
      {/* DSPy configuration */}
      <Accordion sx={{ mt: 4 }}>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography variant="subtitle1">DSPy Configuration</Typography>
        </AccordionSummary>
        <AccordionDetails>
          <Typography paragraph>
            Configure global DSPy settings.
          </Typography>
          
          <Typography variant="subtitle2" gutterBottom>
            Coming soon...
          </Typography>
        </AccordionDetails>
      </Accordion>
    </Box>
  );
};

export default DSPyDashboard;
