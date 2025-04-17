import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Paper,
  Grid,
  Button,
  CircularProgress,
  Tabs,
  Tab,
  Divider,
  useTheme
} from '@mui/material';
import { Add as AddIcon } from '@mui/icons-material';
import { useSnackbar } from 'notistack';
import { useNavigate } from 'react-router-dom';

import PageLayout from '../../components/layout/PageLayout';
import AdaptersList from '../../components/LLM/cl_peft/AdaptersList';
import AdapterDetails from '../../components/LLM/cl_peft/AdapterDetails';
import TaskHistory from '../../components/LLM/cl_peft/TaskHistory';
import Training from '../../components/LLM/cl_peft/Training';
import Evaluation from '../../components/LLM/cl_peft/Evaluation';
import TextGeneration from '../../components/LLM/cl_peft/TextGeneration';
import CreateAdapterDialog from '../../components/LLM/cl_peft/CreateAdapterDialog';
import { 
  fetchAdapters, 
  fetchAdapter, 
  createAdapter, 
  deleteAdapter,
  fetchClStrategies,
  fetchPeftMethods,
  fetchBaseModels
} from '../../services/cl_peft_service';

const CLPEFTDashboard = ({ status, onRefresh }) => {
  const theme = useTheme();
  const { enqueueSnackbar } = useSnackbar();
  const navigate = useNavigate();
  
  const [loading, setLoading] = useState(true);
  const [adapters, setAdapters] = useState([]);
  const [selectedAdapter, setSelectedAdapter] = useState(null);
  const [activeTab, setActiveTab] = useState(0);
  const [createDialogOpen, setCreateDialogOpen] = useState(false);
  const [clStrategies, setClStrategies] = useState([]);
  const [peftMethods, setPeftMethods] = useState([]);
  const [baseModels, setBaseModels] = useState([]);
  
  // Load adapters and metadata
  useEffect(() => {
    const loadData = async () => {
      setLoading(true);
      try {
        const [adaptersData, strategiesData, methodsData, modelsData] = await Promise.all([
          fetchAdapters(),
          fetchClStrategies(),
          fetchPeftMethods(),
          fetchBaseModels()
        ]);
        
        setAdapters(adaptersData);
        setClStrategies(strategiesData);
        setPeftMethods(methodsData);
        setBaseModels(modelsData);
        
        if (adaptersData.length > 0 && !selectedAdapter) {
          setSelectedAdapter(adaptersData[0]);
        }
      } catch (error) {
        console.error('Error loading CL-PEFT data:', error);
        enqueueSnackbar('Failed to load CL-PEFT data', { variant: 'error' });
      } finally {
        setLoading(false);
      }
    };
    
    loadData();
  }, [enqueueSnackbar, selectedAdapter]);
  
  // Handle adapter selection
  const handleAdapterSelect = async (adapterId) => {
    try {
      const adapter = await fetchAdapter(adapterId);
      setSelectedAdapter(adapter);
      setActiveTab(0); // Reset to details tab
    } catch (error) {
      console.error(`Error selecting adapter ${adapterId}:`, error);
      enqueueSnackbar(`Failed to load adapter details`, { variant: 'error' });
    }
  };
  
  // Handle tab change
  const handleTabChange = (event, newValue) => {
    setActiveTab(newValue);
  };
  
  // Handle create adapter
  const handleCreateAdapter = async (adapterData) => {
    try {
      setLoading(true);
      const newAdapter = await createAdapter(adapterData);
      
      setAdapters([...adapters, newAdapter]);
      setSelectedAdapter(newAdapter);
      setCreateDialogOpen(false);
      
      enqueueSnackbar('Adapter created successfully', { variant: 'success' });
    } catch (error) {
      console.error('Error creating adapter:', error);
      enqueueSnackbar('Failed to create adapter', { variant: 'error' });
    } finally {
      setLoading(false);
    }
  };
  
  // Handle delete adapter
  const handleDeleteAdapter = async (adapterId) => {
    try {
      setLoading(true);
      await deleteAdapter(adapterId);
      
      const updatedAdapters = adapters.filter(a => a.adapter_id !== adapterId);
      setAdapters(updatedAdapters);
      
      if (selectedAdapter && selectedAdapter.adapter_id === adapterId) {
        setSelectedAdapter(updatedAdapters.length > 0 ? updatedAdapters[0] : null);
      }
      
      enqueueSnackbar('Adapter deleted successfully', { variant: 'success' });
    } catch (error) {
      console.error(`Error deleting adapter ${adapterId}:`, error);
      enqueueSnackbar('Failed to delete adapter', { variant: 'error' });
    } finally {
      setLoading(false);
    }
  };
  
  // Handle refresh
  const handleRefresh = async () => {
    try {
      setLoading(true);
      const adaptersData = await fetchAdapters();
      setAdapters(adaptersData);
      
      if (selectedAdapter) {
        const refreshedAdapter = await fetchAdapter(selectedAdapter.adapter_id);
        setSelectedAdapter(refreshedAdapter);
      }
      
      enqueueSnackbar('Data refreshed successfully', { variant: 'success' });
      
      if (onRefresh) {
        onRefresh();
      }
    } catch (error) {
      console.error('Error refreshing data:', error);
      enqueueSnackbar('Failed to refresh data', { variant: 'error' });
    } finally {
      setLoading(false);
    }
  };
  
  return (
    <PageLayout
      title="CL-PEFT Dashboard"
      subtitle="Manage Continual Learning with Parameter-Efficient Fine-Tuning"
      icon="🧠"
      actions={
        <>
          <Button
            variant="contained"
            color="primary"
            startIcon={<AddIcon />}
            onClick={() => setCreateDialogOpen(true)}
            sx={{ mr: 1 }}
          >
            Create Adapter
          </Button>
          <Button
            variant="outlined"
            color="primary"
            onClick={handleRefresh}
            disabled={loading}
          >
            Refresh
          </Button>
        </>
      }
    >
      {loading && !selectedAdapter ? (
        <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
          <CircularProgress />
        </Box>
      ) : (
        <Grid container spacing={3}>
          {/* Adapters List */}
          <Grid item xs={12} md={4} lg={3}>
            <Paper sx={{ p: 2, height: '100%' }}>
              <Typography variant="h6" gutterBottom>
                Adapters
              </Typography>
              <AdaptersList
                adapters={adapters}
                selectedAdapter={selectedAdapter}
                onSelect={handleAdapterSelect}
                onDelete={handleDeleteAdapter}
              />
            </Paper>
          </Grid>
          
          {/* Adapter Details and Actions */}
          <Grid item xs={12} md={8} lg={9}>
            {selectedAdapter ? (
              <Paper sx={{ p: 2 }}>
                <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 2 }}>
                  <Tabs value={activeTab} onChange={handleTabChange} aria-label="adapter tabs">
                    <Tab label="Details" id="tab-0" aria-controls="tabpanel-0" />
                    <Tab label="Task History" id="tab-1" aria-controls="tabpanel-1" />
                    <Tab label="Training" id="tab-2" aria-controls="tabpanel-2" />
                    <Tab label="Evaluation" id="tab-3" aria-controls="tabpanel-3" />
                    <Tab label="Text Generation" id="tab-4" aria-controls="tabpanel-4" />
                  </Tabs>
                </Box>
                
                {/* Details Tab */}
                <Box role="tabpanel" hidden={activeTab !== 0} id="tabpanel-0" aria-labelledby="tab-0">
                  {activeTab === 0 && (
                    <AdapterDetails adapter={selectedAdapter} />
                  )}
                </Box>
                
                {/* Task History Tab */}
                <Box role="tabpanel" hidden={activeTab !== 1} id="tabpanel-1" aria-labelledby="tab-1">
                  {activeTab === 1 && (
                    <TaskHistory adapter={selectedAdapter} />
                  )}
                </Box>
                
                {/* Training Tab */}
                <Box role="tabpanel" hidden={activeTab !== 2} id="tabpanel-2" aria-labelledby="tab-2">
                  {activeTab === 2 && (
                    <Training
                      adapter={selectedAdapter}
                      onTrainingComplete={handleRefresh}
                    />
                  )}
                </Box>
                
                {/* Evaluation Tab */}
                <Box role="tabpanel" hidden={activeTab !== 3} id="tabpanel-3" aria-labelledby="tab-3">
                  {activeTab === 3 && (
                    <Evaluation
                      adapter={selectedAdapter}
                      onEvaluationComplete={handleRefresh}
                    />
                  )}
                </Box>
                
                {/* Text Generation Tab */}
                <Box role="tabpanel" hidden={activeTab !== 4} id="tabpanel-4" aria-labelledby="tab-4">
                  {activeTab === 4 && (
                    <TextGeneration adapter={selectedAdapter} />
                  )}
                </Box>
              </Paper>
            ) : (
              <Paper sx={{ p: 4, textAlign: 'center' }}>
                <Typography variant="h6" color="textSecondary" gutterBottom>
                  No adapter selected
                </Typography>
                <Typography variant="body1" color="textSecondary" paragraph>
                  Select an adapter from the list or create a new one to get started.
                </Typography>
                <Button
                  variant="contained"
                  color="primary"
                  startIcon={<AddIcon />}
                  onClick={() => setCreateDialogOpen(true)}
                  sx={{ mt: 2 }}
                >
                  Create Adapter
                </Button>
              </Paper>
            )}
          </Grid>
        </Grid>
      )}
      
      {/* Create Adapter Dialog */}
      <CreateAdapterDialog
        open={createDialogOpen}
        onClose={() => setCreateDialogOpen(false)}
        onSubmit={handleCreateAdapter}
        clStrategies={clStrategies}
        peftMethods={peftMethods}
        baseModels={baseModels}
      />
    </PageLayout>
  );
};

export default CLPEFTDashboard;
