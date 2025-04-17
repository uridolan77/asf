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

import PageLayout from '../../components/Layout/PageLayout';
import AdaptersList from '../../components/LLM/cl_peft/AdaptersList.jsx';
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
  const handleAdapterSelect = async (adapter) => {
    try {
      // If we already have the full adapter object, use it directly
      if (adapter && adapter.adapter_id) {
        setSelectedAdapter(adapter);
        setActiveTab(0); // Reset to details tab
      } else {
        // If we only have the ID, fetch the full adapter
        const adapterId = typeof adapter === 'string' ? adapter : adapter?.adapter_id;
        if (adapterId) {
          const adapterData = await fetchAdapter(adapterId);
          setSelectedAdapter(adapterData);
          setActiveTab(0); // Reset to details tab
        }
      }
    } catch (error) {
      console.error(`Error selecting adapter:`, error);
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
    <Box sx={{ width: '100%' }}>
      <Box sx={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        mb: 3
      }}>
        <Typography variant="h4" component="h1">
          CL-PEFT Dashboard
        </Typography>
        <Box sx={{ display: 'flex', gap: 1 }}>
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
        </Box>
      </Box>
      {loading && !selectedAdapter ? (
        <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
          <CircularProgress />
        </Box>
      ) : (
        <Grid container spacing={3} sx={{ minHeight: 'calc(100vh - 200px)' }}>
          {/* Adapters List */}
          <Grid item xs={12} md={4} lg={3}>
            <Paper sx={{
              p: 0,
              height: '100%',
              overflow: 'hidden',
              display: 'flex',
              flexDirection: 'column',
              borderRadius: theme.shape.borderRadius,
              boxShadow: theme.shadows[2]
            }}>
              <Box sx={{ p: 2, borderBottom: `1px solid ${theme.palette.divider}` }}>
                <Typography variant="h6">
                  Adapters
                </Typography>
              </Box>
              <Box sx={{ flexGrow: 1, overflow: 'auto' }}>
                <AdaptersList
                  adapters={adapters}
                  selectedAdapter={selectedAdapter}
                  onSelect={handleAdapterSelect}
                  onDelete={handleDeleteAdapter}
                />
              </Box>
            </Paper>
          </Grid>

          {/* Adapter Details and Actions */}
          <Grid item xs={12} md={8} lg={9}>
            {selectedAdapter ? (
              <Paper sx={{
                p: 0,
                height: '100%',
                overflow: 'hidden',
                display: 'flex',
                flexDirection: 'column',
                borderRadius: theme.shape.borderRadius,
                boxShadow: theme.shadows[2]
              }}>
                <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
                  <Tabs
                    value={activeTab}
                    onChange={handleTabChange}
                    aria-label="adapter tabs"
                    variant="scrollable"
                    scrollButtons="auto"
                    sx={{
                      px: 2,
                      '& .MuiTab-root': {
                        minHeight: 48,
                        textTransform: 'none',
                        fontWeight: theme.typography.fontWeightMedium,
                      }
                    }}
                  >
                    <Tab label="Details" id="tab-0" aria-controls="tabpanel-0" />
                    <Tab label="Task History" id="tab-1" aria-controls="tabpanel-1" />
                    <Tab label="Training" id="tab-2" aria-controls="tabpanel-2" />
                    <Tab label="Evaluation" id="tab-3" aria-controls="tabpanel-3" />
                    <Tab label="Text Generation" id="tab-4" aria-controls="tabpanel-4" />
                  </Tabs>
                </Box>
                <Box sx={{ flexGrow: 1, overflow: 'auto', p: 3 }}>

                  {/* Details Tab */}
                  <Box role="tabpanel" hidden={activeTab !== 0} id="tabpanel-0" aria-labelledby="tab-0">
                    {activeTab === 0 && (
                      <AdapterDetails adapter={selectedAdapter} onRefresh={handleRefresh} />
                    )}
                  </Box>

                  {/* Task History Tab */}
                  <Box role="tabpanel" hidden={activeTab !== 1} id="tabpanel-1" aria-labelledby="tab-1">
                    {activeTab === 1 && (
                      <TaskHistory adapter={selectedAdapter} onRefresh={handleRefresh} />
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
                </Box>
              </Paper>
            ) : (
              <Paper sx={{
                p: 4,
                textAlign: 'center',
                height: '100%',
                display: 'flex',
                flexDirection: 'column',
                justifyContent: 'center',
                alignItems: 'center',
                borderRadius: theme.shape.borderRadius,
                boxShadow: theme.shadows[2]
              }}>
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
    </Box>
  );
};

export default CLPEFTDashboard;
