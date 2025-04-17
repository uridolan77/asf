import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Paper,
  Grid,
  Tabs,
  Tab,
  CircularProgress,
  Alert,
  Button,
  Divider,
  useTheme
} from '@mui/material';
import {
  Memory as MemoryIcon,
  BarChart as BarChartIcon,
  Psychology as PsychologyIcon,
  Settings as SettingsIcon,
  Add as AddIcon,
  Refresh as RefreshIcon
} from '@mui/icons-material';
import { SnackbarProvider, useSnackbar } from 'notistack';

// Import components
import AdaptersList from '../../components/LLM/cl_peft/AdaptersList';
import AdapterDetails from '../../components/LLM/cl_peft/AdapterDetails';
import TaskHistory from '../../components/LLM/cl_peft/TaskHistory';
import Evaluation from '../../components/LLM/cl_peft/Evaluation';
import TextGeneration from '../../components/LLM/cl_peft/TextGeneration';
import CreateAdapterDialog from '../../components/LLM/cl_peft/CreateAdapterDialog';
import { ForgettingChart } from '../../components/LLM/cl_peft/visualizations';

// Import services
import {
  getAdapters,
  getAdapter,
  createAdapter,
  deleteAdapter
} from '../../services/cl_peft_service';

// Mock data for development
const mockStrategies = [
  { id: 'ewc', name: 'Elastic Weight Consolidation (EWC)', description: 'Prevents forgetting by penalizing changes to important parameters' },
  { id: 'replay_experience', name: 'Experience Replay', description: 'Prevents forgetting by replaying examples from previous tasks' },
  { id: 'generative_replay', name: 'Generative Replay', description: 'Prevents forgetting by generating synthetic examples from previous tasks' },
  { id: 'orthogonal', name: 'Orthogonal Projection', description: 'Prevents forgetting by projecting gradients orthogonally to previous tasks' },
  { id: 'mask_based', name: 'Mask-Based CL', description: 'Prevents forgetting by using binary masks for each task' },
  { id: 'adaptive_svd', name: 'Adaptive SVD', description: 'Prevents forgetting by identifying important parameter subspaces' }
];

const mockPeftMethods = [
  { id: 'lora', name: 'LoRA', description: 'Low-Rank Adaptation of Large Language Models' },
  { id: 'qlora', name: 'QLoRA', description: 'Quantized Low-Rank Adaptation for efficient fine-tuning' },
  { id: 'adalora', name: 'AdaLoRA', description: 'Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning' },
  { id: 'ia3', name: 'IA³', description: 'Infused Adapter by Inhibiting and Amplifying Inner Activations' },
  { id: 'lisa', name: 'LISA', description: 'Learning with Integrated Soft Prompts and Adapters' }
];

const mockBaseModels = [
  { id: 'meta-llama/Llama-2-7b-hf', name: 'Llama 2 (7B)', provider: 'Meta' },
  { id: 'meta-llama/Llama-2-13b-hf', name: 'Llama 2 (13B)', provider: 'Meta' },
  { id: 'mistralai/Mistral-7B-v0.1', name: 'Mistral (7B)', provider: 'Mistral AI' },
  { id: 'microsoft/phi-2', name: 'Phi-2', provider: 'Microsoft' },
  { id: 'google/gemma-7b', name: 'Gemma (7B)', provider: 'Google' }
];

/**
 * TabPanel component for tab content
 */
function TabPanel(props) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`cl-peft-tabpanel-${index}`}
      aria-labelledby={`cl-peft-tab-${index}`}
      {...other}
      style={{ height: '100%' }}
    >
      {value === index && (
        <Box sx={{ p: 3, height: '100%' }}>
          {children}
        </Box>
      )}
    </div>
  );
}

/**
 * CL-PEFT Dashboard component for managing Continual Learning with Parameter-Efficient Fine-Tuning
 */
const CLPEFTDashboard = () => {
  const theme = useTheme();
  const { enqueueSnackbar } = useSnackbar();
  
  const [loading, setLoading] = useState(true);
  const [adapters, setAdapters] = useState([]);
  const [selectedAdapter, setSelectedAdapter] = useState(null);
  const [activeTab, setActiveTab] = useState(0);
  const [openCreateDialog, setOpenCreateDialog] = useState(false);
  
  // Load adapters on mount
  useEffect(() => {
    loadAdapters();
  }, []);
  
  // Load adapters
  const loadAdapters = async () => {
    try {
      setLoading(true);
      const result = await getAdapters();
      setAdapters(result.adapters || []);
      
      // Select first adapter if none selected
      if (!selectedAdapter && result.adapters && result.adapters.length > 0) {
        await handleSelectAdapter(result.adapters[0].adapter_id);
      }
    } catch (error) {
      console.error('Error loading adapters:', error);
      enqueueSnackbar('Failed to load adapters', { variant: 'error' });
    } finally {
      setLoading(false);
    }
  };
  
  // Handle adapter selection
  const handleSelectAdapter = async (adapterId) => {
    try {
      setLoading(true);
      
      // Get adapter details
      const result = await getAdapter(adapterId);
      setSelectedAdapter(result.adapter);
      
      // Switch to details tab
      setActiveTab(0);
    } catch (error) {
      console.error('Error selecting adapter:', error);
      enqueueSnackbar('Failed to load adapter details', { variant: 'error' });
    } finally {
      setLoading(false);
    }
  };
  
  // Handle adapter creation
  const handleCreateAdapter = async (adapterData) => {
    try {
      const result = await createAdapter(adapterData);
      
      // Add new adapter to list
      setAdapters([...adapters, result.adapter]);
      
      // Select new adapter
      setSelectedAdapter(result.adapter);
      
      // Close dialog
      setOpenCreateDialog(false);
      
      enqueueSnackbar('Adapter created successfully', { variant: 'success' });
    } catch (error) {
      console.error('Error creating adapter:', error);
      enqueueSnackbar('Failed to create adapter', { variant: 'error' });
    }
  };
  
  // Handle adapter deletion
  const handleDeleteAdapter = async (adapterId) => {
    try {
      await deleteAdapter(adapterId);
      
      // Remove adapter from list
      const updatedAdapters = adapters.filter(adapter => adapter.adapter_id !== adapterId);
      setAdapters(updatedAdapters);
      
      // Clear selected adapter if it was deleted
      if (selectedAdapter && selectedAdapter.adapter_id === adapterId) {
        setSelectedAdapter(updatedAdapters.length > 0 ? updatedAdapters[0] : null);
      }
      
      enqueueSnackbar('Adapter deleted successfully', { variant: 'success' });
    } catch (error) {
      console.error('Error deleting adapter:', error);
      enqueueSnackbar('Failed to delete adapter', { variant: 'error' });
    }
  };
  
  // Handle tab change
  const handleTabChange = (event, newValue) => {
    setActiveTab(newValue);
  };
  
  return (
    <Box sx={{ height: '100%' }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h5" component="h1">
          CL-PEFT Dashboard
        </Typography>
        <Box sx={{ display: 'flex', gap: 1 }}>
          <Button
            variant="outlined"
            startIcon={<RefreshIcon />}
            onClick={loadAdapters}
          >
            Refresh
          </Button>
          <Button
            variant="contained"
            startIcon={<AddIcon />}
            onClick={() => setOpenCreateDialog(true)}
          >
            Create Adapter
          </Button>
        </Box>
      </Box>
      
      <Grid container spacing={3} sx={{ height: 'calc(100% - 60px)' }}>
        {/* Adapters List */}
        <Grid item xs={12} md={3} sx={{ height: '100%' }}>
          <Paper variant="outlined" sx={{ height: '100%' }}>
            <Typography variant="h6" sx={{ p: 2, pb: 1 }}>
              Adapters
            </Typography>
            <Divider />
            {loading && adapters.length === 0 ? (
              <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
                <CircularProgress />
              </Box>
            ) : adapters.length === 0 ? (
              <Box sx={{ p: 2 }}>
                <Alert severity="info">
                  No adapters found. Create a new adapter to get started.
                </Alert>
              </Box>
            ) : (
              <AdaptersList
                adapters={adapters}
                selectedAdapter={selectedAdapter}
                onSelect={handleSelectAdapter}
                onDelete={handleDeleteAdapter}
              />
            )}
          </Paper>
        </Grid>
        
        {/* Main Content */}
        <Grid item xs={12} md={9} sx={{ height: '100%' }}>
          <Paper variant="outlined" sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
            {/* Tabs */}
            <Tabs
              value={activeTab}
              onChange={handleTabChange}
              aria-label="CL-PEFT tabs"
              variant="scrollable"
              scrollButtons="auto"
              sx={{ borderBottom: 1, borderColor: 'divider' }}
            >
              <Tab 
                icon={<MemoryIcon />} 
                label="Details" 
                id="cl-peft-tab-0" 
                aria-controls="cl-peft-tabpanel-0" 
                disabled={!selectedAdapter}
              />
              <Tab 
                icon={<BarChartIcon />} 
                label="Evaluation" 
                id="cl-peft-tab-1" 
                aria-controls="cl-peft-tabpanel-1" 
                disabled={!selectedAdapter}
              />
              <Tab 
                icon={<PsychologyIcon />} 
                label="Text Generation" 
                id="cl-peft-tab-2" 
                aria-controls="cl-peft-tabpanel-2" 
                disabled={!selectedAdapter}
              />
              <Tab 
                icon={<SettingsIcon />} 
                label="Settings" 
                id="cl-peft-tab-3" 
                aria-controls="cl-peft-tabpanel-3" 
              />
            </Tabs>
            
            {/* Tab Content */}
            <Box sx={{ flexGrow: 1, overflow: 'auto' }}>
              {!selectedAdapter && activeTab !== 3 ? (
                <Box sx={{ p: 3 }}>
                  <Alert severity="info">
                    Select an adapter from the list or create a new one to get started.
                  </Alert>
                </Box>
              ) : (
                <>
                  {/* Details Tab */}
                  <TabPanel value={activeTab} index={0}>
                    <Grid container spacing={3}>
                      <Grid item xs={12}>
                        <AdapterDetails adapter={selectedAdapter} />
                      </Grid>
                      <Grid item xs={12}>
                        <TaskHistory adapter={selectedAdapter} />
                      </Grid>
                    </Grid>
                  </TabPanel>
                  
                  {/* Evaluation Tab */}
                  <TabPanel value={activeTab} index={1}>
                    <Evaluation 
                      adapter={selectedAdapter} 
                      onEvaluationComplete={loadAdapters}
                    />
                  </TabPanel>
                  
                  {/* Text Generation Tab */}
                  <TabPanel value={activeTab} index={2}>
                    <TextGeneration 
                      adapter={selectedAdapter} 
                      onRefresh={loadAdapters}
                    />
                  </TabPanel>
                  
                  {/* Settings Tab */}
                  <TabPanel value={activeTab} index={3}>
                    <Grid container spacing={3}>
                      <Grid item xs={12} md={6}>
                        <Paper variant="outlined" sx={{ p: 2 }}>
                          <Typography variant="h6" gutterBottom>
                            CL Strategies
                          </Typography>
                          <Box sx={{ mt: 2 }}>
                            {mockStrategies.map((strategy) => (
                              <Box key={strategy.id} sx={{ mb: 2 }}>
                                <Typography variant="subtitle1">
                                  {strategy.name}
                                </Typography>
                                <Typography variant="body2" color="textSecondary">
                                  {strategy.description}
                                </Typography>
                              </Box>
                            ))}
                          </Box>
                        </Paper>
                      </Grid>
                      <Grid item xs={12} md={6}>
                        <Paper variant="outlined" sx={{ p: 2 }}>
                          <Typography variant="h6" gutterBottom>
                            PEFT Methods
                          </Typography>
                          <Box sx={{ mt: 2 }}>
                            {mockPeftMethods.map((method) => (
                              <Box key={method.id} sx={{ mb: 2 }}>
                                <Typography variant="subtitle1">
                                  {method.name}
                                </Typography>
                                <Typography variant="body2" color="textSecondary">
                                  {method.description}
                                </Typography>
                              </Box>
                            ))}
                          </Box>
                        </Paper>
                      </Grid>
                    </Grid>
                  </TabPanel>
                </>
              )}
            </Box>
          </Paper>
        </Grid>
      </Grid>
      
      {/* Create Adapter Dialog */}
      <CreateAdapterDialog
        open={openCreateDialog}
        onClose={() => setOpenCreateDialog(false)}
        onSubmit={handleCreateAdapter}
        clStrategies={mockStrategies}
        peftMethods={mockPeftMethods}
        baseModels={mockBaseModels}
      />
    </Box>
  );
};

// Wrap component with SnackbarProvider
const CLPEFTDashboardWithSnackbar = () => (
  <SnackbarProvider maxSnack={3}>
    <CLPEFTDashboard />
  </SnackbarProvider>
);

export default CLPEFTDashboardWithSnackbar;
