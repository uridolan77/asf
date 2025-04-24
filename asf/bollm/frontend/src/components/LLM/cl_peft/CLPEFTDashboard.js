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

import PageLayout from '../../layout/PageLayout';
import AdaptersList from './AdaptersList';
import AdapterDetails from './AdapterDetails';
import TaskHistory from './TaskHistory';
import Training from './Training';
import Evaluation from './Evaluation';
import TextGeneration from './TextGeneration';
import CreateAdapterDialog from './CreateAdapterDialog';
import { fetchAdapters } from '../../../services/cl_peft_service';

function TabPanel(props) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`cl-peft-tabpanel-${index}`}
      aria-labelledby={`cl-peft-tab-${index}`}
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
    id: `cl-peft-tab-${index}`,
    'aria-controls': `cl-peft-tabpanel-${index}`,
  };
}

const CLPEFTDashboard = () => {
  const theme = useTheme();
  const { enqueueSnackbar } = useSnackbar();
  const navigate = useNavigate();

  const [loading, setLoading] = useState(true);
  const [adapters, setAdapters] = useState([]);
  const [selectedAdapter, setSelectedAdapter] = useState(null);
  const [createDialogOpen, setCreateDialogOpen] = useState(false);
  const [tabValue, setTabValue] = useState(0);

  useEffect(() => {
    loadAdapters();
  }, []);

  const loadAdapters = async () => {
    setLoading(true);
    try {
      const data = await fetchAdapters();
      setAdapters(data);
      if (data.length > 0 && !selectedAdapter) {
        setSelectedAdapter(data[0]);
      }
    } catch (error) {
      console.error('Error loading adapters:', error);
      enqueueSnackbar('Failed to load adapters', { variant: 'error' });
    } finally {
      setLoading(false);
    }
  };

  const handleTabChange = (event, newValue) => {
    setTabValue(newValue);
  };

  const handleAdapterSelect = (adapter) => {
    setSelectedAdapter(adapter);
    setTabValue(0); // Reset to details tab
  };

  const handleCreateAdapter = () => {
    setCreateDialogOpen(true);
  };

  const handleCreateDialogClose = () => {
    setCreateDialogOpen(false);
  };

  const handleAdapterCreated = () => {
    loadAdapters();
    setCreateDialogOpen(false);
    enqueueSnackbar('Adapter created successfully', { variant: 'success' });
  };

  return (
    <PageLayout title="CL-PEFT Dashboard" subtitle="Manage Continual Learning with Parameter-Efficient Fine-Tuning">
      <Box sx={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
        <Box sx={{ display: 'flex', justifyContent: 'flex-end', mb: 2 }}>
          <Button
            variant="contained"
            color="primary"
            startIcon={<AddIcon />}
            onClick={handleCreateAdapter}
          >
            Create Adapter
          </Button>
        </Box>

        <Grid container spacing={3} sx={{ flexGrow: 1 }}>
          <Grid item xs={12} md={4} lg={3}>
            <Paper
              elevation={2}
              sx={{
                height: '100%',
                overflow: 'auto',
                borderRadius: theme.shape.borderRadius,
              }}
            >
              <Typography variant="h6" sx={{ p: 2, borderBottom: `1px solid ${theme.palette.divider}` }}>
                Adapters
              </Typography>
              {loading ? (
                <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
                  <CircularProgress />
                </Box>
              ) : (
                <AdaptersList
                  adapters={adapters}
                  selectedAdapter={selectedAdapter}
                  onAdapterSelect={handleAdapterSelect}
                />
              )}
            </Paper>
          </Grid>

          <Grid item xs={12} md={8} lg={9}>
            <Paper
              elevation={2}
              sx={{
                height: '100%',
                display: 'flex',
                flexDirection: 'column',
                borderRadius: theme.shape.borderRadius,
              }}
            >
              {selectedAdapter ? (
                <>
                  <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
                    <Tabs
                      value={tabValue}
                      onChange={handleTabChange}
                      aria-label="adapter tabs"
                      sx={{ px: 2 }}
                    >
                      <Tab label="Details" {...a11yProps(0)} />
                      <Tab label="Tasks" {...a11yProps(1)} />
                      <Tab label="Training" {...a11yProps(2)} />
                      <Tab label="Evaluation" {...a11yProps(3)} />
                      <Tab label="Generation" {...a11yProps(4)} />
                    </Tabs>
                  </Box>

                  <Box sx={{ flexGrow: 1, overflow: 'auto' }}>
                    <TabPanel value={tabValue} index={0}>
                      <AdapterDetails adapter={selectedAdapter} onRefresh={loadAdapters} />
                    </TabPanel>
                    <TabPanel value={tabValue} index={1}>
                      <TaskHistory adapter={selectedAdapter} onRefresh={loadAdapters} />
                    </TabPanel>
                    <TabPanel value={tabValue} index={2}>
                      <Training adapter={selectedAdapter} onRefresh={loadAdapters} />
                    </TabPanel>
                    <TabPanel value={tabValue} index={3}>
                      <Evaluation adapter={selectedAdapter} onRefresh={loadAdapters} />
                    </TabPanel>
                    <TabPanel value={tabValue} index={4}>
                      <TextGeneration adapter={selectedAdapter} onRefresh={loadAdapters} />
                    </TabPanel>
                  </Box>
                </>
              ) : (
                <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}>
                  <Typography variant="body1" color="textSecondary">
                    {loading ? 'Loading adapters...' : 'No adapter selected. Please select an adapter from the list or create a new one.'}
                  </Typography>
                </Box>
              )}
            </Paper>
          </Grid>
        </Grid>
      </Box>

      <CreateAdapterDialog
        open={createDialogOpen}
        onClose={handleCreateDialogClose}
        onAdapterCreated={handleAdapterCreated}
      />
    </PageLayout>
  );
};

export default CLPEFTDashboard;
