import React, { useState, useEffect } from 'react';
import { Container, Typography, Box, Paper, Tabs, Tab, Grid, CircularProgress, Alert } from '@mui/material';
import { useNavigate, useParams } from 'react-router-dom';
import ClientService from '../services/ClientService';
import DSPyClientList from '../components/dspy/DSPyClientList';
import DSPyClientDetails from '../components/dspy/DSPyClientDetails';
import DSPyModuleList from '../components/dspy/DSPyModuleList';
import DSPyModuleDetails from '../components/dspy/DSPyModuleDetails';
import DSPyAuditLogViewer from '../components/dspy/DSPyAuditLogViewer';
import DSPyCircuitBreakers from '../components/dspy/DSPyCircuitBreakers';
import { TabPanel, a11yProps } from '../components/common/TabPanel';

const DSPyDashboard: React.FC = () => {
  const navigate = useNavigate();
  const { clientId, moduleId } = useParams<{ clientId?: string; moduleId?: string }>();
  const [currentTab, setCurrentTab] = useState<number>(0);
  const [clients, setClients] = useState<any[]>([]);
  const [selectedClient, setSelectedClient] = useState<any>(null);
  const [selectedModule, setSelectedModule] = useState<any>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadClients();
  }, []);

  useEffect(() => {
    if (clientId && clients.length > 0) {
      const client = clients.find(c => c.client_id === clientId);
      if (client) {
        setSelectedClient(client);
        // If moduleId is provided, load the module
        if (moduleId) {
          loadModule(moduleId);
          setCurrentTab(2); // Switch to the module details tab
        } else {
          setCurrentTab(1); // Switch to client details tab
        }
      } else {
        setError(`Client not found: ${clientId}`);
        setSelectedClient(null);
      }
    }
  }, [clientId, moduleId, clients]);

  const loadClients = async () => {
    try {
      setLoading(true);
      const response = await ClientService.getAllDSPyClients();
      setClients(response);
      setError(null);
    } catch (err) {
      console.error('Failed to load DSPy clients:', err);
      setError('Failed to load DSPy clients. Please try again later.');
    } finally {
      setLoading(false);
    }
  };

  const loadModule = async (moduleId: string) => {
    try {
      const module = await ClientService.getDSPyModule(moduleId);
      setSelectedModule(module);
    } catch (err) {
      console.error('Failed to load DSPy module:', err);
      setError(`Failed to load module details: ${err}`);
      setSelectedModule(null);
    }
  };

  const handleClientSelect = (client: any) => {
    setSelectedClient(client);
    setSelectedModule(null);
    navigate(`/dspy/${client.client_id}`);
    setCurrentTab(1); // Switch to client details tab
  };

  const handleModuleSelect = (module: any) => {
    setSelectedModule(module);
    navigate(`/dspy/${selectedClient?.client_id}/module/${module.module_id}`);
    setCurrentTab(2); // Switch to module details tab
  };

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setCurrentTab(newValue);
    // Reset navigation if switching back to main tabs
    if (newValue === 0) {
      navigate('/dspy');
      setSelectedClient(null);
      setSelectedModule(null);
    } else if (newValue === 1 && selectedClient) {
      navigate(`/dspy/${selectedClient.client_id}`);
    }
  };

  if (loading && !clients.length) {
    return (
      <Container maxWidth="lg">
        <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '50vh' }}>
          <CircularProgress />
        </Box>
      </Container>
    );
  }

  return (
    <Container maxWidth="lg">
      <Typography variant="h4" gutterBottom>
        DSPy Dashboard
      </Typography>
      
      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      <Box sx={{ width: '100%' }}>
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs value={currentTab} onChange={handleTabChange} aria-label="dspy dashboard tabs">
            <Tab label="Clients" {...a11yProps(0)} />
            {selectedClient && <Tab label="Client Details" {...a11yProps(1)} />}
            {selectedModule && <Tab label="Module Details" {...a11yProps(2)} />}
          </Tabs>
        </Box>
        
        <TabPanel value={currentTab} index={0}>
          <DSPyClientList 
            clients={clients} 
            onClientSelect={handleClientSelect}
            onRefresh={loadClients}
          />
        </TabPanel>
        
        {selectedClient && (
          <TabPanel value={currentTab} index={1}>
            <Grid container spacing={3}>
              <Grid item xs={12}>
                <DSPyClientDetails 
                  client={selectedClient}
                  onRefresh={() => {
                    loadClients();
                    // Reload the selected client with fresh data
                    if (selectedClient?.client_id) {
                      ClientService.getDSPyClient(selectedClient.client_id)
                        .then(updatedClient => setSelectedClient(updatedClient))
                        .catch(err => console.error('Error refreshing client:', err));
                    }
                  }}
                />
              </Grid>
              <Grid item xs={12}>
                <Paper elevation={2} sx={{ p: 2, mb: 2 }}>
                  <Typography variant="h6" gutterBottom>
                    Modules
                  </Typography>
                  <DSPyModuleList 
                    clientId={selectedClient.client_id}
                    onModuleSelect={handleModuleSelect}
                  />
                </Paper>
              </Grid>
              <Grid item xs={12}>
                <Paper elevation={2} sx={{ p: 2, mb: 2 }}>
                  <Typography variant="h6" gutterBottom>
                    Circuit Breakers
                  </Typography>
                  <DSPyCircuitBreakers clientId={selectedClient.client_id} />
                </Paper>
              </Grid>
              <Grid item xs={12}>
                <Paper elevation={2} sx={{ p: 2 }}>
                  <Typography variant="h6" gutterBottom>
                    Audit Logs
                  </Typography>
                  <DSPyAuditLogViewer clientId={selectedClient.client_id} />
                </Paper>
              </Grid>
            </Grid>
          </TabPanel>
        )}
        
        {selectedModule && (
          <TabPanel value={currentTab} index={2}>
            <DSPyModuleDetails 
              module={selectedModule}
              onRefresh={() => {
                if (selectedModule?.module_id) {
                  loadModule(selectedModule.module_id);
                }
              }}
            />
          </TabPanel>
        )}
      </Box>
    </Container>
  );
};

export default DSPyDashboard;