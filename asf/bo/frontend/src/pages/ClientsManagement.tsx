import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Box, Paper, Typography, Grid, Card, CardContent, CardHeader,
  CardActions, Button, Chip, Divider, CircularProgress,
  IconButton, Tooltip, Alert, LinearProgress, Tab, Tabs
} from '@mui/material';
import {
  Refresh as RefreshIcon,
  Settings as SettingsIcon,
  CheckCircle as CheckCircleIcon,
  Error as ErrorIcon,
  Warning as WarningIcon,
  Info as InfoIcon,
  BarChart as BarChartIcon,
  Speed as SpeedIcon
} from '@mui/icons-material';

import PageLayout from '../components/Layout/PageLayout';
import { ClientConfigDialog } from '../components/Clients';
import { ContentLoader } from '../components/UI/LoadingIndicators';
import { FadeIn, StaggeredList } from '../components/UI/Animations';
import { useAuth } from '../hooks/useAuth';
import { useNotification } from '../context/NotificationContext';
import { useMedicalClients } from '../hooks/useMedicalClients';
import { useFeatureFlags } from '../context/FeatureFlagContext';

// Types
interface Client {
  id: string;
  name: string;
  description: string;
  status: string;
  last_checked: string;
  api_version: string;
  endpoints: string[];
  config?: Record<string, any>;
  response_time?: number;
  error_message?: string;
}

interface TestResult {
  clientId: string;
  status: string;
  message: string;
  success: boolean;
  api_version?: string;
  response_time?: number;
}

/**
 * Medical Clients Management page
 *
 * This page provides management functionality for medical clients,
 * including NCBI, UMLS, ClinicalTrials, Cochrane, Crossref, and SNOMED.
 */
const ClientsManagement: React.FC = () => {
  const { user } = useAuth();
  const { showSuccess, showError } = useNotification();
  const navigate = useNavigate();
  const { isEnabled } = useFeatureFlags();
  const useMockData = isEnabled('useMockData');

  // State
  const [selectedClient, setSelectedClient] = useState<Client | null>(null);
  const [configDialogOpen, setConfigDialogOpen] = useState<boolean>(false);
  const [testResults, setTestResults] = useState<TestResult | null>(null);
  const [activeTab, setActiveTab] = useState<number>(0);

  // Medical clients hooks
  const {
    clients,
    isLoadingClients,
    isErrorClients,
    errorClients,
    refetchClients
  } = useMedicalClients();

  // Test client connection
  const {
    mutate: testConnection,
    isPending: isTestingConnection
  } = useMedicalClients().testClientConnection();

  // Update client configuration
  const {
    mutate: updateConfig,
    isPending: isUpdatingConfig
  } = useMedicalClients().updateClientConfig();

  // Handle client configuration
  const handleConfigureClient = (client: Client) => {
    setSelectedClient(client);
    setConfigDialogOpen(true);
  };

  // Handle client configuration update
  const handleUpdateClientConfig = async (clientId: string, config: Record<string, any>) => {
    updateConfig({
      id: clientId,
      ...config
    });
  };

  // Handle client connection test
  const handleTestConnection = async (clientId: string) => {
    setTestResults(null);

    testConnection(clientId, {
      onSuccess: (data) => {
        setTestResults({
          clientId,
          status: data.status,
          message: data.status === 'connected' ? 'Connection successful' : data.error_message || 'Connection failed',
          success: data.status === 'connected',
          api_version: data.api_version,
          response_time: data.response_time
        });
      }
    });
  };

  // Handle tab change
  const handleTabChange = (_: React.SyntheticEvent, newValue: number) => {
    setActiveTab(newValue);
  };

  // Get status chip color
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'connected':
        return 'success';
      case 'disconnected':
        return 'error';
      case 'error':
        return 'error';
      default:
        return 'default';
    }
  };

  // Get status icon
  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'connected':
        return <CheckCircleIcon />;
      case 'disconnected':
        return <ErrorIcon />;
      case 'error':
        return <WarningIcon />;
      default:
        return <InfoIcon />;
    }
  };

  // Render client cards
  const renderClientCards = () => {
    return (
      <StaggeredList>
        <Grid container spacing={3}>
          {clients.map((client) => (
            <Grid item xs={12} md={6} lg={4} key={client.id}>
              <Card
                variant="outlined"
                sx={{
                  height: '100%',
                  display: 'flex',
                  flexDirection: 'column',
                  transition: 'all 0.3s ease',
                  '&:hover': {
                    boxShadow: 3,
                    transform: 'translateY(-4px)'
                  }
                }}
              >
                <CardHeader
                  title={client.name}
                  subheader={`ID: ${client.id}`}
                  action={
                    <Chip
                      label={client.status}
                      color={getStatusColor(client.status)}
                      icon={getStatusIcon(client.status)}
                      size="small"
                    />
                  }
                />
                <CardContent sx={{ flexGrow: 1 }}>
                  <Typography variant="body2" color="text.secondary" gutterBottom>
                    Last checked: {new Date(client.last_checked).toLocaleString()}
                  </Typography>

                  {client.api_version && (
                    <Typography variant="body2" color="text.secondary" gutterBottom>
                      API Version: {client.api_version}
                    </Typography>
                  )}

                  {client.response_time && (
                    <Box sx={{ display: 'flex', alignItems: 'center', mt: 1 }}>
                      <SpeedIcon fontSize="small" sx={{ mr: 1, color: 'text.secondary' }} />
                      <Typography variant="body2" color="text.secondary">
                        Response time: {client.response_time.toFixed(2)}s
                      </Typography>
                    </Box>
                  )}

                  {client.error_message && (
                    <Alert severity="error" sx={{ mt: 2 }}>
                      {client.error_message}
                    </Alert>
                  )}

                  {testResults && testResults.clientId === client.id && (
                    <Box sx={{ mt: 2 }}>
                      <Typography variant="subtitle2" gutterBottom>
                        Test Results:
                      </Typography>
                      <Alert severity={testResults.success ? "success" : "error"}>
                        {testResults.message}
                      </Alert>
                      {testResults.api_version && (
                        <Typography variant="body2" sx={{ mt: 1 }}>
                          API Version: {testResults.api_version}
                        </Typography>
                      )}
                      {testResults.response_time && (
                        <Typography variant="body2">
                          Response Time: {testResults.response_time.toFixed(2)}s
                        </Typography>
                      )}
                    </Box>
                  )}
                </CardContent>
                <CardActions>
                  <Button
                    size="small"
                    startIcon={<SettingsIcon />}
                    onClick={() => handleConfigureClient(client)}
                  >
                    Configure
                  </Button>
                  <Button
                    size="small"
                    startIcon={isTestingConnection ? <CircularProgress size={16} /> : <RefreshIcon />}
                    onClick={() => handleTestConnection(client.id)}
                    disabled={isTestingConnection}
                  >
                    Test Connection
                  </Button>
                  {client.id === 'ncbi' && (
                    <Button
                      size="small"
                      color="primary"
                      onClick={() => navigate('/clients-management/ncbi')}
                    >
                      Manage
                    </Button>
                  )}
                </CardActions>
              </Card>
            </Grid>
          ))}
        </Grid>
      </StaggeredList>
    );
  };

  // Render client usage statistics
  const renderClientUsage = () => {
    return (
      <Typography variant="body1" sx={{ p: 3, textAlign: 'center' }}>
        Client usage statistics will be displayed here.
      </Typography>
    );
  };

  return (
    <PageLayout
      title="Medical Clients Management"
      breadcrumbs={[{ label: 'Clients Management', path: '/clients-management' }]}
      loading={isLoadingClients}
      user={user}
      actions={
        <Button
          variant="outlined"
          startIcon={isLoadingClients ? <CircularProgress size={20} /> : <RefreshIcon />}
          onClick={() => refetchClients()}
          disabled={isLoadingClients}
        >
          Refresh
        </Button>
      }
    >
      {useMockData && (
        <Alert severity="info" sx={{ mb: 2 }}>
          Using mock data. Toggle the "Use Mock Data" feature flag to use real API data.
        </Alert>
      )}

      {isErrorClients && (
        <Alert 
          severity="error" 
          sx={{ mb: 2 }}
          action={
            <Button color="inherit" size="small" onClick={() => refetchClients()}>
              Retry
            </Button>
          }
        >
          Failed to load clients: {errorClients?.message || 'Unknown error'}
        </Alert>
      )}

      <Paper sx={{ mb: 3 }}>
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs
            value={activeTab}
            onChange={handleTabChange}
            aria-label="client management tabs"
          >
            <Tab
              icon={<SettingsIcon />}
              label="Client Configuration"
              id="tab-0"
              aria-controls="tabpanel-0"
            />
            <Tab
              icon={<BarChartIcon />}
              label="Usage Statistics"
              id="tab-1"
              aria-controls="tabpanel-1"
            />
          </Tabs>
        </Box>

        <Box role="tabpanel" hidden={activeTab !== 0} id="tabpanel-0" aria-labelledby="tab-0" sx={{ p: 3 }}>
          {activeTab === 0 && (
            isLoadingClients ? (
              <ContentLoader height={200} message="Loading clients..." />
            ) : clients.length > 0 ? (
              renderClientCards()
            ) : (
              <Alert severity="info">
                No clients found. Please refresh to load clients.
              </Alert>
            )
          )}
        </Box>

        <Box role="tabpanel" hidden={activeTab !== 1} id="tabpanel-1" aria-labelledby="tab-1" sx={{ p: 3 }}>
          {activeTab === 1 && renderClientUsage()}
        </Box>
      </Paper>

      {/* Additional information */}
      <Paper sx={{ p: 3 }}>
        <Typography variant="h6" gutterBottom>About Medical Clients</Typography>
        <Typography paragraph>
          This page allows you to manage the medical clients used by the application to access
          various medical databases and services.
        </Typography>
        <Grid container spacing={2}>
          <Grid item xs={12} md={6}>
            <Typography variant="subtitle1" gutterBottom>Available Clients:</Typography>
            <Box component="ul" sx={{ pl: 2 }}>
              <Box component="li"><Typography>NCBI - National Center for Biotechnology Information</Typography></Box>
              <Box component="li"><Typography>UMLS - Unified Medical Language System</Typography></Box>
              <Box component="li"><Typography>ClinicalTrials.gov - Clinical trials database</Typography></Box>
              <Box component="li"><Typography>Cochrane Library - Systematic reviews database</Typography></Box>
              <Box component="li"><Typography>Crossref - DOI registration agency</Typography></Box>
              <Box component="li"><Typography>SNOMED CT - Clinical terminology</Typography></Box>
            </Box>
          </Grid>
          <Grid item xs={12} md={6}>
            <Typography variant="subtitle1" gutterBottom>Management Features:</Typography>
            <Box component="ul" sx={{ pl: 2 }}>
              <Box component="li"><Typography>Configure client settings</Typography></Box>
              <Box component="li"><Typography>Test client connections</Typography></Box>
              <Box component="li"><Typography>View client status</Typography></Box>
              <Box component="li"><Typography>Monitor usage statistics</Typography></Box>
            </Box>
          </Grid>
        </Grid>
      </Paper>

      {/* Client configuration dialog */}
      <ClientConfigDialog
        open={configDialogOpen}
        onClose={() => setConfigDialogOpen(false)}
        client={selectedClient}
        onUpdate={handleUpdateClientConfig}
      />
    </PageLayout>
  );
};

export default ClientsManagement;
