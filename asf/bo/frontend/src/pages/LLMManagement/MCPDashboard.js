import React, { useState } from 'react';
import {
  Box,
  Paper,
  Typography,
  Tabs,
  Tab,
  Grid,
  Card,
  CardHeader,
  CardContent,
  CardActions,
  Button,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Divider,
  Alert,
  CircularProgress,
  Chip,
  Tooltip,
  MenuItem,
  FormControl,
  InputLabel,
  Select
} from '@mui/material';
import {
  CloudQueue as CloudQueueIcon,
  Settings as SettingsIcon,
  Check as CheckIcon,
  BarChart as BarChartIcon,
  SmartToy as SmartToyIcon,
  Storage as StorageIcon,
  Speed as SpeedIcon,
  Code as CodeIcon,
  Dns as DnsIcon,
  Refresh as RefreshIcon
} from '@mui/icons-material';

import { useAuth } from '../../context/AuthContext';
import { ContentLoader } from '../../components/UI/LoadingIndicators';
import { FadeIn, StaggeredList } from '../../components/UI/Animations';
import { useMCPProviders } from '../../hooks/useMCPProviders';
// Import useMCPInfo with error handling
import { useMCPInfo as useMCPInfoOriginal } from '../../hooks/useMCPInfo';

// Wrap the hook with error handling
const useMCPInfo = () => {
  try {
    return useMCPInfoOriginal();
  } catch (error) {
    console.error('Error using useMCPInfo hook:', error);
    return {
      mcpInfo: null,
      isLoading: false,
      isError: true,
      error: error.message,
      refetch: () => {}
    };
  }
};
import MCPProviderManagement from '../../components/LLM/MCP/MCPProviderManagement';
import MCPStatusMonitor from '../../components/LLM/MCP/MCPStatusMonitor';
import MCPUsageStats from '../../components/LLM/MCP/MCPUsageStats';
import MCPConfigDialog from '../../components/LLM/MCP/MCPConfigDialog';

/**
 * MCP (Model Context Protocol) Dashboard
 *
 * This component provides a comprehensive interface for managing MCP providers,
 * monitoring their status, and viewing usage statistics.
 */
const MCPDashboard = () => {
  const { user } = useAuth();

  // State
  const [activeTab, setActiveTab] = useState(0);
  const [selectedProvider, setSelectedProvider] = useState(null);
  const [configDialogOpen, setConfigDialogOpen] = useState(false);

  // Use MCP hooks with fallback
  const { mcpInfo, isLoading: mcpInfoLoading, isError: mcpInfoError } = useMCPInfo();
  const {
    providers,
    isLoading: providersLoading,
    refetch: refetchProviders,
    testProvider,
    testProviderLoading
  } = useMCPProviders();

  // Filter active providers
  const activeProviders = Array.isArray(providers) ? providers.filter(p =>
    p.status === 'operational' ||
    p.status === 'available' ||
    p.status === 'connected'
  ) : [];

  // Set selected provider if none is selected and active providers exist
  if (activeProviders.length > 0 && !selectedProvider) {
    setSelectedProvider(activeProviders[0].provider_id);
  }

  // Handle tab change
  const handleTabChange = (event, newValue) => {
    setActiveTab(newValue);
  };

  // Handle provider selection change
  const handleProviderChange = (event) => {
    setSelectedProvider(event.target.value);
  };

  // Render provider cards
  const renderProviderCards = () => {
    if (!Array.isArray(providers) || providers.length === 0) {
      return (
        <Alert severity="info">
          No MCP providers found. Click "Add MCP Provider" to create one.
        </Alert>
      );
    }

    return (
      <StaggeredList>
        <Grid container spacing={2}>
          {providers.map((provider) => (
            <Grid item xs={12} md={6} lg={4} key={provider.provider_id}>
              <Card
                variant="outlined"
                sx={{
                  height: '100%',
                  display: 'flex',
                  flexDirection: 'column',
                  transition: 'all 0.2s ease-in-out',
                  '&:hover': {
                    boxShadow: 3,
                    transform: 'translateY(-4px)'
                  },
                  cursor: 'pointer'
                }}
                onClick={() => {
                  setSelectedProvider(provider.provider_id);
                  setActiveTab(1); // Switch to status tab
                }}
              >
                <CardHeader
                  title={provider.display_name || provider.provider_id}
                  subheader={`Transport: ${provider.transport_type}`}
                  avatar={
                    <CloudQueueIcon
                      color={
                        provider.status === 'operational' ||
                        provider.status === 'available' ||
                        provider.status === 'connected' ? 'success' : 'disabled'
                      }
                    />
                  }
                  action={
                    <Chip
                      label={provider.status}
                      color={
                        provider.status === 'operational' ||
                        provider.status === 'available' ||
                        provider.status === 'connected' ? 'success' :
                        provider.status === 'error' ? 'error' : 'default'
                      }
                      size="small"
                      sx={{ mt: 1, mr: 1 }}
                    />
                  }
                />
                <CardContent sx={{ flexGrow: 1 }}>
                  <List dense>
                    <ListItem>
                      <ListItemIcon sx={{ minWidth: '32px' }}>
                        <DnsIcon fontSize="small" />
                      </ListItemIcon>
                      <ListItemText
                        primary="Transport Type"
                        secondary={provider.transport_type}
                      />
                    </ListItem>

                    {provider.circuit_breaker && (
                      <ListItem>
                        <ListItemIcon sx={{ minWidth: '32px' }}>
                          <SpeedIcon fontSize="small" />
                        </ListItemIcon>
                        <ListItemText
                          primary="Circuit Breaker"
                          secondary={provider.circuit_breaker.state}
                        />
                      </ListItem>
                    )}

                    {provider.models && provider.models.length > 0 && (
                      <ListItem>
                        <ListItemIcon sx={{ minWidth: '32px' }}>
                          <SmartToyIcon fontSize="small" />
                        </ListItemIcon>
                        <ListItemText
                          primary="Available Models"
                          secondary={provider.models.length}
                        />
                      </ListItem>
                    )}

                    <ListItem>
                      <ListItemIcon sx={{ minWidth: '32px' }}>
                        <StorageIcon fontSize="small" />
                      </ListItemIcon>
                      <ListItemText
                        primary="Last Checked"
                        secondary={new Date(provider.checked_at).toLocaleString()}
                      />
                    </ListItem>
                  </List>

                  {provider.message && (
                    <Alert
                      severity={
                        provider.status === 'error' ? 'error' :
                        provider.status === 'warning' ? 'warning' : 'info'
                      }
                      sx={{ mt: 1 }}
                    >
                      {provider.message}
                    </Alert>
                  )}
                </CardContent>
                <CardActions>
                  <Button
                    size="small"
                    onClick={(e) => {
                      e.stopPropagation();
                      testProvider(provider.provider_id);
                    }}
                    startIcon={testProviderLoading ? <CircularProgress size={16} /> : <CheckIcon />}
                    disabled={testProviderLoading}
                  >
                    Test Connection
                  </Button>
                  <Button
                    size="small"
                    onClick={(e) => {
                      e.stopPropagation();
                      setSelectedProvider(provider.provider_id);
                      setActiveTab(1);
                    }}
                    startIcon={<BarChartIcon />}
                  >
                    View Usage
                  </Button>
                </CardActions>
              </Card>
            </Grid>
          ))}
        </Grid>
      </StaggeredList>
    );
  };

  return (
    <FadeIn>
      <Box sx={{ mb: 4 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
          <Typography variant="h4" component="h1">
            MCP Provider Management
          </Typography>

          <Box sx={{ display: 'flex', gap: 1 }}>
            <Button
              variant="outlined"
              onClick={() => refetchProviders()}
              disabled={providersLoading}
              startIcon={providersLoading ? <CircularProgress size={20} /> : <RefreshIcon />}
            >
              {providersLoading ? 'Refreshing...' : 'Refresh'}
            </Button>

            <Button
              variant="contained"
              color="primary"
              onClick={() => setConfigDialogOpen(true)}
            >
              Add MCP Provider
            </Button>
          </Box>
        </Box>

        {mcpInfoLoading ? (
          <ContentLoader height={200} message="Loading MCP information..." />
        ) : mcpInfoError ? (
          <Alert severity="warning" sx={{ mb: 3 }}>
            Unable to load MCP information. The service may be unavailable.
          </Alert>
        ) : mcpInfo ? (
          <Paper sx={{ p: 3, mb: 3 }}>
            <Typography variant="h6" gutterBottom>
              {mcpInfo.name} ({mcpInfo.version})
            </Typography>
            <Typography paragraph>
              {mcpInfo.description}
            </Typography>

            <Grid container spacing={2}>
              <Grid item xs={12} md={6}>
                <Typography variant="subtitle1" gutterBottom>Features:</Typography>
                <Box component="ul" sx={{ pl: 2 }}>
                  {mcpInfo?.features?.map((feature, index) => (
                    <Box component="li" key={index}>
                      <Typography>{feature}</Typography>
                    </Box>
                  )) || <Typography>No features available</Typography>}
                </Box>
              </Grid>

              <Grid item xs={12} md={6}>
                <Typography variant="subtitle1" gutterBottom>Transport Types:</Typography>
                <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                  {mcpInfo?.transport_types?.map((type, index) => (
                    <Chip
                      key={index}
                      label={type}
                      icon={<CodeIcon />}
                      variant="outlined"
                    />
                  )) || <Typography>No transport types available</Typography>}
                </Box>

                {mcpInfo?.documentation_url && (
                  <Box sx={{ mt: 2 }}>
                    <Button
                      variant="text"
                      href={mcpInfo.documentation_url}
                      target="_blank"
                      rel="noopener noreferrer"
                    >
                      View Documentation
                    </Button>
                  </Box>
                )}
              </Grid>
            </Grid>
          </Paper>
        ) : (
          <Alert severity="info" sx={{ mb: 3 }}>
            No MCP information available.
          </Alert>
        )}

        <Paper sx={{ mb: 3 }}>
          <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
            <Tabs
              value={activeTab}
              onChange={handleTabChange}
              aria-label="MCP management tabs"
            >
              <Tab
                icon={<CloudQueueIcon />}
                label="Providers"
                id="tab-0"
                aria-controls="tabpanel-0"
              />
              <Tab
                icon={<BarChartIcon />}
                label="Usage Statistics"
                id="tab-1"
                aria-controls="tabpanel-1"
                disabled={!selectedProvider}
              />
              <Tab
                icon={<SettingsIcon />}
                label="Provider Management"
                id="tab-2"
                aria-controls="tabpanel-2"
              />
            </Tabs>
          </Box>

          {/* Providers Tab */}
          <Box role="tabpanel" hidden={activeTab !== 0} id="tabpanel-0" aria-labelledby="tab-0" sx={{ p: 3 }}>
            {activeTab === 0 && (
              providersLoading ? (
                <ContentLoader height={200} message="Loading MCP providers..." />
              ) : Array.isArray(providers) && providers.length > 0 ? (
                renderProviderCards()
              ) : (
                <Alert severity="info">
                  No MCP providers found. Click "Add MCP Provider" to create one.
                </Alert>
              )
            )}
          </Box>

          {/* Status & Usage Tab */}
          <Box role="tabpanel" hidden={activeTab !== 1} id="tabpanel-1" aria-labelledby="tab-1" sx={{ p: 3 }}>
            {activeTab === 1 && (
              <>
                {Array.isArray(providers) && providers.length > 0 ? (
                  <Box sx={{ mb: 3 }}>
                    <FormControl fullWidth sx={{ maxWidth: 300, mb: 3 }}>
                      <InputLabel id="provider-select-label">Select Provider</InputLabel>
                      <Select
                        labelId="provider-select-label"
                        value={selectedProvider || ''}
                        onChange={handleProviderChange}
                        label="Select Provider"
                      >
                        {Array.isArray(providers) && providers.map((provider) => (
                          <MenuItem key={provider.provider_id} value={provider.provider_id}>
                            {provider.display_name || provider.provider_id}
                          </MenuItem>
                        ))}
                      </Select>
                    </FormControl>

                    {selectedProvider ? (
                      <Grid container spacing={3}>
                        <Grid item xs={12}>
                          <MCPStatusMonitor providerId={selectedProvider} />
                        </Grid>
                        <Grid item xs={12}>
                          <Divider sx={{ my: 3 }} />
                          <MCPUsageStats providerId={selectedProvider} />
                        </Grid>
                      </Grid>
                    ) : (
                      <Alert severity="info">
                        Please select a provider to view its status and usage statistics.
                      </Alert>
                    )}
                  </Box>
                ) : (
                  <Alert severity="info">
                    No MCP providers found. Add a provider first to view status and usage statistics.
                  </Alert>
                )}
              </>
            )}
          </Box>

          {/* Provider Management Tab */}
          <Box role="tabpanel" hidden={activeTab !== 2} id="tabpanel-2" aria-labelledby="tab-2" sx={{ p: 3 }}>
            {activeTab === 2 && (
              <MCPProviderManagement />
            )}
          </Box>
        </Paper>

        {/* Additional information */}
        <Paper sx={{ p: 3 }}>
          <Typography variant="h6" gutterBottom>About MCP Providers</Typography>
          <Typography paragraph>
            The Model Context Protocol (MCP) is a standardized protocol for interacting with large language models.
            This page allows you to manage MCP providers, monitor their status, and view usage statistics.
          </Typography>

          <Grid container spacing={2}>
            <Grid item xs={12} md={6}>
              <Typography variant="subtitle1" gutterBottom>Key Features:</Typography>
              <Box component="ul" sx={{ pl: 2 }}>
                <Box component="li"><Typography>Multiple transport options (stdio, gRPC, HTTP/REST)</Typography></Box>
                <Box component="li"><Typography>Streaming & non-streaming support</Typography></Box>
                <Box component="li"><Typography>Advanced resilience with circuit breaker pattern</Typography></Box>
                <Box component="li"><Typography>Comprehensive observability with metrics and tracing</Typography></Box>
              </Box>
            </Grid>

            <Grid item xs={12} md={6}>
              <Typography variant="subtitle1" gutterBottom>Management Features:</Typography>
              <Box component="ul" sx={{ pl: 2 }}>
                <Box component="li"><Typography>Configure provider settings</Typography></Box>
                <Box component="li"><Typography>Test provider connections</Typography></Box>
                <Box component="li"><Typography>Monitor provider status</Typography></Box>
                <Box component="li"><Typography>View usage statistics</Typography></Box>
              </Box>
            </Grid>
          </Grid>
        </Paper>
      </Box>

      {/* Config Dialog */}
      <MCPConfigDialog
        open={configDialogOpen}
        mode="add"
        onClose={() => setConfigDialogOpen(false)}
        onSave={(providerData) => {
          setConfigDialogOpen(false);
          // The provider will be added through the useMCPProviders hook
        }}
      />
    </FadeIn>
  );
};

export default MCPDashboard;
