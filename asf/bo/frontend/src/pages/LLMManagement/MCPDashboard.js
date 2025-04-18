import React, { useState, useEffect } from 'react';
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
  Tooltip
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
  Dns as DnsIcon
} from '@mui/icons-material';

import { useAuth } from '../../context/AuthContext';
import { useNotification } from '../../context/NotificationContext';
import apiService from '../../services/api';
import { ContentLoader } from '../../components/UI/LoadingIndicators';
import { FadeIn, StaggeredList } from '../../components/UI/Animations';
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
  const { showSuccess, showError } = useNotification();
  
  // State
  const [activeTab, setActiveTab] = useState(0);
  const [providers, setProviders] = useState([]);
  const [activeProviders, setActiveProviders] = useState([]);
  const [selectedProvider, setSelectedProvider] = useState(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [mcpInfo, setMcpInfo] = useState(null);
  const [configDialogOpen, setConfigDialogOpen] = useState(false);
  
  // Load MCP info and providers on mount
  useEffect(() => {
    loadMCPInfo();
    loadProviders();
  }, []);
  
  // Load MCP information
  const loadMCPInfo = async () => {
    try {
      const result = await apiService.llm.getMCPInfo();
      
      if (result.success) {
        setMcpInfo(result.data);
      } else {
        showError(`Failed to load MCP information: ${result.error}`);
      }
    } catch (error) {
      console.error('Error loading MCP information:', error);
      showError(`Error loading MCP information: ${error.message}`);
    }
  };
  
  // Load MCP providers
  const loadProviders = async () => {
    setLoading(true);
    
    try {
      const result = await apiService.llm.getMCPProviders();
      
      if (result.success) {
        setProviders(result.data);
        // Filter active providers
        const active = result.data.filter(p =>
          p.status === 'operational' ||
          p.status === 'available' ||
          p.status === 'connected'
        );
        setActiveProviders(active);
        
        if (active.length > 0 && !selectedProvider) {
          setSelectedProvider(active[0].provider_id);
        }
      } else {
        showError(`Failed to load MCP providers: ${result.error}`);
      }
    } catch (error) {
      console.error('Error loading MCP providers:', error);
      showError(`Error loading MCP providers: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };
  
  // Refresh providers
  const refreshProviders = async () => {
    setRefreshing(true);
    
    try {
      await loadProviders();
      showSuccess('MCP providers refreshed successfully');
    } catch (error) {
      console.error('Error refreshing MCP providers:', error);
      showError(`Error refreshing MCP providers: ${error.message}`);
    } finally {
      setRefreshing(false);
    }
  };
  
  // Test provider connection
  const testProvider = async (providerId) => {
    try {
      const result = await apiService.llm.testMCPProvider(providerId);
      
      if (result.success) {
        showSuccess(`Connection to ${providerId} successful (${result.data.latency_ms.toFixed(2)}ms)`);
      } else {
        showError(`Connection to ${providerId} failed: ${result.error}`);
      }
    } catch (error) {
      console.error(`Error testing MCP provider ${providerId}:`, error);
      showError(`Error testing MCP provider: ${error.message}`);
    }
  };
  
  // Handle tab change
  const handleTabChange = (event, newValue) => {
    setActiveTab(newValue);
  };
  
  // Render provider cards
  const renderProviderCards = () => {
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
                  }
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
                    onClick={() => testProvider(provider.provider_id)}
                    startIcon={<CheckIcon />}
                  >
                    Test Connection
                  </Button>
                  <Button
                    size="small"
                    onClick={() => {
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
              onClick={refreshProviders}
              disabled={refreshing}
              startIcon={refreshing ? <CircularProgress size={20} /> : null}
            >
              {refreshing ? 'Refreshing...' : 'Refresh'}
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
        
        {mcpInfo && (
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
                  {mcpInfo.features.map((feature, index) => (
                    <Box component="li" key={index}>
                      <Typography>{feature}</Typography>
                    </Box>
                  ))}
                </Box>
              </Grid>
              
              <Grid item xs={12} md={6}>
                <Typography variant="subtitle1" gutterBottom>Transport Types:</Typography>
                <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                  {mcpInfo.transport_types.map((type, index) => (
                    <Chip
                      key={index}
                      label={type}
                      icon={<CodeIcon />}
                      variant="outlined"
                    />
                  ))}
                </Box>
                
                {mcpInfo.documentation_url && (
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
              loading ? (
                <ContentLoader height={200} message="Loading MCP providers..." />
              ) : providers.length > 0 ? (
                renderProviderCards()
              ) : (
                <Alert severity="info">
                  No MCP providers found. Click "Add MCP Provider" to create one.
                </Alert>
              )
            )}
          </Box>
          
          {/* Usage Statistics Tab */}
          <Box role="tabpanel" hidden={activeTab !== 1} id="tabpanel-1" aria-labelledby="tab-1" sx={{ p: 3 }}>
            {activeTab === 1 && selectedProvider && (
              <MCPUsageStats providerId={selectedProvider} />
            )}
          </Box>
          
          {/* Provider Management Tab */}
          <Box role="tabpanel" hidden={activeTab !== 2} id="tabpanel-2" aria-labelledby="tab-2" sx={{ p: 3 }}>
            {activeTab === 2 && (
              <MCPProviderManagement
                onProviderAdded={loadProviders}
                onProviderUpdated={loadProviders}
                onProviderDeleted={loadProviders}
              />
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
        onClose={() => setConfigDialogOpen(false)}
        onSave={() => {
          setConfigDialogOpen(false);
          loadProviders();
        }}
      />
    </FadeIn>
  );
};

export default MCPDashboard;
