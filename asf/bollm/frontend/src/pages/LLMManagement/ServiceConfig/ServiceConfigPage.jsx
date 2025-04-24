import React, { useState, useEffect } from 'react';
import { useNavigate, useLocation, useParams } from 'react-router-dom';
import {
  Box, Typography, Paper, Tabs, Tab, Switch, FormControlLabel,
  TextField, Button, CircularProgress, Alert, Divider,
  Card, CardHeader, CardContent, Grid, Accordion, AccordionSummary,
  AccordionDetails, Table, TableBody, TableCell, TableContainer,
  TableHead, TableRow, Chip
} from '@mui/material';
import {
  ExpandMore as ExpandMoreIcon,
  Refresh as RefreshIcon,
  Save as SaveIcon,
  Check as CheckIcon,
  Error as ErrorIcon,
  ClearAll as ClearAllIcon,
  Settings as SettingsIcon
} from '@mui/icons-material';
import { useNotification } from '../../../context/NotificationContext';
import apiService from '../../../services/api';

// Import the PageLayout component that includes the sidebar
import PageLayout from '../../../components/Layout/PageLayout';

/**
 * Service Configuration Page
 * 
 * This page allows administrators to configure and monitor the LLM Service Abstraction Layer.
 * 
 * @param {Object} props - Component props
 * @param {number} [props.initialTab=0] - Initial tab index to display
 */
const ServiceConfigPage = ({ initialTab = 0, isEditMode = false }) => {
  const { showSuccess, showError } = useNotification();
  const navigate = useNavigate();
  const location = useLocation();
  const { configId } = useParams();
  
  // State
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [config, setConfig] = useState({
    service_id: 'enhanced_llm_service',
    enable_caching: true,
    enable_resilience: true,
    enable_observability: true,
    enable_events: true,
    enable_progress_tracking: true,
    config: {
      cache: {
        similarity_threshold: 0.92,
        max_entries: 10000,
        ttl_seconds: 3600
      }
    }
  });
  const [health, setHealth] = useState(null);
  const [stats, setStats] = useState(null);
  const [tabValue, setTabValue] = useState(initialTab);
  
  // Tab paths mapping
  const tabPaths = [
    '/llm/settings/service-config',
    '/llm/settings/service-config/caching',
    '/llm/settings/service-config/resilience',
    '/llm/settings/service-config/observability',
    '/llm/settings/service-config/events',
    '/llm/settings/service-config/progress-tracking'
  ];
  
  // Load data on mount
  useEffect(() => {
    loadData();
  }, [configId]);
  
  // Load configuration and health data
  const loadData = async () => {
    setLoading(true);
    
    try {
      // Load configuration based on mode
      if (isEditMode && configId) {
        // Load specific configuration by ID
        const configResult = await apiService.llm.getServiceConfigurationById(configId);
        
        if (configResult.success) {
          setConfig(configResult.data);
        } else {
          console.error('Failed to load configuration:', configResult.error);
          showError(`Failed to load configuration: ${configResult.error}`);
          navigate('/llm/settings/service-config/configurations');
        }
      } else {
        // Load active service configuration
        const configResult = await apiService.llm.getServiceConfig();
        
        if (configResult.success) {
          setConfig(configResult.data);
        } else {
          console.error('Failed to load service configuration:', configResult.error);
          showError(`Failed to load service configuration: ${configResult.error}`);
        }
      }
      
      // Load service health
      const healthResult = await apiService.llm.getServiceHealth();
      
      if (healthResult.success) {
        setHealth(healthResult.data);
      } else {
        console.error('Failed to load service health:', healthResult.error);
        showError(`Failed to load service health: ${healthResult.error}`);
      }
      
      // Load service stats
      const statsResult = await apiService.llm.getServiceStats();
      
      if (statsResult.success) {
        setStats(statsResult.data);
      } else {
        console.error('Failed to load service stats:', statsResult.error);
        showError(`Failed to load service stats: ${statsResult.error}`);
      }
    } catch (error) {
      console.error('Error loading data:', error);
      showError(`Error loading data: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };
  
  // Refresh data
  const handleRefresh = async () => {
    setRefreshing(true);
    await loadData();
    setRefreshing(false);
  };
  
  // Handle tab change
  const handleTabChange = (event, newValue) => {
    setTabValue(newValue);
    // Update URL to match the selected tab
    if (location.pathname !== tabPaths[newValue]) {
      navigate(tabPaths[newValue]);
    }
  };
  
  // Update tab value when URL changes
  useEffect(() => {
    const pathIndex = tabPaths.findIndex(path => location.pathname === path);
    if (pathIndex !== -1 && pathIndex !== tabValue) {
      setTabValue(pathIndex);
    }
  }, [location.pathname]);
  
  // Handle config change
  const handleConfigChange = (e) => {
    const { name, checked, value, type } = e.target;
    
    setConfig(prev => ({
      ...prev,
      [name]: type === 'checkbox' ? checked : value
    }));
  };
  
  // Handle nested config change
  const handleNestedConfigChange = (section, field, value) => {
    setConfig(prev => ({
      ...prev,
      config: {
        ...prev.config,
        [section]: {
          ...prev.config[section],
          [field]: value
        }
      }
    }));
  };
  
  // Save configuration
  const handleSaveConfig = async () => {
    try {
      let result;
      
      if (isEditMode && configId) {
        // Update existing configuration
        result = await apiService.llm.updateServiceConfiguration(configId, config);
        
        if (result.success) {
          showSuccess('Configuration updated successfully');
          await loadData();
        } else {
          showError(`Failed to update configuration: ${result.error}`);
        }
      } else {
        // Update active service configuration
        result = await apiService.llm.updateServiceConfig(config);
        
        if (result.success) {
          showSuccess('Service configuration saved successfully');
          await loadData();
        } else {
          showError(`Failed to save service configuration: ${result.error}`);
        }
      }
    } catch (error) {
      console.error('Error saving configuration:', error);
      showError(`Error saving configuration: ${error.message}`);
    }
  };
  
  // Clear cache
  const handleClearCache = async () => {
    try {
      const result = await apiService.llm.clearCache();
      
      if (result.success) {
        showSuccess('Cache cleared successfully');
        await loadData();
      } else {
        showError(`Failed to clear cache: ${result.error}`);
      }
    } catch (error) {
      console.error('Error clearing cache:', error);
      showError(`Error clearing cache: ${error.message}`);
    }
  };
  
  // Reset circuit breakers
  const handleResetCircuitBreakers = async () => {
    try {
      const result = await apiService.llm.resetCircuitBreakers();
      
      if (result.success) {
        showSuccess('Circuit breakers reset successfully');
        await loadData();
      } else {
        showError(`Failed to reset circuit breakers: ${result.error}`);
      }
    } catch (error) {
      console.error('Error resetting circuit breakers:', error);
      showError(`Error resetting circuit breakers: ${error.message}`);
    }
  };
  
  // Render general settings tab
  const renderGeneralSettings = () => (
    <Paper sx={{ p: 3, mt: 2 }}>
      <Typography variant="h6" gutterBottom>
        Feature Toggles
      </Typography>
      <Grid container spacing={2}>
        <Grid item xs={12} md={4}>
          <FormControlLabel
            control={
              <Switch
                checked={config.enable_caching}
                onChange={handleConfigChange}
                name="enable_caching"
              />
            }
            label="Enable Caching"
          />
        </Grid>
        <Grid item xs={12} md={4}>
          <FormControlLabel
            control={
              <Switch
                checked={config.enable_resilience}
                onChange={handleConfigChange}
                name="enable_resilience"
              />
            }
            label="Enable Resilience Patterns"
          />
        </Grid>
        <Grid item xs={12} md={4}>
          <FormControlLabel
            control={
              <Switch
                checked={config.enable_observability}
                onChange={handleConfigChange}
                name="enable_observability"
              />
            }
            label="Enable Observability"
          />
        </Grid>
        <Grid item xs={12} md={4}>
          <FormControlLabel
            control={
              <Switch
                checked={config.enable_events}
                onChange={handleConfigChange}
                name="enable_events"
              />
            }
            label="Enable Events"
          />
        </Grid>
        <Grid item xs={12} md={4}>
          <FormControlLabel
            control={
              <Switch
                checked={config.enable_progress_tracking}
                onChange={handleConfigChange}
                name="enable_progress_tracking"
              />
            }
            label="Enable Progress Tracking"
          />
        </Grid>
      </Grid>
      
      <Divider sx={{ my: 3 }} />
      
      <Typography variant="h6" gutterBottom>
        {isEditMode ? 'Configuration Details' : 'Service Identity'}
      </Typography>
      <Grid container spacing={2}>
        {isEditMode && (
          <>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Configuration Name"
                name="name"
                value={config.name || ''}
                onChange={handleConfigChange}
                margin="normal"
                required
                helperText="Name for this configuration"
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Description"
                name="description"
                value={config.description || ''}
                onChange={handleConfigChange}
                margin="normal"
                helperText="Optional description for this configuration"
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <FormControlLabel
                control={
                  <Switch
                    checked={config.is_public || false}
                    onChange={handleConfigChange}
                    name="is_public"
                  />
                }
                label="Make this configuration public (visible to all users)"
              />
            </Grid>
          </>
        )}
        <Grid item xs={12} md={6}>
          <TextField
            fullWidth
            label="Service ID"
            name="service_id"
            value={config.service_id}
            onChange={handleConfigChange}
            margin="normal"
            helperText="Unique identifier for this service instance"
          />
        </Grid>
      </Grid>
      
      <Box sx={{ mt: 3, display: 'flex', justifyContent: 'flex-end' }}>
        <Button
          variant="contained"
          startIcon={<SaveIcon />}
          onClick={handleSaveConfig}
        >
          Save Configuration
        </Button>
      </Box>
    </Paper>
  );
  
  // Render caching settings tab
  const renderCachingSettings = () => (
    <Paper sx={{ p: 3, mt: 2 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Typography variant="h6">
          Cache Configuration
        </Typography>
        <Button
          variant="outlined"
          color="error"
          startIcon={<ClearAllIcon />}
          onClick={handleClearCache}
          disabled={!config.enable_caching}
        >
          Clear Cache
        </Button>
      </Box>
      
      <Grid container spacing={2}>
        <Grid item xs={12} md={4}>
          <TextField
            fullWidth
            label="Similarity Threshold"
            type="number"
            inputProps={{ min: 0, max: 1, step: 0.01 }}
            value={config.config?.cache?.similarity_threshold || 0.92}
            onChange={(e) => handleNestedConfigChange('cache', 'similarity_threshold', parseFloat(e.target.value))}
            margin="normal"
            disabled={!config.enable_caching}
            helperText="Threshold for semantic similarity (0-1)"
          />
        </Grid>
        <Grid item xs={12} md={4}>
          <TextField
            fullWidth
            label="Max Entries"
            type="number"
            inputProps={{ min: 100, step: 100 }}
            value={config.config?.cache?.max_entries || 10000}
            onChange={(e) => handleNestedConfigChange('cache', 'max_entries', parseInt(e.target.value))}
            margin="normal"
            disabled={!config.enable_caching}
            helperText="Maximum number of cache entries"
          />
        </Grid>
        <Grid item xs={12} md={4}>
          <TextField
            fullWidth
            label="TTL (seconds)"
            type="number"
            inputProps={{ min: 60, step: 60 }}
            value={config.config?.cache?.ttl_seconds || 3600}
            onChange={(e) => handleNestedConfigChange('cache', 'ttl_seconds', parseInt(e.target.value))}
            margin="normal"
            disabled={!config.enable_caching}
            helperText="Time-to-live for cache entries in seconds"
          />
        </Grid>
      </Grid>
      
      {stats && stats.cache_stats && (
        <Box sx={{ mt: 3 }}>
          <Typography variant="h6" gutterBottom>
            Cache Statistics
          </Typography>
          <TableContainer component={Paper} variant="outlined">
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell>Metric</TableCell>
                  <TableCell align="right">Value</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                <TableRow>
                  <TableCell>Cache Hits</TableCell>
                  <TableCell align="right">{stats.cache_stats.hits || 0}</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell>Cache Misses</TableCell>
                  <TableCell align="right">{stats.cache_stats.misses || 0}</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell>Current Size</TableCell>
                  <TableCell align="right">{stats.cache_stats.size || 0}</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell>Max Size</TableCell>
                  <TableCell align="right">{stats.cache_stats.max_size || 0}</TableCell>
                </TableRow>
              </TableBody>
            </Table>
          </TableContainer>
        </Box>
      )}
      
      <Box sx={{ mt: 3, display: 'flex', justifyContent: 'flex-end' }}>
        <Button
          variant="contained"
          startIcon={<SaveIcon />}
          onClick={handleSaveConfig}
          disabled={!config.enable_caching}
        >
          Save Configuration
        </Button>
      </Box>
    </Paper>
  );
  
  // Render resilience settings tab
  const renderResilienceSettings = () => (
    <Paper sx={{ p: 3, mt: 2 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Typography variant="h6">
          Resilience Configuration
        </Typography>
        <Button
          variant="outlined"
          color="warning"
          startIcon={<RefreshIcon />}
          onClick={handleResetCircuitBreakers}
          disabled={!config.enable_resilience}
        >
          Reset Circuit Breakers
        </Button>
      </Box>
      
      {health && health.components && health.components.resilience && health.components.resilience.circuit_breakers && (
        <Box sx={{ mt: 3 }}>
          <Typography variant="h6" gutterBottom>
            Circuit Breakers
          </Typography>
          {Object.keys(health.components.resilience.circuit_breakers).length > 0 ? (
            <TableContainer component={Paper} variant="outlined">
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>Name</TableCell>
                    <TableCell>State</TableCell>
                    <TableCell>Failure Count</TableCell>
                    <TableCell>Last Failure</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {Object.entries(health.components.resilience.circuit_breakers).map(([name, data]) => (
                    <TableRow key={name}>
                      <TableCell>{name}</TableCell>
                      <TableCell>
                        <Chip
                          size="small"
                          color={data.state === 'open' ? 'error' : 'success'}
                          label={data.state}
                          icon={data.state === 'open' ? <ErrorIcon /> : <CheckIcon />}
                        />
                      </TableCell>
                      <TableCell>{data.failure_count}</TableCell>
                      <TableCell>
                        {data.last_failure_time ? new Date(data.last_failure_time * 1000).toLocaleString() : 'N/A'}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          ) : (
            <Alert severity="info">No active circuit breakers</Alert>
          )}
        </Box>
      )}
      
      <Box sx={{ mt: 3, display: 'flex', justifyContent: 'flex-end' }}>
        <Button
          variant="contained"
          startIcon={<SaveIcon />}
          onClick={handleSaveConfig}
          disabled={!config.enable_resilience}
        >
          Save Configuration
        </Button>
      </Box>
    </Paper>
  );
  
  // Render observability tab
  const renderObservabilityTab = () => (
    <Paper sx={{ p: 3, mt: 2 }}>
      <Typography variant="h6" gutterBottom>
        Metrics
      </Typography>
      
      {stats && stats.metrics && (
        <Box sx={{ mt: 2 }}>
          {Object.keys(stats.metrics).length > 0 ? (
            Object.entries(stats.metrics).map(([metricName, metricValues]) => (
              <Accordion key={metricName} sx={{ mb: 1 }}>
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                  <Typography>{metricName}</Typography>
                </AccordionSummary>
                <AccordionDetails>
                  <TableContainer component={Paper} variant="outlined">
                    <Table size="small">
                      <TableHead>
                        <TableRow>
                          <TableCell>Tags</TableCell>
                          <TableCell align="right">Value</TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {Object.entries(metricValues).map(([tags, value]) => (
                          <TableRow key={tags}>
                            <TableCell>{tags || 'default'}</TableCell>
                            <TableCell align="right">{value}</TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </TableContainer>
                </AccordionDetails>
              </Accordion>
            ))
          ) : (
            <Alert severity="info">No metrics recorded yet</Alert>
          )}
        </Box>
      )}
      
      <Box sx={{ mt: 3, display: 'flex', justifyContent: 'flex-end' }}>
        <Button
          variant="contained"
          startIcon={<SaveIcon />}
          onClick={handleSaveConfig}
          disabled={!config.enable_observability}
        >
          Save Configuration
        </Button>
      </Box>
    </Paper>
  );
  
  // Render events tab
  const renderEventsTab = () => (
    <Paper sx={{ p: 3, mt: 2 }}>
      <Typography variant="h6" gutterBottom>
        Recent Events
      </Typography>
      
      {stats && stats.recent_events && (
        <Box sx={{ mt: 2 }}>
          {stats.recent_events.length > 0 ? (
            <TableContainer component={Paper} variant="outlined">
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>Type</TableCell>
                    <TableCell>Payload</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {stats.recent_events.map((event, index) => (
                    <TableRow key={index}>
                      <TableCell>{event.type}</TableCell>
                      <TableCell>
                        <pre style={{ margin: 0, whiteSpace: 'pre-wrap' }}>
                          {JSON.stringify(event.payload, null, 2)}
                        </pre>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          ) : (
            <Alert severity="info">No recent events</Alert>
          )}
        </Box>
      )}
      
      <Box sx={{ mt: 3, display: 'flex', justifyContent: 'flex-end' }}>
        <Button
          variant="contained"
          startIcon={<SaveIcon />}
          onClick={handleSaveConfig}
          disabled={!config.enable_events}
        >
          Save Configuration
        </Button>
      </Box>
    </Paper>
  );
  
  // Render progress tracking tab
  const renderProgressTrackingTab = () => (
    <Paper sx={{ p: 3, mt: 2 }}>
      <Typography variant="h6" gutterBottom>
        Active Operations
      </Typography>
      
      {stats && stats.active_operations && (
        <Box sx={{ mt: 2 }}>
          {Object.keys(stats.active_operations).length > 0 ? (
            Object.entries(stats.active_operations).map(([operationId, operation]) => (
              <Accordion key={operationId} sx={{ mb: 1 }}>
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                  <Box sx={{ display: 'flex', alignItems: 'center', width: '100%' }}>
                    <Typography sx={{ flexGrow: 1 }}>
                      {operation.operation_type}: {operation.message}
                    </Typography>
                    <Box sx={{ ml: 2, display: 'flex', alignItems: 'center' }}>
                      <Typography variant="body2" color="text.secondary" sx={{ mr: 1 }}>
                        {Math.round((operation.current_step / operation.total_steps) * 100)}%
                      </Typography>
                      <CircularProgress
                        size={24}
                        variant="determinate"
                        value={Math.round((operation.current_step / operation.total_steps) * 100)}
                      />
                    </Box>
                  </Box>
                </AccordionSummary>
                <AccordionDetails>
                  <TableContainer component={Paper} variant="outlined">
                    <Table size="small">
                      <TableHead>
                        <TableRow>
                          <TableCell>Step</TableCell>
                          <TableCell>Message</TableCell>
                          <TableCell>Time</TableCell>
                          <TableCell align="right">Progress</TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {operation.updates.map((update, index) => (
                          <TableRow key={index}>
                            <TableCell>{update.step}</TableCell>
                            <TableCell>{update.message}</TableCell>
                            <TableCell>
                              {new Date(update.time * 1000).toLocaleTimeString()}
                            </TableCell>
                            <TableCell align="right">{update.progress_pct}%</TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </TableContainer>
                </AccordionDetails>
              </Accordion>
            ))
          ) : (
            <Alert severity="info">No active operations</Alert>
          )}
        </Box>
      )}
      
      <Box sx={{ mt: 3, display: 'flex', justifyContent: 'flex-end' }}>
        <Button
          variant="contained"
          startIcon={<SaveIcon />}
          onClick={handleSaveConfig}
          disabled={!config.enable_progress_tracking}
        >
          Save Configuration
        </Button>
      </Box>
    </Paper>
  );
  
  // Define breadcrumbs for PageLayout
  const breadcrumbs = isEditMode
    ? [
        { label: 'LLM Management', path: '/llm/dashboard' },
        { label: 'Settings', path: '/llm/settings/gateway' },
        { label: 'Service Configurations', path: '/llm/settings/service-config/configurations' },
        { label: 'Edit Configuration' }
      ]
    : [
        { label: 'LLM Management', path: '/llm/dashboard' },
        { label: 'Settings', path: '/llm/settings/gateway' },
        { label: 'Service Configuration' }
      ];

  // Get the current tab name for the title
  const getTabTitle = () => {
    const tabNames = ['General', 'Service Caching', 'Resilience', 'Service Metrics', 'Events', 'Progress Tracking'];
    return tabNames[tabValue] || 'General';
  };

  // Get page title
  const getPageTitle = () => {
    if (isEditMode) {
      return `Edit Configuration - ${config.name || 'Loading...'}`;
    } else {
      return `LLM Service Configuration - ${getTabTitle()}`;
    }
  };

  return (
    <PageLayout
      title={getPageTitle()}
      breadcrumbs={breadcrumbs}
      action={
        <Button
          variant="outlined"
          startIcon={refreshing ? <CircularProgress size={16} /> : <RefreshIcon />}
          onClick={handleRefresh}
          disabled={refreshing}
        >
          Refresh
        </Button>
      }
    >
      <Box>
        {health && (
          <Paper sx={{ p: 2, mb: 3 }}>
            <Box sx={{ display: 'flex', alignItems: 'center' }}>
              <Typography variant="h6" sx={{ flexGrow: 1 }}>
                Service Status
              </Typography>
              <Chip
                color={health.status === 'operational' ? 'success' : 'error'}
                icon={health.status === 'operational' ? <CheckIcon /> : <ErrorIcon />}
                label={health.status}
              />
            </Box>
            
            <Grid container spacing={2} sx={{ mt: 1 }}>
              {health.components && Object.entries(health.components).map(([name, component]) => (
                <Grid item xs={6} sm={4} md={2} key={name}>
                  <Card variant="outlined">
                    <CardContent sx={{ p: 1, '&:last-child': { pb: 1 } }}>
                      <Typography variant="body2" color="text.secondary">
                        {name}
                      </Typography>
                      <Box sx={{ display: 'flex', alignItems: 'center', mt: 1 }}>
                        <Switch
                          size="small"
                          checked={component.enabled}
                          disabled
                        />
                        <Chip
                          size="small"
                          sx={{ ml: 1 }}
                          color={component.status === 'operational' ? 'success' : 'error'}
                          label={component.status}
                        />
                      </Box>
                    </CardContent>
                  </Card>
                </Grid>
              ))}
            </Grid>
          </Paper>
        )}
        
        {loading ? (
          <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
            <CircularProgress />
          </Box>
        ) : (
          <>
            <Tabs value={tabValue} onChange={handleTabChange} sx={{ borderBottom: 1, borderColor: 'divider' }}>
              <Tab label="General" />
              <Tab label="Service Caching" disabled={!config.enable_caching} />
              <Tab label="Resilience" disabled={!config.enable_resilience} />
              <Tab label="Service Metrics" disabled={!config.enable_observability} />
              <Tab label="Events" disabled={!config.enable_events} />
              <Tab label="Progress Tracking" disabled={!config.enable_progress_tracking} />
            </Tabs>
            
            {tabValue === 0 && renderGeneralSettings()}
            {tabValue === 1 && renderCachingSettings()}
            {tabValue === 2 && renderResilienceSettings()}
            {tabValue === 3 && renderObservabilityTab()}
            {tabValue === 4 && renderEventsTab()}
            {tabValue === 5 && renderProgressTrackingTab()}
          </>
        )}
      </Box>
    </PageLayout>
  );
};

export default ServiceConfigPage;
