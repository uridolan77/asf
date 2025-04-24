import React, { useState, useEffect } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { 
  Box, 
  Typography, 
  Paper,
  Button,
  Alert,
  CircularProgress,
  TextField,
  FormControlLabel,
  Switch,
  Divider,
  Grid,
  Tabs,
  Tab,
  Accordion,
  AccordionSummary,
  AccordionDetails
} from '@mui/material';
import {
  Save as SaveIcon,
  ExpandMore as ExpandMoreIcon,
  PlayArrow as ApplyIcon,
  ArrowBack as ArrowBackIcon
} from '@mui/icons-material';

// Import the PageLayout component that includes the sidebar
import PageLayout from '../../../components/Layout/PageLayout';
import apiService from '../../../services/api';
import { useNotification } from '../../../context/NotificationContext';

/**
 * Configuration Edit Page
 * Allows editing a specific service configuration
 */
const ConfigurationEditPage = () => {
  const { configId } = useParams();
  const navigate = useNavigate();
  const { showSuccess, showError } = useNotification();
  
  // State
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [applying, setApplying] = useState(false);
  const [config, setConfig] = useState({
    service_id: 'enhanced_llm_service',
    name: '',
    description: '',
    enable_caching: true,
    enable_resilience: true,
    enable_observability: true,
    enable_events: true,
    enable_progress_tracking: true,
    is_public: false,
    config: {
      caching: {
        similarity_threshold: 0.92,
        max_entries: 10000,
        ttl_seconds: 3600,
        persistence_type: 'disk',
        persistence_config: null
      },
      resilience: {
        max_retries: 3,
        retry_delay: 1.0,
        backoff_factor: 2.0,
        circuit_breaker_failure_threshold: 5,
        circuit_breaker_reset_timeout: 30,
        timeout_seconds: 30.0
      },
      observability: {
        metrics_enabled: true,
        tracing_enabled: true,
        logging_level: 'INFO',
        export_metrics: false,
        metrics_export_url: null
      },
      events: {
        max_event_history: 100,
        publish_to_external: false,
        external_event_url: null,
        event_types_filter: null
      },
      progress_tracking: {
        max_active_operations: 100,
        operation_ttl_seconds: 3600,
        publish_updates: true
      }
    }
  });
  const [tabValue, setTabValue] = useState(0);
  
  // Load configuration on mount
  useEffect(() => {
    loadConfiguration();
  }, [configId]);
  
  // Load configuration from API
  const loadConfiguration = async () => {
    setLoading(true);
    try {
      const result = await apiService.llm.getServiceConfigurationById(configId);
      if (result.success) {
        setConfig(result.data);
      } else {
        showError(`Failed to load configuration: ${result.error}`);
        navigate('/llm/settings/service-config/configurations');
      }
    } catch (error) {
      console.error('Error loading configuration:', error);
      showError(`Error loading configuration: ${error.message}`);
      navigate('/llm/settings/service-config/configurations');
    } finally {
      setLoading(false);
    }
  };
  
  // Handle tab change
  const handleTabChange = (event, newValue) => {
    setTabValue(newValue);
  };
  
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
    setSaving(true);
    try {
      const result = await apiService.llm.updateServiceConfiguration(configId, config);
      
      if (result.success) {
        showSuccess('Configuration updated successfully');
        await loadConfiguration();
      } else {
        showError(`Failed to update configuration: ${result.error}`);
      }
    } catch (error) {
      console.error('Error updating configuration:', error);
      showError(`Error updating configuration: ${error.message}`);
    } finally {
      setSaving(false);
    }
  };
  
  // Apply configuration
  const handleApplyConfig = async () => {
    setApplying(true);
    try {
      const result = await apiService.llm.applyServiceConfiguration(configId);
      
      if (result.success) {
        showSuccess('Configuration applied successfully');
      } else {
        showError(`Failed to apply configuration: ${result.error}`);
      }
    } catch (error) {
      console.error('Error applying configuration:', error);
      showError(`Error applying configuration: ${error.message}`);
    } finally {
      setApplying(false);
    }
  };
  
  // Go back to configurations list
  const handleGoBack = () => {
    navigate('/llm/settings/service-config/configurations');
  };
  
  // Render general settings tab
  const renderGeneralSettings = () => (
    <Paper sx={{ p: 3, mt: 2 }}>
      <Typography variant="h6" gutterBottom>
        Configuration Details
      </Typography>
      <Grid container spacing={2}>
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
            label="Service ID"
            name="service_id"
            value={config.service_id}
            onChange={handleConfigChange}
            margin="normal"
            helperText="Unique identifier for this service instance"
          />
        </Grid>
        <Grid item xs={12}>
          <TextField
            fullWidth
            label="Description"
            name="description"
            value={config.description || ''}
            onChange={handleConfigChange}
            margin="normal"
            multiline
            rows={2}
            helperText="Optional description for this configuration"
          />
        </Grid>
        <Grid item xs={12}>
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
      </Grid>
      
      <Divider sx={{ my: 3 }} />
      
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
    </Paper>
  );
  
  // Render caching settings tab
  const renderCachingSettings = () => (
    <Paper sx={{ p: 3, mt: 2 }}>
      <Typography variant="h6" gutterBottom>
        Cache Configuration
      </Typography>
      
      <Grid container spacing={2}>
        <Grid item xs={12} md={4}>
          <TextField
            fullWidth
            label="Similarity Threshold"
            type="number"
            inputProps={{ min: 0, max: 1, step: 0.01 }}
            value={config.config?.caching?.similarity_threshold || 0.92}
            onChange={(e) => handleNestedConfigChange('caching', 'similarity_threshold', parseFloat(e.target.value))}
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
            value={config.config?.caching?.max_entries || 10000}
            onChange={(e) => handleNestedConfigChange('caching', 'max_entries', parseInt(e.target.value))}
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
            value={config.config?.caching?.ttl_seconds || 3600}
            onChange={(e) => handleNestedConfigChange('caching', 'ttl_seconds', parseInt(e.target.value))}
            margin="normal"
            disabled={!config.enable_caching}
            helperText="Time-to-live for cache entries in seconds"
          />
        </Grid>
        <Grid item xs={12} md={4}>
          <TextField
            fullWidth
            label="Persistence Type"
            select
            SelectProps={{ native: true }}
            value={config.config?.caching?.persistence_type || 'disk'}
            onChange={(e) => handleNestedConfigChange('caching', 'persistence_type', e.target.value)}
            margin="normal"
            disabled={!config.enable_caching}
            helperText="Type of persistence for cache"
          >
            <option value="disk">Disk</option>
            <option value="memory">Memory</option>
            <option value="redis">Redis</option>
          </TextField>
        </Grid>
      </Grid>
    </Paper>
  );
  
  // Render resilience settings tab
  const renderResilienceSettings = () => (
    <Paper sx={{ p: 3, mt: 2 }}>
      <Typography variant="h6" gutterBottom>
        Resilience Configuration
      </Typography>
      
      <Grid container spacing={2}>
        <Grid item xs={12} md={4}>
          <TextField
            fullWidth
            label="Max Retries"
            type="number"
            inputProps={{ min: 0, step: 1 }}
            value={config.config?.resilience?.max_retries || 3}
            onChange={(e) => handleNestedConfigChange('resilience', 'max_retries', parseInt(e.target.value))}
            margin="normal"
            disabled={!config.enable_resilience}
            helperText="Maximum number of retry attempts"
          />
        </Grid>
        <Grid item xs={12} md={4}>
          <TextField
            fullWidth
            label="Retry Delay (seconds)"
            type="number"
            inputProps={{ min: 0.1, step: 0.1 }}
            value={config.config?.resilience?.retry_delay || 1.0}
            onChange={(e) => handleNestedConfigChange('resilience', 'retry_delay', parseFloat(e.target.value))}
            margin="normal"
            disabled={!config.enable_resilience}
            helperText="Delay between retry attempts"
          />
        </Grid>
        <Grid item xs={12} md={4}>
          <TextField
            fullWidth
            label="Backoff Factor"
            type="number"
            inputProps={{ min: 1, step: 0.1 }}
            value={config.config?.resilience?.backoff_factor || 2.0}
            onChange={(e) => handleNestedConfigChange('resilience', 'backoff_factor', parseFloat(e.target.value))}
            margin="normal"
            disabled={!config.enable_resilience}
            helperText="Exponential backoff factor"
          />
        </Grid>
        <Grid item xs={12} md={4}>
          <TextField
            fullWidth
            label="Circuit Breaker Failure Threshold"
            type="number"
            inputProps={{ min: 1, step: 1 }}
            value={config.config?.resilience?.circuit_breaker_failure_threshold || 5}
            onChange={(e) => handleNestedConfigChange('resilience', 'circuit_breaker_failure_threshold', parseInt(e.target.value))}
            margin="normal"
            disabled={!config.enable_resilience}
            helperText="Number of failures before circuit opens"
          />
        </Grid>
        <Grid item xs={12} md={4}>
          <TextField
            fullWidth
            label="Circuit Breaker Reset Timeout (seconds)"
            type="number"
            inputProps={{ min: 1, step: 1 }}
            value={config.config?.resilience?.circuit_breaker_reset_timeout || 30}
            onChange={(e) => handleNestedConfigChange('resilience', 'circuit_breaker_reset_timeout', parseInt(e.target.value))}
            margin="normal"
            disabled={!config.enable_resilience}
            helperText="Time before circuit half-opens"
          />
        </Grid>
        <Grid item xs={12} md={4}>
          <TextField
            fullWidth
            label="Timeout (seconds)"
            type="number"
            inputProps={{ min: 1, step: 1 }}
            value={config.config?.resilience?.timeout_seconds || 30.0}
            onChange={(e) => handleNestedConfigChange('resilience', 'timeout_seconds', parseFloat(e.target.value))}
            margin="normal"
            disabled={!config.enable_resilience}
            helperText="Request timeout in seconds"
          />
        </Grid>
      </Grid>
    </Paper>
  );
  
  // Render observability settings tab
  const renderObservabilitySettings = () => (
    <Paper sx={{ p: 3, mt: 2 }}>
      <Typography variant="h6" gutterBottom>
        Observability Configuration
      </Typography>
      
      <Grid container spacing={2}>
        <Grid item xs={12} md={4}>
          <FormControlLabel
            control={
              <Switch
                checked={config.config?.observability?.metrics_enabled || true}
                onChange={(e) => handleNestedConfigChange('observability', 'metrics_enabled', e.target.checked)}
                disabled={!config.enable_observability}
              />
            }
            label="Enable Metrics"
          />
        </Grid>
        <Grid item xs={12} md={4}>
          <FormControlLabel
            control={
              <Switch
                checked={config.config?.observability?.tracing_enabled || true}
                onChange={(e) => handleNestedConfigChange('observability', 'tracing_enabled', e.target.checked)}
                disabled={!config.enable_observability}
              />
            }
            label="Enable Tracing"
          />
        </Grid>
        <Grid item xs={12} md={4}>
          <TextField
            fullWidth
            label="Logging Level"
            select
            SelectProps={{ native: true }}
            value={config.config?.observability?.logging_level || 'INFO'}
            onChange={(e) => handleNestedConfigChange('observability', 'logging_level', e.target.value)}
            margin="normal"
            disabled={!config.enable_observability}
          >
            <option value="DEBUG">DEBUG</option>
            <option value="INFO">INFO</option>
            <option value="WARNING">WARNING</option>
            <option value="ERROR">ERROR</option>
            <option value="CRITICAL">CRITICAL</option>
          </TextField>
        </Grid>
        <Grid item xs={12} md={6}>
          <FormControlLabel
            control={
              <Switch
                checked={config.config?.observability?.export_metrics || false}
                onChange={(e) => handleNestedConfigChange('observability', 'export_metrics', e.target.checked)}
                disabled={!config.enable_observability}
              />
            }
            label="Export Metrics"
          />
        </Grid>
        <Grid item xs={12} md={6}>
          <TextField
            fullWidth
            label="Metrics Export URL"
            value={config.config?.observability?.metrics_export_url || ''}
            onChange={(e) => handleNestedConfigChange('observability', 'metrics_export_url', e.target.value)}
            margin="normal"
            disabled={!config.enable_observability || !config.config?.observability?.export_metrics}
            helperText="URL to export metrics to"
          />
        </Grid>
      </Grid>
    </Paper>
  );
  
  // Render events settings tab
  const renderEventsSettings = () => (
    <Paper sx={{ p: 3, mt: 2 }}>
      <Typography variant="h6" gutterBottom>
        Events Configuration
      </Typography>
      
      <Grid container spacing={2}>
        <Grid item xs={12} md={6}>
          <TextField
            fullWidth
            label="Max Event History"
            type="number"
            inputProps={{ min: 10, step: 10 }}
            value={config.config?.events?.max_event_history || 100}
            onChange={(e) => handleNestedConfigChange('events', 'max_event_history', parseInt(e.target.value))}
            margin="normal"
            disabled={!config.enable_events}
            helperText="Maximum number of events to keep in history"
          />
        </Grid>
        <Grid item xs={12} md={6}>
          <FormControlLabel
            control={
              <Switch
                checked={config.config?.events?.publish_to_external || false}
                onChange={(e) => handleNestedConfigChange('events', 'publish_to_external', e.target.checked)}
                disabled={!config.enable_events}
              />
            }
            label="Publish to External Service"
          />
        </Grid>
        <Grid item xs={12}>
          <TextField
            fullWidth
            label="External Event URL"
            value={config.config?.events?.external_event_url || ''}
            onChange={(e) => handleNestedConfigChange('events', 'external_event_url', e.target.value)}
            margin="normal"
            disabled={!config.enable_events || !config.config?.events?.publish_to_external}
            helperText="URL to publish events to"
          />
        </Grid>
      </Grid>
    </Paper>
  );
  
  // Render progress tracking settings tab
  const renderProgressTrackingSettings = () => (
    <Paper sx={{ p: 3, mt: 2 }}>
      <Typography variant="h6" gutterBottom>
        Progress Tracking Configuration
      </Typography>
      
      <Grid container spacing={2}>
        <Grid item xs={12} md={6}>
          <TextField
            fullWidth
            label="Max Active Operations"
            type="number"
            inputProps={{ min: 10, step: 10 }}
            value={config.config?.progress_tracking?.max_active_operations || 100}
            onChange={(e) => handleNestedConfigChange('progress_tracking', 'max_active_operations', parseInt(e.target.value))}
            margin="normal"
            disabled={!config.enable_progress_tracking}
            helperText="Maximum number of active operations to track"
          />
        </Grid>
        <Grid item xs={12} md={6}>
          <TextField
            fullWidth
            label="Operation TTL (seconds)"
            type="number"
            inputProps={{ min: 60, step: 60 }}
            value={config.config?.progress_tracking?.operation_ttl_seconds || 3600}
            onChange={(e) => handleNestedConfigChange('progress_tracking', 'operation_ttl_seconds', parseInt(e.target.value))}
            margin="normal"
            disabled={!config.enable_progress_tracking}
            helperText="Time-to-live for completed operations"
          />
        </Grid>
        <Grid item xs={12}>
          <FormControlLabel
            control={
              <Switch
                checked={config.config?.progress_tracking?.publish_updates || true}
                onChange={(e) => handleNestedConfigChange('progress_tracking', 'publish_updates', e.target.checked)}
                disabled={!config.enable_progress_tracking}
              />
            }
            label="Publish Progress Updates"
          />
        </Grid>
      </Grid>
    </Paper>
  );
  
  // Define breadcrumbs for PageLayout
  const breadcrumbs = [
    { label: 'LLM Management', path: '/llm/dashboard' },
    { label: 'Settings', path: '/llm/settings/gateway' },
    { label: 'Service Configurations', path: '/llm/settings/service-config/configurations' },
    { label: 'Edit Configuration' }
  ];

  return (
    <PageLayout
      title={`Edit Configuration - ${config.name || 'Loading...'}`}
      breadcrumbs={breadcrumbs}
      action={
        <Box sx={{ display: 'flex', gap: 1 }}>
          <Button
            variant="outlined"
            startIcon={<ArrowBackIcon />}
            onClick={handleGoBack}
          >
            Back
          </Button>
          <Button
            variant="contained"
            color="primary"
            startIcon={<ApplyIcon />}
            onClick={handleApplyConfig}
            disabled={applying || loading}
          >
            {applying ? <CircularProgress size={24} /> : 'Apply Configuration'}
          </Button>
        </Box>
      }
    >
      {loading ? (
        <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
          <CircularProgress />
        </Box>
      ) : (
        <Box>
          <Tabs value={tabValue} onChange={handleTabChange} sx={{ borderBottom: 1, borderColor: 'divider' }}>
            <Tab label="General" />
            <Tab label="Caching" disabled={!config.enable_caching} />
            <Tab label="Resilience" disabled={!config.enable_resilience} />
            <Tab label="Observability" disabled={!config.enable_observability} />
            <Tab label="Events" disabled={!config.enable_events} />
            <Tab label="Progress Tracking" disabled={!config.enable_progress_tracking} />
          </Tabs>
          
          {tabValue === 0 && renderGeneralSettings()}
          {tabValue === 1 && renderCachingSettings()}
          {tabValue === 2 && renderResilienceSettings()}
          {tabValue === 3 && renderObservabilitySettings()}
          {tabValue === 4 && renderEventsSettings()}
          {tabValue === 5 && renderProgressTrackingSettings()}
          
          <Box sx={{ mt: 3, display: 'flex', justifyContent: 'flex-end' }}>
            <Button
              variant="contained"
              startIcon={saving ? <CircularProgress size={24} /> : <SaveIcon />}
              onClick={handleSaveConfig}
              disabled={saving}
            >
              Save Configuration
            </Button>
          </Box>
        </Box>
      )}
    </PageLayout>
  );
};

export default ConfigurationEditPage;
