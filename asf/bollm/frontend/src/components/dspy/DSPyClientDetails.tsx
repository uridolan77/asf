import React, { useState } from 'react';
import {
  Paper,
  Box,
  Typography,
  Button,
  Grid,
  Divider,
  Chip,
  LinearProgress,
  Alert,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  TextField,
  FormControlLabel,
  Switch,
  MenuItem,
  Tooltip,
  IconButton
} from '@mui/material';
import {
  ExpandMore as ExpandMoreIcon,
  Refresh as RefreshIcon,
  Save as SaveIcon,
  CheckCircle as CheckCircleIcon,
  Error as ErrorIcon,
  Settings as SettingsIcon
} from '@mui/icons-material';
import ClientService from '../../services/ClientService';
import { formatDateTime } from '../../utils/formatters';
import UsageChart from '../common/UsageChart';

interface DSPyClientDetailsProps {
  client: any;
  onRefresh: () => void;
}

const DSPyClientDetails: React.FC<DSPyClientDetailsProps> = ({ client, onRefresh }) => {
  const [testing, setTesting] = useState<boolean>(false);
  const [testResult, setTestResult] = useState<any>(null);
  const [editingConfig, setEditingConfig] = useState<boolean>(false);
  const [configData, setConfigData] = useState<any>(client.config || {});
  const [saving, setSaving] = useState<boolean>(false);
  const [usageStats, setUsageStats] = useState<any[]>([]);
  const [usagePeriod, setUsagePeriod] = useState<number>(30); // days
  const [loadingStats, setLoadingStats] = useState<boolean>(false);
  const [expanded, setExpanded] = useState<string | false>('panel-info');

  const handleExpandChange = (panel: string) => (event: React.SyntheticEvent, isExpanded: boolean) => {
    setExpanded(isExpanded ? panel : false);
    if (panel === 'panel-usage' && isExpanded && usageStats.length === 0) {
      loadUsageStats();
    }
  };

  const handleTestConnection = async () => {
    try {
      setTesting(true);
      setTestResult(null);
      
      const result = await ClientService.testDSPyClientConnection(client.client_id);
      setTestResult(result);
    } catch (error) {
      console.error('Error testing connection:', error);
      setTestResult({
        success: false,
        message: 'Failed to test connection: ' + (error as any).message
      });
    } finally {
      setTesting(false);
    }
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value, type, checked } = e.target;
    setConfigData({
      ...configData,
      [name]: type === 'checkbox' ? checked : value
    });
  };

  const handleSaveConfig = async () => {
    try {
      setSaving(true);
      await ClientService.updateDSPyClientConfig(client.client_id, configData);
      setEditingConfig(false);
      onRefresh();
    } catch (error) {
      console.error('Error saving config:', error);
      alert('Failed to save configuration: ' + (error as any).message);
    } finally {
      setSaving(false);
    }
  };

  const loadUsageStats = async () => {
    try {
      setLoadingStats(true);
      const stats = await ClientService.getDSPyClientUsage(client.client_id, usagePeriod);
      setUsageStats(stats);
    } catch (error) {
      console.error('Error loading usage stats:', error);
    } finally {
      setLoadingStats(false);
    }
  };

  const toggleEditConfig = () => {
    if (editingConfig) {
      // Discard changes
      setConfigData(client.config || {});
    }
    setEditingConfig(!editingConfig);
  };

  return (
    <Box sx={{ mb: 4 }}>
      <Paper elevation={2} sx={{ p: 3, mb: 3 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 2 }}>
          <Box>
            <Typography variant="h5" gutterBottom>
              {client.name}
            </Typography>
            <Typography variant="body1" color="textSecondary" paragraph>
              {client.description || 'No description provided'}
            </Typography>
          </Box>
          <Button
            variant="outlined"
            color={testResult?.success ? "success" : "primary"}
            onClick={handleTestConnection}
            startIcon={testing ? null : <RefreshIcon />}
            disabled={testing}
          >
            {testing ? 'Testing...' : 'Test Connection'}
          </Button>
        </Box>

        {testing && <LinearProgress sx={{ mb: 2 }} />}

        {testResult && (
          <Alert 
            severity={testResult.success ? "success" : "error"}
            sx={{ mb: 2 }}
            icon={testResult.success ? <CheckCircleIcon /> : <ErrorIcon />}
          >
            {testResult.message}
            {testResult.response_time !== undefined && (
              <Box component="span" sx={{ ml: 1 }}>
                (Response time: {testResult.response_time.toFixed(2)}s)
              </Box>
            )}
          </Alert>
        )}

        <Divider sx={{ mb: 2 }} />

        <Accordion 
          expanded={expanded === 'panel-info'} 
          onChange={handleExpandChange('panel-info')}
        >
          <AccordionSummary expandIcon={<ExpandMoreIcon />}>
            <Typography variant="h6">Client Information</Typography>
          </AccordionSummary>
          <AccordionDetails>
            <Grid container spacing={2}>
              <Grid item xs={12} md={6}>
                <Typography variant="subtitle2">Client ID</Typography>
                <Typography variant="body1">{client.client_id}</Typography>
              </Grid>
              <Grid item xs={12} md={6}>
                <Typography variant="subtitle2">Base URL</Typography>
                <Typography variant="body1">{client.base_url}</Typography>
              </Grid>
              <Grid item xs={12} md={6}>
                <Typography variant="subtitle2">Created At</Typography>
                <Typography variant="body1">{formatDateTime(client.created_at)}</Typography>
              </Grid>
              <Grid item xs={12} md={6}>
                <Typography variant="subtitle2">Updated At</Typography>
                <Typography variant="body1">{formatDateTime(client.updated_at)}</Typography>
              </Grid>
              <Grid item xs={12}>
                <Typography variant="subtitle2">Status</Typography>
                <Box sx={{ display: 'flex', alignItems: 'center' }}>
                  <Chip 
                    label={client.status?.status || 'UNKNOWN'} 
                    color={
                      client.status?.status === 'CONNECTED' ? 'success' :
                      client.status?.status === 'ERROR' ? 'error' :
                      'default'
                    } 
                    size="small"
                    sx={{ mr: 1 }}
                  />
                  {client.status?.last_checked && (
                    <Typography variant="body2">
                      Last checked: {formatDateTime(client.status.last_checked)}
                    </Typography>
                  )}
                </Box>
                {client.status?.error_message && (
                  <Typography variant="body2" color="error" sx={{ mt: 1 }}>
                    Error: {client.status.error_message}
                  </Typography>
                )}
              </Grid>
            </Grid>
          </AccordionDetails>
        </Accordion>

        <Accordion 
          expanded={expanded === 'panel-config'} 
          onChange={handleExpandChange('panel-config')}
        >
          <AccordionSummary expandIcon={<ExpandMoreIcon />}>
            <Box sx={{ display: 'flex', width: '100%', justifyContent: 'space-between', alignItems: 'center', pr: 2 }}>
              <Typography variant="h6">Configuration</Typography>
              {expanded === 'panel-config' && (
                <Tooltip title={editingConfig ? "Cancel" : "Edit"}>
                  <IconButton 
                    onClick={(e) => { 
                      e.stopPropagation(); 
                      toggleEditConfig(); 
                    }}
                    size="small"
                  >
                    <SettingsIcon />
                  </IconButton>
                </Tooltip>
              )}
            </Box>
          </AccordionSummary>
          <AccordionDetails>
            {!client.config && !editingConfig ? (
              <Alert severity="info">
                No configuration has been set for this client. Edit to add configuration.
              </Alert>
            ) : editingConfig ? (
              <Box>
                <Grid container spacing={2}>
                  <Grid item xs={12}>
                    <Typography variant="subtitle1" gutterBottom>LLM Provider Settings</Typography>
                  </Grid>
                  
                  <Grid item xs={12} md={6}>
                    <TextField
                      name="provider"
                      label="Provider"
                      select
                      fullWidth
                      value={configData.provider || 'openai'}
                      onChange={handleInputChange}
                    >
                      <MenuItem value="openai">OpenAI</MenuItem>
                      <MenuItem value="anthropic">Anthropic</MenuItem>
                      <MenuItem value="azure">Azure OpenAI</MenuItem>
                      <MenuItem value="huggingface">Hugging Face</MenuItem>
                    </TextField>
                  </Grid>
                  
                  <Grid item xs={12} md={6}>
                    <TextField
                      name="default_model"
                      label="Default Model"
                      fullWidth
                      value={configData.default_model || 'gpt-4'}
                      onChange={handleInputChange}
                    />
                  </Grid>
                  
                  <Grid item xs={12} md={6}>
                    <TextField
                      name="api_key"
                      label="API Key"
                      fullWidth
                      type="password"
                      value={configData.api_key || ''}
                      onChange={handleInputChange}
                    />
                  </Grid>
                  
                  <Grid item xs={12} md={6}>
                    <TextField
                      name="organization_id"
                      label="Organization ID"
                      fullWidth
                      value={configData.organization_id || ''}
                      onChange={handleInputChange}
                    />
                  </Grid>

                  <Grid item xs={12}>
                    <Divider sx={{ my: 1 }} />
                    <Typography variant="subtitle1" gutterBottom>Azure Settings</Typography>
                  </Grid>
                  
                  <Grid item xs={12} md={6}>
                    <TextField
                      name="azure_endpoint"
                      label="Azure Endpoint"
                      fullWidth
                      value={configData.azure_endpoint || ''}
                      onChange={handleInputChange}
                    />
                  </Grid>
                  
                  <Grid item xs={12} md={6}>
                    <TextField
                      name="azure_deployment_name"
                      label="Azure Deployment Name"
                      fullWidth
                      value={configData.azure_deployment_name || ''}
                      onChange={handleInputChange}
                    />
                  </Grid>
                  
                  <Grid item xs={12} md={6}>
                    <TextField
                      name="azure_api_version"
                      label="Azure API Version"
                      fullWidth
                      value={configData.azure_api_version || '2023-05-15'}
                      onChange={handleInputChange}
                    />
                  </Grid>

                  <Grid item xs={12}>
                    <Divider sx={{ my: 1 }} />
                    <Typography variant="subtitle1" gutterBottom>Model Parameters</Typography>
                  </Grid>
                  
                  <Grid item xs={12} md={4}>
                    <TextField
                      name="max_tokens"
                      label="Max Tokens"
                      fullWidth
                      type="number"
                      value={configData.max_tokens || 1000}
                      onChange={handleInputChange}
                    />
                  </Grid>
                  
                  <Grid item xs={12} md={4}>
                    <TextField
                      name="temperature"
                      label="Temperature"
                      fullWidth
                      type="number"
                      inputProps={{ step: 0.1, min: 0, max: 2 }}
                      value={configData.temperature || 0.7}
                      onChange={handleInputChange}
                    />
                  </Grid>
                  
                  <Grid item xs={12} md={4}>
                    <TextField
                      name="top_p"
                      label="Top P"
                      fullWidth
                      type="number"
                      inputProps={{ step: 0.1, min: 0, max: 1 }}
                      value={configData.top_p || 1.0}
                      onChange={handleInputChange}
                    />
                  </Grid>

                  <Grid item xs={12}>
                    <Divider sx={{ my: 1 }} />
                    <Typography variant="subtitle1" gutterBottom>Caching Settings</Typography>
                  </Grid>
                  
                  <Grid item xs={12} md={6}>
                    <TextField
                      name="cache_backend"
                      label="Cache Backend"
                      select
                      fullWidth
                      value={configData.cache_backend || 'disk'}
                      onChange={handleInputChange}
                    >
                      <MenuItem value="disk">Disk</MenuItem>
                      <MenuItem value="redis">Redis</MenuItem>
                      <MenuItem value="null">Null (No Cache)</MenuItem>
                    </TextField>
                  </Grid>
                  
                  <Grid item xs={12} md={6}>
                    <TextField
                      name="cache_ttl"
                      label="Cache TTL (seconds)"
                      fullWidth
                      type="number"
                      value={configData.cache_ttl || 3600}
                      onChange={handleInputChange}
                    />
                  </Grid>
                  
                  {configData.cache_backend === 'disk' && (
                    <Grid item xs={12}>
                      <TextField
                        name="cache_directory"
                        label="Cache Directory"
                        fullWidth
                        value={configData.cache_directory || '.dspy_cache'}
                        onChange={handleInputChange}
                      />
                    </Grid>
                  )}
                  
                  {configData.cache_backend === 'redis' && (
                    <>
                      <Grid item xs={12} md={6}>
                        <TextField
                          name="redis_url"
                          label="Redis URL"
                          fullWidth
                          value={configData.redis_url || 'redis://localhost:6379/0'}
                          onChange={handleInputChange}
                        />
                      </Grid>
                      
                      <Grid item xs={12} md={6}>
                        <TextField
                          name="redis_password"
                          label="Redis Password"
                          fullWidth
                          type="password"
                          value={configData.redis_password || ''}
                          onChange={handleInputChange}
                        />
                      </Grid>
                      
                      <Grid item xs={12} md={6}>
                        <TextField
                          name="redis_pool_size"
                          label="Redis Pool Size"
                          fullWidth
                          type="number"
                          value={configData.redis_pool_size || 10}
                          onChange={handleInputChange}
                        />
                      </Grid>
                    </>
                  )}

                  <Grid item xs={12}>
                    <Divider sx={{ my: 1 }} />
                    <Typography variant="subtitle1" gutterBottom>Circuit Breaker Settings</Typography>
                  </Grid>
                  
                  <Grid item xs={12} md={4}>
                    <TextField
                      name="circuit_breaker_failure_threshold"
                      label="Failure Threshold"
                      fullWidth
                      type="number"
                      value={configData.circuit_breaker_failure_threshold || 5}
                      onChange={handleInputChange}
                      helperText="Number of failures before opening circuit"
                    />
                  </Grid>
                  
                  <Grid item xs={12} md={4}>
                    <TextField
                      name="circuit_breaker_reset_timeout"
                      label="Reset Timeout (seconds)"
                      fullWidth
                      type="number"
                      value={configData.circuit_breaker_reset_timeout || 30}
                      onChange={handleInputChange}
                      helperText="Time before attempting recovery"
                    />
                  </Grid>
                  
                  <Grid item xs={12} md={4}>
                    <TextField
                      name="circuit_breaker_success_threshold"
                      label="Success Threshold"
                      fullWidth
                      type="number"
                      value={configData.circuit_breaker_success_threshold || 2}
                      onChange={handleInputChange}
                      helperText="Successes needed to close circuit"
                    />
                  </Grid>

                  <Grid item xs={12}>
                    <Divider sx={{ my: 1 }} />
                    <Typography variant="subtitle1" gutterBottom>Audit Logging Settings</Typography>
                  </Grid>
                  
                  <Grid item xs={12} md={6}>
                    <FormControlLabel
                      control={
                        <Switch
                          name="enable_audit_logging"
                          checked={configData.enable_audit_logging !== false}
                          onChange={handleInputChange}
                        />
                      }
                      label="Enable Audit Logging"
                    />
                  </Grid>
                  
                  <Grid item xs={12} md={6}>
                    <FormControlLabel
                      control={
                        <Switch
                          name="enable_phi_detection"
                          checked={configData.enable_phi_detection !== false}
                          onChange={handleInputChange}
                        />
                      }
                      label="Enable PHI Detection"
                    />
                  </Grid>
                  
                  <Grid item xs={12} md={6}>
                    <TextField
                      name="audit_log_path"
                      label="Audit Log Path"
                      fullWidth
                      value={configData.audit_log_path || 'audit_logs'}
                      onChange={handleInputChange}
                    />
                  </Grid>

                  <Grid item xs={12}>
                    <Divider sx={{ my: 1 }} />
                    <Typography variant="subtitle1" gutterBottom>Retry Settings</Typography>
                  </Grid>
                  
                  <Grid item xs={12} md={4}>
                    <TextField
                      name="max_retries"
                      label="Max Retries"
                      fullWidth
                      type="number"
                      value={configData.max_retries || 3}
                      onChange={handleInputChange}
                    />
                  </Grid>
                  
                  <Grid item xs={12} md={4}>
                    <TextField
                      name="retry_min_wait"
                      label="Min Wait (seconds)"
                      fullWidth
                      type="number"
                      step={0.1}
                      value={configData.retry_min_wait || 1.0}
                      onChange={handleInputChange}
                    />
                  </Grid>
                  
                  <Grid item xs={12} md={4}>
                    <TextField
                      name="retry_max_wait"
                      label="Max Wait (seconds)"
                      fullWidth
                      type="number"
                      step={0.1}
                      value={configData.retry_max_wait || 10.0}
                      onChange={handleInputChange}
                    />
                  </Grid>

                  <Grid item xs={12}>
                    <Divider sx={{ my: 1 }} />
                    <Typography variant="subtitle1" gutterBottom>Other Settings</Typography>
                  </Grid>
                  
                  <Grid item xs={12} md={4}>
                    <TextField
                      name="thread_limit"
                      label="Thread Limit"
                      fullWidth
                      type="number"
                      value={configData.thread_limit || 4}
                      onChange={handleInputChange}
                    />
                  </Grid>
                  
                  <Grid item xs={12} md={4}>
                    <TextField
                      name="timeout"
                      label="Timeout (seconds)"
                      fullWidth
                      type="number"
                      value={configData.timeout || 30}
                      onChange={handleInputChange}
                    />
                  </Grid>
                  
                  <Grid item xs={12} md={4}>
                    <TextField
                      name="max_prompt_length"
                      label="Max Prompt Length"
                      fullWidth
                      type="number"
                      value={configData.max_prompt_length || 16000}
                      onChange={handleInputChange}
                    />
                  </Grid>
                </Grid>

                <Box sx={{ mt: 2, display: 'flex', justifyContent: 'flex-end' }}>
                  <Button 
                    onClick={toggleEditConfig} 
                    sx={{ mr: 1 }}
                  >
                    Cancel
                  </Button>
                  <Button
                    variant="contained"
                    color="primary"
                    startIcon={saving ? null : <SaveIcon />}
                    onClick={handleSaveConfig}
                    disabled={saving}
                  >
                    {saving ? 'Saving...' : 'Save Configuration'}
                  </Button>
                </Box>
              </Box>
            ) : (
              <Grid container spacing={2}>
                <Grid item xs={12}>
                  <Typography variant="subtitle1">LLM Provider Settings</Typography>
                  <Typography variant="body2">Provider: {client.config.provider || 'openai'}</Typography>
                  <Typography variant="body2">Default Model: {client.config.default_model || 'gpt-4'}</Typography>
                  <Typography variant="body2">API Key: {client.config.api_key ? '********' : 'Not set'}</Typography>
                </Grid>
                
                {(client.config.azure_endpoint || client.config.azure_deployment_name) && (
                  <Grid item xs={12}>
                    <Typography variant="subtitle1">Azure Settings</Typography>
                    <Typography variant="body2">Endpoint: {client.config.azure_endpoint || 'Not set'}</Typography>
                    <Typography variant="body2">Deployment: {client.config.azure_deployment_name || 'Not set'}</Typography>
                    <Typography variant="body2">API Version: {client.config.azure_api_version || '2023-05-15'}</Typography>
                  </Grid>
                )}
                
                <Grid item xs={12}>
                  <Typography variant="subtitle1">Model Parameters</Typography>
                  <Typography variant="body2">Max Tokens: {client.config.max_tokens || 1000}</Typography>
                  <Typography variant="body2">Temperature: {client.config.temperature || 0.7}</Typography>
                  <Typography variant="body2">Top P: {client.config.top_p || 1.0}</Typography>
                </Grid>
                
                <Grid item xs={12}>
                  <Typography variant="subtitle1">Caching</Typography>
                  <Typography variant="body2">Cache Backend: {client.config.cache_backend || 'disk'}</Typography>
                  <Typography variant="body2">Cache TTL: {client.config.cache_ttl || 3600} seconds</Typography>
                  {client.config.cache_backend === 'disk' && (
                    <Typography variant="body2">Cache Directory: {client.config.cache_directory || '.dspy_cache'}</Typography>
                  )}
                  {client.config.cache_backend === 'redis' && (
                    <>
                      <Typography variant="body2">Redis URL: {client.config.redis_url || 'Not set'}</Typography>
                      <Typography variant="body2">Redis Pool Size: {client.config.redis_pool_size || 10}</Typography>
                    </>
                  )}
                </Grid>
                
                <Grid item xs={12}>
                  <Typography variant="subtitle1">Circuit Breaker</Typography>
                  <Typography variant="body2">Failure Threshold: {client.config.circuit_breaker_failure_threshold || 5}</Typography>
                  <Typography variant="body2">Reset Timeout: {client.config.circuit_breaker_reset_timeout || 30} seconds</Typography>
                  <Typography variant="body2">Success Threshold: {client.config.circuit_breaker_success_threshold || 2}</Typography>
                </Grid>
                
                <Grid item xs={12}>
                  <Typography variant="subtitle1">Audit Logging</Typography>
                  <Typography variant="body2">Enabled: {client.config.enable_audit_logging !== false ? 'Yes' : 'No'}</Typography>
                  <Typography variant="body2">PHI Detection: {client.config.enable_phi_detection !== false ? 'Yes' : 'No'}</Typography>
                  <Typography variant="body2">Log Path: {client.config.audit_log_path || 'audit_logs'}</Typography>
                </Grid>
              </Grid>
            )}
          </AccordionDetails>
        </Accordion>

        <Accordion 
          expanded={expanded === 'panel-usage'} 
          onChange={handleExpandChange('panel-usage')}
        >
          <AccordionSummary expandIcon={<ExpandMoreIcon />}>
            <Typography variant="h6">Usage Statistics</Typography>
          </AccordionSummary>
          <AccordionDetails>
            <Box sx={{ display: 'flex', justifyContent: 'flex-end', mb: 2 }}>
              <Button
                variant="outlined"
                size="small"
                onClick={loadUsageStats}
                startIcon={<RefreshIcon />}
                disabled={loadingStats}
              >
                Refresh
              </Button>
            </Box>
            
            {loadingStats ? (
              <LinearProgress />
            ) : usageStats.length === 0 ? (
              <Alert severity="info">No usage data available for this client.</Alert>
            ) : (
              <Box sx={{ height: 300 }}>
                <UsageChart data={usageStats} />
              </Box>
            )}
          </AccordionDetails>
        </Accordion>
      </Paper>
    </Box>
  );
};

export default DSPyClientDetails;