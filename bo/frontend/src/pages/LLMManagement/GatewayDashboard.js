import React, { useState, useEffect } from 'react';
import {
  Box,
  Paper,
  Typography,
  Button,
  Grid,
  Card,
  CardHeader,
  CardContent,
  CardActions,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Alert,
  Chip,
  Divider,
  CircularProgress,
  IconButton,
  Tooltip,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Tab,
  Tabs,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Switch,
  FormControlLabel,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions
} from '@mui/material';
import {
  Refresh as RefreshIcon,
  ExpandMore as ExpandMoreIcon,
  Check as CheckIcon,
  Error as ErrorIcon,
  CloudQueue as CloudQueueIcon,
  Settings as SettingsIcon,
  Send as SendIcon,
  History as HistoryIcon,
  Delete as DeleteIcon,
  SmartToy as SmartToyIcon
} from '@mui/icons-material';

import apiService from '../../services/api';
import { useNotification } from '../../context/NotificationContext';

/**
 * Dashboard for LLM Gateway operations and monitoring
 */
const GatewayDashboard = ({ status, onRefresh }) => {
  const [loading, setLoading] = useState(false);
  const [providers, setProviders] = useState([]);
  const [activeProviders, setActiveProviders] = useState([]);
  const [selectedProvider, setSelectedProvider] = useState('');
  const [providerModels, setProviderModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState('');
  const [prompt, setPrompt] = useState('');
  const [responseText, setResponseText] = useState('');
  const [generating, setGenerating] = useState(false);
  const [activeTab, setActiveTab] = useState(0);
  const [historyDialogOpen, setHistoryDialogOpen] = useState(false);
  const [history, setHistory] = useState([]);
  const [advancedSettings, setAdvancedSettings] = useState({
    temperature: 0.7,
    max_tokens: 500,
    stream: false,
    top_p: 1,
    presence_penalty: 0,
    frequency_penalty: 0,
    system_prompt: ''
  });

  const { showSuccess, showError } = useNotification();

  // Load providers on mount
  useEffect(() => {
    loadProviders();
  }, []);

  // Update providers when status changes
  useEffect(() => {
    if (status?.details?.providers) {
      setProviders(status.details.providers);
      const active = status.details.active_providers || [];
      setActiveProviders(active);

      if (active.length > 0 && !selectedProvider) {
        setSelectedProvider(active[0].id);
      }
    }
  }, [status, selectedProvider]);

  // Load models when provider changes
  useEffect(() => {
    if (selectedProvider) {
      loadModels(selectedProvider);
    }
  }, [selectedProvider]);

  // Load gateway providers
  const loadProviders = async () => {
    setLoading(true);

    try {
      const result = await apiService.llm.getProviders();

      if (result.success) {
        setProviders(result.data);
        // Filter active providers (those with status 'operational' or 'available')
        const active = result.data.filter(p =>
          p.status === 'operational' ||
          p.status === 'available' ||
          p.is_active
        );
        setActiveProviders(active);

        if (active.length > 0 && !selectedProvider) {
          setSelectedProvider(active[0].provider_id || active[0].id);
        }
      } else {
        showError(`Failed to load LLM providers: ${result.error}`);
      }
    } catch (error) {
      console.error('Error loading LLM providers:', error);
      showError(`Error loading LLM providers: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  // Load models for a provider
  const loadModels = async (providerId) => {
    try {
      const result = await apiService.llm.getModels(providerId);

      if (result.success) {
        // Transform the data to match the expected format if needed
        const models = result.data.map(model => ({
          id: model.model_id,
          name: model.display_name || model.model_id,
          provider_id: model.provider_id,
          type: model.model_type,
          capabilities: model.capabilities || [],
          context_window: model.context_window,
          max_output_tokens: model.max_output_tokens
        }));

        setProviderModels(models);

        if (models.length > 0) {
          setSelectedModel(models[0].id);
        } else {
          setSelectedModel('');
        }
      } else {
        showError(`Failed to load models: ${result.error}`);
        setProviderModels([]);
        setSelectedModel('');
      }
    } catch (error) {
      console.error('Error loading models:', error);
      showError(`Error loading models: ${error.message}`);
      setProviderModels([]);
      setSelectedModel('');
    }
  };

  // Test provider connection
  const testProvider = async (providerId) => {
    try {
      const result = await apiService.llm.testProvider(providerId);

      if (result.success) {
        showSuccess(`Connection to ${providerId} successful`);
        // Refresh providers list to update status
        loadProviders();
      } else {
        showError(`Connection to ${providerId} failed: ${result.error}`);
      }
    } catch (error) {
      console.error('Error testing provider connection:', error);
      showError(`Error testing provider connection: ${error.message}`);
    }
  };

  // Generate text
  const generateText = async () => {
    if (!selectedProvider || !selectedModel || !prompt.trim()) {
      showError('Please select a provider, model, and enter a prompt');
      return;
    }

    setGenerating(true);
    setResponseText('');

    try {
      const result = await apiService.llm.generateLLMResponse({
        provider_id: selectedProvider,
        model: selectedModel,  // Backend expects 'model', not 'model_id'
        prompt: prompt,
        temperature: advancedSettings.temperature,
        max_tokens: advancedSettings.max_tokens,
        stream: advancedSettings.stream,
        system_prompt: advancedSettings.system_prompt || undefined
      });

      if (result.success) {
        // The backend returns content in result.data.content
        setResponseText(result.data.content || result.data.text || '');
        showSuccess('Text generated successfully');

        // Add to history with usage statistics
        setHistory(prev => [{
          id: Date.now().toString(),
          timestamp: new Date().toISOString(),
          provider: selectedProvider,
          model: selectedModel,
          prompt,
          response: result.data.content || result.data.text || '',
          stats: {
            prompt_tokens: result.data.prompt_tokens,
            completion_tokens: result.data.completion_tokens,
            total_tokens: result.data.total_tokens,
            latency_ms: result.data.latency_ms
          }
        }, ...prev]);
      } else {
        showError(`Failed to generate text: ${result.error}`);
      }
    } catch (error) {
      console.error('Error generating text:', error);
      showError(`Error generating text: ${error.message}`);
    } finally {
      setGenerating(false);
    }
  };

  // Handle provider change
  const handleProviderChange = (event) => {
    setSelectedProvider(event.target.value);
    setSelectedModel('');
  };

  // Handle model change
  const handleModelChange = (event) => {
    setSelectedModel(event.target.value);
  };

  // Handle prompt change
  const handlePromptChange = (event) => {
    setPrompt(event.target.value);
  };

  // Handle tab change
  const handleTabChange = (_, newValue) => {
    setActiveTab(newValue);
  };

  // Handle advanced setting change
  const handleSettingChange = (setting, value) => {
    setAdvancedSettings(prev => ({
      ...prev,
      [setting]: value
    }));
  };

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h5">
          <SmartToyIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
          LLM Gateway Dashboard
        </Typography>
        <Box>
          <Button
            variant="outlined"
            startIcon={<HistoryIcon />}
            onClick={() => setHistoryDialogOpen(true)}
            sx={{ mr: 1 }}
          >
            History
          </Button>
          <Button
            variant="outlined"
            startIcon={loading ? <CircularProgress size={20} /> : <RefreshIcon />}
            onClick={loadProviders}
            disabled={loading}
          >
            Refresh
          </Button>
        </Box>
      </Box>

      {status?.status !== 'available' && (
        <Alert severity="error" sx={{ mb: 3 }}>
          LLM Gateway service is currently unavailable. Please check the server status.
        </Alert>
      )}

      <Tabs
        value={activeTab}
        onChange={handleTabChange}
        aria-label="Gateway tabs"
        sx={{ mb: 2, borderBottom: 1, borderColor: 'divider' }}
      >
        <Tab label="Text Generation" id="tab-0" aria-controls="tabpanel-0" />
        <Tab label="Provider Status" id="tab-1" aria-controls="tabpanel-1" />
        <Tab label="Settings" id="tab-2" aria-controls="tabpanel-2" />
      </Tabs>

      {/* Text Generation Panel */}
      <Box role="tabpanel" hidden={activeTab !== 0} id="tabpanel-0" aria-labelledby="tab-0">
        {activeTab === 0 && (
          <Grid container spacing={3}>
            <Grid item xs={12}>
              <Paper sx={{ p: 2 }}>
                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                  <Box sx={{ display: 'flex', gap: 2 }}>
                    <FormControl fullWidth>
                      <InputLabel id="provider-select-label">LLM Provider</InputLabel>
                      <Select
                        labelId="provider-select-label"
                        id="provider-select"
                        value={selectedProvider}
                        label="LLM Provider"
                        onChange={handleProviderChange}
                        disabled={loading || generating || activeProviders.length === 0}
                      >
                        {activeProviders.map((provider) => (
                          <MenuItem key={provider.provider_id || provider.id} value={provider.provider_id || provider.id}>
                            {provider.display_name || provider.name || provider.provider_id || provider.id}
                            {` (${provider.provider_type || provider.type || 'Unknown'})`}
                          </MenuItem>
                        ))}
                      </Select>
                    </FormControl>

                    <FormControl fullWidth>
                      <InputLabel id="model-select-label">Model</InputLabel>
                      <Select
                        labelId="model-select-label"
                        id="model-select"
                        value={selectedModel}
                        label="Model"
                        onChange={handleModelChange}
                        disabled={loading || generating || !selectedProvider || providerModels.length === 0}
                      >
                        {providerModels.map((model) => (
                          <MenuItem key={model.model_id || model.id} value={model.model_id || model.id}>
                            {model.display_name || model.name || model.model_id || model.id}
                            {model.type && ` (${model.type})`}
                          </MenuItem>
                        ))}
                      </Select>
                    </FormControl>
                  </Box>

                  <TextField
                    label="Prompt"
                    multiline
                    rows={4}
                    fullWidth
                    variant="outlined"
                    value={prompt}
                    onChange={handlePromptChange}
                    placeholder="Enter your prompt here..."
                    disabled={generating}
                  />

                  <Accordion>
                    <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                      <Typography>Advanced Settings</Typography>
                    </AccordionSummary>
                    <AccordionDetails>
                      <Grid container spacing={2}>
                        <Grid item xs={12}>
                          <TextField
                            label="System Prompt"
                            multiline
                            rows={2}
                            fullWidth
                            variant="outlined"
                            value={advancedSettings.system_prompt}
                            onChange={(e) => handleSettingChange('system_prompt', e.target.value)}
                            placeholder="Optional system instructions for the LLM"
                            helperText="Instructions that define the LLM's behavior and context"
                          />
                        </Grid>
                        <Grid item xs={12} sm={6} md={4}>
                          <TextField
                            label="Temperature"
                            type="number"
                            fullWidth
                            variant="outlined"
                            value={advancedSettings.temperature}
                            onChange={(e) => handleSettingChange('temperature', parseFloat(e.target.value))}
                            inputProps={{ step: 0.1, min: 0, max: 2 }}
                            helperText="Controls randomness (0=deterministic, 1=creative)"
                          />
                        </Grid>
                        <Grid item xs={12} sm={6} md={4}>
                          <TextField
                            label="Max Tokens"
                            type="number"
                            fullWidth
                            variant="outlined"
                            value={advancedSettings.max_tokens}
                            onChange={(e) => handleSettingChange('max_tokens', parseInt(e.target.value))}
                            inputProps={{ min: 1, max: 4096 }}
                            helperText="Maximum length of the generated response"
                          />
                        </Grid>
                        <Grid item xs={12} sm={6} md={4}>
                          <TextField
                            label="Top P"
                            type="number"
                            fullWidth
                            variant="outlined"
                            value={advancedSettings.top_p}
                            onChange={(e) => handleSettingChange('top_p', parseFloat(e.target.value))}
                            inputProps={{ step: 0.1, min: 0, max: 1 }}
                            helperText="Controls diversity via nucleus sampling"
                          />
                        </Grid>
                        <Grid item xs={12} sm={6} md={4}>
                          <TextField
                            label="Presence Penalty"
                            type="number"
                            fullWidth
                            variant="outlined"
                            value={advancedSettings.presence_penalty}
                            onChange={(e) => handleSettingChange('presence_penalty', parseFloat(e.target.value))}
                            inputProps={{ step: 0.1, min: -2, max: 2 }}
                            helperText="Penalizes new tokens based on presence in text"
                          />
                        </Grid>
                        <Grid item xs={12} sm={6} md={4}>
                          <TextField
                            label="Frequency Penalty"
                            type="number"
                            fullWidth
                            variant="outlined"
                            value={advancedSettings.frequency_penalty}
                            onChange={(e) => handleSettingChange('frequency_penalty', parseFloat(e.target.value))}
                            inputProps={{ step: 0.1, min: -2, max: 2 }}
                            helperText="Penalizes new tokens based on frequency in text"
                          />
                        </Grid>
                        <Grid item xs={12} sm={6} md={4}>
                          <FormControlLabel
                            control={
                              <Switch
                                checked={advancedSettings.stream}
                                onChange={(e) => handleSettingChange('stream', e.target.checked)}
                              />
                            }
                            label="Stream Response"
                          />
                        </Grid>
                      </Grid>
                    </AccordionDetails>
                  </Accordion>

                  <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                    <Button
                      variant="contained"
                      startIcon={generating ? <CircularProgress size={20} /> : <SendIcon />}
                      onClick={generateText}
                      disabled={loading || generating || !selectedProvider || !selectedModel || !prompt.trim()}
                    >
                      {generating ? 'Generating...' : 'Generate'}
                    </Button>

                    <Button
                      variant="outlined"
                      onClick={() => {
                        setPrompt('');
                        setResponseText('');
                      }}
                      disabled={generating}
                    >
                      Clear
                    </Button>
                  </Box>

                  <Divider sx={{ my: 2 }} />

                  <Typography variant="h6" gutterBottom>
                    Response
                  </Typography>

                  {generating ? (
                    <Box sx={{ display: 'flex', justifyContent: 'center', my: 3 }}>
                      <CircularProgress />
                    </Box>
                  ) : responseText ? (
                    <Paper
                      variant="outlined"
                      sx={{
                        p: 2,
                        backgroundColor: 'grey.50',
                        maxHeight: '300px',
                        overflowY: 'auto',
                        whiteSpace: 'pre-wrap'
                      }}
                    >
                      {responseText}
                    </Paper>
                  ) : (
                    <Alert severity="info">
                      Response will appear here after generation
                    </Alert>
                  )}
                </Box>
              </Paper>
            </Grid>
          </Grid>
        )}
      </Box>

      {/* Provider Status Panel */}
      <Box role="tabpanel" hidden={activeTab !== 1} id="tabpanel-1" aria-labelledby="tab-1">
        {activeTab === 1 && (
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              LLM Provider Status
            </Typography>

            {loading ? (
              <Box sx={{ display: 'flex', justifyContent: 'center', my: 3 }}>
                <CircularProgress />
              </Box>
            ) : providers.length === 0 ? (
              <Alert severity="info">No LLM providers configured</Alert>
            ) : (
              <Grid container spacing={2}>
                {providers.map((provider) => (
                  <Grid item xs={12} md={6} lg={4} key={provider.provider_id || provider.id}>
                    <Card variant="outlined">
                      <CardHeader
                        title={provider.display_name || provider.name || provider.provider_id || provider.id}
                        subheader={provider.provider_type || provider.type}
                        avatar={
                          <CloudQueueIcon
                            color={
                              provider.status === 'operational' ||
                              provider.status === 'available' ||
                              provider.is_active ? 'success' : 'disabled'
                            }
                          />
                        }
                        action={
                          <Chip
                            label={
                              provider.status === 'operational' ? 'Operational' :
                              provider.status === 'available' ? 'Available' :
                              provider.status === 'degraded' ? 'Degraded' :
                              provider.status === 'error' ? 'Error' :
                              provider.is_active ? 'Active' : 'Inactive'
                            }
                            color={
                              provider.status === 'operational' || provider.status === 'available' ? 'success' :
                              provider.status === 'degraded' ? 'warning' :
                              provider.is_active ? 'success' : 'error'
                            }
                            size="small"
                          />
                        }
                      />
                      <CardContent>
                        <Typography variant="body2" color="text.secondary" gutterBottom>
                          {provider.message || provider.description || 'No description available'}
                        </Typography>

                        <List dense>
                          <ListItem>
                            <ListItemIcon sx={{ minWidth: '32px' }}>
                              {(provider.connection_params?.api_key_env_var || provider.requires_api_key) ? (
                                <CheckIcon fontSize="small" color="success" />
                              ) : (
                                <ErrorIcon fontSize="small" color="warning" />
                              )}
                            </ListItemIcon>
                            <ListItemText
                              primary="API Key Required"
                              secondary={(provider.connection_params?.api_key_env_var || provider.requires_api_key) ? 'Yes' : 'No'}
                            />
                          </ListItem>

                          {(provider.models_count !== undefined || (provider.models && Object.keys(provider.models).length > 0)) && (
                            <ListItem>
                              <ListItemIcon sx={{ minWidth: '32px' }}>
                                <SmartToyIcon fontSize="small" />
                              </ListItemIcon>
                              <ListItemText
                                primary="Available Models"
                                secondary={provider.models_count || (provider.models ? Object.keys(provider.models).length : 0)}
                              />
                            </ListItem>
                          )}
                        </List>
                      </CardContent>
                      <CardActions>
                        <Button
                          size="small"
                          onClick={() => testProvider(provider.provider_id || provider.id)}
                          startIcon={<CheckIcon />}
                        >
                          Test Connection
                        </Button>
                        <Button
                          size="small"
                          onClick={() => {
                            setSelectedProvider(provider.provider_id || provider.id);
                            setActiveTab(0);
                          }}
                        >
                          Use Provider
                        </Button>
                      </CardActions>
                    </Card>
                  </Grid>
                ))}
              </Grid>
            )}
          </Paper>
        )}
      </Box>

      {/* Settings Panel */}
      <Box role="tabpanel" hidden={activeTab !== 2} id="tabpanel-2" aria-labelledby="tab-2">
        {activeTab === 2 && (
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Gateway Settings
            </Typography>

            <Alert severity="info" sx={{ mb: 3 }}>
              Gateway settings can be configured on the Providers and Models tabs in the main LLM Management panel.
            </Alert>
          </Paper>
        )}
      </Box>

      {/* History Dialog */}
      <Dialog
        open={historyDialogOpen}
        onClose={() => setHistoryDialogOpen(false)}
        fullWidth
        maxWidth="md"
      >
        <DialogTitle>
          <Box display="flex" alignItems="center" justifyContent="space-between">
            <Typography variant="h6">Generation History</Typography>
            <Button
              size="small"
              startIcon={<DeleteIcon />}
              onClick={() => setHistory([])}
              disabled={history.length === 0}
            >
              Clear
            </Button>
          </Box>
        </DialogTitle>
        <DialogContent dividers>
          {history.length === 0 ? (
            <Alert severity="info">No history available</Alert>
          ) : (
            <List>
              {history.map((item) => (
                <Box key={item.id} sx={{ mb: 2 }}>
                  <Accordion>
                    <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                      <Box sx={{ display: 'flex', alignItems: 'center', width: '100%' }}>
                        <Typography sx={{ flexGrow: 1 }}>
                          {item.prompt.length > 50 ? `${item.prompt.slice(0, 50)}...` : item.prompt}
                        </Typography>
                        <Chip
                          label={item.provider}
                          size="small"
                          sx={{ mr: 1 }}
                        />
                        <Typography variant="caption" color="text.secondary">
                          {new Date(item.timestamp).toLocaleString()}
                        </Typography>
                      </Box>
                    </AccordionSummary>
                    <AccordionDetails>
                      <Box>
                        <Typography variant="subtitle2" gutterBottom>Prompt:</Typography>
                        <Paper variant="outlined" sx={{ p: 1, mb: 2, backgroundColor: 'grey.50' }}>
                          <Typography variant="body2">{item.prompt}</Typography>
                        </Paper>

                        <Typography variant="subtitle2" gutterBottom>Response:</Typography>
                        <Paper variant="outlined" sx={{ p: 1, mb: 2, backgroundColor: 'grey.50', maxHeight: '200px', overflowY: 'auto' }}>
                          <Typography variant="body2" sx={{ whiteSpace: 'pre-wrap' }}>{item.response}</Typography>
                        </Paper>

                        <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                          <Typography variant="caption">
                            Provider: {item.provider} | Model: {item.model}
                          </Typography>

                          {item.stats && Object.keys(item.stats).length > 0 && (
                            <Typography variant="caption">
                              Tokens: {item.stats.total_tokens || 'N/A'} |
                              Time: {item.stats.latency_ms ? `${item.stats.latency_ms}ms` : 'N/A'}
                            </Typography>
                          )}
                        </Box>

                        <Box sx={{ display: 'flex', justifyContent: 'flex-end', mt: 1 }}>
                          <Button
                            size="small"
                            onClick={() => {
                              setPrompt(item.prompt);
                              setSelectedProvider(item.provider);
                              setSelectedModel(item.model);
                              setActiveTab(0);
                              setHistoryDialogOpen(false);
                            }}
                          >
                            Reuse
                          </Button>
                        </Box>
                      </Box>
                    </AccordionDetails>
                  </Accordion>
                </Box>
              ))}
            </List>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setHistoryDialogOpen(false)}>Close</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default GatewayDashboard;
