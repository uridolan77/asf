import React, { useState, useEffect, useRef } from 'react';
import {
  Box,
  Paper,
  Typography,
  Button,
  Grid,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  TextField,
  Alert,
  CircularProgress,
  Divider,
  Avatar,
  IconButton,
  Chip,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  List,
  ListItem
} from '@mui/material';
import {
  Send as SendIcon,
  Refresh as RefreshIcon,
  SmartToy as SmartToyIcon,
  Person as PersonIcon,
  Settings as SettingsIcon,
  ExpandMore as ExpandMoreIcon,
  Delete as DeleteIcon,
  Save as SaveIcon,
  History as HistoryIcon
} from '@mui/icons-material';

import PageLayout from '../../components/Layout/PageLayout';
import apiService from '../../services/api';
import { useNotification } from '../../context/NotificationContext';
import { useAuth } from '../../context/AuthContext';

/**
 * TextPlayground component for chatting with LLM providers
 */
const TextPlayground = () => {
  const { user } = useAuth();
  const { showSuccess, showError } = useNotification();
  const messagesEndRef = useRef(null);
  
  // States for providers and models
  const [loading, setLoading] = useState(false);
  const [providers, setProviders] = useState([]);
  const [activeProviders, setActiveProviders] = useState([]);
  const [selectedProvider, setSelectedProvider] = useState('');
  const [providerModels, setProviderModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState('');
  
  // States for chat
  const [messages, setMessages] = useState([]);
  const [currentMessage, setCurrentMessage] = useState('');
  const [sending, setSending] = useState(false);
  const [conversationTitle, setConversationTitle] = useState('New Conversation');
  const [conversations, setConversations] = useState([]);
  
  // States for settings and history
  const [advancedSettings, setAdvancedSettings] = useState({
    temperature: 0.7,
    max_tokens: 500,
    stream: false,
    top_p: 1,
    presence_penalty: 0,
    frequency_penalty: 0,
    system_prompt: ''
  });
  const [historyDialogOpen, setHistoryDialogOpen] = useState(false);

  // Load providers on mount
  useEffect(() => {
    loadProviders();
  }, []);

  // Scroll to bottom of messages on new message
  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages]);

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
        // Check if result.data is an array
        const providersArray = Array.isArray(result.data) ? result.data : [];
        setProviders(providersArray);
        
        // Filter active providers (those with status 'operational' or 'available')
        const active = providersArray.filter(p =>
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
        // Check if result.data is an array
        const modelsArray = Array.isArray(result.data) ? result.data : [];
        
        // Transform the data to match the expected format if needed
        const models = modelsArray.map(model => ({
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

  // Send message to LLM
  const sendMessage = async () => {
    if (!selectedProvider || !selectedModel || !currentMessage.trim()) {
      showError('Please select a provider, model, and enter a message');
      return;
    }

    // Add user message to chat
    const userMessage = {
      id: Date.now().toString(),
      role: 'user',
      content: currentMessage,
      timestamp: new Date().toISOString()
    };
    
    setMessages(prevMessages => [...prevMessages, userMessage]);
    setCurrentMessage('');
    setSending(true);

    try {
      // Create a prompt that combines conversation history and the current message
      // This creates a text prompt from the chat history that the API can understand
      let promptText = '';
      
      // Include previous messages as context if any exist
      if (messages.length > 0) {
        promptText += "Previous messages:\n";
        messages.forEach(msg => {
          promptText += `${msg.role === 'user' ? 'User' : 'Assistant'}: ${msg.content}\n`;
        });
        promptText += "\nCurrent message:\n";
      }
      
      // Add the current user message
      promptText += `User: ${userMessage.content}`;
      
      const result = await apiService.llm.generateLLMResponse({
        provider_id: selectedProvider,
        model: selectedModel,
        prompt: promptText,  // Using prompt instead of messages array
        temperature: advancedSettings.temperature,
        max_tokens: advancedSettings.max_tokens,
        stream: advancedSettings.stream,
        system_prompt: advancedSettings.system_prompt || undefined
      });

      if (result.success) {
        // Check if the response indicates an error from the provider
        if (result.data?.finish_reason === 'error') {
          const errorMessage = result.data.error_details?.message || 'The LLM provider returned an error.';
          const errorCode = result.data.error_details?.code || '';
          
          // Create an error message in the chat as if it came from the assistant
          const errorResponseMessage = {
            id: `${Date.now()}-error`,
            role: 'assistant',
            isError: true,
            content: `Error${errorCode ? ` (${errorCode})` : ''}: ${errorMessage}`,
            timestamp: new Date().toISOString(),
            stats: {
              latency_ms: result.data.latency_ms || 0
            }
          };
          
          setMessages(prevMessages => [...prevMessages, errorResponseMessage]);
          showError(`LLM Error: ${errorMessage}`);
          return;
        }
        
        // The backend returns content in result.data.content or result.data.text
        const responseContent = result.data.content || result.data.text || result.data.generated_text || '';
        
        // If we got an empty content but no explicit error, treat it as an error
        if (!responseContent && result.data.finish_reason !== 'stop') {
          const errorResponseMessage = {
            id: `${Date.now()}-error`,
            role: 'assistant',
            isError: true,
            content: `Error: The LLM provider returned an empty response with reason: "${result.data.finish_reason || 'unknown'}". This might indicate a configuration issue with the model or provider.`,
            timestamp: new Date().toISOString(),
            stats: {
              latency_ms: result.data.latency_ms || 0
            }
          };
          
          setMessages(prevMessages => [...prevMessages, errorResponseMessage]);
          showError(`LLM returned an empty response: ${result.data.finish_reason || 'unknown reason'}`);
          return;
        }
        
        const aiMessage = {
          id: `${Date.now()}-ai`,
          role: 'assistant',
          content: responseContent,
          timestamp: new Date().toISOString(),
          stats: {
            prompt_tokens: result.data.prompt_tokens || result.data.usage?.prompt_tokens || 0,
            completion_tokens: result.data.completion_tokens || result.data.usage?.completion_tokens || 0,
            total_tokens: result.data.total_tokens || result.data.usage?.total_tokens || 0,
            latency_ms: result.data.latency_ms || result.data.generation_time_ms || 0
          }
        };
        
        setMessages(prevMessages => [...prevMessages, aiMessage]);
        
        // Auto-generate title if this is the first exchange
        if (messages.length === 0 && conversationTitle === 'New Conversation') {
          setConversationTitle(userMessage.content.substring(0, 30) + (userMessage.content.length > 30 ? '...' : ''));
        }
      } else {
        // Handle API error (not LLM error)
        const errorMessage = result.error || 'Unknown error generating response';
        const errorResponseMessage = {
          id: `${Date.now()}-error`,
          role: 'assistant',
          isError: true,
          content: `API Error: ${errorMessage}`,
          timestamp: new Date().toISOString()
        };
        
        setMessages(prevMessages => [...prevMessages, errorResponseMessage]);
        showError(`Failed to generate response: ${errorMessage}`);
      }
    } catch (error) {
      // Handle exception
      console.error('Error generating response:', error);
      const errorResponseMessage = {
        id: `${Date.now()}-error`,
        role: 'assistant',
        isError: true,
        content: `Exception: ${error.message || 'Unknown error occurred'}`,
        timestamp: new Date().toISOString()
      };
      
      setMessages(prevMessages => [...prevMessages, errorResponseMessage]);
      showError(`Error generating response: ${error.message}`);
    } finally {
      setSending(false);
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

  // Handle message change
  const handleMessageChange = (event) => {
    setCurrentMessage(event.target.value);
  };

  // Handle setting change
  const handleSettingChange = (setting, value) => {
    setAdvancedSettings(prev => ({
      ...prev,
      [setting]: value
    }));
  };

  // Handle key press in message input
  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  // Clear conversation
  const clearConversation = () => {
    setMessages([]);
    setConversationTitle('New Conversation');
  };

  // Save conversation
  const saveConversation = () => {
    const conversation = {
      id: Date.now().toString(),
      title: conversationTitle,
      messages: [...messages],
      provider: selectedProvider,
      model: selectedModel,
      timestamp: new Date().toISOString()
    };
    
    setConversations(prev => [conversation, ...prev]);
    showSuccess('Conversation saved');
  };

  // Load conversation
  const loadConversation = (conversation) => {
    setMessages(conversation.messages);
    setConversationTitle(conversation.title);
    setSelectedProvider(conversation.provider);
    setSelectedModel(conversation.model);
    setHistoryDialogOpen(false);
  };

  // Delete conversation
  const deleteConversation = (id) => {
    setConversations(prev => prev.filter(conv => conv.id !== id));
  };

  return (
    <PageLayout
      title="Text Playground"
      breadcrumbs={[
        { label: 'LLM Management', path: '/llm-management' },
        { label: 'Playground', path: '/llm/playground/text' }
      ]}
      user={user}
      actions={
        <>
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
        </>
      }
    >
      <Grid container spacing={3}>
        {/* Left sidebar with settings */}
        <Grid item xs={12} md={3}>
          <Paper sx={{ p: 2, height: '100%' }}>
            <Typography variant="h6" gutterBottom>
              Playground Settings
            </Typography>
            <Divider sx={{ my: 2 }} />
            
            <FormControl fullWidth sx={{ mb: 2 }}>
              <InputLabel id="provider-select-label">LLM Provider</InputLabel>
              <Select
                labelId="provider-select-label"
                id="provider-select"
                value={selectedProvider}
                label="LLM Provider"
                onChange={handleProviderChange}
                disabled={loading || sending || activeProviders.length === 0}
              >
                {activeProviders.map((provider) => (
                  <MenuItem key={provider.provider_id || provider.id} value={provider.provider_id || provider.id}>
                    {provider.display_name || provider.name || provider.provider_id || provider.id}
                    {` (${provider.provider_type || provider.type || 'Unknown'})`}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>

            <FormControl fullWidth sx={{ mb: 3 }}>
              <InputLabel id="model-select-label">Model</InputLabel>
              <Select
                labelId="model-select-label"
                id="model-select"
                value={selectedModel}
                label="Model"
                onChange={handleModelChange}
                disabled={loading || sending || !selectedProvider || providerModels.length === 0}
              >
                {providerModels.map((model) => (
                  <MenuItem key={model.model_id || model.id} value={model.model_id || model.id}>
                    {model.display_name || model.name || model.model_id || model.id}
                    {model.type && ` (${model.type})`}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>

            <Accordion>
              <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                <Typography>Advanced Settings</Typography>
              </AccordionSummary>
              <AccordionDetails>
                <TextField
                  label="System Prompt"
                  multiline
                  rows={3}
                  fullWidth
                  variant="outlined"
                  value={advancedSettings.system_prompt}
                  onChange={(e) => handleSettingChange('system_prompt', e.target.value)}
                  placeholder="Optional system instructions for the LLM"
                  sx={{ mb: 2 }}
                  helperText="Instructions that define the LLM's behavior and context"
                />
                
                <TextField
                  label="Temperature"
                  type="number"
                  fullWidth
                  variant="outlined"
                  value={advancedSettings.temperature}
                  onChange={(e) => handleSettingChange('temperature', parseFloat(e.target.value))}
                  inputProps={{ step: 0.1, min: 0, max: 2 }}
                  sx={{ mb: 2 }}
                  helperText="Controls randomness (0=deterministic, 1=creative)"
                />
                
                <TextField
                  label="Max Tokens"
                  type="number"
                  fullWidth
                  variant="outlined"
                  value={advancedSettings.max_tokens}
                  onChange={(e) => handleSettingChange('max_tokens', parseInt(e.target.value))}
                  inputProps={{ min: 1, max: 4096 }}
                  sx={{ mb: 2 }}
                  helperText="Maximum length of the generated response"
                />
              </AccordionDetails>
            </Accordion>

            <Box sx={{ mt: 3 }}>
              <Button
                variant="outlined"
                color="secondary"
                fullWidth
                onClick={clearConversation}
                disabled={messages.length === 0}
                startIcon={<DeleteIcon />}
              >
                Clear Conversation
              </Button>

              <Button
                variant="outlined"
                color="primary"
                fullWidth
                onClick={saveConversation}
                disabled={messages.length === 0}
                startIcon={<SaveIcon />}
                sx={{ mt: 1 }}
              >
                Save Conversation
              </Button>
            </Box>
          </Paper>
        </Grid>
        
        {/* Chat section */}
        <Grid item xs={12} md={9}>
          <Paper sx={{ p: 2, height: '100%', display: 'flex', flexDirection: 'column' }}>
            <Typography variant="h6" gutterBottom>
              {conversationTitle}
            </Typography>
            <Divider sx={{ mb: 2 }} />
            
            {/* Messages area */}
            <Box 
              sx={{
                flexGrow: 1,
                overflowY: 'auto',
                display: 'flex',
                flexDirection: 'column',
                gap: 2,
                mb: 2,
                p: 1,
                height: '60vh'
              }}
            >
              {messages.length === 0 ? (
                <Box 
                  sx={{
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    justifyContent: 'center',
                    height: '100%',
                    color: 'text.secondary'
                  }}
                >
                  <SmartToyIcon sx={{ fontSize: 50, mb: 2, opacity: 0.7 }} />
                  <Typography variant="body1">
                    Start a conversation by typing a message below
                  </Typography>
                </Box>
              ) : (
                messages.map((message) => (
                  <Box
                    key={message.id}
                    sx={{
                      display: 'flex',
                      alignItems: 'flex-start',
                      gap: 1,
                      mb: 2
                    }}
                  >
                    <Avatar
                      sx={{
                        bgcolor: message.role === 'user' ? 'primary.main' : 'secondary.main',
                      }}
                    >
                      {message.role === 'user' ? <PersonIcon /> : <SmartToyIcon />}
                    </Avatar>
                    <Box sx={{ flexGrow: 1 }}>
                      <Box 
                        sx={{
                          backgroundColor: message.role === 'user' ? 'grey.100' : 'grey.50',
                          p: 2,
                          borderRadius: 2,
                          whiteSpace: 'pre-wrap',
                          overflowWrap: 'break-word',
                          '& code': {
                            backgroundColor: 'background.paper',
                            p: 1,
                            borderRadius: 1,
                            display: 'block',
                            overflowX: 'auto',
                            fontFamily: 'monospace'
                          }
                        }}
                      >
                        {message.content}
                      </Box>
                      <Typography variant="caption" color="text.secondary" sx={{ ml: 1 }}>
                        {new Date(message.timestamp).toLocaleTimeString()}
                        {message.stats && (
                          ` • ${message.stats.completion_tokens || 0} tokens • ${message.stats.latency_ms ? (message.stats.latency_ms / 1000).toFixed(2) + 's' : 'N/A'}`
                        )}
                      </Typography>
                    </Box>
                  </Box>
                ))
              )}
              <div ref={messagesEndRef} />
            </Box>
            
            {/* Input area */}
            <Box sx={{ display: 'flex', gap: 1 }}>
              <TextField
                fullWidth
                multiline
                maxRows={4}
                placeholder="Type your message..."
                value={currentMessage}
                onChange={handleMessageChange}
                onKeyPress={handleKeyPress}
                disabled={sending || !selectedProvider || !selectedModel}
                variant="outlined"
              />
              <Button
                variant="contained"
                color="primary"
                endIcon={sending ? <CircularProgress size={20} /> : <SendIcon />}
                onClick={sendMessage}
                disabled={sending || !currentMessage.trim() || !selectedProvider || !selectedModel}
                sx={{ alignSelf: 'flex-start' }}
              >
                {sending ? 'Sending...' : 'Send'}
              </Button>
            </Box>
          </Paper>
        </Grid>
      </Grid>

      {/* Conversations history dialog */}
      <Dialog
        open={historyDialogOpen}
        onClose={() => setHistoryDialogOpen(false)}
        fullWidth
        maxWidth="md"
      >
        <DialogTitle>
          <Box display="flex" alignItems="center" justifyContent="space-between">
            <Typography variant="h6">Conversation History</Typography>
            <Button
              size="small"
              startIcon={<DeleteIcon />}
              onClick={() => setConversations([])}
              disabled={conversations.length === 0}
            >
              Clear All
            </Button>
          </Box>
        </DialogTitle>
        <DialogContent dividers>
          {conversations.length === 0 ? (
            <Alert severity="info">No saved conversations</Alert>
          ) : (
            <List>
              {conversations.map((conversation) => (
                <ListItem
                  key={conversation.id}
                  secondaryAction={
                    <>
                      <Button
                        size="small"
                        onClick={() => deleteConversation(conversation.id)}
                      >
                        Delete
                      </Button>
                      <Button
                        size="small"
                        variant="contained"
                        onClick={() => loadConversation(conversation)}
                      >
                        Load
                      </Button>
                    </>
                  }
                >
                  <Box sx={{ display: 'flex', flexDirection: 'column', width: '100%' }}>
                    <Typography variant="subtitle1">{conversation.title}</Typography>
                    <Typography variant="caption" color="text.secondary">
                      {new Date(conversation.timestamp).toLocaleString()} • 
                      {` ${conversation.messages.length} messages • `}
                      {conversation.provider} / {conversation.model}
                    </Typography>
                    <Typography variant="body2" noWrap sx={{ opacity: 0.7 }}>
                      {conversation.messages[0]?.content.substring(0, 60)}...
                    </Typography>
                  </Box>
                </ListItem>
              ))}
            </List>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setHistoryDialogOpen(false)}>Close</Button>
        </DialogActions>
      </Dialog>
    </PageLayout>
  );
};

export default TextPlayground;
