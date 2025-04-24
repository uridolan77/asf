// filepath: c:\code\asf\asf\bollm\frontend\src\pages\LLMManagement\Models\ModelTestPage.jsx
import React, { useState, useEffect, useRef } from 'react';
import { 
  Box, 
  Typography, 
  Container, 
  Paper,
  TextField,
  Button,
  Divider,
  Grid,
  Alert,
  IconButton,
  LinearProgress,
  Card,
  CardContent,
  Tab,
  Tabs,
  FormControlLabel,
  Switch,
  Tooltip,
  Collapse
} from '@mui/material';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import StopIcon from '@mui/icons-material/Stop';
import CodeIcon from '@mui/icons-material/Code';
import SettingsIcon from '@mui/icons-material/Settings';
import ContentCopyIcon from '@mui/icons-material/ContentCopy';
import CheckCircleOutlineIcon from '@mui/icons-material/CheckCircleOutline';
import { useNavigate, useParams, useLocation } from 'react-router-dom';
import apiService from '../../../services/api';
import ReactMarkdown from 'react-markdown';

/**
 * Model Test Page
 * For interactively testing and evaluating LLM models
 */
const ModelTestPage = () => {
  const navigate = useNavigate();
  const { modelId, providerId } = useParams();
  const location = useLocation();
  
  // Refs for streaming
  const streamingRef = useRef(false);
  const abortControllerRef = useRef(null);
  
  // Model and parameters
  const [model, setModel] = useState(null);
  const [parameters, setParameters] = useState(
    location.state?.parameters || {
      temperature: 0.7,
      top_p: 1.0,
      presence_penalty: 0.0,
      frequency_penalty: 0.0,
      max_tokens: 1000,
      stop_sequences: [],
      stream: true
    }
  );
  
  // UI state
  const [loading, setLoading] = useState(true);
  const [generating, setGenerating] = useState(false);
  const [showParameters, setShowParameters] = useState(false);
  const [currentTab, setCurrentTab] = useState(0);
  const [promptTemplates, setPromptTemplates] = useState([
    { name: 'General', prompt: 'Write a short paragraph about artificial intelligence.' },
    { name: 'Summarize', prompt: 'Summarize the following text:\n\n{{input}}' },
    { name: 'Answer Question', prompt: 'Answer the following question:\n\n{{input}}' },
    { name: 'Write Code', prompt: 'Write a function in {{language}} that {{task}}' }
  ]);
  
  // Input/Output state
  const [prompt, setPrompt] = useState('');
  const [systemPrompt, setSystemPrompt] = useState('You are a helpful AI assistant.');
  const [response, setResponse] = useState('');
  const [error, setError] = useState('');
  const [responseMetrics, setResponseMetrics] = useState({
    totalTokens: 0,
    inputTokens: 0,
    outputTokens: 0,
    startTime: null,
    endTime: null
  });
  const [copied, setCopied] = useState(false);
  
  // Load model data on mount
  useEffect(() => {
    const loadModel = async () => {
      setLoading(true);
      try {
        if (!modelId || !providerId) {
          setError('Model ID and Provider ID are required');
          return;
        }
        
        const response = await apiService.llm.getModelById(modelId, providerId);
        if (response.success) {
          setModel(response.data);
          // Set some default prompt text
          setPrompt(promptTemplates[0].prompt);
        } else {
          setError(`Failed to load model data: ${response.error}`);
        }
      } catch (err) {
        setError(`Error loading model: ${err.message}`);
      } finally {
        setLoading(false);
      }
    };
    
    loadModel();
    
    // Cleanup function to abort any pending requests
    return () => {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
    };
  }, [modelId, providerId]);
  
  // Handle parameter change
  const handleParameterChange = (param, value) => {
    setParameters(prev => ({
      ...prev,
      [param]: value
    }));
  };
  
  // Handle number parameter change
  const handleNumberChange = (e) => {
    const { name, value } = e.target;
    handleParameterChange(name, value === '' ? '' : Number(value));
  };
  
  // Copy response to clipboard
  const handleCopyResponse = () => {
    navigator.clipboard.writeText(response).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  };
  
  // Handle template change
  const handleTemplateChange = (_, newValue) => {
    setCurrentTab(newValue);
    setPrompt(promptTemplates[newValue].prompt);
  };
  
  // Generate response
  const handleGenerate = async () => {
    if (!prompt.trim()) {
      setError('Please enter a prompt.');
      return;
    }
    
    setError('');
    setResponse('');
    setGenerating(true);
    
    // Create metrics
    const startTime = new Date();
    setResponseMetrics({
      totalTokens: 0,
      inputTokens: 0,
      outputTokens: 0,
      startTime,
      endTime: null
    });
    
    // Create a new AbortController for this request
    abortControllerRef.current = new AbortController();
    streamingRef.current = parameters.stream;
    
    try {
      if (parameters.stream) {
        // Streaming generation
        const stream = await apiService.llm.generateStreamingCompletion(
          providerId,
          modelId,
          {
            prompt,
            system_prompt: systemPrompt,
            parameters: {
              temperature: parameters.temperature,
              top_p: parameters.top_p,
              presence_penalty: parameters.presence_penalty,
              frequency_penalty: parameters.frequency_penalty,
              max_tokens: parameters.max_tokens,
              stop: parameters.stop_sequences
            }
          },
          abortControllerRef.current
        );
        
        if (stream) {
          let accumulatedResponse = '';
          
          for await (const chunk of stream) {
            if (!streamingRef.current) break; // Stop if streaming was disabled
            
            try {
              accumulatedResponse += chunk.text || '';
              setResponse(accumulatedResponse);
              
              // Update token count if available
              if (chunk.stats) {
                setResponseMetrics(prev => ({
                  ...prev,
                  outputTokens: (prev.outputTokens || 0) + (chunk.stats.tokens || 0)
                }));
              }
            } catch (err) {
              console.error('Error processing chunk:', err);
            }
          }
          
          // Set final metrics
          const endTime = new Date();
          setResponseMetrics(prev => ({
            ...prev,
            endTime,
            totalTokens: (prev.inputTokens || 0) + (prev.outputTokens || 0)
          }));
        }
      } else {
        // Non-streaming generation
        const result = await apiService.llm.generateCompletion(
          providerId,
          modelId,
          {
            prompt,
            system_prompt: systemPrompt,
            parameters: {
              temperature: parameters.temperature,
              top_p: parameters.top_p,
              presence_penalty: parameters.presence_penalty,
              frequency_penalty: parameters.frequency_penalty,
              max_tokens: parameters.max_tokens,
              stop: parameters.stop_sequences
            }
          }
        );
        
        if (result.success) {
          setResponse(result.data.text || '');
          
          // Set final metrics
          const endTime = new Date();
          setResponseMetrics({
            inputTokens: result.data.usage?.prompt_tokens || 0,
            outputTokens: result.data.usage?.completion_tokens || 0,
            totalTokens: result.data.usage?.total_tokens || 0,
            startTime,
            endTime
          });
        } else {
          setError(`Error generating response: ${result.error}`);
        }
      }
    } catch (err) {
      if (err.name !== 'AbortError') {
        setError(`Error: ${err.message}`);
      }
    } finally {
      abortControllerRef.current = null;
      streamingRef.current = false;
      setGenerating(false);
    }
  };
  
  // Stop generation
  const handleStopGeneration = () => {
    if (abortControllerRef.current) {
      streamingRef.current = false;
      abortControllerRef.current.abort();
      setGenerating(false);
    }
  };
  
  // Calculate response time
  const getResponseTime = () => {
    if (!responseMetrics.startTime || !responseMetrics.endTime) return 'N/A';
    const ms = responseMetrics.endTime - responseMetrics.startTime;
    return (ms / 1000).toFixed(2) + 's';
  };
  
  return (
    <Container maxWidth="lg">
      <Box sx={{ pt: 3, pb: 5 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 4 }}>
          <IconButton 
            onClick={() => navigate('/llm/models')}
            sx={{ mr: 2 }}
          >
            <ArrowBackIcon />
          </IconButton>
          <Typography variant="h4" component="h1">
            Test Model
          </Typography>
        </Box>
        
        {error && (
          <Alert severity="error" sx={{ mb: 3 }}>
            {error}
          </Alert>
        )}
        
        {loading ? (
          <Paper sx={{ p: 3, textAlign: 'center' }}>
            <Typography>Loading model data...</Typography>
            <LinearProgress sx={{ mt: 2 }} />
          </Paper>
        ) : model ? (
          <Grid container spacing={3}>
            {/* Model Info */}
            <Grid item xs={12}>
              <Paper sx={{ p: 3 }}>
                <Grid container spacing={2} alignItems="center">
                  <Grid item xs={12} sm={8}>
                    <Typography variant="h5">{model.display_name || model.model_id}</Typography>
                    <Typography variant="body2" color="text.secondary">
                      Provider: {model.provider_id} | Type: {model.model_type || 'chat'} | 
                      Context: {model.context_window?.toLocaleString() || 'Unknown'} tokens
                    </Typography>
                  </Grid>
                  <Grid item xs={12} sm={4} sx={{ textAlign: 'right' }}>
                    <Button
                      variant="outlined"
                      startIcon={<SettingsIcon />}
                      onClick={() => setShowParameters(!showParameters)}
                      sx={{ mr: 2 }}
                    >
                      {showParameters ? 'Hide Parameters' : 'Show Parameters'}
                    </Button>
                  </Grid>
                </Grid>
                
                {/* Parameters Panel */}
                <Collapse in={showParameters}>
                  <Box sx={{ mt: 3 }}>
                    <Divider sx={{ mb: 2 }} />
                    <Typography variant="h6" gutterBottom>
                      Generation Parameters
                    </Typography>
                    
                    <Grid container spacing={2}>
                      <Grid item xs={12} sm={6} md={3}>
                        <TextField
                          label="Temperature"
                          name="temperature"
                          type="number"
                          value={parameters.temperature}
                          onChange={handleNumberChange}
                          inputProps={{
                            step: 0.05,
                            min: 0,
                            max: 2
                          }}
                          fullWidth
                          size="small"
                          margin="normal"
                          helperText="0-2.0. Higher = more random"
                        />
                      </Grid>
                      <Grid item xs={12} sm={6} md={3}>
                        <TextField
                          label="Top P"
                          name="top_p"
                          type="number"
                          value={parameters.top_p}
                          onChange={handleNumberChange}
                          inputProps={{
                            step: 0.01,
                            min: 0,
                            max: 1
                          }}
                          fullWidth
                          size="small"
                          margin="normal"
                          helperText="0-1.0. Nucleus sampling"
                        />
                      </Grid>
                      <Grid item xs={12} sm={6} md={3}>
                        <TextField
                          label="Max Tokens"
                          name="max_tokens"
                          type="number"
                          value={parameters.max_tokens}
                          onChange={handleNumberChange}
                          inputProps={{
                            step: 50,
                            min: 1
                          }}
                          fullWidth
                          size="small"
                          margin="normal"
                          helperText="Maximum response length"
                        />
                      </Grid>
                      <Grid item xs={12} sm={6} md={3}>
                        <FormControlLabel
                          control={
                            <Switch
                              checked={parameters.stream}
                              onChange={(e) => handleParameterChange('stream', e.target.checked)}
                            />
                          }
                          label="Stream Response"
                          sx={{ mt: 2 }}
                        />
                      </Grid>
                    </Grid>
                  </Box>
                </Collapse>
              </Paper>
            </Grid>
            
            {/* Prompt Input */}
            <Grid item xs={12}>
              <Paper sx={{ p: 3 }}>
                <Typography variant="h6" gutterBottom>
                  Prompt
                </Typography>
                
                <Tabs value={currentTab} onChange={handleTemplateChange} sx={{ mb: 2 }}>
                  {promptTemplates.map((template, index) => (
                    <Tab key={index} label={template.name} />
                  ))}
                </Tabs>
                
                <TextField
                  label="System Prompt (optional)"
                  value={systemPrompt}
                  onChange={(e) => setSystemPrompt(e.target.value)}
                  fullWidth
                  margin="normal"
                  helperText="Instruction to the model about how to behave"
                />
                
                <TextField
                  label="User Prompt"
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                  multiline
                  rows={5}
                  fullWidth
                  margin="normal"
                  required
                />
                
                <Box sx={{ mt: 2, display: 'flex', justifyContent: 'flex-end' }}>
                  {generating ? (
                    <Button
                      variant="outlined"
                      color="error"
                      startIcon={<StopIcon />}
                      onClick={handleStopGeneration}
                    >
                      Stop Generation
                    </Button>
                  ) : (
                    <Button
                      variant="contained"
                      startIcon={<PlayArrowIcon />}
                      onClick={handleGenerate}
                      disabled={!prompt.trim()}
                    >
                      Generate Response
                    </Button>
                  )}
                </Box>
              </Paper>
            </Grid>
            
            {/* Response Output */}
            <Grid item xs={12}>
              <Paper sx={{ p: 3 }}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                  <Typography variant="h6">
                    Response
                  </Typography>
                  {response && (
                    <Tooltip title={copied ? "Copied!" : "Copy to clipboard"}>
                      <IconButton onClick={handleCopyResponse} size="small">
                        {copied ? <CheckCircleOutlineIcon color="success" /> : <ContentCopyIcon />}
                      </IconButton>
                    </Tooltip>
                  )}
                </Box>
                
                {generating && (
                  <LinearProgress sx={{ mb: 2 }} />
                )}
                
                <Card variant="outlined" sx={{ minHeight: '200px' }}>
                  <CardContent>
                    {response ? (
                      <Box sx={{ fontFamily: 'monospace', whiteSpace: 'pre-wrap' }}>
                        <ReactMarkdown>
                          {response}
                        </ReactMarkdown>
                      </Box>
                    ) : (
                      <Typography color="text.secondary" sx={{ fontStyle: 'italic' }}>
                        Response will appear here
                      </Typography>
                    )}
                  </CardContent>
                </Card>
                
                {/* Response metrics */}
                {response && (
                  <Box sx={{ mt: 2, display: 'flex', flexWrap: 'wrap', gap: 2 }}>
                    <Typography variant="caption" color="text.secondary">
                      <strong>Time:</strong> {getResponseTime()}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      <strong>Input tokens:</strong> {responseMetrics.inputTokens || 'N/A'}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      <strong>Output tokens:</strong> {responseMetrics.outputTokens || 'N/A'}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      <strong>Total tokens:</strong> {responseMetrics.totalTokens || 'N/A'}
                    </Typography>
                  </Box>
                )}
              </Paper>
            </Grid>
          </Grid>
        ) : (
          <Alert severity="error">Model not found</Alert>
        )}
      </Box>
    </Container>
  );
};

export default ModelTestPage;