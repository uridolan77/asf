import React, { useState, useEffect, useRef } from 'react';
import {
  Box,
  Typography,
  Paper,
  Grid,
  TextField,
  Button,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  FormHelperText,
  Divider,
  Card,
  CardContent,
  CardHeader,
  CircularProgress,
  Alert,
  Chip,
  Slider,
  IconButton,
  Tooltip,
  Switch,
  FormControlLabel,
  useTheme
} from '@mui/material';
import {
  Send as SendIcon,
  ContentCopy as CopyIcon,
  Delete as DeleteIcon,
  Save as SaveIcon,
  Refresh as RefreshIcon,
  Settings as SettingsIcon,
  History as HistoryIcon,
  Autorenew as AutorenewIcon
} from '@mui/icons-material';
import { useSnackbar } from 'notistack';
import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { atomDark } from 'react-syntax-highlighter/dist/esm/styles/prism';
import remarkGfm from 'remark-gfm';

import { generateText } from '../../../services/cl_peft_service';

// Sample prompts for different domains
const SAMPLE_PROMPTS = {
  medical: [
    "Explain the pathophysiology of type 2 diabetes mellitus.",
    "What are the latest treatment options for Alzheimer's disease?",
    "Describe the mechanism of action of ACE inhibitors.",
    "What are the clinical manifestations of rheumatoid arthritis?",
    "Explain the differences between CT and MRI imaging techniques."
  ],
  scientific: [
    "Explain the concept of quantum entanglement in simple terms.",
    "What are the potential applications of CRISPR-Cas9 technology?",
    "Describe the process of neural network training in machine learning.",
    "What are the environmental impacts of climate change?",
    "Explain the principles behind mRNA vaccine technology."
  ],
  general: [
    "Write a summary of the latest advancements in artificial intelligence.",
    "Explain the concept of continual learning in language models.",
    "What are the ethical considerations in medical AI applications?",
    "Describe the challenges in implementing personalized medicine.",
    "What is the future of telemedicine in healthcare?"
  ]
};

const TextGeneration = ({ adapter, onRefresh }) => {
  const theme = useTheme();
  const { enqueueSnackbar } = useSnackbar();
  const responseRef = useRef(null);
  
  const [prompt, setPrompt] = useState('');
  const [generating, setGenerating] = useState(false);
  const [generatedText, setGeneratedText] = useState('');
  const [generationHistory, setGenerationHistory] = useState([]);
  const [showSettings, setShowSettings] = useState(false);
  const [promptCategory, setPromptCategory] = useState('medical');
  
  // Generation parameters
  const [params, setParams] = useState({
    max_new_tokens: 200,
    temperature: 0.7,
    top_p: 0.9,
    do_sample: true,
    num_beams: 1,
    repetition_penalty: 1.1
  });
  
  // Scroll to response when generated
  useEffect(() => {
    if (generatedText && responseRef.current) {
      responseRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [generatedText]);
  
  const handlePromptChange = (e) => {
    setPrompt(e.target.value);
  };
  
  const handleParamChange = (param) => (event, newValue) => {
    setParams({
      ...params,
      [param]: newValue
    });
  };
  
  const handleSwitchChange = (param) => (event) => {
    setParams({
      ...params,
      [param]: event.target.checked
    });
  };
  
  const handleSelectChange = (e) => {
    setParams({
      ...params,
      [e.target.name]: e.target.value
    });
  };
  
  const handlePromptCategoryChange = (e) => {
    setPromptCategory(e.target.value);
  };
  
  const handleSamplePrompt = (samplePrompt) => {
    setPrompt(samplePrompt);
  };
  
  const handleGenerate = async () => {
    if (!prompt.trim()) {
      enqueueSnackbar('Please enter a prompt', { variant: 'warning' });
      return;
    }
    
    setGenerating(true);
    setGeneratedText('');
    
    try {
      const result = await generateText(adapter.adapter_id, {
        prompt,
        ...params
      });
      
      setGeneratedText(result.generated_text);
      
      // Add to history
      const historyItem = {
        id: Date.now(),
        prompt,
        response: result.generated_text,
        timestamp: new Date().toISOString(),
        params: { ...params }
      };
      
      setGenerationHistory(prev => [historyItem, ...prev]);
      
      enqueueSnackbar('Text generated successfully', { variant: 'success' });
    } catch (error) {
      console.error('Error generating text:', error);
      enqueueSnackbar('Failed to generate text', { variant: 'error' });
    } finally {
      setGenerating(false);
    }
  };
  
  const handleCopyText = (text) => {
    navigator.clipboard.writeText(text);
    enqueueSnackbar('Text copied to clipboard', { variant: 'success' });
  };
  
  const handleClearPrompt = () => {
    setPrompt('');
    setGeneratedText('');
  };
  
  const handleClearHistory = () => {
    setGenerationHistory([]);
    enqueueSnackbar('History cleared', { variant: 'info' });
  };
  
  const handleUseHistoryItem = (historyItem) => {
    setPrompt(historyItem.prompt);
    setParams(historyItem.params);
    setGeneratedText(historyItem.response);
  };
  
  const toggleSettings = () => {
    setShowSettings(!showSettings);
  };
  
  if (!adapter) {
    return null;
  }
  
  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h5" component="h2">
          Text Generation
        </Typography>
        <Box sx={{ display: 'flex', gap: 1 }}>
          <Button
            variant="outlined"
            size="small"
            startIcon={<SettingsIcon />}
            onClick={toggleSettings}
          >
            {showSettings ? 'Hide Settings' : 'Show Settings'}
          </Button>
          <Button
            variant="outlined"
            size="small"
            startIcon={<RefreshIcon />}
            onClick={onRefresh}
          >
            Refresh
          </Button>
        </Box>
      </Box>
      
      <Grid container spacing={3}>
        <Grid item xs={12} md={showSettings ? 8 : 12}>
          <Card variant="outlined">
            <CardHeader title="Generate Text" />
            <Divider />
            <CardContent>
              <Box sx={{ mb: 3 }}>
                <Typography variant="subtitle2" gutterBottom>
                  Sample Prompts
                </Typography>
                <FormControl size="small" sx={{ mb: 2, minWidth: 200 }}>
                  <InputLabel>Category</InputLabel>
                  <Select
                    value={promptCategory}
                    onChange={handlePromptCategoryChange}
                    label="Category"
                  >
                    <MenuItem value="medical">Medical</MenuItem>
                    <MenuItem value="scientific">Scientific</MenuItem>
                    <MenuItem value="general">General</MenuItem>
                  </Select>
                </FormControl>
                <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                  {SAMPLE_PROMPTS[promptCategory].map((samplePrompt, index) => (
                    <Chip
                      key={index}
                      label={samplePrompt.length > 40 ? samplePrompt.substring(0, 40) + '...' : samplePrompt}
                      onClick={() => handleSamplePrompt(samplePrompt)}
                      clickable
                    />
                  ))}
                </Box>
              </Box>
              
              <TextField
                label="Prompt"
                multiline
                rows={4}
                fullWidth
                value={prompt}
                onChange={handlePromptChange}
                placeholder="Enter your prompt here..."
                variant="outlined"
                sx={{ mb: 2 }}
              />
              
              <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 3 }}>
                <Button
                  variant="outlined"
                  color="secondary"
                  startIcon={<DeleteIcon />}
                  onClick={handleClearPrompt}
                >
                  Clear
                </Button>
                <Button
                  variant="contained"
                  color="primary"
                  startIcon={generating ? <CircularProgress size={20} /> : <SendIcon />}
                  onClick={handleGenerate}
                  disabled={generating || !prompt.trim()}
                >
                  {generating ? 'Generating...' : 'Generate'}
                </Button>
              </Box>
              
              {generatedText && (
                <Box ref={responseRef}>
                  <Typography variant="subtitle1" gutterBottom>
                    Generated Text
                  </Typography>
                  <Paper 
                    variant="outlined" 
                    sx={{ 
                      p: 2, 
                      maxHeight: 400, 
                      overflow: 'auto',
                      bgcolor: theme.palette.mode === 'dark' ? 'rgba(0, 0, 0, 0.1)' : 'rgba(0, 0, 0, 0.02)'
                    }}
                  >
                    <Box sx={{ mb: 2, display: 'flex', justifyContent: 'flex-end' }}>
                      <Tooltip title="Copy to clipboard">
                        <IconButton size="small" onClick={() => handleCopyText(generatedText)}>
                          <CopyIcon fontSize="small" />
                        </IconButton>
                      </Tooltip>
                    </Box>
                    <ReactMarkdown
                      remarkPlugins={[remarkGfm]}
                      components={{
                        code({node, inline, className, children, ...props}) {
                          const match = /language-(\w+)/.exec(className || '')
                          return !inline && match ? (
                            <SyntaxHighlighter
                              style={atomDark}
                              language={match[1]}
                              PreTag="div"
                              {...props}
                            >
                              {String(children).replace(/\n$/, '')}
                            </SyntaxHighlighter>
                          ) : (
                            <code className={className} {...props}>
                              {children}
                            </code>
                          )
                        }
                      }}
                    >
                      {generatedText}
                    </ReactMarkdown>
                  </Paper>
                </Box>
              )}
            </CardContent>
          </Card>
          
          {generationHistory.length > 0 && (
            <Card variant="outlined" sx={{ mt: 3 }}>
              <CardHeader 
                title="Generation History" 
                action={
                  <Button
                    variant="text"
                    size="small"
                    startIcon={<DeleteIcon />}
                    onClick={handleClearHistory}
                  >
                    Clear History
                  </Button>
                }
              />
              <Divider />
              <CardContent sx={{ maxHeight: 300, overflow: 'auto' }}>
                {generationHistory.map((item) => (
                  <Paper 
                    key={item.id} 
                    variant="outlined" 
                    sx={{ p: 2, mb: 2, bgcolor: theme.palette.mode === 'dark' ? 'rgba(0, 0, 0, 0.1)' : 'rgba(0, 0, 0, 0.02)' }}
                  >
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                      <Typography variant="subtitle2" noWrap sx={{ maxWidth: '70%' }}>
                        {item.prompt.length > 50 ? item.prompt.substring(0, 50) + '...' : item.prompt}
                      </Typography>
                      <Box>
                        <Tooltip title="Use this prompt and settings">
                          <IconButton size="small" onClick={() => handleUseHistoryItem(item)}>
                            <AutorenewIcon fontSize="small" />
                          </IconButton>
                        </Tooltip>
                        <Tooltip title="Copy response">
                          <IconButton size="small" onClick={() => handleCopyText(item.response)}>
                            <CopyIcon fontSize="small" />
                          </IconButton>
                        </Tooltip>
                      </Box>
                    </Box>
                    <Typography variant="caption" color="textSecondary">
                      {new Date(item.timestamp).toLocaleString()}
                    </Typography>
                    <Box sx={{ mt: 1, display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                      <Chip 
                        label={`temp: ${item.params.temperature}`} 
                        size="small" 
                        variant="outlined" 
                        sx={{ height: 20, fontSize: '0.7rem' }}
                      />
                      <Chip 
                        label={`top_p: ${item.params.top_p}`} 
                        size="small" 
                        variant="outlined" 
                        sx={{ height: 20, fontSize: '0.7rem' }}
                      />
                      <Chip 
                        label={`tokens: ${item.params.max_new_tokens}`} 
                        size="small" 
                        variant="outlined" 
                        sx={{ height: 20, fontSize: '0.7rem' }}
                      />
                    </Box>
                  </Paper>
                ))}
              </CardContent>
            </Card>
          )}
        </Grid>
        
        {showSettings && (
          <Grid item xs={12} md={4}>
            <Card variant="outlined">
              <CardHeader title="Generation Settings" />
              <Divider />
              <CardContent>
                <Box sx={{ mb: 3 }}>
                  <Typography gutterBottom>
                    Maximum Length: {params.max_new_tokens} tokens
                  </Typography>
                  <Slider
                    value={params.max_new_tokens}
                    onChange={handleParamChange('max_new_tokens')}
                    step={10}
                    marks={[
                      { value: 50, label: '50' },
                      { value: 200, label: '200' },
                      { value: 500, label: '500' }
                    ]}
                    min={10}
                    max={500}
                    valueLabelDisplay="auto"
                  />
                  <FormHelperText>
                    Maximum number of tokens to generate
                  </FormHelperText>
                </Box>
                
                <Box sx={{ mb: 3 }}>
                  <Typography gutterBottom>
                    Temperature: {params.temperature}
                  </Typography>
                  <Slider
                    value={params.temperature}
                    onChange={handleParamChange('temperature')}
                    step={0.1}
                    marks={[
                      { value: 0, label: '0' },
                      { value: 0.7, label: '0.7' },
                      { value: 1.5, label: '1.5' }
                    ]}
                    min={0}
                    max={1.5}
                    valueLabelDisplay="auto"
                  />
                  <FormHelperText>
                    Higher values produce more diverse outputs
                  </FormHelperText>
                </Box>
                
                <Box sx={{ mb: 3 }}>
                  <Typography gutterBottom>
                    Top-p: {params.top_p}
                  </Typography>
                  <Slider
                    value={params.top_p}
                    onChange={handleParamChange('top_p')}
                    step={0.05}
                    marks={[
                      { value: 0.5, label: '0.5' },
                      { value: 0.9, label: '0.9' },
                      { value: 1, label: '1.0' }
                    ]}
                    min={0.1}
                    max={1}
                    valueLabelDisplay="auto"
                  />
                  <FormHelperText>
                    Nucleus sampling parameter
                  </FormHelperText>
                </Box>
                
                <Box sx={{ mb: 3 }}>
                  <FormControlLabel
                    control={
                      <Switch
                        checked={params.do_sample}
                        onChange={handleSwitchChange('do_sample')}
                      />
                    }
                    label="Use sampling"
                  />
                  <FormHelperText>
                    If disabled, uses greedy decoding
                  </FormHelperText>
                </Box>
                
                <Divider sx={{ my: 2 }} />
                
                <Typography variant="subtitle2" gutterBottom>
                  Advanced Settings
                </Typography>
                
                <Box sx={{ mb: 3 }}>
                  <Typography gutterBottom>
                    Repetition Penalty: {params.repetition_penalty}
                  </Typography>
                  <Slider
                    value={params.repetition_penalty}
                    onChange={handleParamChange('repetition_penalty')}
                    step={0.1}
                    marks={[
                      { value: 1, label: '1.0' },
                      { value: 1.5, label: '1.5' },
                      { value: 2, label: '2.0' }
                    ]}
                    min={1}
                    max={2}
                    valueLabelDisplay="auto"
                  />
                  <FormHelperText>
                    Penalizes repetition in generated text
                  </FormHelperText>
                </Box>
                
                <Box sx={{ mb: 3 }}>
                  <FormControl fullWidth size="small">
                    <InputLabel>Number of Beams</InputLabel>
                    <Select
                      name="num_beams"
                      value={params.num_beams}
                      onChange={handleSelectChange}
                      label="Number of Beams"
                    >
                      <MenuItem value={1}>1 (No beam search)</MenuItem>
                      <MenuItem value={2}>2</MenuItem>
                      <MenuItem value={4}>4</MenuItem>
                      <MenuItem value={8}>8</MenuItem>
                    </Select>
                    <FormHelperText>
                      Number of beams for beam search
                    </FormHelperText>
                  </FormControl>
                </Box>
                
                <Button
                  variant="outlined"
                  fullWidth
                  startIcon={<RefreshIcon />}
                  onClick={() => setParams({
                    max_new_tokens: 200,
                    temperature: 0.7,
                    top_p: 0.9,
                    do_sample: true,
                    num_beams: 1,
                    repetition_penalty: 1.1
                  })}
                >
                  Reset to Defaults
                </Button>
              </CardContent>
            </Card>
            
            <Card variant="outlined" sx={{ mt: 3 }}>
              <CardHeader title="Adapter Information" />
              <Divider />
              <CardContent>
                <Typography variant="subtitle2" gutterBottom>
                  Adapter Name
                </Typography>
                <Typography variant="body2" paragraph>
                  {adapter.adapter_name}
                </Typography>
                
                <Typography variant="subtitle2" gutterBottom>
                  Base Model
                </Typography>
                <Typography variant="body2" paragraph>
                  {adapter.base_model_name}
                </Typography>
                
                <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                  <Chip 
                    label={adapter.cl_strategy} 
                    size="small" 
                    color="primary"
                    variant="outlined"
                  />
                  <Chip 
                    label={adapter.peft_method} 
                    size="small" 
                    color="primary"
                    variant="outlined"
                  />
                </Box>
              </CardContent>
            </Card>
          </Grid>
        )}
      </Grid>
    </Box>
  );
};

export default TextGeneration;
