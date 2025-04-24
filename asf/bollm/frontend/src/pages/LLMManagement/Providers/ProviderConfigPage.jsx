import React, { useState } from 'react';
import { 
  Box, 
  Typography, 
  Container, 
  Paper,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Button,
  FormHelperText,
  Divider,
  Grid,
  Alert,
  IconButton,
  Tooltip
} from '@mui/material';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';
import InfoIcon from '@mui/icons-material/Info';
import { useNavigate } from 'react-router-dom';

/**
 * Provider Configuration Page
 * For adding/editing LLM provider configurations
 */
const ProviderConfigPage = () => {
  const navigate = useNavigate();
  
  // Form state
  const [formData, setFormData] = useState({
    name: '',
    providerType: '',
    baseUrl: '',
    authType: 'api_key',
    requestTimeout: 30,
    retryAttempts: 3,
    rateLimit: 10,
    description: ''
  });
  
  const [saved, setSaved] = useState(false);
  
  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };
  
  const handleSubmit = (e) => {
    e.preventDefault();
    // API call would go here in a real implementation
    console.log('Form submitted:', formData);
    setSaved(true);
    setTimeout(() => {
      setSaved(false);
    }, 3000);
  };
  
  return (
    <Container maxWidth="md">
      <Box sx={{ pt: 3, pb: 5 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 4 }}>
          <IconButton 
            onClick={() => navigate('/llm/providers')}
            sx={{ mr: 2 }}
          >
            <ArrowBackIcon />
          </IconButton>
          <Typography variant="h4" component="h1">
            Provider Configuration
          </Typography>
        </Box>
        
        {saved && (
          <Alert severity="success" sx={{ mb: 3 }}>
            Provider configuration saved successfully!
          </Alert>
        )}
        
        <Paper sx={{ p: 3 }}>
          <form onSubmit={handleSubmit}>
            <Grid container spacing={3}>
              {/* Basic Info */}
              <Grid item xs={12}>
                <Typography variant="h6">Basic Information</Typography>
                <Divider sx={{ mt: 1, mb: 2 }} />
              </Grid>
              
              <Grid item xs={12} md={6}>
                <TextField
                  required
                  fullWidth
                  label="Provider Name"
                  name="name"
                  value={formData.name}
                  onChange={handleChange}
                  helperText="A unique name for this provider"
                />
              </Grid>
              
              <Grid item xs={12} md={6}>
                <FormControl fullWidth required>
                  <InputLabel>Provider Type</InputLabel>
                  <Select
                    name="providerType"
                    value={formData.providerType}
                    onChange={handleChange}
                    label="Provider Type"
                  >
                    <MenuItem value="openai">OpenAI</MenuItem>
                    <MenuItem value="anthropic">Anthropic</MenuItem>
                    <MenuItem value="cohere">Cohere</MenuItem>
                    <MenuItem value="huggingface">HuggingFace</MenuItem>
                    <MenuItem value="azure_openai">Azure OpenAI</MenuItem>
                    <MenuItem value="custom">Custom</MenuItem>
                  </Select>
                  <FormHelperText>Select the LLM service provider</FormHelperText>
                </FormControl>
              </Grid>
              
              <Grid item xs={12}>
                <TextField
                  fullWidth
                  label="Description"
                  name="description"
                  value={formData.description}
                  onChange={handleChange}
                  multiline
                  rows={2}
                />
              </Grid>
              
              {/* Connection Settings */}
              <Grid item xs={12} sx={{ mt: 2 }}>
                <Typography variant="h6">Connection Settings</Typography>
                <Divider sx={{ mt: 1, mb: 2 }} />
              </Grid>
              
              <Grid item xs={12}>
                <TextField
                  fullWidth
                  label="Base URL"
                  name="baseUrl"
                  value={formData.baseUrl}
                  onChange={handleChange}
                  helperText="Override the default API endpoint URL (optional)"
                />
              </Grid>
              
              <Grid item xs={12} md={6}>
                <FormControl fullWidth required>
                  <InputLabel>Authentication Type</InputLabel>
                  <Select
                    name="authType"
                    value={formData.authType}
                    onChange={handleChange}
                    label="Authentication Type"
                  >
                    <MenuItem value="api_key">API Key</MenuItem>
                    <MenuItem value="oauth">OAuth 2.0</MenuItem>
                    <MenuItem value="bearer">Bearer Token</MenuItem>
                    <MenuItem value="basic">Basic Auth</MenuItem>
                    <MenuItem value="none">No Auth</MenuItem>
                  </Select>
                </FormControl>
              </Grid>
              
              {/* Request Settings */}
              <Grid item xs={12} sx={{ mt: 2 }}>
                <Typography variant="h6">
                  Request Settings 
                  <Tooltip title="These settings affect how requests to this provider are handled by the LLM gateway">
                    <IconButton size="small">
                      <InfoIcon fontSize="small" />
                    </IconButton>
                  </Tooltip>
                </Typography>
                <Divider sx={{ mt: 1, mb: 2 }} />
              </Grid>
              
              <Grid item xs={12} md={4}>
                <TextField
                  fullWidth
                  type="number"
                  label="Request Timeout (sec)"
                  name="requestTimeout"
                  value={formData.requestTimeout}
                  onChange={handleChange}
                  inputProps={{ min: 1, max: 300 }}
                />
              </Grid>
              
              <Grid item xs={12} md={4}>
                <TextField
                  fullWidth
                  type="number"
                  label="Retry Attempts"
                  name="retryAttempts"
                  value={formData.retryAttempts}
                  onChange={handleChange}
                  inputProps={{ min: 0, max: 10 }}
                />
              </Grid>
              
              <Grid item xs={12} md={4}>
                <TextField
                  fullWidth
                  type="number"
                  label="Rate Limit (req/min)"
                  name="rateLimit"
                  value={formData.rateLimit}
                  onChange={handleChange}
                  inputProps={{ min: 0 }}
                  helperText="0 = no limit"
                />
              </Grid>
              
              {/* Submit */}
              <Grid item xs={12} sx={{ mt: 3 }}>
                <Box sx={{ display: 'flex', justifyContent: 'flex-end', gap: 2 }}>
                  <Button 
                    variant="outlined" 
                    onClick={() => navigate('/llm/providers')}
                  >
                    Cancel
                  </Button>
                  <Button 
                    type="submit" 
                    variant="contained" 
                    color="primary"
                  >
                    Save Configuration
                  </Button>
                </Box>
              </Grid>
            </Grid>
          </form>
        </Paper>
      </Box>
    </Container>
  );
};

export default ProviderConfigPage;