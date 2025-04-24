import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Grid,
  Paper,
  Typography,
  Box,
  Button,
  Card,
  CardContent,
  CardActions,
  Divider,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Chip,
  CircularProgress,
  Alert,
  AlertTitle
} from '@mui/material';
import {
  Cloud as CloudIcon,
  Memory as MemoryIcon,
  Storage as StorageIcon,
  SmartToy as SmartToyIcon,
  Speed as SpeedIcon,
  AutoFixHigh as AutoFixHighIcon,
  Code as CodeIcon,
  Biotech as BiotechIcon,
  PsychologyAlt as PsychologyAltIcon,
  BarChart as BarChartIcon,
  Warning as WarningIcon,
  CheckCircle as CheckCircleIcon
} from '@mui/icons-material';

import PageLayout from '../../components/Layout/PageLayout';
import { useAuth } from '../../context/AuthContext';

/**
 * LLM Management Dashboard
 * Main dashboard for the LLM Management interface
 */
const Dashboard = () => {
  const [loading, setLoading] = useState(false);
  const [systemStatus, setSystemStatus] = useState({
    gateway: 'healthy', // healthy, degraded, offline
    providers: {
      total: 5,
      active: 4,
      offline: 1
    },
    models: {
      total: 12,
      available: 10,
      unavailable: 2
    },
    recentRequests: {
      total: 1245,
      success: 1232,
      error: 13,
      latestTimestamp: new Date().toISOString()
    },
    cache: {
      hitRate: 0.68,
      size: '45MB',
      items: 12450
    }
  });

  const navigate = useNavigate();
  const { user } = useAuth();

  // Mock stats for the dashboard
  const mockStats = {
    providersData: [
      { name: 'OpenAI', status: 'active', modelsCount: 5, icon: <MemoryIcon /> },
      { name: 'Anthropic', status: 'active', modelsCount: 3, icon: <MemoryIcon /> },
      { name: 'Google', status: 'active', modelsCount: 2, icon: <MemoryIcon /> },
      { name: 'Mistral', status: 'active', modelsCount: 4, icon: <MemoryIcon /> },
      { name: 'Cohere', status: 'offline', modelsCount: 0, icon: <MemoryIcon /> }
    ],
    modelUsageData: [
      { name: 'gpt-4', provider: 'OpenAI', usage: 542, icon: <SmartToyIcon /> },
      { name: 'claude-3-opus', provider: 'Anthropic', usage: 324, icon: <SmartToyIcon /> },
      { name: 'gemini-pro', provider: 'Google', usage: 213, icon: <SmartToyIcon /> },
      { name: 'gpt-4-turbo', provider: 'OpenAI', usage: 187, icon: <SmartToyIcon /> },
      { name: 'mistral-large', provider: 'Mistral', usage: 128, icon: <SmartToyIcon /> }
    ],
    quickAccessItems: [
      { title: 'Provider Management', description: 'Configure providers and API keys', icon: <CloudIcon color="primary" />, path: '/llm/providers' },
      { title: 'Model Management', description: 'Configure and monitor LLM models', icon: <MemoryIcon color="primary" />, path: '/llm/models' },
      { title: 'Cache Management', description: 'Configure and monitor cache', icon: <StorageIcon color="primary" />, path: '/llm/cache/config' },
      { title: 'Text Completion', description: 'Interactive playground for text completion', icon: <SmartToyIcon color="primary" />, path: '/llm/playground/text' },
      { title: 'Chat Playground', description: 'Interactive chat playground', icon: <CodeIcon color="primary" />, path: '/llm/playground/chat' },
      { title: 'DSPy Integration', description: 'Manage DSPy modules', icon: <PsychologyAltIcon color="primary" />, path: '/llm/advanced/dspy' }
    ],
    recentErrors: [
      { id: 'err-001', timestamp: '2025-04-20T10:15:00Z', model: 'gpt-4', message: 'Rate limit exceeded', severity: 'warning' },
      { id: 'err-002', timestamp: '2025-04-20T09:45:12Z', model: 'claude-3-opus', message: 'Timeout waiting for response', severity: 'error' },
      { id: 'err-003', timestamp: '2025-04-20T08:32:45Z', model: 'mistral-large', message: 'Invalid API key', severity: 'error' }
    ]
  };

  return (
    <PageLayout
      title="LLM Management Dashboard"
      breadcrumbs={[]}
      user={user}
    >
      {loading ? (
        <Box sx={{ display: 'flex', justifyContent: 'center', p: 5 }}>
          <CircularProgress />
        </Box>
      ) : (
        <Grid container spacing={3}>
          {/* System Status */}
          <Grid item xs={12}>
            <Paper sx={{ p: 3, mb: 2 }}>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                <Typography variant="h5">System Status</Typography>
                <Chip 
                  label={systemStatus.gateway === 'healthy' ? 'All Systems Operational' : 'System Degraded'} 
                  color={systemStatus.gateway === 'healthy' ? 'success' : 'warning'}
                  icon={systemStatus.gateway === 'healthy' ? <CheckCircleIcon /> : <WarningIcon />}
                />
              </Box>
              
              <Grid container spacing={2}>
                <Grid item xs={12} sm={6} md={3}>
                  <Card variant="outlined">
                    <CardContent>
                      <Typography variant="h6" color="primary" gutterBottom>
                        Providers
                      </Typography>
                      <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                        <CloudIcon color="primary" sx={{ mr: 1 }} />
                        <Typography variant="h4">{systemStatus.providers.total}</Typography>
                      </Box>
                      <Typography variant="body2" color="text.secondary">
                        {systemStatus.providers.active} active, {systemStatus.providers.offline} offline
                      </Typography>
                    </CardContent>
                    <CardActions>
                      <Button size="small" onClick={() => navigate('/llm/providers')}>Manage Providers</Button>
                    </CardActions>
                  </Card>
                </Grid>
                <Grid item xs={12} sm={6} md={3}>
                  <Card variant="outlined">
                    <CardContent>
                      <Typography variant="h6" color="primary" gutterBottom>
                        Models
                      </Typography>
                      <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                        <MemoryIcon color="primary" sx={{ mr: 1 }} />
                        <Typography variant="h4">{systemStatus.models.total}</Typography>
                      </Box>
                      <Typography variant="body2" color="text.secondary">
                        {systemStatus.models.available} available, {systemStatus.models.unavailable} unavailable
                      </Typography>
                    </CardContent>
                    <CardActions>
                      <Button size="small" onClick={() => navigate('/llm/models')}>Manage Models</Button>
                    </CardActions>
                  </Card>
                </Grid>
                <Grid item xs={12} sm={6} md={3}>
                  <Card variant="outlined">
                    <CardContent>
                      <Typography variant="h6" color="primary" gutterBottom>
                        Recent Requests
                      </Typography>
                      <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                        <CodeIcon color="primary" sx={{ mr: 1 }} />
                        <Typography variant="h4">{systemStatus.recentRequests.total}</Typography>
                      </Box>
                      <Typography variant="body2" color="text.secondary">
                        {systemStatus.recentRequests.success} successful, {systemStatus.recentRequests.error} failed
                      </Typography>
                    </CardContent>
                    <CardActions>
                      <Button size="small" onClick={() => navigate('/llm/requests/logs')}>View Logs</Button>
                    </CardActions>
                  </Card>
                </Grid>
                <Grid item xs={12} sm={6} md={3}>
                  <Card variant="outlined">
                    <CardContent>
                      <Typography variant="h6" color="primary" gutterBottom>
                        Cache Performance
                      </Typography>
                      <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                        <StorageIcon color="primary" sx={{ mr: 1 }} />
                        <Typography variant="h4">{(systemStatus.cache.hitRate * 100).toFixed(1)}%</Typography>
                      </Box>
                      <Typography variant="body2" color="text.secondary">
                        {systemStatus.cache.items} items ({systemStatus.cache.size})
                      </Typography>
                    </CardContent>
                    <CardActions>
                      <Button size="small" onClick={() => navigate('/llm/cache/stats')}>View Cache Stats</Button>
                    </CardActions>
                  </Card>
                </Grid>
              </Grid>
            </Paper>
          </Grid>

          {/* Quick Access Cards */}
          <Grid item xs={12}>
            <Typography variant="h6" sx={{ mb: 2 }}>Quick Access</Typography>
            <Grid container spacing={2}>
              {mockStats.quickAccessItems.map((item, index) => (
                <Grid item xs={12} sm={6} md={4} key={index}>
                  <Card 
                    variant="outlined" 
                    sx={{ 
                      cursor: 'pointer', 
                      transition: 'transform 0.2s', 
                      '&:hover': { 
                        transform: 'translateY(-4px)', 
                        boxShadow: 3 
                      } 
                    }}
                    onClick={() => navigate(item.path)}
                  >
                    <CardContent>
                      <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                        {item.icon}
                        <Typography variant="h6" sx={{ ml: 1 }}>
                          {item.title}
                        </Typography>
                      </Box>
                      <Typography variant="body2" color="text.secondary">
                        {item.description}
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
              ))}
            </Grid>
          </Grid>

          {/* Provider Status & Model Usage */}
          <Grid item xs={12} md={6}>
            <Paper sx={{ p: 3, height: '100%' }}>
              <Typography variant="h6" gutterBottom>Provider Status</Typography>
              <List>
                {mockStats.providersData.map((provider, index) => (
                  <React.Fragment key={index}>
                    <ListItem>
                      <ListItemIcon>
                        {provider.icon}
                      </ListItemIcon>
                      <ListItemText 
                        primary={provider.name} 
                        secondary={`${provider.modelsCount} models`} 
                      />
                      <Chip 
                        label={provider.status} 
                        color={provider.status === 'active' ? 'success' : 'error'} 
                        size="small" 
                      />
                    </ListItem>
                    {index < mockStats.providersData.length - 1 && <Divider variant="inset" component="li" />}
                  </React.Fragment>
                ))}
              </List>
              <Box sx={{ mt: 2, textAlign: 'right' }}>
                <Button size="small" onClick={() => navigate('/llm/providers')}>
                  View All Providers
                </Button>
              </Box>
            </Paper>
          </Grid>
          
          <Grid item xs={12} md={6}>
            <Paper sx={{ p: 3, height: '100%' }}>
              <Typography variant="h6" gutterBottom>Top Models by Usage</Typography>
              <List>
                {mockStats.modelUsageData.map((model, index) => (
                  <React.Fragment key={index}>
                    <ListItem>
                      <ListItemIcon>
                        {model.icon}
                      </ListItemIcon>
                      <ListItemText 
                        primary={model.name} 
                        secondary={`Provider: ${model.provider}`} 
                      />
                      <Chip 
                        label={`${model.usage} requests`} 
                        color="primary" 
                        size="small" 
                        variant="outlined"
                      />
                    </ListItem>
                    {index < mockStats.modelUsageData.length - 1 && <Divider variant="inset" component="li" />}
                  </React.Fragment>
                ))}
              </List>
              <Box sx={{ mt: 2, textAlign: 'right' }}>
                <Button size="small" onClick={() => navigate('/llm/models/usage')}>
                  View Usage Stats
                </Button>
              </Box>
            </Paper>
          </Grid>

          {/* Recent Errors */}
          <Grid item xs={12}>
            <Paper sx={{ p: 3 }}>
              <Typography variant="h6" gutterBottom>Recent Errors</Typography>
              {mockStats.recentErrors.length > 0 ? (
                <Grid container spacing={2}>
                  {mockStats.recentErrors.map((error, index) => (
                    <Grid item xs={12} key={index}>
                      <Alert 
                        severity={error.severity}
                        sx={{ mb: index < mockStats.recentErrors.length - 1 ? 1 : 0 }}
                      >
                        <AlertTitle>{error.model} - {new Date(error.timestamp).toLocaleString()}</AlertTitle>
                        {error.message}
                      </Alert>
                    </Grid>
                  ))}
                  <Grid item xs={12}>
                    <Box sx={{ mt: 1, textAlign: 'right' }}>
                      <Button size="small" onClick={() => navigate('/llm/requests/logs?filter=errors')}>
                        View All Errors
                      </Button>
                    </Box>
                  </Grid>
                </Grid>
              ) : (
                <Box sx={{ p: 2, textAlign: 'center' }}>
                  <CheckCircleIcon color="success" sx={{ fontSize: 40, mb: 1 }} />
                  <Typography variant="body1">No errors in the past 24 hours</Typography>
                </Box>
              )}
            </Paper>
          </Grid>

          {/* Advanced Features Promotion */}
          <Grid item xs={12}>
            <Paper sx={{ p: 3, bgcolor: 'primary.dark', color: 'white' }}>
              <Grid container spacing={2} alignItems="center">
                <Grid item xs={12} md={8}>
                  <Typography variant="h5" gutterBottom>Advanced LLM Features</Typography>
                  <Typography variant="body1" paragraph>
                    Explore our advanced LLM management features including DSPy integration, 
                    CL-PEFT for continual learning, intervention strategies, and more.
                  </Typography>
                  <Button 
                    variant="contained" 
                    color="secondary"
                    onClick={() => navigate('/llm/advanced/dspy')}
                  >
                    Explore Advanced Features
                  </Button>
                </Grid>
                <Grid item xs={12} md={4} sx={{ textAlign: 'center' }}>
                  <Box sx={{ display: { xs: 'none', md: 'block' } }}>
                    <AutoFixHighIcon sx={{ fontSize: 100, opacity: 0.8 }} />
                  </Box>
                </Grid>
              </Grid>
            </Paper>
          </Grid>
        </Grid>
      )}
    </PageLayout>
  );
};

export default Dashboard;