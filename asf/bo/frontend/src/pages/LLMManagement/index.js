import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { 
  Box, 
  Tab, 
  Tabs, 
  Paper, 
  Typography, 
  Alert, 
  Button,
  CircularProgress
} from '@mui/material';
import { 
  SmartToy as SmartToyIcon,
  Psychology as PsychologyIcon,
  Biotech as BiotechIcon,
  BarChart as BarChartIcon,
  Refresh as RefreshIcon
} from '@mui/icons-material';

import PageLayout from '../../components/Layout/PageLayout';
import { ContentLoader } from '../../components/UI/LoadingIndicators';
import apiService from '../../services/api';
import { useNotification } from '../../context/NotificationContext';
import useApi from '../../hooks/useApi';

// Import LLM components
import GatewayDashboard from './GatewayDashboard';
import DSPyDashboard from './DSPyDashboard';
import BiomedLMDashboard from './BiomedLMDashboard';
import UsageDashboard from './UsageDashboard';

/**
 * LLM Management page with tabs for different LLM components
 */
const LLMManagement = () => {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState(0);
  const [refreshing, setRefreshing] = useState(false);
  const [llmStatus, setLlmStatus] = useState(null);
  
  const navigate = useNavigate();
  const { showSuccess, showError } = useNotification();
  
  // Use API hook for fetching user data
  const { 
    data: userData, 
    loading: userLoading, 
    error: userError,
    execute: fetchUser 
  } = useApi(apiService.auth.me, {
    loadOnMount: true,
    onSuccess: (data) => {
      setUser(data);
      setLoading(false);
    },
    onError: (error) => {
      console.error('Failed to fetch user data:', error);
      if (error.includes('401') || error.includes('403')) {
        handleLogout();
      }
      setLoading(false);
    }
  });
  
  // Load LLM status on mount
  useEffect(() => {
    if (!loading) {
      loadLlmStatus();
    }
  }, [loading]);
  
  // Handle logout
  const handleLogout = () => {
    localStorage.removeItem('token');
    navigate('/');
  };
  
  // Handle tab change
  const handleTabChange = (_, newValue) => {
    setActiveTab(newValue);
  };
  
  // Load LLM status
  const loadLlmStatus = async () => {
    setRefreshing(true);
    
    try {
      const result = await apiService.llm.getStatus();
      
      if (result.success) {
        setLlmStatus(result.data);
        showSuccess('LLM status loaded successfully');
      } else {
        showError(`Failed to load LLM status: ${result.error}`);
      }
    } catch (error) {
      console.error('Error loading LLM status:', error);
      showError(`Error loading LLM status: ${error.message}`);
    } finally {
      setRefreshing(false);
    }
  };
  
  if (loading) {
    return (
      <PageLayout
        title="LLM Management"
        breadcrumbs={[{ label: 'LLM Management', path: '/llm-management' }]}
        loading={true}
      />
    );
  }
  
  return (
    <PageLayout
      title="LLM Management"
      breadcrumbs={[{ label: 'LLM Management', path: '/llm-management' }]}
      user={user}
      actions={
        <Button
          variant="outlined"
          startIcon={refreshing ? <CircularProgress size={20} /> : <RefreshIcon />}
          onClick={loadLlmStatus}
          disabled={refreshing}
        >
          Refresh
        </Button>
      }
    >
      {/* Status overview */}
      {llmStatus && (
        <Paper sx={{ mb: 3, p: 2 }}>
          <Typography variant="h6" gutterBottom>
            LLM System Status: {' '}
            <Box component="span" sx={{ 
              color: 
                llmStatus.overall_status === 'operational' ? 'success.main' : 
                llmStatus.overall_status === 'degraded' ? 'warning.main' : 
                'error.main' 
            }}>
              {llmStatus.overall_status.toUpperCase()}
            </Box>
          </Typography>
          
          <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
            {Object.entries(llmStatus.components).map(([component, status]) => (
              <Paper 
                key={component}
                sx={{ 
                  p: 1.5, 
                  minWidth: 200,
                  bgcolor: 
                    status.status === 'available' ? 'success.light' : 
                    'error.light',
                  color: 'common.white'
                }}
              >
                <Typography variant="subtitle1">
                  {component.toUpperCase()}
                </Typography>
                <Typography variant="body2">
                  Status: {status.status}
                </Typography>
                {status.status === 'available' && component === 'gateway' && (
                  <Typography variant="body2">
                    Providers: {status.details.active_providers.length}
                  </Typography>
                )}
                {status.status === 'available' && component === 'dspy' && (
                  <Typography variant="body2">
                    Modules: {status.modules_count}
                  </Typography>
                )}
                {status.status === 'available' && component === 'biomedlm' && (
                  <Typography variant="body2">
                    Models: {status.models_count}
                  </Typography>
                )}
              </Paper>
            ))}
          </Box>
        </Paper>
      )}
      
      {/* Main tabs */}
      <Paper sx={{ mb: 3 }}>
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs 
            value={activeTab} 
            onChange={handleTabChange} 
            aria-label="LLM management tabs"
            variant="fullWidth"
          >
            <Tab 
              icon={<SmartToyIcon />} 
              label="LLM Gateway" 
              id="tab-0" 
              aria-controls="tabpanel-0" 
            />
            <Tab 
              icon={<PsychologyIcon />} 
              label="DSPy" 
              id="tab-1" 
              aria-controls="tabpanel-1" 
            />
            <Tab 
              icon={<BiotechIcon />} 
              label="BiomedLM" 
              id="tab-2" 
              aria-controls="tabpanel-2" 
            />
            <Tab 
              icon={<BarChartIcon />} 
              label="Usage" 
              id="tab-3" 
              aria-controls="tabpanel-3" 
            />
          </Tabs>
        </Box>

        <Box role="tabpanel" hidden={activeTab !== 0} id="tabpanel-0" aria-labelledby="tab-0" sx={{ p: 3 }}>
          {activeTab === 0 && (
            <GatewayDashboard 
              status={llmStatus?.components?.gateway} 
              onRefresh={loadLlmStatus}
            />
          )}
        </Box>

        <Box role="tabpanel" hidden={activeTab !== 1} id="tabpanel-1" aria-labelledby="tab-1" sx={{ p: 3 }}>
          {activeTab === 1 && (
            <DSPyDashboard 
              status={llmStatus?.components?.dspy}
              onRefresh={loadLlmStatus}
            />
          )}
        </Box>

        <Box role="tabpanel" hidden={activeTab !== 2} id="tabpanel-2" aria-labelledby="tab-2" sx={{ p: 3 }}>
          {activeTab === 2 && (
            <BiomedLMDashboard 
              status={llmStatus?.components?.biomedlm}
              onRefresh={loadLlmStatus}
            />
          )}
        </Box>
        
        <Box role="tabpanel" hidden={activeTab !== 3} id="tabpanel-3" aria-labelledby="tab-3" sx={{ p: 3 }}>
          {activeTab === 3 && (
            <UsageDashboard 
              status={llmStatus}
              onRefresh={loadLlmStatus}
            />
          )}
        </Box>
      </Paper>

      {/* Additional information */}
      <Paper sx={{ p: 3 }}>
        <Typography variant="h6" gutterBottom>About LLM Management</Typography>
        <Typography paragraph>
          This page provides management functionality for Large Language Model (LLM) components,
          including LLM Gateway, DSPy, and BiomedLM.
        </Typography>
        <Typography component="div" sx={{ mb: 2 }}>
          <strong>LLM Gateway</strong> - Manage LLM providers and configurations, test connections,
          and execute direct LLM requests. The gateway provides a unified interface for interacting
          with various LLM providers like OpenAI, Anthropic, and more.
        </Typography>
        <Typography component="div" sx={{ mb: 2 }}>
          <strong>DSPy</strong> - Manage DSPy modules for advanced LLM programming. DSPy provides
          a framework for building complex LLM applications with optimized prompts, fine-tuned
          for specific tasks like medical RAG, contradiction detection, and more.
        </Typography>
        <Typography component="div" sx={{ mb: 2 }}>
          <strong>BiomedLM</strong> - Manage BiomedLM models and adapters for medical-specific
          LLM capabilities. BiomedLM is a specialized language model for biomedical text, with
          adapters for specific medical tasks.
        </Typography>
        <Typography component="div">
          <strong>Usage</strong> - Monitor usage statistics for all LLM components, including
          token usage, request counts, and latency metrics.
        </Typography>
      </Paper>
    </PageLayout>
  );
};

export default LLMManagement;
