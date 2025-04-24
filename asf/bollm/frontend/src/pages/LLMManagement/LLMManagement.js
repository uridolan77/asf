import React, { useState, useEffect, lazy, Suspense } from 'react';
import { useNavigate, Link, useLocation } from 'react-router-dom';
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
  Refresh as RefreshIcon,
  Cloud as CloudIcon,
  Memory as MemoryIcon,
  AutoFixHigh as AutoFixHighIcon,
  Code as CodeIcon
} from '@mui/icons-material';

import PageLayout from '../../components/Layout/PageLayout';
import { ContentLoader } from '../../components/UI/LoadingIndicators.js';
import apiService from '../../services/api';
import { useNotification } from '../../context/NotificationContext.jsx';
import useApi from '../../hooks/useApi';

// Lazy load all components to avoid circular dependencies
const GatewayDashboard = lazy(() => import('./GatewayDashboard'));
const DSPyDashboard = lazy(() => import('./DSPyDashboard'));
const BiomedLMDashboard = lazy(() => import('./BiomedLMDashboard'));
const CLPEFTDashboard = lazy(() => import('./CLPEFTDashboard'));
const MCPDashboard = lazy(() => import('./MCPDashboard'));
const UsageDashboard = lazy(() => import('./UsageDashboard'));
const ProviderManagement = lazy(() => import('../../components/LLM/Providers/ProviderManagement'));
const ModelManagement = lazy(() => import('../../components/LLM/Models/ModelManagement'));

/**
 * LLM Management page with tabs for different LLM components
 * @deprecated Use the new LLM Management interface at /llm/dashboard instead
 */
const LLMManagement = () => {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState(0);
  const [refreshing, setRefreshing] = useState(false);
  const [llmStatus, setLlmStatus] = useState({
    overall_status: 'unknown',
    components: {
      gateway: { status: 'unknown', details: {} },
      dspy: { status: 'unknown', modules: [], modules_count: 0 },
      biomedlm: { status: 'unknown', models: [], models_count: 0 },
      cl_peft: { status: 'unknown', adapters: [], adapters_count: 0 },
      mcp: { status: 'unknown', providers: [], providers_count: 0 }
    }
  });

  const navigate = useNavigate();
  const location = useLocation();
  const { showSuccess, showError } = useNotification();

  // Use API hook for fetching user data
  const {
    data: userData,
    loading: userLoading,
    error: userError,
    execute: fetchUser
  } = useApi(apiService.auth.me, {
    params: {
      loadOnMount: true
    },
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

  // Prompt user to use new interface
  useEffect(() => {
    showSuccess('Try our new LLM Management interface at /llm/dashboard');
  }, []);

  // Load LLM status on mount - separate from user loading
  useEffect(() => {
    loadLlmStatus();
  }, []);

  // Set active tab based on query parameter
  useEffect(() => {
    const params = new URLSearchParams(location.search);
    const tabParam = params.get('tab');

    if (tabParam) {
      switch (tabParam.toLowerCase()) {
        case 'providers':
          setActiveTab(0);
          break;
        case 'models':
          setActiveTab(1);
          break;
        case 'gateway':
          setActiveTab(2);
          break;
        case 'dspy':
          setActiveTab(3);
          break;
        case 'biomedlm':
          setActiveTab(4);
          break;
        case 'cl-peft':
        case 'clpeft':
          setActiveTab(5);
          break;
        case 'mcp':
          setActiveTab(6);
          break;
        case 'usage':
          setActiveTab(7);
          break;
        default:
          break;
      }
    }
  }, [location]);

  // Handle logout
  const handleLogout = () => {
    localStorage.removeItem('token');
    navigate('/');
  };

  // Handle tab change
  const handleTabChange = (_, newValue) => {
    setActiveTab(newValue);

    // Update URL with tab parameter
    let tabName = '';
    switch (newValue) {
      case 0:
        tabName = 'providers';
        break;
      case 1:
        tabName = 'models';
        break;
      case 2:
        tabName = 'gateway';
        break;
      case 3:
        tabName = 'dspy';
        break;
      case 4:
        tabName = 'biomedlm';
        break;
      case 5:
        tabName = 'cl-peft';
        break;
      case 6:
        tabName = 'mcp';
        break;
      case 7:
        tabName = 'usage';
        break;
      default:
        break;
    }

    if (tabName) {
      const params = new URLSearchParams(location.search);
      params.set('tab', tabName);
      navigate({ search: params.toString() }, { replace: true });
    }
  };

  // Load LLM status
  const loadLlmStatus = async () => {
    setRefreshing(true);

    try {
      // We'll simulate a successful response for development
      // Once backend is ready, uncomment the actual API call

      /*
      const result = await apiService.llm.getStatus();

      if (result.success) {
        setLlmStatus(result.data);
        showSuccess('LLM status loaded successfully');

        // Fetch additional information from each component
        await Promise.allSettled([
          fetchGatewayStatus(),
          fetchDspyStatus(),
          fetchBiomedLMStatus()
        ]);
      } else {
        showError(`Failed to load LLM status: ${result.error}`);
      }
      */

      // Simulated response for development
      setTimeout(() => {
        setLlmStatus({
          overall_status: 'operational',
          components: {
            gateway: {
              status: 'available',
              details: {
                providers: [
                  { id: 'openai', name: 'OpenAI', type: 'API', is_active: true, requires_api_key: true, models_count: 5 },
                  { id: 'anthropic', name: 'Anthropic', type: 'API', is_active: true, requires_api_key: true, models_count: 3 },
                  { id: 'local', name: 'Local Models', type: 'Local', is_active: true, requires_api_key: false, models_count: 2 }
                ],
                active_providers: [
                  { id: 'openai', name: 'OpenAI', type: 'API', is_active: true, requires_api_key: true, models_count: 5 },
                  { id: 'anthropic', name: 'Anthropic', type: 'API', is_active: true, requires_api_key: true, models_count: 3 },
                  { id: 'local', name: 'Local Models', type: 'Local', is_active: true, requires_api_key: false, models_count: 2 }
                ]
              }
            },
            dspy: {
              status: 'available',
              modules: [
                { name: 'MedicalSummarizer', module_type: 'Summarization', optimized: true, description: 'Summarizes medical text', tags: ['medical', 'nlp'] },
                { name: 'EvidenceExtractor', module_type: 'Extraction', optimized: false, description: 'Extracts evidences from medical literature', tags: ['evidence', 'medical'] },
                { name: 'ContradictionDetector', module_type: 'Classification', optimized: true, description: 'Detects contradictions in medical claims', tags: ['contradiction', 'medical'] }
              ],
              modules_count: 3
            },
            biomedlm: {
              status: 'available',
              models: [
                { id: 'biomedlm-base', name: 'BioMedLM Base', status: 'active', size: '7B', description: 'Base biomedical language model', tags: ['medical', 'base'] },
                { id: 'biomedlm-clinical', name: 'BioMedLM Clinical', status: 'active', size: '13B', description: 'Clinical-focused biomedical model', tags: ['clinical', 'medical'] }
              ],
              models_count: 2
            },
            cl_peft: {
              status: 'available',
              adapters: [
                { adapter_id: 'adapter_12345678', adapter_name: 'Medical QA Adapter', base_model_name: 'meta-llama/Llama-2-7b-hf', cl_strategy: 'generative_replay', peft_method: 'lora' },
                { adapter_id: 'adapter_87654321', adapter_name: 'Clinical Notes Adapter', base_model_name: 'mistralai/Mistral-7B-v0.1', cl_strategy: 'ewc', peft_method: 'qlora' }
              ],
              adapters_count: 2
            },
            mcp: {
              status: 'available',
              providers: [
                { provider_id: 'anthropic_mcp', display_name: 'Anthropic MCP', transport_type: 'stdio', status: 'connected' },
                { provider_id: 'openai_mcp', display_name: 'OpenAI MCP', transport_type: 'http', status: 'connected' }
              ],
              providers_count: 2
            }
          }
        });
        showSuccess('LLM status loaded successfully');
        setRefreshing(false);
      }, 1000);
    } catch (error) {
      console.error('Error loading LLM status:', error);
      showError(`Error loading LLM status: ${error.message}`);
      setRefreshing(false);
    }
  };

  // Mock implementation of fetchGatewayStatus, fetchDspyStatus, fetchBiomedLMStatus
  // These would typically make actual API calls
  const fetchGatewayStatus = async () => {
    // Implementation will be replaced with actual API calls
    console.log('Fetching gateway status');
  };

  const fetchDspyStatus = async () => {
    // Implementation will be replaced with actual API calls
    console.log('Fetching DSPy status');
  };

  const fetchBiomedLMStatus = async () => {
    // Implementation will be replaced with actual API calls
    console.log('Fetching BiomedLM status');
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
        <>
          <Button
            variant="contained"
            color="primary"
            onClick={() => navigate('/llm/dashboard')}
            sx={{ mr: 2 }}
          >
            Try New Dashboard
          </Button>
          <Button
            variant="outlined"
            startIcon={refreshing ? <CircularProgress size={20} /> : <RefreshIcon />}
            onClick={loadLlmStatus}
            disabled={refreshing}
          >
            Refresh
          </Button>
        </>
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
                    Providers: {status.details?.active_providers?.length || 0}
                  </Typography>
                )}
                {status.status === 'available' && component === 'dspy' && (
                  <Typography variant="body2">
                    Modules: {status.modules_count || 0}
                  </Typography>
                )}
                {status.status === 'available' && component === 'biomedlm' && (
                  <Typography variant="body2">
                    Models: {status.models_count || 0}
                  </Typography>
                )}
                {status.status === 'available' && component === 'cl_peft' && (
                  <Typography variant="body2">
                    Adapters: {status.adapters_count || 0}
                  </Typography>
                )}
                {status.status === 'available' && component === 'mcp' && (
                  <Typography variant="body2">
                    Providers: {status.providers_count || 0}
                  </Typography>
                )}
              </Paper>
            ))}
          </Box>
        </Paper>
      )}

      {/* Notice about new interface */}
      <Alert severity="info" sx={{ mb: 3 }}>
        <Typography variant="body1">
          This is the legacy LLM Management interface. We recommend using our new, improved interface for better organization and more features.
        </Typography>
        <Button 
          variant="contained" 
          color="primary" 
          size="small" 
          onClick={() => navigate('/llm/dashboard')}
          sx={{ mt: 1 }}
        >
          Go to New Dashboard
        </Button>
      </Alert>

      {/* Main tabs */}
      <Paper sx={{ mb: 3 }}>
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs
            value={activeTab}
            onChange={handleTabChange}
            aria-label="LLM management tabs"
            variant="scrollable"
            scrollButtons="auto"
          >
            <Tab
              icon={<CloudIcon />}
              label="Providers"
              id="tab-0"
              aria-controls="tabpanel-0"
            />
            <Tab
              icon={<MemoryIcon />}
              label="Models"
              id="tab-1"
              aria-controls="tabpanel-1"
            />
            <Tab
              icon={<SmartToyIcon />}
              label="Gateway"
              id="tab-2"
              aria-controls="tabpanel-2"
            />
            <Tab
              icon={<PsychologyIcon />}
              label="DSPy"
              id="tab-3"
              aria-controls="tabpanel-3"
            />
            <Tab
              icon={<BiotechIcon />}
              label="BiomedLM"
              id="tab-4"
              aria-controls="tabpanel-4"
            />
            <Tab
              icon={<AutoFixHighIcon />}
              label="CL-PEFT"
              id="tab-5"
              aria-controls="tabpanel-5"
            />
            <Tab
              icon={<CodeIcon />}
              label="MCP"
              id="tab-6"
              aria-controls="tabpanel-6"
            />
            <Tab
              icon={<BarChartIcon />}
              label="Usage"
              id="tab-7"
              aria-controls="tabpanel-7"
            />
          </Tabs>
        </Box>

        <Box role="tabpanel" hidden={activeTab !== 0} id="tabpanel-0" aria-labelledby="tab-0" sx={{ p: 3 }}>
          {activeTab === 0 && (
            <Suspense fallback={<ContentLoader />}>
              <ProviderManagement />
            </Suspense>
          )}
        </Box>

        <Box role="tabpanel" hidden={activeTab !== 1} id="tabpanel-1" aria-labelledby="tab-1" sx={{ p: 3 }}>
          {activeTab === 1 && (
            <Suspense fallback={<ContentLoader />}>
              <ModelManagement />
            </Suspense>
          )}
        </Box>

        <Box role="tabpanel" hidden={activeTab !== 2} id="tabpanel-2" aria-labelledby="tab-2" sx={{ p: 3 }}>
          {activeTab === 2 && (
            <Suspense fallback={<ContentLoader />}>
              <GatewayDashboard
                status={llmStatus?.components?.gateway}
                onRefresh={loadLlmStatus}
              />
            </Suspense>
          )}
        </Box>

        <Box role="tabpanel" hidden={activeTab !== 3} id="tabpanel-3" aria-labelledby="tab-3" sx={{ p: 3 }}>
          {activeTab === 3 && (
            <Suspense fallback={<ContentLoader />}>
              <DSPyDashboard
                status={llmStatus?.components?.dspy}
                onRefresh={loadLlmStatus}
              />
            </Suspense>
          )}
        </Box>

        <Box role="tabpanel" hidden={activeTab !== 4} id="tabpanel-4" aria-labelledby="tab-4" sx={{ p: 3 }}>
          {activeTab === 4 && (
            <Suspense fallback={<ContentLoader />}>
              <BiomedLMDashboard
                status={llmStatus?.components?.biomedlm}
                onRefresh={loadLlmStatus}
              />
            </Suspense>
          )}
        </Box>

        <Box role="tabpanel" hidden={activeTab !== 5} id="tabpanel-5" aria-labelledby="tab-5" sx={{ p: 3 }}>
          {activeTab === 5 && (
            <Suspense fallback={<ContentLoader />}>
              <CLPEFTDashboard
                status={llmStatus?.components?.cl_peft}
                onRefresh={loadLlmStatus}
              />
            </Suspense>
          )}
        </Box>

        <Box role="tabpanel" hidden={activeTab !== 6} id="tabpanel-6" aria-labelledby="tab-6" sx={{ p: 3 }}>
          {activeTab === 6 && (
            <Suspense fallback={<ContentLoader />}>
              <MCPDashboard
                status={llmStatus?.components?.mcp}
                onRefresh={loadLlmStatus}
              />
            </Suspense>
          )}
        </Box>

        <Box role="tabpanel" hidden={activeTab !== 7} id="tabpanel-7" aria-labelledby="tab-7" sx={{ p: 3 }}>
          {activeTab === 7 && (
            <Suspense fallback={<ContentLoader />}>
              <UsageDashboard
                status={llmStatus}
                onRefresh={loadLlmStatus}
              />
            </Suspense>
          )}
        </Box>
      </Paper>

      {/* Additional information */}
      <Paper sx={{ p: 3 }}>
        <Typography variant="h6" gutterBottom>About LLM Management</Typography>
        <Typography paragraph>
          This page provides management functionality for Large Language Model (LLM) components,
          including LLM Gateway, DSPy, BiomedLM, CL-PEFT, and MCP.
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
        <Typography component="div" sx={{ mb: 2 }}>
          <strong>CL-PEFT</strong> - Manage Continual Learning with Parameter-Efficient Fine-Tuning
          adapters. CL-PEFT combines continual learning strategies with efficient fine-tuning
          techniques like LoRA and QLoRA to adapt language models to new tasks while mitigating
          catastrophic forgetting.
        </Typography>
        <Typography component="div" sx={{ mb: 2 }}>
          <strong>MCP</strong> - Manage Model Context Protocol (MCP) providers for standardized
          interaction with large language models. MCP provides a unified protocol with support for
          multiple transport options, streaming, and advanced resilience features.
          <Button
            variant="outlined"
            size="small"
            component={Link}
            to="/mcp-dashboard"
            sx={{ ml: 2 }}
          >
            Open Standalone MCP Dashboard Page
          </Button>
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