import React, { useState } from 'react';
import { Box, Typography, Paper, Tabs, Tab } from '@mui/material';
import PageLayout from '../components/Layout/PageLayout';
import { useAuth } from '../context/AuthContext';

/**
 * LLM Management page with tabs for different LLM components
 */
const LLMManagement = () => {
  const [activeTab, setActiveTab] = useState(0);
  const { user } = useAuth();

  const handleTabChange = (event, newValue) => {
    setActiveTab(newValue);
  };

  return (
    <PageLayout
      title="LLM Management"
      breadcrumbs={[{ label: 'LLM Management', path: '/llm-management' }]}
      user={user}
    >
      <Paper sx={{ mb: 3 }}>
        <Tabs
          value={activeTab}
          onChange={handleTabChange}
          variant="fullWidth"
          aria-label="LLM management tabs"
        >
          <Tab label="LLM Gateway" id="tab-0" />
          <Tab label="DSPy" id="tab-1" />
          <Tab label="BiomedLM" id="tab-2" />
          <Tab label="Usage" id="tab-3" />
        </Tabs>

        <Box role="tabpanel" hidden={activeTab !== 0} id="tabpanel-0" sx={{ p: 3 }}>
          {activeTab === 0 && (
            <Box>
              <Typography variant="h6" gutterBottom>
                LLM Gateway Management
              </Typography>
              <Typography paragraph>
                Manage LLM providers and configurations, test connections, and execute direct LLM requests.
                The gateway provides a unified interface for interacting with various LLM providers like
                OpenAI, Anthropic, and more.
              </Typography>
              <Typography paragraph>
                Configure API keys, model preferences, and usage limits for each provider. Test connections
                and monitor performance metrics in real-time.
              </Typography>
            </Box>
          )}
        </Box>

        <Box role="tabpanel" hidden={activeTab !== 1} id="tabpanel-1" sx={{ p: 3 }}>
          {activeTab === 1 && (
            <Box>
              <Typography variant="h6" gutterBottom>
                DSPy Module Management
              </Typography>
              <Typography paragraph>
                Manage DSPy modules for advanced LLM programming. DSPy provides a framework for building
                complex LLM applications with optimized prompts, fine-tuned for specific tasks like
                medical RAG, contradiction detection, and more.
              </Typography>
              <Typography paragraph>
                Register new modules, optimize existing ones, and monitor their performance across
                different tasks and datasets.
              </Typography>
            </Box>
          )}
        </Box>

        <Box role="tabpanel" hidden={activeTab !== 2} id="tabpanel-2" sx={{ p: 3 }}>
          {activeTab === 2 && (
            <Box>
              <Typography variant="h6" gutterBottom>
                BiomedLM Model Management
              </Typography>
              <Typography paragraph>
                Manage BiomedLM models and adapters for medical-specific LLM capabilities. BiomedLM is a
                specialized language model for biomedical text, with adapters for specific medical tasks.
              </Typography>
              <Typography paragraph>
                Fine-tune models on specialized medical datasets, create and manage adapters for specific
                tasks, and evaluate model performance on medical benchmarks.
              </Typography>
            </Box>
          )}
        </Box>

        <Box role="tabpanel" hidden={activeTab !== 3} id="tabpanel-3" sx={{ p: 3 }}>
          {activeTab === 3 && (
            <Box>
              <Typography variant="h6" gutterBottom>
                LLM Usage Statistics
              </Typography>
              <Typography paragraph>
                Monitor usage statistics for all LLM components, including token usage, request counts,
                and latency metrics. Track usage patterns over time and identify optimization opportunities.
              </Typography>
              <Typography paragraph>
                View detailed breakdowns by provider, model, and application. Export reports for billing
                and analysis purposes.
              </Typography>
            </Box>
          )}
        </Box>
      </Paper>

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
