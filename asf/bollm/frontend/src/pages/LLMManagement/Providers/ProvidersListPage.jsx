import React from 'react';
import { 
  Box, 
  Typography, 
  Paper
} from '@mui/material';

// Import the PageLayout component that includes the sidebar
import PageLayout from '../../../components/Layout/PageLayout';

// Import the full-featured ProviderManagement component
import ProviderManagement from '../../../components/LLM/Providers/ProviderManagement.js';

/**
 * LLM Providers List Page
 * Shows all configured LLM providers and their status
 * Uses the full-featured ProviderManagement component
 */
const ProvidersListPage = () => {
  // Define breadcrumbs for PageLayout
  const breadcrumbs = [
    { label: 'LLM Management', path: '/llm/dashboard' },
    { label: 'Providers' }
  ];

  return (
    <PageLayout
      title="LLM Providers"
      breadcrumbs={breadcrumbs}
    >
      <Box sx={{ mb: 3 }}>
        <Typography variant="body1" color="text.secondary" paragraph>
          Manage your LLM providers, configurations, and API keys. Add new providers or modify existing ones.
        </Typography>
        
        {/* Main content - using the full-featured ProviderManagement component */}
        <Paper sx={{ mt: 3, p: 3 }}>
          <ProviderManagement />
        </Paper>
      </Box>
    </PageLayout>
  );
};

export default ProvidersListPage;