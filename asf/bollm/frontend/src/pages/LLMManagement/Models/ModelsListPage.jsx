// filepath: c:\code\asf\asf\bollm\frontend\src\pages\LLMManagement\Models\ModelsListPage.jsx
import React, { useState, useEffect } from 'react';
import { 
  Box, 
  Typography, 
  Paper,
  Button,
  Alert,
  CircularProgress
} from '@mui/material';

// Import the PageLayout component that includes the sidebar
import PageLayout from '../../../components/Layout/PageLayout';

// Import the full-featured ModelManagement component
import ModelManagement from '../../../components/LLM/Models/ModelManagement.js';
import apiService from '../../../services/api';

/**
 * LLM Models List Page
 * Shows all configured LLM models and their details
 * Uses the full-featured ModelManagement component
 */
const ModelsListPage = () => {
  // Add state for API status
  const [loading, setLoading] = useState(true);
  const [apiAvailable, setApiAvailable] = useState(true);
  const [usingMockData, setUsingMockData] = useState(false);

  // Check if the API is available
  useEffect(() => {
    const checkApiStatus = async () => {
      try {
        // Use the gateway providers endpoint which we know is working
        const providersResult = await apiService.llm.getProviders();
        setApiAvailable(providersResult.success);
        
        if (providersResult.success) {
          setUsingMockData(false);
        } else {
          console.warn("LLM Gateway providers endpoint not available");
          setApiAvailable(false);
          setUsingMockData(true);
        }
      } catch (error) {
        console.error("Error checking API status:", error);
        setApiAvailable(false);
        setUsingMockData(true);
      } finally {
        setLoading(false);
      }
    };

    checkApiStatus();
  }, []);

  // Define breadcrumbs for PageLayout
  const breadcrumbs = [
    { label: 'LLM Management', path: '/llm/dashboard' },
    { label: 'Models' }
  ];

  return (
    <PageLayout
      title="LLM Models"
      breadcrumbs={breadcrumbs}
    >
      <Box sx={{ mb: 3 }}>
        <Typography variant="body1" color="text.secondary" paragraph>
          Manage your LLM models, capabilities, and parameters. Add new models or modify existing ones.
        </Typography>
        
        {/* Show mock data warning if needed */}
        {usingMockData && apiAvailable && (
          <Alert severity="warning" sx={{ mb: 3 }}>
            Connection to the models API endpoint failed. Displaying demo data for reference.
          </Alert>
        )}
        
        {/* Main content */}
        {loading ? (
          <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
            <CircularProgress />
          </Box>
        ) : apiAvailable ? (
          <Paper sx={{ mt: 3, p: 3 }}>
            <ModelManagement />
          </Paper>
        ) : (
          <Paper sx={{ mt: 3, p: 3 }}>
            <Alert severity="warning" sx={{ mb: 3 }}>
              Unable to connect to the LLM API. Make sure the backend server is running and accessible.
            </Alert>
            <Typography variant="h6" gutterBottom>
              LLM Models Management
            </Typography>
            <Typography paragraph>
              This page allows you to manage your LLM models, including:
            </Typography>
            <ul>
              <li>View all registered models across providers</li>
              <li>Add new models to your providers</li>
              <li>Configure model parameters and capabilities</li>
              <li>Delete models that are no longer needed</li>
            </ul>
            <Button 
              variant="contained"
              onClick={() => window.location.reload()}
              sx={{ mt: 2 }}
            >
              Retry Connection
            </Button>
          </Paper>
        )}
      </Box>
    </PageLayout>
  );
};

export default ModelsListPage;