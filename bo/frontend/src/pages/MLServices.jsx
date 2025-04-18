import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { Box, Tab, Tabs, Paper, Typography, Alert } from '@mui/material';
import { 
  Compare as CompareIcon,
  AccessTime as AccessTimeIcon,
  WarningAmber as WarningIcon
} from '@mui/icons-material';

import PageLayout from '../components/Layout/PageLayout';
import { 
  ContradictionDetection, 
  TemporalAnalysis, 
  BiasAssessment 
} from '../components/ML';
import { ExportDialog } from '../components/Export';
import { useAuth } from '../context/AuthContext.jsx';

/**
 * ML Services page with tabs for different ML services
 */
const MLServices = () => {
  const { user, api } = useAuth();
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState(0);
  const [exportDialogOpen, setExportDialogOpen] = useState(false);
  const [exportParams, setExportParams] = useState({});
  const [processingInProgress, setProcessingInProgress] = useState(false);
  const [apiServiceError, setApiServiceError] = useState(false);
  
  // Track mounted state to prevent state updates after unmount
  const isMounted = useRef(true);
  
  const navigate = useNavigate();

  // Handle component mount/unmount
  useEffect(() => {
    // Component mounted
    isMounted.current = true;
    
    // Create an interceptor to prevent continuous API calls
    const originalRequest = api?.interceptors?.request?.use || null;
    let requestInterceptor = null;
    let mlServicesCalls = 0;
    
    if (originalRequest) {
      requestInterceptor = api.interceptors.request.use(
        (config) => {
          // Check if this is an ML services status or metrics call
          if (config.url?.includes('/ml/services/') || config.url?.includes('/medical/ml/services/')) {
            mlServicesCalls++;
            
            // If we're seeing too many consecutive ML service calls, block them
            if (mlServicesCalls > 5) {
              setApiServiceError(true);
              // Cancel the request
              const error = new Error('Too many consecutive ML service calls - request cancelled to prevent infinite loop');
              error.name = 'CanceledError';
              return Promise.reject(error);
            }
          } else {
            // Reset counter for non-ML service calls
            mlServicesCalls = 0;
          }
          return config;
        },
        (error) => {
          return Promise.reject(error);
        }
      );
    }
    
    // Cleanup on unmount
    return () => {
      isMounted.current = false;
      if (requestInterceptor !== null && api?.interceptors?.request) {
        api.interceptors.request.eject(requestInterceptor);
      }
    };
  }, [api]);

  // Handle tab change
  const handleTabChange = (_, newValue) => {
    // Only allow tab change if no processing is in progress or user confirms
    if (processingInProgress) {
      if (window.confirm("Processing is in progress. Changing tabs may interrupt your analysis. Continue?")) {
        setActiveTab(newValue);
      }
    } else {
      setActiveTab(newValue);
    }
  };
  
  // Handle export
  const handleExport = (format, params) => {
    setExportParams({
      format,
      ...params
    });
    setExportDialogOpen(true);
  };

  // Handle processing state changes
  const handleProcessingStateChange = (isProcessing) => {
    if (isMounted.current) {
      setProcessingInProgress(isProcessing);
      setLoading(isProcessing); // Update page loading state to show indicator
    }
  };

  // Handle navigation prompt
  React.useEffect(() => {
    // Set up a navigation guard
    const unblock = navigate((nextLocation) => {
      if (processingInProgress) {
        if (window.confirm("Processing is in progress. Navigating away will cancel your analysis. Continue?")) {
          return true; // Allow navigation
        } else {
          return false; // Block navigation
        }
      }
      return true; // Allow navigation if no processing
    });

    // Cleanup
    return () => {
      if (unblock) unblock();
    };
  }, [processingInProgress, navigate]);
  
  // Function to render the active tab content
  const renderActiveTabContent = () => {
    switch (activeTab) {
      case 0:
        return (
          <ContradictionDetection 
            onExport={handleExport} 
            api={api} 
            onProcessingStateChange={handleProcessingStateChange}
          />
        );
      case 1:
        return (
          <TemporalAnalysis 
            onExport={handleExport} 
            api={api}
            onProcessingStateChange={handleProcessingStateChange}
          />
        );
      case 2:
        return (
          <BiasAssessment 
            onExport={handleExport} 
            api={api}
            onProcessingStateChange={handleProcessingStateChange}
          />
        );
      default:
        return null;
    }
  };
  
  return (
    <PageLayout
      title="Machine Learning Services"
      breadcrumbs={[{ label: 'ML Services', path: '/ml-services' }]}
      loading={loading}
      user={user}
    >
      {apiServiceError && (
        <Alert 
          severity="warning" 
          sx={{ mb: 3 }}
          onClose={() => setApiServiceError(false)}
        >
          Detected excessive API calls to the ML services. Some requests were blocked to prevent performance issues.
          This may indicate a configuration issue with the ML services. Try refreshing the page if functionality is limited.
        </Alert>
      )}
      
      <Paper sx={{ mb: 3 }}>
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs 
            value={activeTab} 
            onChange={handleTabChange} 
            aria-label="ML services tabs"
            variant="fullWidth"
          >
            <Tab 
              icon={<CompareIcon />} 
              label="Contradiction Detection" 
              id="tab-0" 
              aria-controls="tabpanel-0" 
            />
            <Tab 
              icon={<AccessTimeIcon />} 
              label="Temporal Analysis" 
              id="tab-1" 
              aria-controls="tabpanel-1" 
            />
            <Tab 
              icon={<WarningIcon />} 
              label="Bias Assessment" 
              id="tab-2" 
              aria-controls="tabpanel-2" 
            />
          </Tabs>
        </Box>

        <Box role="tabpanel" id={`tabpanel-${activeTab}`} aria-labelledby={`tab-${activeTab}`} sx={{ p: 3 }}>
          {renderActiveTabContent()}
        </Box>
      </Paper>

      {/* Additional information or help section */}
      <Paper sx={{ p: 3 }}>
        <Typography variant="h6" gutterBottom>About Machine Learning Services</Typography>
        <Typography paragraph>
          This section provides access to advanced machine learning services for medical research:
        </Typography>
        <Typography component="div" sx={{ mb: 2 }}>
          <strong>Contradiction Detection</strong> - Detect contradictions between medical claims using
          various ML models including BioMedLM, TSMixer, and Lorentz embeddings. This helps researchers
          identify conflicting evidence in the medical literature.
        </Typography>
        <Typography component="div" sx={{ mb: 2 }}>
          <strong>Temporal Analysis</strong> - Analyze the temporal confidence of medical claims based on
          publication dates and domain-specific characteristics. This helps researchers understand how
          the reliability of medical evidence changes over time.
        </Typography>
        <Typography component="div">
          <strong>Bias Assessment</strong> - Assess bias in medical articles using various bias assessment
          tools like ROBINS-I, RoB 2, etc. This helps researchers evaluate the quality of evidence and
          identify potential sources of bias.
        </Typography>
      </Paper>
      
      {/* Export Dialog */}
      <ExportDialog
        open={exportDialogOpen}
        onClose={() => setExportDialogOpen(false)}
        resultId={exportParams.result_id}
        analysisId={exportParams.analysis_id}
        query={exportParams.query}
        title="Export ML Analysis Results"
        api={api}
      />
    </PageLayout>
  );
};

export default MLServices;