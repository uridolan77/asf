import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Box, Tab, Tabs, Paper, Typography } from '@mui/material';
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
  
  const navigate = useNavigate();

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
    setProcessingInProgress(isProcessing);
    setLoading(isProcessing); // Update page loading state to show indicator
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
  
  return (
    <PageLayout
      title="Machine Learning Services"
      breadcrumbs={[{ label: 'ML Services', path: '/ml-services' }]}
      loading={loading}
      user={user}
    >
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

        <Box role="tabpanel" hidden={activeTab !== 0} id="tabpanel-0" aria-labelledby="tab-0" sx={{ p: 3 }}>
          {activeTab === 0 && (
            <ContradictionDetection 
              onExport={handleExport} 
              api={api} 
              onProcessingStateChange={handleProcessingStateChange}
            />
          )}
        </Box>

        <Box role="tabpanel" hidden={activeTab !== 1} id="tabpanel-1" aria-labelledby="tab-1" sx={{ p: 3 }}>
          {activeTab === 1 && (
            <TemporalAnalysis 
              onExport={handleExport} 
              api={api}
              onProcessingStateChange={handleProcessingStateChange}
            />
          )}
        </Box>

        <Box role="tabpanel" hidden={activeTab !== 2} id="tabpanel-2" aria-labelledby="tab-2" sx={{ p: 3 }}>
          {activeTab === 2 && (
            <BiasAssessment 
              onExport={handleExport} 
              api={api}
              onProcessingStateChange={handleProcessingStateChange}
            />
          )}
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