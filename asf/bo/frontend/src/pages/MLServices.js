import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Box, Tab, Tabs, Paper, Typography, Alert } from '@mui/material';
import { 
  Science as ScienceIcon, 
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
import apiService from '../services/api';
import useApi from '../hooks/useApi';

/**
 * ML Services page with tabs for different ML services
 */
const MLServices = () => {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState(0);
  const [exportDialogOpen, setExportDialogOpen] = useState(false);
  const [exportParams, setExportParams] = useState({});
  
  const navigate = useNavigate();
  
  // Use API hook for fetching user data
  const { 
    data: userData, 
    loading: userLoading, 
    error: userError,
    execute: fetchUser 
  } = useApi(apiService.auth.me, {
    params: { loadOnMount: true }, // FIX: pass loadOnMount inside params
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
  
  // Handle logout
  const handleLogout = () => {
    localStorage.removeItem('token');
    navigate('/');
  };
  
  // Handle tab change
  const handleTabChange = (_, newValue) => {
    setActiveTab(newValue);
  };
  
  // Handle export
  const handleExport = (format, params) => {
    setExportParams({
      format,
      ...params
    });
    setExportDialogOpen(true);
  };
  
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
            <ContradictionDetection onExport={handleExport} />
          )}
        </Box>

        <Box role="tabpanel" hidden={activeTab !== 1} id="tabpanel-1" aria-labelledby="tab-1" sx={{ p: 3 }}>
          {activeTab === 1 && (
            <TemporalAnalysis onExport={handleExport} />
          )}
        </Box>

        <Box role="tabpanel" hidden={activeTab !== 2} id="tabpanel-2" aria-labelledby="tab-2" sx={{ p: 3 }}>
          {activeTab === 2 && (
            <BiasAssessment onExport={handleExport} />
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
      />
    </PageLayout>
  );
};

export default MLServices;
