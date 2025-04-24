import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Box, Tab, Tabs, Paper, Typography, Alert } from '@mui/material';
import {
  Compare as CompareIcon,
  MedicalInformation as MedicalIcon,
  History as HistoryIcon
} from '@mui/icons-material';

import PageLayout from '../components/Layout/PageLayout';
import {
  ContradictionAnalysis,
  CAPAnalysis,
  AnalysisHistory
} from '../components/Analysis';
import { ExportDialog } from '../components/Export';
import { useAuth } from '../context/AuthContext.jsx';
import { useNotification } from '../context/NotificationContext.jsx';

/**
 * Analysis page with tabs for different analysis methods
 */
const Analysis = () => {
  const { user, api } = useAuth();
  const { showError } = useNotification();
  const [loading, setLoading] = useState(true); // Start with loading true
  const [activeTab, setActiveTab] = useState(0);
  const [selectedAnalysisId, setSelectedAnalysisId] = useState(null);
  const [exportDialogOpen, setExportDialogOpen] = useState(false);
  const [exportParams, setExportParams] = useState({});

  const navigate = useNavigate();

  // Set loading to false after component mounts
  useEffect(() => {
    // Short timeout to ensure the component is fully mounted
    const timer = setTimeout(() => {
      setLoading(false);
    }, 500);

    return () => clearTimeout(timer);
  }, []);

  // Handle tab change
  const handleTabChange = (_, newValue) => {
    setActiveTab(newValue);
  };

  // Handle view analysis
  const handleViewAnalysis = (analysisId) => {
    setSelectedAnalysisId(analysisId);

    // Determine which tab to show based on analysis type
    // This would require fetching the analysis details first
    // For now, we'll just switch to the first tab
    setActiveTab(0);
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
      title="Medical Literature Analysis"
      breadcrumbs={[{ label: 'Analysis', path: '/analysis' }]}
      loading={loading}
      user={user}
    >
      <Paper sx={{ mb: 3 }}>
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs
            value={activeTab}
            onChange={handleTabChange}
            aria-label="analysis method tabs"
            variant="fullWidth"
          >
            <Tab
              icon={<CompareIcon />}
              label="Contradiction Analysis"
              id="tab-0"
              aria-controls="tabpanel-0"
            />
            <Tab
              icon={<MedicalIcon />}
              label="CAP Analysis"
              id="tab-1"
              aria-controls="tabpanel-1"
            />
            <Tab
              icon={<HistoryIcon />}
              label="Analysis History"
              id="tab-2"
              aria-controls="tabpanel-2"
            />
          </Tabs>
        </Box>

        <Box role="tabpanel" hidden={activeTab !== 0} id="tabpanel-0" aria-labelledby="tab-0" sx={{ p: 3 }}>
          {activeTab === 0 && (
            <ContradictionAnalysis onExport={handleExport} api={api} />
          )}
        </Box>

        <Box role="tabpanel" hidden={activeTab !== 1} id="tabpanel-1" aria-labelledby="tab-1" sx={{ p: 3 }}>
          {activeTab === 1 && (
            <CAPAnalysis onExport={handleExport} api={api} />
          )}
        </Box>

        <Box role="tabpanel" hidden={activeTab !== 2} id="tabpanel-2" aria-labelledby="tab-2" sx={{ p: 3 }}>
          {activeTab === 2 && (
            <AnalysisHistory
              onViewAnalysis={handleViewAnalysis}
              onExport={handleExport}
              api={api}
            />
          )}
        </Box>
      </Paper>

      {/* Additional information or help section */}
      <Paper sx={{ p: 3 }}>
        <Typography variant="h6" gutterBottom>About Medical Literature Analysis</Typography>
        <Typography paragraph>
          This tool provides different methods for analyzing medical literature:
        </Typography>
        <Typography component="div" sx={{ mb: 2 }}>
          <strong>Contradiction Analysis</strong> - Identifies contradictory statements in medical literature
          based on a search query. This helps researchers identify areas of disagreement in the literature.
        </Typography>
        <Typography component="div" sx={{ mb: 2 }}>
          <strong>CAP Analysis</strong> - Provides comprehensive analysis of Community-Acquired Pneumonia literature,
          including treatment effectiveness, patient populations, and clinical outcomes.
        </Typography>
        <Typography component="div">
          <strong>Analysis History</strong> - View and manage your previous analyses, with the ability to
          export results in various formats.
        </Typography>
      </Paper>

      {/* Export Dialog */}
      <ExportDialog
        open={exportDialogOpen}
        onClose={() => setExportDialogOpen(false)}
        resultId={exportParams.result_id}
        analysisId={exportParams.analysis_id}
        query={exportParams.query}
        title="Export Analysis Results"
        api={api}
      />
    </PageLayout>
  );
};

export default Analysis;