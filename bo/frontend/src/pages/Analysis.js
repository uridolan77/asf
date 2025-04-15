import React, { useState } from 'react';
import { Box, Typography, Paper, Tabs, Tab } from '@mui/material';
import PageLayout from '../components/Layout/PageLayout';
import { useAuth } from '../context/AuthContext';

/**
 * Analysis page with tabs for different analysis methods
 */
const Analysis = () => {
  const [activeTab, setActiveTab] = useState(0);
  const { user } = useAuth();

  const handleTabChange = (event, newValue) => {
    setActiveTab(newValue);
  };

  return (
    <PageLayout
      title="Medical Literature Analysis"
      breadcrumbs={[{ label: 'Analysis', path: '/analysis' }]}
      user={user}
    >
      <Paper sx={{ mb: 3 }}>
        <Tabs
          value={activeTab}
          onChange={handleTabChange}
          variant="fullWidth"
          aria-label="Analysis tabs"
        >
          <Tab label="Contradiction Analysis" id="tab-0" />
          <Tab label="CAP Analysis" id="tab-1" />
          <Tab label="Analysis History" id="tab-2" />
        </Tabs>

        <Box role="tabpanel" hidden={activeTab !== 0} id="tabpanel-0" sx={{ p: 3 }}>
          {activeTab === 0 && (
            <Box>
              <Typography variant="h6" gutterBottom>
                Contradiction Analysis
              </Typography>
              <Typography paragraph>
                Identify contradictory statements in medical literature based on a search query.
                This helps researchers identify areas of disagreement in the literature.
              </Typography>
              <Typography paragraph>
                To use this tool, enter a medical topic or question, and the system will search for
                contradictory statements in the literature.
              </Typography>
            </Box>
          )}
        </Box>

        <Box role="tabpanel" hidden={activeTab !== 1} id="tabpanel-1" sx={{ p: 3 }}>
          {activeTab === 1 && (
            <Box>
              <Typography variant="h6" gutterBottom>
                Community-Acquired Pneumonia (CAP) Analysis
              </Typography>
              <Typography paragraph>
                Comprehensive analysis of Community-Acquired Pneumonia literature, including
                treatment effectiveness, patient populations, and clinical outcomes.
              </Typography>
              <Typography paragraph>
                This specialized tool focuses on CAP research, providing insights into treatment
                protocols, antibiotic resistance patterns, and patient outcomes.
              </Typography>
            </Box>
          )}
        </Box>

        <Box role="tabpanel" hidden={activeTab !== 2} id="tabpanel-2" sx={{ p: 3 }}>
          {activeTab === 2 && (
            <Box>
              <Typography variant="h6" gutterBottom>
                Analysis History
              </Typography>
              <Typography paragraph>
                View and manage your previous analyses, with the ability to export results in
                various formats.
              </Typography>
              <Typography paragraph>
                Your analysis history is stored securely and can be accessed at any time. You can
                also share analyses with colleagues or export them for use in publications.
              </Typography>
            </Box>
          )}
        </Box>
      </Paper>

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
    </PageLayout>
  );
};

export default Analysis;
