import React, { useState } from 'react';
import { Box, Typography, Paper, Tabs, Tab } from '@mui/material';
import PageLayout from '../components/Layout/PageLayout';
import { useAuth } from '../context/AuthContext';

/**
 * ML Services page with tabs for different ML services
 */
const MLServices = () => {
  const [activeTab, setActiveTab] = useState(0);
  const { user } = useAuth();

  const handleTabChange = (event, newValue) => {
    setActiveTab(newValue);
  };

  return (
    <PageLayout
      title="Machine Learning Services"
      breadcrumbs={[{ label: 'ML Services', path: '/ml-services' }]}
      user={user}
    >
      <Paper sx={{ mb: 3 }}>
        <Tabs
          value={activeTab}
          onChange={handleTabChange}
          variant="fullWidth"
          aria-label="ML services tabs"
        >
          <Tab label="Contradiction Detection" id="tab-0" />
          <Tab label="Temporal Analysis" id="tab-1" />
          <Tab label="Bias Assessment" id="tab-2" />
        </Tabs>

        <Box role="tabpanel" hidden={activeTab !== 0} id="tabpanel-0" sx={{ p: 3 }}>
          {activeTab === 0 && (
            <Box>
              <Typography variant="h6" gutterBottom>
                Contradiction Detection
              </Typography>
              <Typography paragraph>
                Detect contradictions between medical claims using various ML models including
                BioMedLM, TSMixer, and Lorentz embeddings. This helps researchers identify
                conflicting evidence in the medical literature.
              </Typography>
              <Typography paragraph>
                To use this tool, enter a medical topic or question, and the system will analyze
                the literature for contradictory claims and evidence.
              </Typography>
            </Box>
          )}
        </Box>

        <Box role="tabpanel" hidden={activeTab !== 1} id="tabpanel-1" sx={{ p: 3 }}>
          {activeTab === 1 && (
            <Box>
              <Typography variant="h6" gutterBottom>
                Temporal Analysis
              </Typography>
              <Typography paragraph>
                Analyze the temporal confidence of medical claims based on publication dates
                and domain-specific characteristics. This helps researchers understand how the
                reliability of medical evidence changes over time.
              </Typography>
              <Typography paragraph>
                This tool uses advanced time-series analysis to track how medical knowledge
                evolves, identifying trends, breakthroughs, and obsolete information.
              </Typography>
            </Box>
          )}
        </Box>

        <Box role="tabpanel" hidden={activeTab !== 2} id="tabpanel-2" sx={{ p: 3 }}>
          {activeTab === 2 && (
            <Box>
              <Typography variant="h6" gutterBottom>
                Bias Assessment
              </Typography>
              <Typography paragraph>
                Assess bias in medical articles using various bias assessment tools like
                ROBINS-I, RoB 2, etc. This helps researchers evaluate the quality of evidence
                and identify potential sources of bias.
              </Typography>
              <Typography paragraph>
                Our ML models can automatically detect common biases in research methodology,
                reporting, and conclusions, providing a comprehensive bias profile for each study.
              </Typography>
            </Box>
          )}
        </Box>
      </Paper>

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
    </PageLayout>
  );
};

export default MLServices;
