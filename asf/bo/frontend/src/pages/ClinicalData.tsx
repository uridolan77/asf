import React, { useState } from 'react';
import {
  Box,
  Paper,
  Tabs,
  Tab,
  Typography,
  Alert
} from '@mui/material';
import {
  Search as SearchIcon,
  Psychology as PsychologyIcon,
  Science as ScienceIcon
} from '@mui/icons-material';

import PageLayout from '../components/Layout/PageLayout';
import TermSearch from '../components/ClinicalData/TermSearch';
import SemanticSearch from '../components/ClinicalData/SemanticSearch';
import ConceptExplorer from '../components/ClinicalData/ConceptExplorer';
import { useAuth } from '../hooks/useAuth';
import { useFeatureFlags } from '../context/FeatureFlagContext';

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`clinical-data-tabpanel-${index}`}
      aria-labelledby={`clinical-data-tab-${index}`}
      {...other}
    >
      {value === index && (
        <Box sx={{ p: 3 }}>
          {children}
        </Box>
      )}
    </div>
  );
}

/**
 * ClinicalData page component
 * 
 * This page provides access to clinical data search tools:
 * - Term Search: Search for medical terms and find related concepts and trials
 * - Semantic Search: Search with semantic expansion of medical terms
 * - Concept Explorer: Find trials by medical concept ID
 */
const ClinicalData: React.FC = () => {
  const [activeTab, setActiveTab] = useState(0);
  const { user } = useAuth();
  const { isEnabled } = useFeatureFlags();
  const useMockData = isEnabled('useMockData');

  const handleTabChange = (_: React.SyntheticEvent, newValue: number) => {
    setActiveTab(newValue);
  };

  return (
    <PageLayout
      title="Clinical Data"
      breadcrumbs={[{ label: 'Clinical Data', path: '/clinical-data' }]}
      user={user}
    >
      {useMockData && (
        <Alert severity="info" sx={{ mb: 2 }}>
          Using mock data. Toggle the "Use Mock Data" feature flag to use real API data.
        </Alert>
      )}

      <Paper sx={{ width: '100%' }}>
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs
            value={activeTab}
            onChange={handleTabChange}
            aria-label="clinical data tabs"
            variant="fullWidth"
          >
            <Tab
              icon={<SearchIcon />}
              label="Term Search"
              id="clinical-data-tab-0"
              aria-controls="clinical-data-tabpanel-0"
            />
            <Tab
              icon={<PsychologyIcon />}
              label="Semantic Search"
              id="clinical-data-tab-1"
              aria-controls="clinical-data-tabpanel-1"
            />
            <Tab
              icon={<ScienceIcon />}
              label="Concept Explorer"
              id="clinical-data-tab-2"
              aria-controls="clinical-data-tabpanel-2"
            />
          </Tabs>
        </Box>

        <TabPanel value={activeTab} index={0}>
          <TermSearch />
        </TabPanel>

        <TabPanel value={activeTab} index={1}>
          <SemanticSearch />
        </TabPanel>

        <TabPanel value={activeTab} index={2}>
          <ConceptExplorer />
        </TabPanel>
      </Paper>

      <Paper sx={{ p: 3, mt: 3 }}>
        <Typography variant="h6" gutterBottom>About Clinical Data</Typography>
        <Typography paragraph>
          This page provides tools for searching and exploring clinical data from various sources,
          including SNOMED CT, ICD-10, and ClinicalTrials.gov.
        </Typography>
        <Typography paragraph>
          <strong>Term Search:</strong> Search for medical terms to find related SNOMED CT concepts
          and clinical trials. This tool helps you understand the standardized terminology for
          medical conditions and find relevant clinical trials.
        </Typography>
        <Typography paragraph>
          <strong>Semantic Search:</strong> Search for clinical trials with semantic expansion of
          medical terms. This tool uses natural language processing to find trials related to your
          search term, even if they don't use the exact same terminology.
        </Typography>
        <Typography paragraph>
          <strong>Concept Explorer:</strong> Find clinical trials by medical concept ID. This tool
          is useful when you already know the specific SNOMED CT or other terminology concept ID
          and want to find related trials.
        </Typography>
      </Paper>
    </PageLayout>
  );
};

export default ClinicalData;
