// ClinicalData.jsx
import React, { useState } from 'react';
import {
  Box, Typography, Divider,
  Card, CardContent, CardHeader, Tabs, Tab, Alert
} from '@mui/material';

// Material Icons
import {
  Search as SearchIcon,
  Biotech as BiotechIcon,
  Science as ScienceIcon,
  LocalHospital as LocalHospitalIcon
} from '@mui/icons-material';

import PageLayout from '../components/Layout/PageLayout';
import { useAuth } from '../context/AuthContext.jsx';

// Import components
import TermSearch from '../components/ClinicalData/TermSearch';
import ConceptExplorer from '../components/ClinicalData/ConceptExplorer';
import TrialMapping from '../components/ClinicalData/TrialMapping';
import SemanticSearch from '../components/ClinicalData/SemanticSearch';

/**
 * Clinical Data page component
 * Provides access to clinical data functionality including:
 * - Term search with SNOMED CT concepts and clinical trials
 * - Concept explorer for finding trials by medical concept
 * - Trial mapping for mapping conditions to SNOMED CT
 * - Semantic search with term expansion
 */
const ClinicalData = () => {
  // State
  const { user } = useAuth();
  const [loading] = useState(false);
  const [error] = useState('');
  const [tabValue, setTabValue] = useState(0);

  // Handle tab change
  const handleTabChange = (_, newValue) => {
    setTabValue(newValue);
  };

  return (
    <PageLayout
      title="Clinical Data Integration"
      breadcrumbs={[{ label: 'Clinical Data', path: '/clinical-data' }]}
      loading={loading}
      user={user}
    >
      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}

      <Card sx={{ mb: 4 }}>
        <CardHeader
          title="Clinical Data Integration"
          subheader="Connecting medical terminology with clinical trials data"
        />
        <Divider />
        <CardContent>
          <Typography variant="body1" paragraph>
            These tools combine medical terminology (SNOMED CT) with clinical trials data to provide
            powerful insights and connections between standardized medical concepts and real-world clinical research.
          </Typography>

          <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 3 }}>
            <Tabs
              value={tabValue}
              onChange={handleTabChange}
              aria-label="clinical data tabs"
            >
              <Tab label="Term Search" icon={<SearchIcon />} iconPosition="start" />
              <Tab label="Concept Explorer" icon={<ScienceIcon />} iconPosition="start" />
              <Tab label="Trial Mapping" icon={<LocalHospitalIcon />} iconPosition="start" />
              <Tab label="Semantic Search" icon={<BiotechIcon />} iconPosition="start" />
            </Tabs>
          </Box>

          {/* Tab content */}
          {tabValue === 0 && <TermSearch />}
          {tabValue === 1 && <ConceptExplorer />}
          {tabValue === 2 && <TrialMapping />}
          {tabValue === 3 && <SemanticSearch />}
        </CardContent>
      </Card>
    </PageLayout>
  );
};

export default ClinicalData;