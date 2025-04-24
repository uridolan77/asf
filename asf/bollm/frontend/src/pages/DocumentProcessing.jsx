import React, { useState } from 'react';
import { Box, Tab, Tabs, Paper, Typography, Container, Card, CardContent } from '@mui/material';
import { 
  Description as DescriptionIcon,
  Collections as CollectionsIcon,
  History as HistoryIcon,
  Settings as SettingsIcon
} from '@mui/icons-material';

import PageLayout from '../components/Layout/PageLayout';
import { useAuth } from '../context/AuthContext.jsx';
import { SingleDocumentProcessor, BatchDocumentProcessor, ProcessingHistory, ProcessingSettings } from '../components/DocumentProcessing';

/**
 * Document Processing page component
 * 
 * This page provides access to document processing functionality including:
 * - Single document processing
 * - Batch document processing
 * - Processing history
 * - Processing settings
 */
const DocumentProcessing = () => {
  const { user } = useAuth();
  const [activeTab, setActiveTab] = useState(0);
  const [loading, setLoading] = useState(false);

  const handleTabChange = (event, newValue) => {
    setActiveTab(newValue);
  };

  return (
    <PageLayout
      title="Document Processing"
      breadcrumbs={[{ label: 'Document Processing', path: '/document-processing' }]}
      loading={loading}
      user={user}
    >
      <Paper sx={{ mb: 3 }}>
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs 
            value={activeTab} 
            onChange={handleTabChange} 
            aria-label="Document processing tabs"
            variant="fullWidth"
          >
            <Tab 
              icon={<DescriptionIcon />} 
              label="Single Document" 
              id="tab-0" 
              aria-controls="tabpanel-0" 
            />
            <Tab 
              icon={<CollectionsIcon />} 
              label="Batch Processing" 
              id="tab-1" 
              aria-controls="tabpanel-1" 
            />
            <Tab 
              icon={<HistoryIcon />} 
              label="Processing History" 
              id="tab-2" 
              aria-controls="tabpanel-2" 
            />
            <Tab 
              icon={<SettingsIcon />} 
              label="Settings" 
              id="tab-3" 
              aria-controls="tabpanel-3" 
            />
          </Tabs>
        </Box>

        <Box role="tabpanel" hidden={activeTab !== 0} id="tabpanel-0" aria-labelledby="tab-0" sx={{ p: 3 }}>
          {activeTab === 0 && (
            <SingleDocumentProcessor />
          )}
        </Box>

        <Box role="tabpanel" hidden={activeTab !== 1} id="tabpanel-1" aria-labelledby="tab-1" sx={{ p: 3 }}>
          {activeTab === 1 && (
            <BatchDocumentProcessor />
          )}
        </Box>

        <Box role="tabpanel" hidden={activeTab !== 2} id="tabpanel-2" aria-labelledby="tab-2" sx={{ p: 3 }}>
          {activeTab === 2 && (
            <ProcessingHistory />
          )}
        </Box>

        <Box role="tabpanel" hidden={activeTab !== 3} id="tabpanel-3" aria-labelledby="tab-3" sx={{ p: 3 }}>
          {activeTab === 3 && (
            <ProcessingSettings />
          )}
        </Box>
      </Paper>

      {/* Additional information or help section */}
      <Paper sx={{ p: 3 }}>
        <Typography variant="h6" gutterBottom>About Document Processing</Typography>
        <Typography paragraph>
          This section provides access to advanced document processing capabilities for medical research papers:
        </Typography>
        <Typography component="div" sx={{ mb: 2 }}>
          <strong>Single Document Processing</strong> - Process individual medical research papers to extract
          structured information including entities, relations, and summaries. This helps researchers
          quickly understand the key findings and implications of a paper.
        </Typography>
        <Typography component="div" sx={{ mb: 2 }}>
          <strong>Batch Processing</strong> - Process multiple documents at once with configurable
          parallelism. This is useful for processing large collections of papers for literature reviews
          or systematic analyses.
        </Typography>
        <Typography component="div" sx={{ mb: 2 }}>
          <strong>Processing History</strong> - View the history of processed documents, including
          processing time, entity counts, and relation counts. You can also access the results of
          previous processing tasks.
        </Typography>
        <Typography component="div" sx={{ mb: 2 }}>
          <strong>Settings</strong> - Configure document processing settings, including PDF parsing
          preferences, entity extraction models, relation extraction models, and summarization options.
        </Typography>
      </Paper>
    </PageLayout>
  );
};

export default DocumentProcessing;
