import React from 'react';
import { Box, Typography, Paper } from '@mui/material';
import PageLayout from '../components/Layout/PageLayout';

/**
 * Analysis page with tabs for different analysis methods
 */
const Analysis = () => {
  // Mock user data
  const mockUser = {
    username: 'testuser',
    role_id: 1
  };

  return (
    <PageLayout
      title="Medical Literature Analysis"
      breadcrumbs={[{ label: 'Analysis', path: '/analysis' }]}
      user={mockUser}
    >
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
