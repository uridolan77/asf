import React from 'react';
import { Box, Typography, Paper } from '@mui/material';
import PageLayout from '../../components/Layout/PageLayout';

/**
 * LLM Management page with tabs for different LLM components
 */
const LLMManagement = () => {
  return (
    <PageLayout
      title="LLM Management"
      breadcrumbs={[{ label: 'LLM Management', path: '/llm-management' }]}
    >
      <Paper sx={{ p: 3 }}>
        <Typography variant="h6" gutterBottom>LLM Management</Typography>
        <Typography paragraph>
          This page provides management functionality for Large Language Model (LLM) components,
          including LLM Gateway, DSPy, and BiomedLM.
        </Typography>
      </Paper>
    </PageLayout>
  );
};

export default LLMManagement;
