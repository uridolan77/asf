import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Box, Tab, Tabs, Paper, Typography, Alert } from '@mui/material';
import { Search as SearchIcon, Science as ScienceIcon } from '@mui/icons-material';

import PageLayout from '../components/Layout/PageLayout';
import { PICOSearch as PICOSearchComponent, AdvancedSearch } from '../components/Search';
import { useAuth } from '../context/AuthContext.jsx';
import { useNotification } from '../context/NotificationContext.jsx';

/**
 * PICO Search page with tabs for different search methods
 */
const PICOSearch = () => {
  const { user, api } = useAuth();
  const { showError } = useNotification();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [searchResults, setSearchResults] = useState(null);
  const [activeTab, setActiveTab] = useState(0);
  const navigate = useNavigate();

  // Handle tab change
  const handleTabChange = (_, newValue) => {
    setActiveTab(newValue);
  };

  // Handle search results
  const handleSearchResults = (results) => {
    setSearchResults(results);
  };

  // Define page actions if needed
  const pageActions = null;

  return (
    <PageLayout
      title="Medical Literature Search"
      breadcrumbs={[{ label: 'PICO Search', path: '/pico-search' }]}
      loading={loading}
      user={user}
      actions={pageActions}
    >
      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}

      <Paper sx={{ mb: 3 }}>
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs
            value={activeTab}
            onChange={handleTabChange}
            aria-label="search method tabs"
            variant="fullWidth"
          >
            <Tab
              icon={<ScienceIcon />}
              label="PICO Search"
              id="tab-0"
              aria-controls="tabpanel-0"
            />
            <Tab
              icon={<SearchIcon />}
              label="Advanced Search"
              id="tab-1"
              aria-controls="tabpanel-1"
            />
          </Tabs>
        </Box>

        <Box role="tabpanel" hidden={activeTab !== 0} id="tabpanel-0" aria-labelledby="tab-0" sx={{ p: 3 }}>
          {activeTab === 0 && (
            <PICOSearchComponent onSearchResults={handleSearchResults} api={api} />
          )}
        </Box>

        <Box role="tabpanel" hidden={activeTab !== 1} id="tabpanel-1" aria-labelledby="tab-1" sx={{ p: 3 }}>
          {activeTab === 1 && (
            <AdvancedSearch onSearchResults={handleSearchResults} api={api} />
          )}
        </Box>
      </Paper>

      {/* Additional information or help section */}
      {!searchResults && (
        <Paper sx={{ p: 3 }}>
          <Typography variant="h6" gutterBottom>About Medical Literature Search</Typography>
          <Typography paragraph>
            This tool provides two different approaches to searching medical literature:
          </Typography>
          <Typography component="div" sx={{ mb: 2 }}>
            <strong>PICO Search</strong> - Uses the Population, Intervention, Comparison, Outcome framework
            to structure clinical questions and find evidence-based answers. Ideal for clinical decision making.
          </Typography>
          <Typography component="div">
            <strong>Advanced Search</strong> - Provides more flexible search options with filters for
            publication years, article types, sources, and more. Ideal for comprehensive literature reviews.
          </Typography>
        </Paper>
      )}
    </PageLayout>
  );
};

export default PICOSearch;