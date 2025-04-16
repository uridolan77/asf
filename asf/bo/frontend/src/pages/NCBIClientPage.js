import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Box, Paper, Typography, Grid, Card, CardContent,
  Button, CircularProgress, Alert, Divider
} from '@mui/material';
import {
  Refresh as RefreshIcon,
  ArrowBack as ArrowBackIcon
} from '@mui/icons-material';

import PageLayout from '../components/Layout/PageLayout';
import { NCBIClient, NCBIPubMedSearch } from '../components/Clients';
import { ContentLoader } from '../components/UI/LoadingIndicators';
import { FadeIn } from '../components/UI/Animations';
import apiService from '../services/api';
import { useNotification } from '../context/NotificationContext.jsx';

/**
 * NCBI Client Management Page
 *
 * This page provides detailed management functionality for the NCBI client,
 * including configuration, testing, and usage statistics.
 */
const NCBIClientPage = () => {
  const { showSuccess, showError } = useNotification();
  const navigate = useNavigate();

  // State
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [client, setClient] = useState(null);
  const [refreshing, setRefreshing] = useState(false);

  // Load user data and client on mount
  useEffect(() => {
    const loadData = async () => {
      try {
        // Load user data
        const userData = await apiService.auth.me();
        if (userData.success) {
          setUser(userData.data);
        } else {
          if (userData.isAuthError) {
            handleLogout();
          }
        }

        // Load NCBI client
        await loadClient();
      } catch (error) {
        console.error('Error loading data:', error);
        showError('Failed to load data');
      } finally {
        setLoading(false);
      }
    };

    loadData();
  }, []);

  // Handle logout
  const handleLogout = () => {
    localStorage.removeItem('token');
    navigate('/');
  };

  // Load NCBI client
  const loadClient = async () => {
    setRefreshing(true);

    try {
      const result = await apiService.clients.getClient('ncbi');

      if (result.success) {
        setClient(result.data);
      } else {
        showError(`Failed to load NCBI client: ${result.error}`);
      }
    } catch (error) {
      console.error('Error loading NCBI client:', error);
      showError(`Error loading NCBI client: ${error.message}`);
    } finally {
      setRefreshing(false);
    }
  };

  // Handle client update
  const handleClientUpdate = (updatedClient) => {
    setClient(updatedClient);
  };

  if (loading) {
    return (
      <PageLayout
        title="NCBI Client Management"
        breadcrumbs={[
          { label: 'Clients Management', path: '/clients-management' },
          { label: 'NCBI', path: '/clients-management/ncbi' }
        ]}
        loading={true}
      />
    );
  }

  return (
    <PageLayout
      title="NCBI Client Management"
      breadcrumbs={[
        { label: 'Clients Management', path: '/clients-management' },
        { label: 'NCBI', path: '/clients-management/ncbi' }
      ]}
      user={user}
      actions={
        <>
          <Button
            variant="outlined"
            startIcon={<ArrowBackIcon />}
            onClick={() => navigate('/clients-management')}
            sx={{ mr: 1 }}
          >
            Back to Clients
          </Button>
          <Button
            variant="outlined"
            startIcon={refreshing ? <CircularProgress size={20} /> : <RefreshIcon />}
            onClick={loadClient}
            disabled={refreshing}
          >
            Refresh
          </Button>
        </>
      }
    >
      <FadeIn>
        {refreshing ? (
          <ContentLoader height={200} message="Loading NCBI client..." />
        ) : client ? (
          <>
            <NCBIClient
              client={client}
              onRefresh={loadClient}
              onConfigUpdate={handleClientUpdate}
            />

            <Box sx={{ mt: 3 }}>
              <NCBIPubMedSearch />
            </Box>
          </>
        ) : (
          <Alert severity="error">
            NCBI client not found. Please refresh to try again.
          </Alert>
        )}

        <Paper sx={{ mt: 3, p: 3 }}>
          <Typography variant="h6" gutterBottom>About NCBI</Typography>
          <Typography paragraph>
            The National Center for Biotechnology Information (NCBI) is part of the United States National Library of Medicine (NLM),
            a branch of the National Institutes of Health (NIH). It houses a series of databases relevant to biotechnology and biomedicine
            and is a vital resource for biomedical and genomic information.
          </Typography>

          <Divider sx={{ my: 2 }} />

          <Grid container spacing={2}>
            <Grid item xs={12} md={6}>
              <Typography variant="subtitle1" gutterBottom>Key Features:</Typography>
              <Box component="ul" sx={{ pl: 2 }}>
                <Box component="li"><Typography>Access to PubMed, a database of biomedical literature</Typography></Box>
                <Box component="li"><Typography>GenBank, a genetic sequence database</Typography></Box>
                <Box component="li"><Typography>BLAST, a sequence similarity search tool</Typography></Box>
                <Box component="li"><Typography>PubChem, a database of chemical molecules</Typography></Box>
                <Box component="li"><Typography>OMIM, a catalog of human genes and genetic disorders</Typography></Box>
              </Box>
            </Grid>
            <Grid item xs={12} md={6}>
              <Typography variant="subtitle1" gutterBottom>API Usage:</Typography>
              <Box component="ul" sx={{ pl: 2 }}>
                <Box component="li"><Typography>E-utilities for programmatic access to NCBI databases</Typography></Box>
                <Box component="li"><Typography>RESTful API for data retrieval</Typography></Box>
                <Box component="li"><Typography>Support for various data formats (XML, JSON, ASN.1)</Typography></Box>
                <Box component="li"><Typography>Rate limits apply - API key recommended</Typography></Box>
              </Box>
            </Grid>
          </Grid>

          <Box sx={{ mt: 2 }}>
            <Button
              variant="text"
              color="primary"
              onClick={() => window.open('https://www.ncbi.nlm.nih.gov/', '_blank')}
            >
              Visit NCBI Website
            </Button>
            <Button
              variant="text"
              color="primary"
              onClick={() => window.open('https://www.ncbi.nlm.nih.gov/books/NBK25500/', '_blank')}
              sx={{ ml: 2 }}
            >
              API Documentation
            </Button>
          </Box>
        </Paper>
      </FadeIn>
    </PageLayout>
  );
};

export default NCBIClientPage;
