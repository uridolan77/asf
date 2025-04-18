import React, { useState } from 'react';
import {
  Box,
  Paper,
  Typography,
  Button,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  IconButton,
  Chip,
  Alert,
  CircularProgress,
  Tooltip
} from '@mui/material';
import {
  Edit as EditIcon,
  Delete as DeleteIcon,
  Refresh as RefreshIcon,
  Check as CheckIcon,
  Warning as WarningIcon
} from '@mui/icons-material';

import { useLLM } from '../../../hooks/useLLM';
import { useFeatureFlags } from '../../../context/FeatureFlagContext';
import MCPConfigDialog from './MCPConfigDialog';

/**
 * MCPProviderList component
 * 
 * Displays a list of MCP providers with options to add, edit, and delete providers.
 * Uses React Query for data fetching and caching.
 */
const MCPProviderList: React.FC = () => {
  // State
  const [configDialogOpen, setConfigDialogOpen] = useState<boolean>(false);
  const [configDialogMode, setConfigDialogMode] = useState<'add' | 'edit'>('add');
  const [selectedProvider, setSelectedProvider] = useState<any | null>(null);
  
  // Feature flags
  const { isEnabled } = useFeatureFlags();
  const useMockData = isEnabled('useMockData');
  
  // LLM hooks
  const {
    providers,
    isLoadingProviders,
    isErrorProviders,
    errorProviders,
    refetchProviders,
    createProvider,
    isCreatingProvider,
    updateProvider,
    isUpdatingProvider,
    deleteProvider,
    isDeletingProvider
  } = useLLM();
  
  // Handle provider creation
  const handleCreateProvider = () => {
    setSelectedProvider(null);
    setConfigDialogMode('add');
    setConfigDialogOpen(true);
  };
  
  // Handle provider edit
  const handleEditProvider = (provider: any) => {
    setSelectedProvider(provider);
    setConfigDialogMode('edit');
    setConfigDialogOpen(true);
  };
  
  // Handle provider save
  const handleSaveProvider = (provider: any) => {
    if (configDialogMode === 'add') {
      createProvider(provider);
    } else {
      updateProvider(provider);
    }
    setConfigDialogOpen(false);
  };
  
  // Handle provider delete
  const handleDeleteProvider = (id: string) => {
    if (window.confirm('Are you sure you want to delete this provider?')) {
      deleteProvider({ id });
    }
  };
  
  // Render loading state
  if (isLoadingProviders) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
        <CircularProgress />
      </Box>
    );
  }
  
  // Render error state
  if (isErrorProviders) {
    return (
      <Alert 
        severity="error" 
        action={
          <Button color="inherit" size="small" onClick={() => refetchProviders()}>
            Retry
          </Button>
        }
      >
        Failed to load providers: {errorProviders?.message || 'Unknown error'}
      </Alert>
    );
  }
  
  return (
    <Box>
      {useMockData && (
        <Alert severity="info" sx={{ mb: 2 }}>
          Using mock data. Toggle the "Use Mock Data" feature flag to use real API data.
        </Alert>
      )}
      
      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
        <Typography variant="h6">MCP Providers</Typography>
        <Box>
          <Button
            variant="contained"
            color="primary"
            onClick={handleCreateProvider}
            disabled={isCreatingProvider}
            startIcon={isCreatingProvider ? <CircularProgress size={20} /> : undefined}
          >
            Add Provider
          </Button>
          <Tooltip title="Refresh providers">
            <IconButton 
              onClick={() => refetchProviders()} 
              sx={{ ml: 1 }}
              disabled={isLoadingProviders}
            >
              <RefreshIcon />
            </IconButton>
          </Tooltip>
        </Box>
      </Box>
      
      <TableContainer component={Paper}>
        <Table>
          <TableHead>
            <TableRow>
              <TableCell>ID</TableCell>
              <TableCell>Name</TableCell>
              <TableCell>Transport</TableCell>
              <TableCell>Status</TableCell>
              <TableCell>Actions</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {providers.length === 0 ? (
              <TableRow>
                <TableCell colSpan={5} align="center">
                  No providers found. Click "Add Provider" to create one.
                </TableCell>
              </TableRow>
            ) : (
              providers.map((provider) => (
                <TableRow key={provider.id}>
                  <TableCell>{provider.id}</TableCell>
                  <TableCell>{provider.name}</TableCell>
                  <TableCell>{provider.transport_type}</TableCell>
                  <TableCell>
                    <Chip
                      icon={provider.status === 'connected' ? <CheckIcon /> : <WarningIcon />}
                      label={provider.status}
                      color={provider.status === 'connected' ? 'success' : 'error'}
                      size="small"
                    />
                  </TableCell>
                  <TableCell>
                    <Tooltip title="Edit provider">
                      <IconButton 
                        onClick={() => handleEditProvider(provider)}
                        disabled={isUpdatingProvider}
                      >
                        <EditIcon />
                      </IconButton>
                    </Tooltip>
                    <Tooltip title="Delete provider">
                      <IconButton 
                        onClick={() => handleDeleteProvider(provider.id)}
                        disabled={isDeletingProvider}
                      >
                        <DeleteIcon />
                      </IconButton>
                    </Tooltip>
                  </TableCell>
                </TableRow>
              ))
            )}
          </TableBody>
        </Table>
      </TableContainer>
      
      <MCPConfigDialog
        open={configDialogOpen}
        mode={configDialogMode}
        provider={selectedProvider}
        onClose={() => setConfigDialogOpen(false)}
        onSave={handleSaveProvider}
      />
    </Box>
  );
};

export default MCPProviderList;
