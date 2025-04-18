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
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
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

import { useNotification } from '../../../context/NotificationContext';
import { ContentLoader } from '../../UI/LoadingIndicators';
import { useMCPProviders } from '../../../hooks/useMCPProviders';
import MCPConfigDialog from './MCPConfigDialog';

/**
 * MCP Provider Management Component
 *
 * This component provides functionality for managing MCP providers,
 * including adding, editing, and deleting providers.
 */
const MCPProviderManagement = ({ onProviderAdded, onProviderUpdated, onProviderDeleted }) => {
  // State
  const [configDialogOpen, setConfigDialogOpen] = useState(false);
  const [configDialogMode, setConfigDialogMode] = useState('add');
  const [selectedProvider, setSelectedProvider] = useState(null);
  const [deleteConfirmOpen, setDeleteConfirmOpen] = useState(false);
  const [providerToDelete, setProviderToDelete] = useState(null);

  // Use the MCP providers hook
  const {
    providers,
    isLoading,
    refetch,
    addProvider,
    updateProvider,
    deleteProvider,
    testProvider,
    testProviderLoading
  } = useMCPProviders();

  // Open edit dialog
  const handleEditProvider = (provider) => {
    setSelectedProvider(provider);
    setConfigDialogMode('edit');
    setConfigDialogOpen(true);
  };

  // Open delete confirmation dialog
  const handleDeleteClick = (provider) => {
    setProviderToDelete(provider);
    setDeleteConfirmOpen(true);
  };

  // Delete provider
  const handleDeleteProvider = () => {
    if (!providerToDelete) return;

    deleteProvider(providerToDelete.provider_id, {
      onSuccess: () => {
        setDeleteConfirmOpen(false);
        setProviderToDelete(null);
        if (onProviderDeleted) {
          onProviderDeleted();
        }
      }
    });
  };

  // Handle dialog close
  const handleDialogClose = () => {
    setConfigDialogOpen(false);
    setSelectedProvider(null);
  };

  // Handle provider save
  const handleProviderSave = (providerData) => {
    setConfigDialogOpen(false);

    if (configDialogMode === 'add') {
      addProvider(providerData, {
        onSuccess: () => {
          if (onProviderAdded) {
            onProviderAdded();
          }
        }
      });
    } else {
      updateProvider({
        providerId: providerData.provider_id,
        config: providerData
      }, {
        onSuccess: () => {
          if (onProviderUpdated) {
            onProviderUpdated();
          }
        }
      });
    }
  };

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h6">MCP Providers</Typography>

        <Box sx={{ display: 'flex', gap: 1 }}>
          <Button
            variant="outlined"
            onClick={() => refetch()}
            disabled={isLoading}
            startIcon={isLoading ? <CircularProgress size={20} /> : <RefreshIcon />}
          >
            {isLoading ? 'Refreshing...' : 'Refresh'}
          </Button>

          <Button
            variant="contained"
            color="primary"
            onClick={() => {
              setConfigDialogMode('add');
              setSelectedProvider(null);
              setConfigDialogOpen(true);
            }}
          >
            Add Provider
          </Button>
        </Box>
      </Box>

      {isLoading ? (
        <ContentLoader height={200} message="Loading MCP providers..." />
      ) : providers.length === 0 ? (
        <Alert severity="info">
          No MCP providers found. Click "Add Provider" to create one.
        </Alert>
      ) : (
        <TableContainer component={Paper}>
          <Table>
            <TableHead>
              <TableRow>
                <TableCell>Provider ID</TableCell>
                <TableCell>Display Name</TableCell>
                <TableCell>Transport Type</TableCell>
                <TableCell>Status</TableCell>
                <TableCell>Last Checked</TableCell>
                <TableCell>Actions</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {providers.map((provider) => (
                <TableRow key={provider.provider_id}>
                  <TableCell>{provider.provider_id}</TableCell>
                  <TableCell>{provider.display_name || '-'}</TableCell>
                  <TableCell>{provider.transport_type}</TableCell>
                  <TableCell>
                    <Chip
                      label={provider.status}
                      color={
                        provider.status === 'operational' ||
                        provider.status === 'available' ||
                        provider.status === 'connected' ? 'success' :
                        provider.status === 'error' ? 'error' : 'default'
                      }
                      size="small"
                    />
                  </TableCell>
                  <TableCell>
                    {new Date(provider.checked_at).toLocaleString()}
                  </TableCell>
                  <TableCell>
                    <Box sx={{ display: 'flex', gap: 1 }}>
                      <Tooltip title="Test Connection">
                        <IconButton
                          color="primary"
                          onClick={() => testProvider(provider.provider_id)}
                          disabled={testProviderLoading}
                        >
                          {testProviderLoading ? (
                            <CircularProgress size={20} />
                          ) : (
                            <CheckIcon />
                          )}
                        </IconButton>
                      </Tooltip>

                      <Tooltip title="Edit Provider">
                        <IconButton
                          color="primary"
                          onClick={() => handleEditProvider(provider)}
                        >
                          <EditIcon />
                        </IconButton>
                      </Tooltip>

                      <Tooltip title="Delete Provider">
                        <IconButton
                          color="error"
                          onClick={() => handleDeleteClick(provider)}
                        >
                          <DeleteIcon />
                        </IconButton>
                      </Tooltip>
                    </Box>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      )}

      {/* Config Dialog */}
      <MCPConfigDialog
        open={configDialogOpen}
        mode={configDialogMode}
        provider={selectedProvider}
        onClose={handleDialogClose}
        onSave={handleProviderSave}
      />

      {/* Delete Confirmation Dialog */}
      <Dialog
        open={deleteConfirmOpen}
        onClose={() => setDeleteConfirmOpen(false)}
      >
        <DialogTitle>Confirm Delete</DialogTitle>
        <DialogContent>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 2 }}>
            <WarningIcon color="warning" />
            <Typography>
              Are you sure you want to delete the provider "{providerToDelete?.display_name || providerToDelete?.provider_id}"?
              This action cannot be undone.
            </Typography>
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDeleteConfirmOpen(false)}>Cancel</Button>
          <Button onClick={handleDeleteProvider} color="error" variant="contained">
            Delete
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default MCPProviderManagement;
