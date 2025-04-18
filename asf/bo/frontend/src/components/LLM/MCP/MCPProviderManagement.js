import React, { useState, useEffect } from 'react';
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
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  FormControlLabel,
  Switch,
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
import apiService from '../../../services/api';
import { ContentLoader } from '../../UI/LoadingIndicators';
import MCPConfigDialog from './MCPConfigDialog';

/**
 * MCP Provider Management Component
 * 
 * This component provides functionality for managing MCP providers,
 * including adding, editing, and deleting providers.
 */
const MCPProviderManagement = ({ onProviderAdded, onProviderUpdated, onProviderDeleted }) => {
  const { showSuccess, showError } = useNotification();
  
  // State
  const [providers, setProviders] = useState([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [configDialogOpen, setConfigDialogOpen] = useState(false);
  const [configDialogMode, setConfigDialogMode] = useState('add');
  const [selectedProvider, setSelectedProvider] = useState(null);
  const [deleteConfirmOpen, setDeleteConfirmOpen] = useState(false);
  const [providerToDelete, setProviderToDelete] = useState(null);
  const [testingProvider, setTestingProvider] = useState(null);
  
  // Load providers on mount
  useEffect(() => {
    loadProviders();
  }, []);
  
  // Load MCP providers
  const loadProviders = async () => {
    setLoading(true);
    
    try {
      const result = await apiService.llm.getMCPProviders();
      
      if (result.success) {
        setProviders(result.data);
      } else {
        showError(`Failed to load MCP providers: ${result.error}`);
      }
    } catch (error) {
      console.error('Error loading MCP providers:', error);
      showError(`Error loading MCP providers: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };
  
  // Refresh providers
  const refreshProviders = async () => {
    setRefreshing(true);
    
    try {
      await loadProviders();
      showSuccess('MCP providers refreshed successfully');
    } catch (error) {
      console.error('Error refreshing MCP providers:', error);
      showError(`Error refreshing MCP providers: ${error.message}`);
    } finally {
      setRefreshing(false);
    }
  };
  
  // Test provider connection
  const testProvider = async (providerId) => {
    setTestingProvider(providerId);
    
    try {
      const result = await apiService.llm.testMCPProvider(providerId);
      
      if (result.success) {
        showSuccess(`Connection to ${providerId} successful (${result.data.latency_ms.toFixed(2)}ms)`);
      } else {
        showError(`Connection to ${providerId} failed: ${result.error}`);
      }
    } catch (error) {
      console.error(`Error testing MCP provider ${providerId}:`, error);
      showError(`Error testing MCP provider: ${error.message}`);
    } finally {
      setTestingProvider(null);
    }
  };
  
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
  const handleDeleteProvider = async () => {
    if (!providerToDelete) return;
    
    try {
      const result = await apiService.llm.deleteMCPProvider(providerToDelete.provider_id);
      
      if (result.success) {
        showSuccess(`Provider "${providerToDelete.display_name || providerToDelete.provider_id}" deleted successfully`);
        setDeleteConfirmOpen(false);
        setProviderToDelete(null);
        loadProviders();
        
        if (onProviderDeleted) {
          onProviderDeleted();
        }
      } else {
        showError(`Failed to delete provider: ${result.error}`);
      }
    } catch (error) {
      console.error('Error deleting provider:', error);
      showError(`Error deleting provider: ${error.message}`);
    }
  };
  
  // Handle dialog close
  const handleDialogClose = () => {
    setConfigDialogOpen(false);
    setSelectedProvider(null);
  };
  
  // Handle provider save
  const handleProviderSave = () => {
    setConfigDialogOpen(false);
    loadProviders();
    
    if (configDialogMode === 'add' && onProviderAdded) {
      onProviderAdded();
    } else if (configDialogMode === 'edit' && onProviderUpdated) {
      onProviderUpdated();
    }
  };
  
  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h6">MCP Providers</Typography>
        
        <Box sx={{ display: 'flex', gap: 1 }}>
          <Button
            variant="outlined"
            onClick={refreshProviders}
            disabled={refreshing}
            startIcon={refreshing ? <CircularProgress size={20} /> : <RefreshIcon />}
          >
            {refreshing ? 'Refreshing...' : 'Refresh'}
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
      
      {loading ? (
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
                          disabled={testingProvider === provider.provider_id}
                        >
                          {testingProvider === provider.provider_id ? (
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
