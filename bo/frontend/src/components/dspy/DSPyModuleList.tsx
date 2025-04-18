import React, { useState, useEffect } from 'react';
import {
  Box,
  Button,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  IconButton,
  LinearProgress,
  Alert,
  Tooltip
} from '@mui/material';
import {
  Refresh as RefreshIcon,
  Sync as SyncIcon,
  RemoveRedEye as ViewIcon
} from '@mui/icons-material';
import ClientService from '../../services/ClientService';
import { formatDateTime } from '../../utils/formatters';

interface DSPyModuleListProps {
  clientId: string;
  onModuleSelect: (module: any) => void;
}

const DSPyModuleList: React.FC<DSPyModuleListProps> = ({ clientId, onModuleSelect }) => {
  const [modules, setModules] = useState<any[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [syncing, setSyncing] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadModules();
  }, [clientId]);

  const loadModules = async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await ClientService.getDSPyModules(clientId);
      setModules(data);
    } catch (err) {
      console.error('Error loading modules:', err);
      setError('Failed to load modules. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleSyncModules = async () => {
    try {
      setSyncing(true);
      setError(null);
      
      const result = await ClientService.syncDSPyModules(clientId);
      
      if (result.success) {
        await loadModules();
      } else {
        setError(`Failed to sync modules: ${result.message}`);
      }
    } catch (err) {
      console.error('Error syncing modules:', err);
      setError(`Failed to sync modules: ${(err as any).message}`);
    } finally {
      setSyncing(false);
    }
  };

  if (loading) {
    return <LinearProgress />;
  }

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'flex-end', mb: 2 }}>
        <Tooltip title="Refresh module list">
          <IconButton onClick={loadModules} disabled={loading} size="small" sx={{ mr: 1 }}>
            <RefreshIcon />
          </IconButton>
        </Tooltip>
        <Button
          variant="outlined"
          size="small"
          startIcon={<SyncIcon />}
          onClick={handleSyncModules}
          disabled={syncing}
        >
          {syncing ? 'Syncing...' : 'Sync Modules'}
        </Button>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      {modules.length === 0 ? (
        <Alert severity="info">
          No modules found for this client. Use the Sync Modules button to retrieve modules from the DSPy server.
        </Alert>
      ) : (
        <TableContainer component={Paper}>
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell>Name</TableCell>
                <TableCell>Type</TableCell>
                <TableCell>Description</TableCell>
                <TableCell>Registered</TableCell>
                <TableCell>Actions</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {modules.map((module) => (
                <TableRow key={module.module_id} hover>
                  <TableCell>{module.name}</TableCell>
                  <TableCell>{module.module_type || module.class_name || 'Unknown'}</TableCell>
                  <TableCell>{module.description || 'No description'}</TableCell>
                  <TableCell>{formatDateTime(module.registered_at)}</TableCell>
                  <TableCell>
                    <IconButton
                      size="small"
                      onClick={() => onModuleSelect(module)}
                      title="View Details"
                    >
                      <ViewIcon fontSize="small" />
                    </IconButton>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      )}
    </Box>
  );
};

export default DSPyModuleList;