import React, { useState } from 'react';
import {
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Button,
  Box,
  Typography,
  Chip,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  Grid
} from '@mui/material';
import {
  Add as AddIcon,
  Refresh as RefreshIcon,
  Edit as EditIcon,
  Delete as DeleteIcon
} from '@mui/icons-material';
import ClientService from '../../services/ClientService';
import { formatDateTime } from '../../utils/formatters';

interface DSPyClientListProps {
  clients: any[];
  onClientSelect: (client: any) => void;
  onRefresh: () => void;
}

const DSPyClientList: React.FC<DSPyClientListProps> = ({ clients, onClientSelect, onRefresh }) => {
  const [openDialog, setOpenDialog] = useState<boolean>(false);
  const [deleteDialog, setDeleteDialog] = useState<boolean>(false);
  const [currentClient, setCurrentClient] = useState<any>(null);
  const [formData, setFormData] = useState<any>({
    name: '',
    description: '',
    base_url: ''
  });
  
  const handleOpenDialog = (client?: any) => {
    if (client) {
      setCurrentClient(client);
      setFormData({
        name: client.name,
        description: client.description || '',
        base_url: client.base_url
      });
    } else {
      setCurrentClient(null);
      setFormData({
        name: '',
        description: '',
        base_url: ''
      });
    }
    setOpenDialog(true);
  };

  const handleCloseDialog = () => {
    setOpenDialog(false);
    setCurrentClient(null);
  };

  const handleOpenDeleteDialog = (client: any) => {
    setCurrentClient(client);
    setDeleteDialog(true);
  };

  const handleCloseDeleteDialog = () => {
    setDeleteDialog(false);
    setCurrentClient(null);
  };

  const handleDeleteClient = async () => {
    try {
      await ClientService.deleteDSPyClient(currentClient.client_id);
      handleCloseDeleteDialog();
      onRefresh();
    } catch (error) {
      console.error('Failed to delete client:', error);
      alert('Failed to delete client. Please try again.');
    }
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setFormData({
      ...formData,
      [name]: value
    });
  };

  const handleSubmit = async () => {
    try {
      if (currentClient) {
        // Update existing client
        await ClientService.updateDSPyClient(currentClient.client_id, formData);
      } else {
        // Create new client
        await ClientService.createDSPyClient(formData);
      }
      handleCloseDialog();
      onRefresh();
    } catch (error) {
      console.error('Failed to save client:', error);
      alert('Failed to save client. Please try again.');
    }
  };

  const getStatusColor = (status: string) => {
    switch (status?.toUpperCase()) {
      case 'CONNECTED': return 'success';
      case 'ERROR': return 'error';
      default: return 'default';
    }
  };

  return (
    <>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Typography variant="h5">DSPy Clients</Typography>
        <Box>
          <Button
            variant="contained"
            color="primary"
            startIcon={<AddIcon />}
            onClick={() => handleOpenDialog()}
            sx={{ mr: 1 }}
          >
            Add Client
          </Button>
          <IconButton onClick={onRefresh} color="primary">
            <RefreshIcon />
          </IconButton>
        </Box>
      </Box>

      <TableContainer component={Paper}>
        <Table>
          <TableHead>
            <TableRow>
              <TableCell>Name</TableCell>
              <TableCell>Base URL</TableCell>
              <TableCell>Status</TableCell>
              <TableCell>Last Checked</TableCell>
              <TableCell>Actions</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {clients.length === 0 ? (
              <TableRow>
                <TableCell colSpan={5} align="center">
                  No DSPy clients found. Add a new one to get started.
                </TableCell>
              </TableRow>
            ) : (
              clients.map((client) => (
                <TableRow key={client.client_id} hover onClick={() => onClientSelect(client)} style={{ cursor: 'pointer' }}>
                  <TableCell>{client.name}</TableCell>
                  <TableCell>{client.base_url}</TableCell>
                  <TableCell>
                    <Chip
                      label={client.status?.status || 'UNKNOWN'}
                      color={getStatusColor(client.status?.status) as any}
                      size="small"
                    />
                  </TableCell>
                  <TableCell>
                    {client.status?.last_checked ? formatDateTime(client.status.last_checked) : 'Never'}
                  </TableCell>
                  <TableCell>
                    <IconButton
                      onClick={(e) => {
                        e.stopPropagation();
                        handleOpenDialog(client);
                      }}
                      size="small"
                    >
                      <EditIcon fontSize="small" />
                    </IconButton>
                    <IconButton
                      onClick={(e) => {
                        e.stopPropagation();
                        handleOpenDeleteDialog(client);
                      }}
                      size="small"
                      color="error"
                    >
                      <DeleteIcon fontSize="small" />
                    </IconButton>
                  </TableCell>
                </TableRow>
              ))
            )}
          </TableBody>
        </Table>
      </TableContainer>
      
      {/* Add/Edit Dialog */}
      <Dialog open={openDialog} onClose={handleCloseDialog} maxWidth="sm" fullWidth>
        <DialogTitle>{currentClient ? `Edit Client: ${currentClient.name}` : 'Add New DSPy Client'}</DialogTitle>
        <DialogContent>
          <Grid container spacing={2} sx={{ mt: 1 }}>
            <Grid item xs={12}>
              <TextField
                name="name"
                label="Client Name"
                fullWidth
                value={formData.name}
                onChange={handleInputChange}
                required
              />
            </Grid>
            <Grid item xs={12}>
              <TextField
                name="description"
                label="Description"
                fullWidth
                value={formData.description}
                onChange={handleInputChange}
                multiline
                rows={2}
              />
            </Grid>
            <Grid item xs={12}>
              <TextField
                name="base_url"
                label="Base URL"
                fullWidth
                value={formData.base_url}
                onChange={handleInputChange}
                required
                placeholder="http://localhost:8000"
                helperText="The base URL of the DSPy server"
              />
            </Grid>
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button onClick={handleCloseDialog}>Cancel</Button>
          <Button onClick={handleSubmit} variant="contained" color="primary">
            {currentClient ? 'Update' : 'Add'}
          </Button>
        </DialogActions>
      </Dialog>

      {/* Delete Confirmation Dialog */}
      <Dialog open={deleteDialog} onClose={handleCloseDeleteDialog}>
        <DialogTitle>Confirm Delete</DialogTitle>
        <DialogContent>
          Are you sure you want to delete the client "{currentClient?.name}"?
          This action cannot be undone.
        </DialogContent>
        <DialogActions>
          <Button onClick={handleCloseDeleteDialog}>Cancel</Button>
          <Button onClick={handleDeleteClient} color="error" variant="contained">
            Delete
          </Button>
        </DialogActions>
      </Dialog>
    </>
  );
};

export default DSPyClientList;