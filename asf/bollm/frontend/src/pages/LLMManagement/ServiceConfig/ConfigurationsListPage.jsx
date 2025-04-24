import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { 
  Box, 
  Typography, 
  Paper,
  Button,
  Alert,
  CircularProgress,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  IconButton,
  Chip,
  Dialog,
  DialogActions,
  DialogContent,
  DialogContentText,
  DialogTitle,
  TextField,
  FormControlLabel,
  Switch,
  Tooltip
} from '@mui/material';
import {
  Add as AddIcon,
  Edit as EditIcon,
  Delete as DeleteIcon,
  PlayArrow as ApplyIcon,
  Public as PublicIcon,
  Lock as PrivateIcon
} from '@mui/icons-material';

// Import the PageLayout component that includes the sidebar
import PageLayout from '../../../components/Layout/PageLayout';
import apiService from '../../../services/api';
import { useNotification } from '../../../context/NotificationContext';

/**
 * Service Configurations List Page
 * Shows all service configurations and allows managing them
 */
const ConfigurationsListPage = () => {
  const navigate = useNavigate();
  const { showSuccess, showError } = useNotification();
  
  // State
  const [loading, setLoading] = useState(true);
  const [configurations, setConfigurations] = useState([]);
  const [openCreateDialog, setOpenCreateDialog] = useState(false);
  const [openDeleteDialog, setOpenDeleteDialog] = useState(false);
  const [selectedConfig, setSelectedConfig] = useState(null);
  const [newConfig, setNewConfig] = useState({
    service_id: 'enhanced_llm_service',
    name: '',
    description: '',
    enable_caching: true,
    enable_resilience: true,
    enable_observability: true,
    enable_events: true,
    enable_progress_tracking: true,
    is_public: false
  });
  
  // Load configurations on mount
  useEffect(() => {
    loadConfigurations();
  }, []);
  
  // Load configurations from API
  const loadConfigurations = async () => {
    setLoading(true);
    try {
      const result = await apiService.llm.getServiceConfigurations();
      if (result.success) {
        setConfigurations(result.data);
      } else {
        showError(`Failed to load configurations: ${result.error}`);
      }
    } catch (error) {
      console.error('Error loading configurations:', error);
      showError(`Error loading configurations: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };
  
  // Handle create dialog open
  const handleOpenCreateDialog = () => {
    setOpenCreateDialog(true);
  };
  
  // Handle create dialog close
  const handleCloseCreateDialog = () => {
    setOpenCreateDialog(false);
    setNewConfig({
      service_id: 'enhanced_llm_service',
      name: '',
      description: '',
      enable_caching: true,
      enable_resilience: true,
      enable_observability: true,
      enable_events: true,
      enable_progress_tracking: true,
      is_public: false
    });
  };
  
  // Handle create configuration
  const handleCreateConfiguration = async () => {
    try {
      const result = await apiService.llm.createServiceConfiguration(newConfig);
      if (result.success) {
        showSuccess('Configuration created successfully');
        handleCloseCreateDialog();
        loadConfigurations();
      } else {
        showError(`Failed to create configuration: ${result.error}`);
      }
    } catch (error) {
      console.error('Error creating configuration:', error);
      showError(`Error creating configuration: ${error.message}`);
    }
  };
  
  // Handle delete dialog open
  const handleOpenDeleteDialog = (config) => {
    setSelectedConfig(config);
    setOpenDeleteDialog(true);
  };
  
  // Handle delete dialog close
  const handleCloseDeleteDialog = () => {
    setOpenDeleteDialog(false);
    setSelectedConfig(null);
  };
  
  // Handle delete configuration
  const handleDeleteConfiguration = async () => {
    if (!selectedConfig) return;
    
    try {
      const result = await apiService.llm.deleteServiceConfiguration(selectedConfig.id);
      if (result.success) {
        showSuccess('Configuration deleted successfully');
        handleCloseDeleteDialog();
        loadConfigurations();
      } else {
        showError(`Failed to delete configuration: ${result.error}`);
      }
    } catch (error) {
      console.error('Error deleting configuration:', error);
      showError(`Error deleting configuration: ${error.message}`);
    }
  };
  
  // Handle edit configuration
  const handleEditConfiguration = (config) => {
    navigate(`/llm/settings/service-config/configurations/${config.id}`);
  };
  
  // Handle apply configuration
  const handleApplyConfiguration = async (config) => {
    try {
      const result = await apiService.llm.applyServiceConfiguration(config.id);
      if (result.success) {
        showSuccess('Configuration applied successfully');
      } else {
        showError(`Failed to apply configuration: ${result.error}`);
      }
    } catch (error) {
      console.error('Error applying configuration:', error);
      showError(`Error applying configuration: ${error.message}`);
    }
  };
  
  // Handle new config change
  const handleNewConfigChange = (e) => {
    const { name, value, checked, type } = e.target;
    setNewConfig(prev => ({
      ...prev,
      [name]: type === 'checkbox' ? checked : value
    }));
  };
  
  // Define breadcrumbs for PageLayout
  const breadcrumbs = [
    { label: 'LLM Management', path: '/llm/dashboard' },
    { label: 'Settings', path: '/llm/settings/gateway' },
    { label: 'Service Configurations' }
  ];

  return (
    <PageLayout
      title="LLM Service Configurations"
      breadcrumbs={breadcrumbs}
      action={
        <Button
          variant="contained"
          startIcon={<AddIcon />}
          onClick={handleOpenCreateDialog}
        >
          Create Configuration
        </Button>
      }
    >
      <Box sx={{ mb: 3 }}>
        <Typography variant="body1" color="text.secondary" paragraph>
          Manage your LLM service configurations. Create, edit, and apply configurations to customize the service behavior.
        </Typography>
        
        {/* Main content */}
        {loading ? (
          <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
            <CircularProgress />
          </Box>
        ) : configurations.length === 0 ? (
          <Paper sx={{ p: 3, textAlign: 'center' }}>
            <Typography variant="h6" gutterBottom>
              No configurations found
            </Typography>
            <Typography paragraph>
              Create a new configuration to get started.
            </Typography>
            <Button
              variant="contained"
              startIcon={<AddIcon />}
              onClick={handleOpenCreateDialog}
            >
              Create Configuration
            </Button>
          </Paper>
        ) : (
          <TableContainer component={Paper}>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>Name</TableCell>
                  <TableCell>Description</TableCell>
                  <TableCell>Features</TableCell>
                  <TableCell>Visibility</TableCell>
                  <TableCell>Actions</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {configurations.map((config) => (
                  <TableRow key={config.id}>
                    <TableCell>{config.name}</TableCell>
                    <TableCell>{config.description}</TableCell>
                    <TableCell>
                      <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                        {config.enable_caching && (
                          <Chip size="small" label="Caching" color="primary" />
                        )}
                        {config.enable_resilience && (
                          <Chip size="small" label="Resilience" color="secondary" />
                        )}
                        {config.enable_observability && (
                          <Chip size="small" label="Observability" color="info" />
                        )}
                        {config.enable_events && (
                          <Chip size="small" label="Events" color="success" />
                        )}
                        {config.enable_progress_tracking && (
                          <Chip size="small" label="Progress" color="warning" />
                        )}
                      </Box>
                    </TableCell>
                    <TableCell>
                      {config.is_public ? (
                        <Chip 
                          icon={<PublicIcon />} 
                          label="Public" 
                          color="success" 
                          size="small" 
                        />
                      ) : (
                        <Chip 
                          icon={<PrivateIcon />} 
                          label="Private" 
                          color="default" 
                          size="small" 
                        />
                      )}
                    </TableCell>
                    <TableCell>
                      <Box sx={{ display: 'flex' }}>
                        <Tooltip title="Apply Configuration">
                          <IconButton 
                            color="primary" 
                            onClick={() => handleApplyConfiguration(config)}
                          >
                            <ApplyIcon />
                          </IconButton>
                        </Tooltip>
                        <Tooltip title="Edit Configuration">
                          <IconButton 
                            color="info" 
                            onClick={() => handleEditConfiguration(config)}
                          >
                            <EditIcon />
                          </IconButton>
                        </Tooltip>
                        <Tooltip title="Delete Configuration">
                          <IconButton 
                            color="error" 
                            onClick={() => handleOpenDeleteDialog(config)}
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
      </Box>
      
      {/* Create Configuration Dialog */}
      <Dialog open={openCreateDialog} onClose={handleCloseCreateDialog} maxWidth="sm" fullWidth>
        <DialogTitle>Create New Configuration</DialogTitle>
        <DialogContent>
          <DialogContentText>
            Create a new service configuration with the desired settings.
          </DialogContentText>
          <TextField
            autoFocus
            margin="dense"
            name="name"
            label="Configuration Name"
            type="text"
            fullWidth
            value={newConfig.name}
            onChange={handleNewConfigChange}
            required
          />
          <TextField
            margin="dense"
            name="description"
            label="Description"
            type="text"
            fullWidth
            multiline
            rows={2}
            value={newConfig.description}
            onChange={handleNewConfigChange}
          />
          <Box sx={{ mt: 2 }}>
            <Typography variant="subtitle2" gutterBottom>
              Features
            </Typography>
            <Box sx={{ display: 'flex', flexWrap: 'wrap' }}>
              <FormControlLabel
                control={
                  <Switch
                    checked={newConfig.enable_caching}
                    onChange={handleNewConfigChange}
                    name="enable_caching"
                  />
                }
                label="Caching"
              />
              <FormControlLabel
                control={
                  <Switch
                    checked={newConfig.enable_resilience}
                    onChange={handleNewConfigChange}
                    name="enable_resilience"
                  />
                }
                label="Resilience"
              />
              <FormControlLabel
                control={
                  <Switch
                    checked={newConfig.enable_observability}
                    onChange={handleNewConfigChange}
                    name="enable_observability"
                  />
                }
                label="Observability"
              />
              <FormControlLabel
                control={
                  <Switch
                    checked={newConfig.enable_events}
                    onChange={handleNewConfigChange}
                    name="enable_events"
                  />
                }
                label="Events"
              />
              <FormControlLabel
                control={
                  <Switch
                    checked={newConfig.enable_progress_tracking}
                    onChange={handleNewConfigChange}
                    name="enable_progress_tracking"
                  />
                }
                label="Progress Tracking"
              />
            </Box>
          </Box>
          <Box sx={{ mt: 2 }}>
            <Typography variant="subtitle2" gutterBottom>
              Visibility
            </Typography>
            <FormControlLabel
              control={
                <Switch
                  checked={newConfig.is_public}
                  onChange={handleNewConfigChange}
                  name="is_public"
                />
              }
              label="Make this configuration public (visible to all users)"
            />
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={handleCloseCreateDialog}>Cancel</Button>
          <Button 
            onClick={handleCreateConfiguration} 
            variant="contained"
            disabled={!newConfig.name}
          >
            Create
          </Button>
        </DialogActions>
      </Dialog>
      
      {/* Delete Configuration Dialog */}
      <Dialog open={openDeleteDialog} onClose={handleCloseDeleteDialog}>
        <DialogTitle>Delete Configuration</DialogTitle>
        <DialogContent>
          <DialogContentText>
            Are you sure you want to delete the configuration "{selectedConfig?.name}"? This action cannot be undone.
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button onClick={handleCloseDeleteDialog}>Cancel</Button>
          <Button onClick={handleDeleteConfiguration} color="error" variant="contained">
            Delete
          </Button>
        </DialogActions>
      </Dialog>
    </PageLayout>
  );
};

export default ConfigurationsListPage;
