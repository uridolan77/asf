import React, { useState } from 'react';
import {
  Box,
  Paper,
  Typography,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
  IconButton,
  Button,
  Divider,
  Tooltip,
  CircularProgress,
  Alert,
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
  SelectChangeEvent
} from '@mui/material';
import {
  Add as AddIcon,
  Delete as DeleteIcon,
  Edit as EditIcon,
  Refresh as RefreshIcon,
  Search as SearchIcon,
  Download as DownloadIcon,
  Info as InfoIcon
} from '@mui/icons-material';

import { useKnowledgeBase } from '../../hooks/useKnowledgeBase';
import { useFeatureFlags } from '../../context/FeatureFlagContext';
import { ButtonLoader } from '../UI/LoadingIndicators';

interface CreateKnowledgeBaseFormData {
  name: string;
  query: string;
  update_schedule: string;
}

/**
 * KnowledgeBaseList Component
 * 
 * Displays a list of knowledge bases with options to create, edit, and delete.
 */
const KnowledgeBaseList: React.FC = () => {
  // Feature flags
  const { isEnabled } = useFeatureFlags();
  const useMockData = isEnabled('useMockData');

  // State for dialogs
  const [createDialogOpen, setCreateDialogOpen] = useState<boolean>(false);
  const [deleteDialogOpen, setDeleteDialogOpen] = useState<boolean>(false);
  const [selectedKnowledgeBaseId, setSelectedKnowledgeBaseId] = useState<string | null>(null);
  
  // Form state
  const [formData, setFormData] = useState<CreateKnowledgeBaseFormData>({
    name: '',
    query: '',
    update_schedule: 'daily'
  });

  // Use the knowledge base hook
  const {
    knowledgeBases,
    isLoadingKnowledgeBases,
    isErrorKnowledgeBases,
    errorKnowledgeBases,
    refetchKnowledgeBases,
    createKnowledgeBase,
    isCreatingKnowledgeBase,
    deleteKnowledgeBase
  } = useKnowledgeBase();

  // Get the delete mutation function
  const { mutate: deleteMutate, isPending: isDeleting } = deleteKnowledgeBase(selectedKnowledgeBaseId || '');

  // Handle form input changes
  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
  };

  // Handle select changes
  const handleSelectChange = (e: SelectChangeEvent<string>) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
  };

  // Handle create dialog open
  const handleOpenCreateDialog = () => {
    setFormData({
      name: '',
      query: '',
      update_schedule: 'daily'
    });
    setCreateDialogOpen(true);
  };

  // Handle create dialog close
  const handleCloseCreateDialog = () => {
    setCreateDialogOpen(false);
  };

  // Handle create knowledge base
  const handleCreateKnowledgeBase = () => {
    createKnowledgeBase(formData);
    setCreateDialogOpen(false);
  };

  // Handle delete dialog open
  const handleOpenDeleteDialog = (id: string) => {
    setSelectedKnowledgeBaseId(id);
    setDeleteDialogOpen(true);
  };

  // Handle delete dialog close
  const handleCloseDeleteDialog = () => {
    setDeleteDialogOpen(false);
    setSelectedKnowledgeBaseId(null);
  };

  // Handle delete knowledge base
  const handleDeleteKnowledgeBase = () => {
    if (selectedKnowledgeBaseId) {
      deleteMutate({});
      setDeleteDialogOpen(false);
    }
  };

  // Format date
  const formatDate = (dateString: string) => {
    try {
      return new Date(dateString).toLocaleString();
    } catch (error) {
      return dateString;
    }
  };

  if (useMockData) {
    return (
      <Alert severity="info">
        Using mock data. Toggle the "Use Mock Data" feature flag to use real API data.
      </Alert>
    );
  }

  if (isLoadingKnowledgeBases) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
        <CircularProgress />
      </Box>
    );
  }

  if (isErrorKnowledgeBases) {
    return (
      <Alert 
        severity="error" 
        action={
          <Button color="inherit" size="small" onClick={() => refetchKnowledgeBases()}>
            Retry
          </Button>
        }
      >
        Error loading knowledge bases: {errorKnowledgeBases?.message || 'Unknown error'}
      </Alert>
    );
  }

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Typography variant="h6">Knowledge Bases</Typography>
        
        <Box sx={{ display: 'flex', gap: 1 }}>
          <Button
            variant="outlined"
            startIcon={<RefreshIcon />}
            onClick={() => refetchKnowledgeBases()}
            size="small"
          >
            Refresh
          </Button>
          
          <Button
            variant="contained"
            color="primary"
            startIcon={<AddIcon />}
            onClick={handleOpenCreateDialog}
            size="small"
          >
            Create New
          </Button>
        </Box>
      </Box>
      
      {knowledgeBases.length === 0 ? (
        <Alert severity="info">
          No knowledge bases found. Create a new one to get started.
        </Alert>
      ) : (
        <Paper variant="outlined">
          <List>
            {knowledgeBases.map((kb, index) => (
              <React.Fragment key={kb.id}>
                {index > 0 && <Divider />}
                <ListItem>
                  <ListItemText
                    primary={
                      <Box sx={{ display: 'flex', alignItems: 'center' }}>
                        <Typography variant="subtitle1">{kb.name}</Typography>
                        <Chip 
                          label={`${kb.article_count} articles`} 
                          size="small" 
                          color="primary" 
                          sx={{ ml: 1 }} 
                        />
                      </Box>
                    }
                    secondary={
                      <Box sx={{ mt: 1 }}>
                        <Typography variant="body2" color="text.secondary">
                          Query: {kb.query}
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          Created: {formatDate(kb.created_at)} | Last Updated: {formatDate(kb.last_updated)} | 
                          Update Schedule: {kb.update_schedule}
                        </Typography>
                      </Box>
                    }
                  />
                  <ListItemSecondaryAction>
                    <Tooltip title="View Details">
                      <IconButton edge="end" aria-label="details" href={`/knowledge-base/${kb.id}`}>
                        <InfoIcon />
                      </IconButton>
                    </Tooltip>
                    <Tooltip title="Search">
                      <IconButton edge="end" aria-label="search" href={`/search?kb=${kb.id}`}>
                        <SearchIcon />
                      </IconButton>
                    </Tooltip>
                    <Tooltip title="Export">
                      <IconButton edge="end" aria-label="export">
                        <DownloadIcon />
                      </IconButton>
                    </Tooltip>
                    <Tooltip title="Edit">
                      <IconButton edge="end" aria-label="edit">
                        <EditIcon />
                      </IconButton>
                    </Tooltip>
                    <Tooltip title="Delete">
                      <IconButton 
                        edge="end" 
                        aria-label="delete" 
                        onClick={() => handleOpenDeleteDialog(kb.id)}
                      >
                        <DeleteIcon />
                      </IconButton>
                    </Tooltip>
                  </ListItemSecondaryAction>
                </ListItem>
              </React.Fragment>
            ))}
          </List>
        </Paper>
      )}
      
      {/* Create Knowledge Base Dialog */}
      <Dialog open={createDialogOpen} onClose={handleCloseCreateDialog} maxWidth="sm" fullWidth>
        <DialogTitle>Create New Knowledge Base</DialogTitle>
        <DialogContent>
          <Box sx={{ pt: 1 }}>
            <TextField
              autoFocus
              margin="dense"
              name="name"
              label="Knowledge Base Name"
              type="text"
              fullWidth
              variant="outlined"
              value={formData.name}
              onChange={handleInputChange}
              sx={{ mb: 2 }}
            />
            
            <TextField
              margin="dense"
              name="query"
              label="Search Query"
              type="text"
              fullWidth
              variant="outlined"
              value={formData.query}
              onChange={handleInputChange}
              helperText="Enter a search query to populate this knowledge base"
              sx={{ mb: 2 }}
            />
            
            <FormControl fullWidth margin="dense">
              <InputLabel id="update-schedule-label">Update Schedule</InputLabel>
              <Select
                labelId="update-schedule-label"
                name="update_schedule"
                value={formData.update_schedule}
                label="Update Schedule"
                onChange={handleSelectChange}
              >
                <MenuItem value="hourly">Hourly</MenuItem>
                <MenuItem value="daily">Daily</MenuItem>
                <MenuItem value="weekly">Weekly</MenuItem>
                <MenuItem value="monthly">Monthly</MenuItem>
                <MenuItem value="manual">Manual Only</MenuItem>
              </Select>
            </FormControl>
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={handleCloseCreateDialog}>Cancel</Button>
          <Button 
            onClick={handleCreateKnowledgeBase} 
            variant="contained" 
            color="primary"
            disabled={!formData.name || !formData.query || isCreatingKnowledgeBase}
            startIcon={isCreatingKnowledgeBase ? <ButtonLoader size={20} /> : null}
          >
            {isCreatingKnowledgeBase ? 'Creating...' : 'Create'}
          </Button>
        </DialogActions>
      </Dialog>
      
      {/* Delete Confirmation Dialog */}
      <Dialog open={deleteDialogOpen} onClose={handleCloseDeleteDialog}>
        <DialogTitle>Delete Knowledge Base</DialogTitle>
        <DialogContent>
          <Typography>
            Are you sure you want to delete this knowledge base? This action cannot be undone.
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={handleCloseDeleteDialog}>Cancel</Button>
          <Button 
            onClick={handleDeleteKnowledgeBase} 
            color="error"
            disabled={isDeleting}
            startIcon={isDeleting ? <ButtonLoader size={20} /> : null}
          >
            {isDeleting ? 'Deleting...' : 'Delete'}
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default KnowledgeBaseList;
