import React, { useState } from 'react';
import { 
  Box, 
  Typography, 
  Grid, 
  Paper, 
  Chip, 
  Button, 
  Divider,
  Card,
  CardContent,
  CardHeader,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogContentText,
  DialogActions,
  Alert,
  useTheme
} from '@mui/material';
import { 
  Delete as DeleteIcon,
  Edit as EditIcon,
  Refresh as RefreshIcon,
  ContentCopy as CopyIcon,
  Download as DownloadIcon,
  Info as InfoIcon
} from '@mui/icons-material';
import { useSnackbar } from 'notistack';
import moment from 'moment';

import { deleteAdapter } from '../../../services/cl_peft_service';

const AdapterDetails = ({ adapter, onRefresh }) => {
  const theme = useTheme();
  const { enqueueSnackbar } = useSnackbar();
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [deleting, setDeleting] = useState(false);
  
  if (!adapter) {
    return null;
  }
  
  const handleCopyId = () => {
    navigator.clipboard.writeText(adapter.adapter_id);
    enqueueSnackbar('Adapter ID copied to clipboard', { variant: 'success' });
  };
  
  const handleDeleteClick = () => {
    setDeleteDialogOpen(true);
  };
  
  const handleDeleteCancel = () => {
    setDeleteDialogOpen(false);
  };
  
  const handleDeleteConfirm = async () => {
    setDeleting(true);
    try {
      await deleteAdapter(adapter.adapter_id);
      setDeleteDialogOpen(false);
      enqueueSnackbar('Adapter deleted successfully', { variant: 'success' });
      onRefresh();
    } catch (error) {
      console.error('Error deleting adapter:', error);
      enqueueSnackbar('Failed to delete adapter', { variant: 'error' });
    } finally {
      setDeleting(false);
    }
  };
  
  const getStatusColor = (status) => {
    switch (status) {
      case 'ready':
        return theme.palette.success.main;
      case 'error':
        return theme.palette.error.main;
      case 'initializing':
      case 'training':
        return theme.palette.warning.main;
      default:
        return theme.palette.grey[500];
    }
  };
  
  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h5" component="h2">
          {adapter.adapter_name}
        </Typography>
        <Box sx={{ display: 'flex', gap: 1 }}>
          <Button
            variant="outlined"
            size="small"
            startIcon={<RefreshIcon />}
            onClick={onRefresh}
          >
            Refresh
          </Button>
          <Button
            variant="outlined"
            size="small"
            color="error"
            startIcon={<DeleteIcon />}
            onClick={handleDeleteClick}
          >
            Delete
          </Button>
        </Box>
      </Box>
      
      {adapter.status === 'error' && (
        <Alert severity="error" sx={{ mb: 3 }}>
          This adapter is in an error state. Check the logs for more information.
        </Alert>
      )}
      
      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Card variant="outlined">
            <CardHeader title="Basic Information" />
            <Divider />
            <CardContent>
              <Grid container spacing={2}>
                <Grid item xs={12}>
                  <Box sx={{ display: 'flex', alignItems: 'center' }}>
                    <Typography variant="subtitle2" sx={{ width: 120 }}>Adapter ID:</Typography>
                    <Typography variant="body2" sx={{ flexGrow: 1 }}>{adapter.adapter_id}</Typography>
                    <IconButton size="small" onClick={handleCopyId}>
                      <CopyIcon fontSize="small" />
                    </IconButton>
                  </Box>
                </Grid>
                
                <Grid item xs={12}>
                  <Box sx={{ display: 'flex', alignItems: 'center' }}>
                    <Typography variant="subtitle2" sx={{ width: 120 }}>Base Model:</Typography>
                    <Typography variant="body2">{adapter.base_model_name}</Typography>
                  </Box>
                </Grid>
                
                <Grid item xs={12}>
                  <Box sx={{ display: 'flex', alignItems: 'center' }}>
                    <Typography variant="subtitle2" sx={{ width: 120 }}>Status:</Typography>
                    <Chip 
                      label={adapter.status} 
                      size="small" 
                      sx={{ 
                        backgroundColor: getStatusColor(adapter.status),
                        color: '#fff'
                      }} 
                    />
                  </Box>
                </Grid>
                
                <Grid item xs={12}>
                  <Box sx={{ display: 'flex', alignItems: 'center' }}>
                    <Typography variant="subtitle2" sx={{ width: 120 }}>Created:</Typography>
                    <Typography variant="body2">
                      {moment(adapter.created_at).format('MMMM D, YYYY h:mm A')}
                    </Typography>
                  </Box>
                </Grid>
                
                {adapter.updated_at && (
                  <Grid item xs={12}>
                    <Box sx={{ display: 'flex', alignItems: 'center' }}>
                      <Typography variant="subtitle2" sx={{ width: 120 }}>Last Updated:</Typography>
                      <Typography variant="body2">
                        {moment(adapter.updated_at).format('MMMM D, YYYY h:mm A')}
                      </Typography>
                    </Box>
                  </Grid>
                )}
                
                {adapter.description && (
                  <Grid item xs={12}>
                    <Typography variant="subtitle2" sx={{ mb: 1 }}>Description:</Typography>
                    <Typography variant="body2">{adapter.description}</Typography>
                  </Grid>
                )}
              </Grid>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} md={6}>
          <Card variant="outlined">
            <CardHeader title="Technical Details" />
            <Divider />
            <CardContent>
              <Grid container spacing={2}>
                <Grid item xs={12} sm={6}>
                  <Typography variant="subtitle2">CL Strategy:</Typography>
                  <Chip 
                    label={adapter.cl_strategy} 
                    size="small" 
                    sx={{ mt: 1 }} 
                  />
                </Grid>
                
                <Grid item xs={12} sm={6}>
                  <Typography variant="subtitle2">PEFT Method:</Typography>
                  <Chip 
                    label={adapter.peft_method} 
                    size="small" 
                    sx={{ mt: 1 }} 
                  />
                </Grid>
                
                <Grid item xs={12}>
                  <Divider sx={{ my: 1 }} />
                </Grid>
                
                <Grid item xs={12}>
                  <Typography variant="subtitle2" sx={{ mb: 1 }}>Tags:</Typography>
                  <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                    {adapter.tags && adapter.tags.length > 0 ? (
                      adapter.tags.map((tag, index) => (
                        <Chip 
                          key={index} 
                          label={tag} 
                          size="small" 
                          variant="outlined" 
                        />
                      ))
                    ) : (
                      <Typography variant="body2" color="textSecondary">No tags</Typography>
                    )}
                  </Box>
                </Grid>
              </Grid>
            </CardContent>
          </Card>
          
          <Card variant="outlined" sx={{ mt: 3 }}>
            <CardHeader title="Task History" />
            <Divider />
            <CardContent>
              {adapter.task_history && adapter.task_history.length > 0 ? (
                <Box>
                  <Typography variant="body2">
                    This adapter has been trained on {adapter.task_history.length} task(s).
                  </Typography>
                  <Button 
                    variant="text" 
                    size="small" 
                    sx={{ mt: 1 }}
                  >
                    View Task History
                  </Button>
                </Box>
              ) : (
                <Typography variant="body2" color="textSecondary">
                  This adapter has not been trained on any tasks yet.
                </Typography>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>
      
      <Dialog
        open={deleteDialogOpen}
        onClose={handleDeleteCancel}
      >
        <DialogTitle>Delete Adapter</DialogTitle>
        <DialogContent>
          <DialogContentText>
            Are you sure you want to delete the adapter "{adapter.adapter_name}"? This action cannot be undone.
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button onClick={handleDeleteCancel} disabled={deleting}>
            Cancel
          </Button>
          <Button 
            onClick={handleDeleteConfirm} 
            color="error" 
            disabled={deleting}
            autoFocus
          >
            {deleting ? 'Deleting...' : 'Delete'}
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default AdapterDetails;
